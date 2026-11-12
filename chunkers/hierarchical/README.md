# Hierarchical Chunker (parent/child)

Parent-child chunking: **small, precise chunks for retrieval**, backed by
**larger chunks for context**.

A 300-character fragment is good at matching a specific question but often
too thin to answer it. Hierarchical chunking stores the small fragments
(which are what you embed and search) and keeps the larger surrounding
section around, so once a fragment matches, the surrounding context can be
pulled in and handed to the generation model.

## When to use it

Use hierarchical chunking when:

- queries are **specific** and the answer needs **broader context** from
  the same document (e.g. "what was the refund policy?" inside a long terms
  of service);
- you want a **small embedding budget** but still want to show the model
  enough surrounding text;
- the source documents have **clear section structure** (headings, numbered
  sections) that should delimit the context window.

Use the **structure-aware** chunker instead when a single flat list of
section-sized chunks is enough — it is simpler and produces one chunk per
section rather than a two-level hierarchy.

## Output design

`HierarchicalChunker.chunk()` returns a `HierarchicalResult`:

```python
from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig

result = HierarchicalChunker(HierarchicalConfig(parent_size=2000, child_size=500)).chunk(doc)

result.parents    # list[Chunk] — larger, section-aware units (context)
result.children   # list[Chunk] — smaller units (embed and index these)
```

Relationship information is carried in the chunk metadata, in **both**
directions:

| Where | Field | Meaning |
|---|---|---|
| child | `metadata["parent_id"]` | id of the parent this child belongs to |
| child | `metadata["parent_index"]` | index of the parent in `result.parents` |
| child | `metadata["child_index"]` | index of the child within its parent |
| parent | `metadata["child_ids"]` | ids of the children belonging to this parent |
| parent | `metadata["child_count"]` | number of children |

Helpers avoid digging through metadata:

```python
parent = result.parent_of(result.children[0])   # parent Chunk or None
kids   = result.children_of(result.parents[0])   # list[Chunk]
for parent, children in chunker.iter_pairs(doc):
    ...
```

### Indexing only the children

A vector store does **not** need to hold the parents. Index the children
(each carries `parent_id` in its payload); at query time, look up the parent
by id and use it as context:

```python
# Index time
for child in result.children:
    store.upsert(id=child.id, vector=embed(child.text), payload={
        "parent_id": child.metadata["parent_id"],
        "doc_id": child.doc_id,
    })

# Query time (pseudo-code)
hits = store.search(query_vector, top_k=5)
context = "\n\n".join(
    parent_by_id[hit.payload["parent_id"]].text for hit in hits
)
```

## Configuration

```python
HierarchicalConfig(
    parent_size=2000,   # max size of a parent chunk
    child_size=500,     # max size of a child chunk (must be <= parent_size)
    child_overlap=50,   # trailing context carried between children (must be < child_size)
    tokenizer=None,     # None = characters; pass a TokenCounter or `str -> int`
)
```

Sizes are measured with `tokenizer` (characters by default) and reported in
`Chunk.length` / `Chunk.length_unit`, exactly like the other chunkers.
Validation raises `ChunkerConfigError` for invalid sizes, an overlap that is
not smaller than the child size, or a child size larger than the parent size.

## How it works

1. **Parents** are produced with the structure-aware chunker at
   `parent_size`, so a parent respects major section boundaries, carries its
   heading context (`Chunk.heading`, `metadata["heading_path"]`) and never
   orphans a heading from its body.
2. **Children** are carved out of each parent with the shared recursive
   splitter, sized to `child_size` with `child_overlap` of trailing context.
   A child never crosses a parent boundary, so a child is always a
   contiguous piece of exactly one parent.
3. **Linking** is explicit in both directions, and every parent is guaranteed
   to have at least one child (there are no orphaned children and no empty
   parents).

Existing chunking logic is reused rather than duplicated: the parent level
comes from `chunkers.structure_aware`, the child level from the shared
`chunkers._textsplit` primitives.

## Provenance

Both levels keep the same provenance as the other chunkers:

- `Chunk.doc_id` — the source document id (`ParsedDocument.source`);
- `Chunk.block_refs` — the source blocks each chunk was built from, with the
  exact character range inside the block;
- `Chunk.metadata["pages"]` — the pages the chunk touches;
- `Chunk.metadata["char_start"]` / `["char_end"]` — offsets into the
  flattened document text, so `text[char_start:char_end] == chunk.text`
  always holds.

## Guarantees

- deterministic: the same document and configuration always produce the same
  ids and the same text;
- stable ids: `<document-digest>-hierarchical-parent-0000` and
  `<document-digest>-hierarchical-child-0000`;
- the input document is never mutated;
- no content is dropped: the children tile their parent;
- Unicode-safe.

## Example

```python
from parsers.pdf.pymupdf import PyMuPDFParser
from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig

doc = PyMuPDFParser().parse("terms-of-service.pdf")
result = HierarchicalChunker(
    HierarchicalConfig(parent_size=2000, child_size=400, child_overlap=40)
).chunk(doc)

print(f"{len(result.parents)} parents, {len(result.children)} children")
for child in result.children[:5]:
    print(child.id, child.metadata["parent_id"], len(child.text))
```

## Known limitations

- **One level only.** This is a deliberate scope choice: a parent-child
  hierarchy answers most retrieval needs without the complexity of an
  arbitrary-depth tree. Deeper hierarchies are not supported.
- **Parent boundaries are section-driven.** Parents follow the structure-aware
  rules, so a document with no headings produces parents that are pure size
  windows.
- **A single oversized block** (a huge paragraph, a very long table) is
  split by the recursive splitter, so a parent may not correspond to a
  single logical section.
- **Children are sized, not semantic.** The child level uses size-based
  splitting; if you want meaning-aware boundaries, use the
  [semantic chunker](../semantic/README.md) instead (or for the child level).
- No vector store, retrieval, reranking or generation is included.
