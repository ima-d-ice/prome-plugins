# Prome RAG Chunkers

Five core chunking strategies behind one shared contract, built on top of the
parser ecosystem. Every chunker accepts a `ParsedDocument` (the same object
returned by all Prome parsers) and returns `Chunk` objects with full
provenance back to the parsed blocks.

| Strategy | Location | When to use |
|---|---|---|
| Fixed-size | [`fixed/`](fixed/README.md) | Uniform context windows; predictable sizes; simplest behavior |
| Recursive | [`recursive/`](recursive/README.md) | Structure-respecting splits without relying on block metadata |
| Structure-aware | [`structure_aware/`](structure_aware/README.md) | Parser-extracted headings/blocks; keeps sections, lists and tables intact |
| Hierarchical | [`hierarchical/`](hierarchical/README.md) | Small chunks for retrieval + larger parent chunks for context (parent/child) |
| Semantic | [`semantic/`](semantic/README.md) | Meaning-aware boundaries driven by an injected embedding model |

The first three are pure text strategies and need no dependencies at all:
the base install is dependency-free. The hierarchical chunker is also
dependency-free. The semantic chunker needs an *embedding model*, but not a
particular one — any object implementing the shared
[`embeddings.contracts`](../embeddings/contracts) interface works, and the
chunker never loads a model itself.

## Install

Chunkers live in the same distribution as the parsers. No extra is needed:

```bash
pip install -e .
```

For the optional tokenizer-backed sizing you may pass any `str -> int`
callable (e.g. a `tiktoken` encoder) or install one of your own; nothing is
mandatory.

## Quick start

```python
from parsers.pdf.pymupdf import PyMuPDFParser
from chunkers.fixed import FixedSizeChunker, FixedSizeConfig
from chunkers.recursive import RecursiveChunker, RecursiveConfig
from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig
from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig
from chunkers.semantic import SemanticChunker, SemanticConfig
from embeddings.deterministic import DeterministicEmbeddingModel

doc = PyMuPDFParser().parse("report.pdf")

chunks = FixedSizeChunker().chunk(doc)                                  # defaults
chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=500, overlap=50)).chunk(doc)
chunks = RecursiveChunker(RecursiveConfig(chunk_size=500, overlap=50)).chunk(doc)
chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=500)).chunk(doc)

# Parent/child: index the children, fetch the parent for context.
result = HierarchicalChunker(HierarchicalConfig(parent_size=2000, child_size=500)).chunk(doc)
for child in result.children:
    print(child.id, child.metadata["parent_id"], len(child.text))

# Semantic: boundaries where the topic changes.
chunks = SemanticChunker(
    SemanticConfig(embedding_model=DeterministicEmbeddingModel(), max_chunk_size=800)
).chunk(doc)

for chunk in chunks:
    print(chunk.id, chunk.index + 1, "/", chunk.count, len(chunk.text), chunk.heading)
```

Every strategy accepts the same `ParsedDocument`, **never mutates it**, and is
deterministic for a given document and configuration.

## Shared contract

`chunkers.contracts` defines:

- **`Chunk`** — unique `id` (deterministic, derived from the source
  document and strategy), `doc_id` (= `ParsedDocument.source`), `text`,
  `index` (0-based), `count` (total chunks in the run), `length`
  (measured in `length_unit`), `length_unit`, `block_refs`, optional
  `heading` (section context; `None` for the fixed/recursive strategies),
  and `metadata`.
- **`BlockReference`** — `(page_number, block_index)` identifies the block
  (`block_index` is the index inside `page.blocks` for paged formats, or
  `doc.blocks` for pageless formats), plus `block_type` and the `[char_start,
  char_end)` range of the referenced slice inside the block's `content`.
- **`Chunker`** — runtime-checkable protocol: `chunk(document) -> list[Chunk]`.
- **`TokenCounter`** protocol + built-ins `CharacterCounter` (default, honest
  about measuring characters), `ApproxTokenCounter` (rough ~4 chars/token
  heuristic, never presented as a real token count) and `CallableCounter`
  (adapts any `str -> int`). Pass any plain callable as `tokenizer=` and it
  is wrapped automatically.
- **`flatten_document(doc)`** — canonical text + block-span flattening used
  by all chunkers; assigns each block `[start, end)` offsets into the
  concatenated text. Chunks are exact slices of this text, so coverage is
  verifiable exactly.

Block references and empty content: blocks whose content is empty or all
whitespace are excluded from the flattened text (and therefore from block
references); pageless formats reference `doc.blocks`, paged formats reference
`(page_number, page.blocks[index])`.

## Length counting (characters vs tokens)

`chunk_size` and `overlap` are measured with the configured counter. The
default is **characters** — reported in chunk metadata as
`length_unit="characters"`, never passed off as model tokens. To budget in
tokenizer units:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
config = FixedSizeConfig(chunk_size=512, overlap=64, tokenizer=enc.encode)
```

A plain `str -> int` callable works out of the box; a `TokenCounter`
implementation can be passed instead. `ApproxTokenCounter` is a rough
heuristic for budgeting only — it is not exact for any model.

## Shared chunk metadata

Every built-in chunk sets `source`, `parser_name`, `parser_version`,
`strategy`, `char_start`/`char_end` (offsets into the flattened document
text) and `pages` (sorted page numbers covered). Structure-aware chunks
additionally carry `heading_path` (list of heading texts), `heading_level`
(innermost section level), `block_types` and `block_count`.

The hierarchical chunker adds `level` (`"parent"`/`"child"`), `parent_id`
and `child_index` on children, and `child_ids`/`child_count` on parents.
The semantic chunker adds `embedding_model`, `embedding_dimension` and a
`semantic_config` snapshot (threshold, sizes, overlap, batch size,
structure flag) so any chunk set can be traced back to the model and
settings that produced it.

## Comparison of the five strategies

| | Fixed-size | Recursive | Structure-aware | Hierarchical | Semantic |
|---|---|---|---|---|---|
| Split basis | Character/tokenizer windows | Separator hierarchy, then packing | Parsed block structure (headings, blocks) | Parent level: parsed structure. Child level: separator hierarchy within a parent | Embedding similarity between adjacent segments |
| Respects extracted headings | No | No | Yes | Yes (parents) | Yes (when `respect_structure=True`) |
| Guaranteed uniform size | Best effort (single unbreakable units may exceed) | Best effort; pieces bounded by separators | Best effort; large blocks are split by descending separators | Best effort on both levels | Best effort; a single segment may exceed |
| Keeps tables/lists atomic | No | No | Yes (split only when larger than `chunk_size`) | Yes (children split oversized blocks) | Yes where possible; oversized tables are split on row boundaries |
| Overlap support | Yes | Yes | No (sections are a natural boundary) | Yes, between children | Yes, between chunks |
| Two output levels | No | No | No | Yes (parents + children) | No |
| Needs an embedding model | No | No | No | No | Yes (injected; never loaded internally) |
| Overhead characters | overlap-driven | minimal (boundary packs) | none | overlap-driven | minimal (segment-boundary packing) |
| Best when | you want simple, predictable windows | documents lack reliable block metadata but have natural separators | parsers already produced structure you can trust | a query is specific but the answer needs broader context | the document has no reliable structure but covers several topics |
| Weakness | cuts across sections/lists/tables freely | still cuts across structure | depends on parser heading quality; one bad level flip causes over-splitting | one level only; child level is size-based, not semantic | costs an embedding pass; depends on embedding quality |

### Parent-child vs. semantic chunking

They solve different problems and can be combined.

| | Hierarchical | Semantic |
|---|---|---|
| Question it answers | "How do I give the model more context than the chunk I matched?" | "Where does the topic change?" |
| Output shape | Two levels: parents + children | One flat list of chunks |
| Decides boundaries by | Document structure and size | Embedding similarity |
| Extra cost | None | One embedding pass over the candidate segments |
| Needs an embedding model | No | Yes |
| Typical use | Retrieve small children, answer with the parent | Produce topically coherent chunks for indexing |

A common combination: use the **hierarchical** chunker for the parent level
and the **semantic** chunker for the child level.

See [benchmarks/chunkers/README.md](../benchmarks/chunkers/README.md) for
how to measure on your own corpus. No strategy is universally superior; the
reports describe behavior on a synthetic corpus only.

## Known limitations

- **`length` is not a token count** unless a tokenizer is configured;
  `characters`/`approx_tokens` are reported honestly as such.
- The flattened document joins blocks with a blank line; a document whose
  *pages* and *top-level blocks* are both populated is flattened pages-first,
  mirroring `ParsedDocument.text`.
- The fixed and recursive strategies ignore block types: they will split
  tables, lists and code mid-block. Use the structure-aware chunker when
  block integrity matters.
- Structure-aware treats headings without a parseable `heading_level` as
  level 1 (major section) — correct for PyMuPDF's level-less heading output,
  but coarse if your parser emits many unleveled headings.
- A single unbreakable token/word longer than `chunk_size` is kept whole in
  tokenizer mode (limits are best-effort); in character mode it is cut hard.
- IDs are derived from `ParsedDocument.source` + strategy + index; two
  documents with the same `source` produce overlapping ID spaces.
- Empty or whitespace-only blocks carry no content and are excluded from the
  flattened text and from block references.
- The **hierarchical** chunker is deliberately one level deep; it is not an
  arbitrary-depth tree.
- The **semantic** chunker costs an embedding pass and its boundary quality
  is only as good as the injected embedding model. It never falls back to
  another strategy when inference fails — it raises.
- No vector store, retrieval, reranking or generation is included in any
  strategy.

## Tests

```bash
python -m pytest tests/chunkers -v
```
