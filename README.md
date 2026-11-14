# Prome RAG Plugins

Modular, extensible components for building and experimenting with production
RAG systems. This repository contains the **parser ecosystem** — independent
parsers behind a common contract, covering PDF, HTML, Office (DOCX/PPTX),
spreadsheets (XLSX/CSV), email (EML/MBOX), JSON, XML, source code, images,
audio and video — plus the **core chunking strategies** (fixed-size,
recursive, structure-aware) built on the same contract, plus tests and
benchmark harnesses for each.

The base install is dependency-free; each parser is an optional extra. Selecting
one parser never falls back to another. Stdlib-only parsers (CSV, email,
JSON, XML ElementTree, Python AST) work out of the box.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[pymupdf]'        # PyMuPDF PDF parser
pip install -e '.[pypdf]'          # PyPDF parser (lightweight)
pip install -e '.[docling]'        # Docling parser (large, downloads models)
pip install -e '.[beautifulsoup]'  # BeautifulSoup HTML parser
pip install -e '.[trafilatura]'    # Trafilatura HTML parser
pip install -e '.[docx]'           # python-docx DOCX parser
pip install -e '.[pptx]'           # python-pptx PPTX parser
pip install -e '.[office]'         # both python office parsers
pip install -e '.[spreadsheets]'   # openpyxl + pandas XLSX parsers
pip install -e '.[xml]'            # lxml XML parser (ElementTree is stdlib)
pip install -e '.[code]'           # tree-sitter multi-language parser
pip install -e '.[images]'         # Tesseract OCR (pytesseract + Pillow)
pip install -e '.[paddleocr]'      # PaddleOCR (large; NOT in `images`)
pip install -e '.[audio]'          # faster-whisper transcription
pip install -e '.[video]'          # PyAV/ffmpeg video parser
pip install -e '.[qdrant]'         # Qdrant vector store backend
pip install -e '.[lancedb]'        # LanceDB vector store backend
pip install -e '.[faiss]'          # FAISS vector store backend
pip install -e '.[pgvector]'       # PostgreSQL (pgvector) vector store backend
pip install -e '.[dev]'            # pytest + most parsers (test suite)
pip install -e '.[all]'            # everything except paddleocr
```

Notes on installation:

- **Tesseract** requires the system binary and language packs (e.g.
  `brew install tesseract` / `apt-get install tesseract-ocr`). pip does not
  install it.
- **Audio**: model weights are downloaded on first `parse()`, not at install
  time. Default model is `tiny` (~75MB); larger models are bigger downloads.
  No GPU is required (CPU + int8 works).
- **Docling** (PDF/DOCX/PPTX) downloads model weights on first use.
- **PaddleOCR** is intentionally excluded from `images` and `all`; install it
  explicitly as `[paddleocr]` if you want it. It is not installed/tested in
  this environment; the parser reports itself as unavailable without it.
- **Video**: PyAV wheels bundle ffmpeg — no system ffmpeg needed. (System
  ffmpeg is only used to regenerate benchmark fixtures.)
- Email, JSON, CSV and Python-AST parsers use only the standard library.

## Usage

```python
from parsers.pdf.pymupdf import PyMuPDFParser
from parsers.html.beautifulsoup import BeautifulSoupParser
from parsers.office.docx.python_docx import PythonDocxParser
from parsers.office.pptx.python_pptx import PythonPptxParser
from parsers.spreadsheets.xlsx.openpyxl import OpenpyxlParser
from parsers.spreadsheets.csv import CsvParser
from parsers.email.eml import EmlParser
from parsers.json import JsonParser
from parsers.xml.elementtree import ElementTreeParser
from parsers.code.python_ast import PythonAstParser
from parsers.images.tesseract import TesseractParser
from parsers.audio.faster_whisper import FasterWhisperParser
from parsers.video.ffmpeg import FFmpegVideoParser
```

All parsers return the same `ParsedDocument` (see `parsers/contracts/models.py`):
paged formats keep `pages[].blocks[]`; pageless formats place blocks on
`doc.blocks` directly with `page_number=None` and `doc.pages == []`.
`block_type` is `text | heading | table | image | list | code | link | record |
key_value | attachment | definition | segment | frame | other`, plus `content`,
optional `bbox`, per-block `metadata` (e.g. `dom_order`, `heading_level`,
`href`, `row_cells`, `start`/`end` for audio segments, `timestamp` for video
frames, `line_start`/`line_end` for code definitions, cell coordinates/types
for spreadsheets), `warnings`, and per-document `parser_name`/`parser_version`.

Per-family usage details, extracted fields, and limitations live in each
parser's README (e.g. `parsers/audio/faster_whisper/README.md`).

### Error handling

All parsers raise subclasses of `parsers.contracts.ParserError`:

| Exception | Cause |
|---|---|
| `ParserDependencyError` | Optional dependency not installed (message includes install instructions) |
| `CorruptDocumentError` | Unreadable / invalid document, or no usable stream/content (e.g. audio-only MP4 for the video parser) |
| `EmptyDocumentError` | Zero-byte file or no extractable/decodable content |
| `EncryptedDocumentError` | Password-protected PDF (PyMuPDF/PyPDF raise this exactly; Docling raises it via message detection, otherwise `CorruptDocumentError`) |
| `FileNotFoundError` | Missing source path |

### Integration example (video → audio)

The video and audio parsers are independent; bridge them explicitly if needed:

```python
video = FFmpegVideoParser()
wav = video.extract_audio_to_wav("clip.mp4", "/tmp/clip.wav")  # 16 kHz mono PCM
audio_doc = FasterWhisperParser(model_size="tiny").parse(wav)
```

## Chunkers

Five core chunking strategies behind a shared contract (`chunkers/contracts/`):
all accept a `ParsedDocument` and return `Chunk` objects with provenance back
to the parsed blocks (page numbers, block indices, sub-block character
ranges). Deterministic; dependency-free by default; token counting is
pluggable (`tokenizer=`) and never claimed to be exact when it is not.

| Strategy | What it does |
|---|---|
| Fixed-size | Uniform character/tokenizer windows |
| Recursive | Splits along a separator hierarchy, then packs |
| Structure-aware | Uses the parser's headings and blocks; keeps sections, lists and tables intact |
| Hierarchical | Parent/child: small chunks to index, larger parent chunks as context |
| Semantic | Meaning-aware boundaries from a pluggable embedding model |

The semantic chunker needs an embedding model, but not a particular one: it
depends only on the shared `embeddings/contracts` interface and never loads a
model itself.

```python
from parsers.pdf.pymupdf import PyMuPDFParser
from chunkers.fixed import FixedSizeChunker, FixedSizeConfig
from chunkers.recursive import RecursiveChunker, RecursiveConfig
from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig
from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig
from chunkers.semantic import SemanticChunker, SemanticConfig
from embeddings.deterministic import DeterministicEmbeddingModel

doc = PyMuPDFParser().parse("report.pdf")
chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=1000, overlap=200)).chunk(doc)
chunks = RecursiveChunker(RecursiveConfig(chunk_size=1000, overlap=200)).chunk(doc)
chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=1000)).chunk(doc)

# Parent/child: index the children, fetch the parent for context.
result = HierarchicalChunker(
    HierarchicalConfig(parent_size=2000, child_size=500)
).chunk(doc)

# Semantic: boundaries where the topic changes.
chunks = SemanticChunker(
    SemanticConfig(embedding_model=DeterministicEmbeddingModel(), max_chunk_size=1000)
).chunk(doc)
```

See `chunkers/README.md` for the shared contract, `chunkers/<strategy>/README.md`
for per-strategy details, and `benchmarks/chunkers/` for the comparison
harness (`python benchmarks/chunkers/benchmark.py`).

## Vector stores

Backend-independent contracts for storing and searching chunk embeddings
(`vector_stores/contracts/`), plus four adapters: **Qdrant**, **LanceDB**,
**FAISS** and **PostgreSQL/pgvector**. A vector store accepts precomputed
embeddings only - it never embeds text and never loads a model.

```python
from vector_stores.contracts import CollectionConfig, SearchRequest, VectorSpaceConfig
from vector_stores.qdrant import QdrantVectorStore

store = QdrantVectorStore.local("/tmp/qdrant")   # or .memory() / .server(url)
store.create_collection(config)                  # one or more named vector spaces
store.upsert("docs", records)                    # StoredRecord objects (chunk + vectors)
hits = store.search(SearchRequest(collection="docs", space="dense", dense=vector, limit=10))
```

| Backend | Install extra | Dense | ANN | Sparse | Multi-vector | Persistence |
|---|---|---|---|---|---|---|
| Qdrant | `qdrant` | yes | HNSW | native | native max-sim | memory / local / server |
| LanceDB | `lancedb` | yes | IVF (incl. IVF_HNSW) | stored only | native max-sim | memory / local |
| FAISS | `faiss` | yes | HNSW / IVF | no | no | memory / local (FAISS + SQLite) |
| pgvector | `pgvector` | yes | HNSW / IVFFlat | native (pgvector ≥ 0.7) | stored only | server (PostgreSQL) |

A collection declares one or more **named vector spaces**; each space fixes
the vector kind (dense / sparse / multi-vector), the width, the distance
metric and the embedding space (model + revision) allowed in it. Vectors of
another model are rejected even when the dimensions match. The original
chunk id is never replaced: when a backend needs its own identifier
(Qdrant uses UUIDs, FAISS uses integers) the id is mapped deterministically
and the chunk id is kept in the payload.

All four adapters run the same conformance suite
(`tests/vector_stores/conformance.py`); the differences in the table above
are real differences between the backends.

See `vector_stores/README.md` for the contracts,
`vector_stores/<backend>/README.md` for the adapters, and
`benchmarks/vector_stores/` for the benchmark harness
(`python benchmarks/vector_stores/benchmark.py --backend lancedb`).

## Retrieval

Retrieval turns a query into ranked passages. The retrieval layer reuses the
contracts above and never duplicates them: a chunk keeps its identity from
`chunkers`, the embeddings come from an injected `EmbeddingModel`, and the
storage and the search come from the `VectorStore` protocol. Nothing in the
retrieval package loads a model or downloads weights.

Two strategies are implemented and both expose the same standardized result
interface:

* **Hybrid** — dense semantic retrieval (Qwen3-Embedding-0.6B) fused with
  BM25 lexical retrieval (SQLite FTS5) through Reciprocal Rank Fusion.
  This is the general-purpose strategy; use it when queries mix
  natural-language questions with exact identifiers, error codes or API
  names.
* **Hierarchical** — a coarse-to-fine strategy that first selects the
  parent *sections* that look relevant, then retrieves the evidence
  *children* inside those sections only.

```python
from embeddings.qwen3 import Qwen3EmbeddingModel
from retrieval.strategies.hybrid import HybridRetrievalStrategy
from retrieval.lexical.bm25.sqlite_fts5 import SQLiteFTS5Index, BM25Retriever
from retrieval.dense.qwen import QwenDenseRetriever
from vector_stores.qdrant import QdrantVectorStore

model = Qwen3EmbeddingModel()
store = QdrantVectorStore.local("/tmp/qdrant")
# (index the corpus into `store` and into the lexical index first)

index = SQLiteFTS5Index("/tmp/lexical.db").open()
dense = QwenDenseRetriever(store, model, collection="docs")
lexical = BM25Retriever(index)
strategy = HybridRetrievalStrategy(dense, lexical)

for result in strategy.retrieve("how are the signing keys rotated?").results:
    print(result.rank, result.text)
```

The hybrid engine keeps the two channels independent: each one applies the
same metadata filter to its own candidates, and a channel failure is reported
loudly (strict by default) rather than hidden.

```python
from embeddings.qwen3 import Qwen3EmbeddingModel
from retrieval.strategies.hierarchical import (
    HierarchicalConfig, HierarchicalIndexer, HierarchicalRetrievalStrategy,
)
from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig
from vector_stores.qdrant import QdrantVectorStore

model = Qwen3EmbeddingModel()
store = QdrantVectorStore.local("/tmp/qdrant")

# Index both levels of the hierarchy.
indexer = HierarchicalIndexer(store, model, collection="docs")
indexer.ensure_collection()
chunker = HierarchicalChunker(ChunkerConfig(parent_size=2000, child_size=500))
indexer.index_result(chunker.chunk(document))

# Retrieve.
strategy = HierarchicalRetrievalStrategy.create(
    store, model, collection="docs",
    config=HierarchicalConfig(parent_top_k=8, child_top_k=40, final_top_k=10),
)
for result in strategy.retrieve("how are the signing keys rotated?").results:
    print(result.rank, result.text)              # the evidence (a child chunk)
    print(result.parent_context.text)           # the section it sits in
```

Both levels live in the same collection and the same named dense vector
space; the `chunk_level` metadata field tells them apart. The result keeps
the evidence (the matched child) and the context (the parent section and,
optionally, the neighbouring children) apart, so a parent is never returned
as if it were a precise match.

See `retrieval/README.md` for the full contracts,
`benchmarks/retrieval/hybrid/` for the hybrid evaluation and benchmark
harnesses, and `benchmarks/retrieval/hierarchical/` for the hierarchical
evaluation and benchmark harnesses.

## Layout

```text
parsers/
├── contracts/          # shared models, Parser protocol, exceptions, source validation
├── pdf/
│   ├── pymupdf/        # fast digital-PDF extraction
│   ├── pypdf/          # lightweight extraction
│   ├── docling/        # layout-aware extraction
│   └── ocr/            # reserved (not implemented)
├── html/
│   ├── beautifulsoup/  # full-DOM HTML extraction (structure-preserving)
│   └── trafilatura/    # main-content extraction (boilerplate removal)
├── office/
│   ├── docx/
│   │   ├── python_docx/    # native Word extraction
│   │   └── docling/        # Docling DOCX engine
│   └── pptx/
│       ├── python_pptx/    # native PowerPoint extraction (slides preserved)
│       └── docling/        # Docling PPTX engine (slide-per-page)
├── spreadsheets/
│   ├── xlsx/
│   │   ├── openpyxl/       # structure-preserving cells/types/formulas/merges
│   │   └── pandas/         # dataframe-oriented extraction
│   └── csv/                # stdlib csv, streaming, configurable delimiter/encoding
├── email/
│   ├── eml/                # stdlib email: headers, bodies, MIME, attachments
│   └── mbox/               # stdlib mailbox: multi-message .mbox archives
├── json/                   # stdlib JSON (+ NDJSON/JSONL mode), path-preserving
├── xml/
│   ├── elementtree/        # stdlib ElementTree
│   └── lxml/               # lxml: XPath, richer namespace handling
├── code/
│   ├── python_ast/         # stdlib ast: defs, docstrings, line ranges
│   └── tree_sitter/        # tree-sitter: Python/JS/TS/Java/C/C++/Go
├── images/
│   ├── tesseract/          # Tesseract OCR (text, bboxes, confidence)
│   └── paddleocr/          # PaddleOCR (not installed/tested here)
├── audio/
│   └── faster_whisper/     # faster-whisper transcription (SEGMENT blocks)
├── video/
│   └── ffmpeg/             # PyAV: metadata + selective FRAME blocks
└── chunkers/
    ├── contracts/          # shared Chunk/BlockReference models, Chunker protocol,
    │                       # counters (characters/approx/pluggable tokenizer), flattening
    ├── fixed/              # fixed-size windows (word-boundary aware, overlap)
    ├── recursive/          # separator-hierarchy splitting + greedy packing
    ├── structure_aware/    # heading/section-aware, tables & lists kept atomic
    ├── hierarchical/       # parent/child: small chunks to index + parent chunks as context
    ├── semantic/           # meaning-aware boundaries via a pluggable embedding model
└── vector_stores/
    ├── contracts/          # shared collection/record/filter/index contracts,
    │                       # the VectorStore protocol, capability descriptors
    ├── qdrant/             # Qdrant adapter (dense, sparse, multi-vector, filters)
    ├── lancedb/            # LanceDB adapter (embedded local, IVF indexes)
    ├── faiss/              # FAISS adapter (dense vectors + SQLite records)
    └── pgvector/           # PostgreSQL adapter (pgvector, HNSW/IVFFlat, sparsevec)
└── retrieval/
    ├── contracts/          # shared RetrievalQuery/RetrievedItem/Response/Diagnostics
    │                       # and the Retriever protocol + capabilities
    ├── dense/              # DenseRetriever: embed + filtered search, capabilities checked
    │   └── qwen/           # Qwen3-Embedding-0.6B production channel
    ├── lexical/bm25/sqlite_fts5/   # BM25 over a SQLite FTS5 index
    ├── fusion/rrf/         # Reciprocal Rank Fusion (default k=60)
    ├── strategies/
    │   ├── hybrid/         # dense + BM25, fused with RRF (general purpose)
    │   └── hierarchical/   # coarse-to-fine: select parent sections, then the
    │                       # evidence children inside them
    ├── postprocessing/
    │   ├── deduplication/  # merge repeated chunk ids, keep best-ranked
    │   ├── adjacent_merge/ # merge adjacent chunks (configurable)
    │   └── context_expansion/  # attach the parent section and the neighbours
    └── indexing/           # coordinator: writes the same chunks into both indexes

tests/parsers/              # per-family pytest suites
tests/chunkers/             # chunker pytest suites + fixture integration tests
tests/embeddings/           # embedding contract + adapter suites
tests/vector_stores/        # vector-store contract + Qdrant adapter suites
tests/retrieval/            # retrieval contract + hierarchical strategy suites
benchmarks/parsers/         # per-family fixture corpora, runners, reports
benchmarks/chunkers/        # shared chunker comparison harness + reports
benchmarks/vector_stores/   # vector-store search/upsert harness + reports
benchmarks/retrieval/       # hierarchical retrieval evaluation + benchmark harness
```

## Capability matrix

Verified capabilities only; "—" means not applicable/supported. Unsupported
features are reported as N/A in benchmarks, never as a fake zero.

| Format | Implementation | Structure | Tables | Media | Metadata | Install extra |
|---|---|---|---|---|---|---|
| PDF | PyMuPDF | Yes | Yes | Image refs | Yes | `pymupdf` |
| PDF | PyPDF | Basic | N/A | No | Yes | `pypdf` |
| PDF | Docling | Yes | Yes | Yes | Yes | `docling` |
| HTML | BeautifulSoup | Yes | Yes | Image/link refs | Yes | `beautifulsoup` |
| HTML | Trafilatura | Main content | Partial | N/A | Yes | `trafilatura` |
| DOCX | python-docx | Yes | Yes | Image refs | Yes | `docx` |
| DOCX | Docling | Yes | Yes | Refs | Yes | `docling` |
| PPTX | python-pptx | Yes (slides) | Yes | Image refs | Yes | `pptx` |
| PPTX | Docling | Yes (slide pages) | Yes | Image refs | Yes | `docling` |
| XLSX | openpyxl | Yes (cells+types) | Yes | N/A | Yes | `spreadsheets` |
| XLSX | pandas | Tabular | N/A (merged/format) | N/A | Limited | `spreadsheets` |
| CSV | stdlib | Rows/cols | N/A | N/A | N/A | stdlib |
| EML | stdlib email | Headers/body/MIME | N/A | Attachment refs | Yes | stdlib |
| MBOX | stdlib mailbox | Same as EML | N/A | Attachment refs | Yes | stdlib |
| JSON | stdlib | Full hierarchy | N/A | N/A | N/A | stdlib |
| XML | ElementTree | Hierarchy+attrs | N/A | N/A | Partial | stdlib |
| XML | lxml | Hierarchy+attrs+ns | N/A | N/A | Yes | `xml` |
| Code | Python AST | Defs/scopes | N/A | N/A | N/A | stdlib |
| Code | Tree-sitter | Defs (7 langs) | N/A | N/A | N/A | `code` |
| Images | Tesseract | Text lines+bboxes | N/A | N/A | Yes | `images` |
| Images | PaddleOCR | Text regions+bboxes | N/A | N/A | Yes | `paddleocr` |
| Audio | faster-whisper | Segments+timestamps | N/A | N/A | Yes | `audio` |
| Video | PyAV | Metadata+selected frames | N/A | Audio stream info | Yes | `video` |

## Tests

```bash
pip install -e '.[dev]'
python -m pytest tests/
```

Parsers whose optional dependency (or external binary/model) is missing skip
cleanly instead of failing. The audio suite needs the `tiny` model (downloaded
on first use, ~75MB); without network access those tests skip. The vector
store suite needs `qdrant-client` (`[qdrant]`); the server-only tests skip
when no Qdrant server is reachable.

**The FAISS suite runs on its own** (`pytest tests/vector_stores/faiss`):
the `faiss-cpu` wheel and torch each link their own copy of the OpenMP
runtime, so they cannot share a process on macOS. In a combined run the
FAISS tests skip with an explicit reason instead of aborting the
interpreter.

Regenerate fixtures if missing (deterministic scripts live next to each
benchmark suite):

```bash
python benchmarks/parsers/<family>/generate_fixtures.py
```

## Benchmarks

Every family has a benchmark suite under `benchmarks/parsers/<family>/` with
synthetic fixtures, ground-truth expectations, a runner, and JSON/CSV reports
(previous reports archived to `reports/archive/`). The chunker comparison
harness lives in `benchmarks/chunkers/` and reports speed and quality
separately on the same fixture corpus. The vector-store harness lives in
`benchmarks/vector_stores/` and reports search latency (p50/p95) and the
recall of the approximate path against the exact baseline separately — no
combined score.

```bash
python benchmarks/parsers/pdf/benchmark.py --repeats 3
python benchmarks/parsers/html/benchmark.py --repeats 3
python benchmarks/parsers/office/benchmark.py
python benchmarks/parsers/spreadsheets/benchmark.py
python benchmarks/parsers/email/benchmark.py
python benchmarks/parsers/json/benchmark.py
python benchmarks/parsers/xml/benchmark.py
python benchmarks/parsers/code/benchmark.py
python benchmarks/parsers/images/benchmark.py
python benchmarks/parsers/audio/benchmark.py        # needs [audio], downloads tiny model
python benchmarks/parsers/video/benchmark.py
python benchmarks/vector_stores/benchmark.py --backend qdrant  --mode memory
python benchmarks/vector_stores/benchmark.py --backend lancedb --mode local
python benchmarks/vector_stores/benchmark.py --backend faiss    --mode local
python benchmarks/vector_stores/benchmark.py --backend pgvector --mode server
python benchmarks/retrieval/hybrid/evaluate.py                 # hybrid quality
python benchmarks/retrieval/hybrid/evaluate.py --embedding qwen
python benchmarks/retrieval/hybrid/benchmark.py                  # hybrid performance
python benchmarks/retrieval/hierarchical/evaluate.py             # hierarchical quality
python benchmarks/retrieval/hierarchical/evaluate.py --embedding qwen
python benchmarks/retrieval/hierarchical/benchmark.py             # hierarchical performance
```

Unsupported capabilities are reported as N/A (not 0); parsers that fail a
fixture are reported as failed. Latest measured runs on synthetic fixtures
(macOS arm64, 2026-11-12; see each suite's `reports/` for raw numbers):

| Family | Parser | Success | Quality |
|---|---|---|---|
| PDF | PyMuPDF | 11/11 | 1.00 |
| PDF | PyPDF | 11/11 | 1.00 (structure N/A) |
| PDF | Docling | 11/11 | 0.946 |
| HTML | BeautifulSoup | 12/12 | 0.975 |
| HTML | Trafilatura | 12/12 | 0.967 |
| Office | python-docx | 1.00 | 1.00 |
| Office | Docling DOCX | 1.00 | 1.00 |
| Office | python-pptx | 1.00 | 1.00 |
| Office | Docling PPTX | 1.00 | 0.92 |
| Spreadsheets | openpyxl / pandas / stdlib CSV | 1.00 | 1.00 |
| Email | EML / MBOX | 1.00 | 1.00 |
| JSON | stdlib | 1.00 | 1.00 |
| XML | ElementTree / lxml | 1.00 | 1.00 |
| Code | Python AST / Tree-sitter | 1.00 | 1.00 |
| Images | Tesseract | 1.00 | 1.00 |
| Images | PaddleOCR | skipped (not installed) | — |
| Audio | faster-whisper (tiny) | 1.00 | 1.00 (WER 0.07 on English fixtures) |
| Video | PyAV | 1.00 | 1.00 |

These are tiny synthetic documents; treat the numbers as regression indicators,
not production baselines.

## Design decisions

- **Common contract, no framework**: plain dataclasses (`ParsedDocument` /
  `ParsedPage` / `ParsedBlock`) and a runtime-checkable `Parser` protocol. No
  factories, registries or inheritance hierarchies.
- **Lazy optional dependencies**: each parser imports its library inside
  `parse()` and raises `ParserDependencyError` with install instructions;
  contracts import nothing third-party. Stdlib-only parsers (CSV, email, JSON,
  ElementTree, Python AST) have no third-party dependency at all.
- **No silent fallback**: each parser uses only its own backend.
- **Contract extensions are additive**: `CODE`, `LINK`, `RECORD`, `KEY_VALUE`,
  `ATTACHMENT`, `DEFINITION`, `SEGMENT`, `FRAME` block types were added for the
  new families; existing PDF/HTML behavior is unchanged.
- **Structured, not flattened**: spreadsheet cells keep coordinates/types;
  JSON/XML keep paths; code definitions keep line ranges and source text;
  audio keeps timestamps; video keeps frame numbers/timestamps; email keeps
  MIME/attachment relationships. Spreadsheets are never blindly stringified
  into "text chunks".
- **Warnings, not exceptions**: individually blank/undecodable units (e.g. a
  no-speech audio file) yield warnings or empty output, never invented content.
- **Benchmarks are synthetic and deterministic**: fixtures are generated by
  scripts; binary fixtures (audio/video) are committed.
- **Model/weight downloads are explicit**: faster-whisper and Docling download
  weights on first use only; nothing downloads at import or install time.
- **faster-whisper + PyAV ≥ 19 compatibility**: the audio parser decodes to
  16 kHz mono float32 itself and passes a NumPy array to faster-whisper,
  bypassing the `metadata_errors` kwarg that PyAV 19 removed. No monkey-patching.
- **Video frame extraction never emits every frame by default**: interval
  strategy (default, 1s) or hard-cut scene detection, both capped by
  `max_frames`. Decoding is sequential.
- **Hallucinated content is never produced**: no-speech audio → no blocks +
  warning; missing language confidence → reported as measured; diarization is
  never faked.

## Limitations / deferred work

- OCR parsers for PDFs (`parsers/pdf/ocr/`) are not implemented.
- **PaddleOCR is not installed or benchmarked** in this environment; its parser
  exists and is covered by tests that skip without the dependency.
- No reranking, generation, API server, database or frontend. Chunking,
  embedding models, the vector store and the retrieval strategies are
  provided; nothing consumes them yet.
- Retrieval: the **hybrid** (dense + BM25 fused with RRF) and **hierarchical**
  (coarse-to-fine) strategies are implemented and evaluated. A lexical arm
  (BM25) and a fusion step (RRF) are provided by the hybrid engine; the
  hierarchical strategy remains dense-only in this milestone. The hierarchy is
  deliberately one level deep (parent/child), not an arbitrary-depth tree, and
  the context expansion never summarizes or rewrites the source text.
  The hybrid engine writes the same chunks into the dense index (vector store)
  and the lexical index (SQLite FTS5); there is no distributed transaction
  between the two, so the coordinator reports per-backend failures and can
  reconcile the lexical side.
- Vector stores: Qdrant, LanceDB, FAISS and pgvector are implemented. IVF,
  PQ, scalar / binary quantization and custom ANN algorithms are not
  implemented everywhere; BM25 and full-text retrieval are out of scope
  (lexical retrieval layer). The in-process Qdrant modes (`memory`, `local`)
  scan every vector, so its HNSW path and its payload indexes are only
  exercised against a server. FAISS keeps the vectors in SQLite as well, so
  its index can always be rebuilt.
- The hierarchical chunker is deliberately one level deep (parent/child), not
  an arbitrary-depth tree.
- The semantic chunker costs an embedding pass and its boundary quality
  depends entirely on the injected embedding model. It never falls back to
  another strategy when inference fails.
- PyMuPDF heading detection is a font-size heuristic; Docling OCR is disabled.
- Audio: no speaker diarization (faster-whisper cannot; possible future
  extension). `silence.wav` reports `language="nn"` with low probability from
  `tiny` — treat `language_probability` as a signal, not ground truth.
- Video: frame extraction is not visual understanding — no object/scene/action
  semantics. Scene detection is a luma heuristic (hard cuts only). Rotation
  metadata is not surfaced.
- Code: tree-sitter grammar availability per language must be installed
  (`[code]` extra ships Python/JS/TS/Java/C/C++/Go); Python AST covers Python
  only.
- Spreadsheet formulas: openpyxl reports formulas as stored (cached results
  only when `data_only` data is present); pandas does not preserve formulas,
  merged cells, formats or hidden-sheet flags (documented in its README).
- fixtures are synthetic; real-world variety (huge DOMs, long audio, complex
  video) is untested. Benchmarks measure parsing only.
- HTML parsers do not download URLs, execute scripts, render pages or crawl.

## License

MIT
# prome-plugins

## Integration Hardening (feature/integration-hardening)

This branch hardens the `integrations/` package for correct packaging, independent
testability, provenance safety, and reliability across vector stores and observability
configurations.

### Changes

- **Packaging**: `integrations/` added to setuptools package discovery; wheel builds
  correctly with all 6 adapter modules importable from the installed distribution.

- **Provenance Chain**: Heuristic doc_id parsing replaced with explicit `.` delimiter
  requirement (e.g., `doc77.chunk-003`). `source_id`, `doc_id`, `chunk_id` preserved
  separately without heuristic parsing.

- **Deletion Coordination**: `IndexSyncCoordinator` supports `delete_by_document()` when
  available on the vector store, with `delete_by_filter()` fallback when
  `TypeError` is raised. Document ID derived from `source_id` when not supplied.
  Idempotent repeated deletion is safe. Both dense (vector) and lexical (BM25/FTS5)
  indexes are updated.

- **Observability**: Trace ID generation fixed (module-level function scoping);
  `PipelineTracer` generates `trace_id` directly; `record_stage`/`record_error` use
  `hasattr` for `SpanScope` attribute safety. One request ID and one coherent trace
  propagated per pipeline. No-op when observability is not configured.

- **Test Coverage**: 6 adapter modules with ~52 deterministic tests using FakeVectorStore
  doubles. Full suite: 51/52 integration tests pass (1 skipped: ragas optional
  dependency). 414 tests pass across loaders, retrieval, reranking, and observability.

### Packaging Verification

```bash
pip install -e '.[dev]'
python -c "from integrations import parse_loaded_resource, retrieved_item_to_rerank_candidate, reranked_to_generation_context, eval_sample_from_pipeline, experiment_config_from_pipeline, pipeline_outputs_to_eval_sample, IndexSyncCoordinator, ensure_observability, shutdown_observability, set_request_context, get_request_context, span, PipelineTracer, trace_pipeline_stage, trace_request_id, trace_trace_id"
python -m pytest tests/integrations -q  # 51 passed, 1 skipped
```
