# aumai-ragoptimizer

**Benchmark and optimize RAG pipeline configurations.** Compare chunking strategies,
retrieval methods, and measure Precision@k, Recall@k, MRR, and NDCG so you ship the
best-performing retrieval pipeline — not just the first one that seemed to work.

Part of the [AumAI](https://github.com/aumai) open-source agentic AI infrastructure suite.

---

## What is this?

Imagine you are building a search engine for your company's internal documents. You have
thousands of pages of text and you want an AI to answer questions about them accurately.
To do that, you first *chunk* the documents (cut them into smaller pieces), then *retrieve*
the most relevant pieces for each question, and finally feed those pieces to the AI.

The problem is that every single decision in that pipeline matters enormously:

- Should each chunk be exactly 512 characters? 256? 1024?
- Should you cut on paragraph boundaries, sentence boundaries, or fixed character counts?
- Should you retrieve 3 results or 10?

**aumai-ragoptimizer** gives you a systematic, evidence-based way to answer those
questions. You supply your documents and a set of labelled test questions (questions where
you already know which documents are relevant), and the tool measures how well each
configuration actually performs. Then it runs a grid search across all your candidate
configurations and tells you which one is best — complete with precision, recall, MRR,
latency, and cost estimates for every candidate.

Think of it like A/B testing for your retrieval pipeline, automated and quantitative.

---

## Why does this matter?

### The problem from first principles

A RAG (Retrieval-Augmented Generation) pipeline has two failure modes that compound:

1. **Poor retrieval** — the relevant document chunks are not in the top-k results, so the
   language model never sees the information it needs. The model then either hallucinates
   or says "I don't know."
2. **Wrong granularity** — chunks are too large (they dilute the signal with irrelevant
   text) or too small (they lose sentence-level context needed for meaning).

Most teams pick a chunking strategy once, early in development, then never revisit it.
They do not measure retrieval quality independently — they only look at final answer
quality and cannot distinguish a retrieval failure from a generation failure.

aumai-ragoptimizer makes retrieval quality measurable and improvable through standard
information-retrieval metrics:

- **Precision@k** — of the k chunks retrieved, what fraction were actually relevant?
- **Recall@k** — of all the relevant chunks, what fraction did we retrieve?
- **MRR** — Mean Reciprocal Rank: was the most relevant chunk ranked first?
- **NDCG@k** — Normalised Discounted Cumulative Gain: a graded ranking-quality metric.

With those numbers you can iterate scientifically rather than by gut feeling.

---

## Architecture

```mermaid
flowchart TD
    A[Source Documents\ncorpus.jsonl] --> B[ChunkerFactory]
    Q[Labelled Queries\nqueries.jsonl] --> E

    B --> C{ChunkingStrategy}
    C -->|fixed_size| D1[FixedSizeChunker]
    C -->|sentence| D2[SentenceChunker]
    C -->|paragraph| D3[ParagraphChunker]
    C -->|recursive| D4[RecursiveChunker]
    C -->|semantic| D2

    D1 & D2 & D3 & D4 --> F[Chunk Index\nchunk_to_doc mapping]
    F --> E[Retrieval Simulation\ntoken-overlap scoring]
    E --> G[RAGBenchmark.run\(\)]

    G --> H[RAGBenchmarkResult\nPrecision · Recall · MRR\nLatency · Cost]

    I[search_space: list of RAGConfig] --> J[RAGOptimizer.optimize\(\)]
    J -->|evaluates each config| G
    J --> K[OptimizationResult\nbest_config · all_results\nimprovement_pct]
```

---

## Features

| Feature | Description |
|---|---|
| Four chunking strategies | Fixed-size windows, sentence-grouped, paragraph-based, and recursive hierarchical splitting — each with configurable size and overlap |
| `TextChunker` protocol | Any class with a `chunk(text) -> list[str]` method is a valid chunker — bring your own |
| `RAGBenchmark.run()` | Single-config evaluation: Precision@k, Recall@k, MRR, latency (ms), and cost estimate |
| `RAGOptimizer.optimize()` | Grid search across any list of `RAGConfig` objects; scoring formula: 0.4×Precision + 0.4×Recall + 0.2×MRR |
| Four IR metrics | `precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k` available as standalone functions |
| Pydantic v2 models | All inputs and outputs are fully validated; serialise/deserialise via `.model_dump()` / `.model_validate()` |
| CLI interface | `benchmark`, `optimize`, and `chunk` commands with JSONL I/O and JSON/YAML config support |

---

## Quick Start

### Install

```bash
pip install aumai-ragoptimizer
```

### Prepare your data

**corpus.jsonl** — one document per line with a `text` field:

```json
{"text": "Retrieval-Augmented Generation (RAG) combines a retrieval step with an LLM."}
{"text": "Chunking strategies determine how documents are split before indexing."}
{"text": "Dense retrieval uses embedding similarity; sparse retrieval uses term frequency."}
```

**queries.jsonl** — one query per line with ground-truth corpus indices:

```json
{"query": "What is RAG?", "relevant_docs": [0]}
{"query": "How does chunking work?", "relevant_docs": [1]}
{"query": "dense vs sparse retrieval", "relevant_docs": [2]}
```

### Run a benchmark in under 30 seconds

```bash
# Benchmark the sentence strategy at top-3 retrieval
aumai-ragoptimizer benchmark \
  --corpus corpus.jsonl \
  --queries queries.jsonl \
  --strategy sentence \
  --top-k 3

# Find the best configuration automatically
aumai-ragoptimizer optimize \
  --corpus corpus.jsonl \
  --queries queries.jsonl \
  --output results.json
```

---

## CLI Reference

### `aumai-ragoptimizer benchmark`

Run a single RAG configuration against a labelled corpus and print metrics.

```
Usage: aumai-ragoptimizer benchmark [OPTIONS]

Options:
  --config PATH     YAML or JSON file with a RAGConfig definition.
  --corpus PATH     JSONL file with documents (field: 'text').         [required]
  --queries PATH    JSONL file with queries ('query', 'relevant_docs').[required]
  --top-k INT       Number of results to retrieve.          [default: 5]
  --strategy TEXT   Chunking strategy: fixed_size | sentence | paragraph
                    | semantic | recursive              [default: fixed_size]
  --output PATH     Output file for results JSON. '-' for stdout. [default: -]
  --version         Show version and exit.
  --help            Show this message and exit.
```

**Example — use sentence chunking with top-5 retrieval:**

```bash
aumai-ragoptimizer benchmark \
  --corpus docs.jsonl \
  --queries eval.jsonl \
  --strategy sentence \
  --top-k 5
```

**Example — load a full config from a JSON file:**

```bash
aumai-ragoptimizer benchmark \
  --config my_config.json \
  --corpus docs.jsonl \
  --queries eval.jsonl \
  --output results.json
```

Config file (`my_config.json`):

```json
{
  "chunking_strategy": "recursive",
  "chunk_size": 512,
  "chunk_overlap": 64,
  "retrieval_method": "dense",
  "top_k": 5,
  "embedding_model": "text-embedding-3-small"
}
```

YAML config is also supported when `PyYAML` is installed:

```yaml
chunking_strategy: recursive
chunk_size: 512
chunk_overlap: 64
retrieval_method: dense
top_k: 5
```

---

### `aumai-ragoptimizer optimize`

Grid-search across 24 built-in configurations (3 strategies × 2 sizes × 2 overlap
settings × 2 retrieval methods) and report the winner.

```
Usage: aumai-ragoptimizer optimize [OPTIONS]

Options:
  --corpus PATH    JSONL file with documents.                  [required]
  --queries PATH   JSONL file with queries and ground truth.   [required]
  --output PATH    Output file for the optimization result JSON. [default: -]
  --help           Show this message and exit.
```

**Example:**

```bash
aumai-ragoptimizer optimize \
  --corpus corpus.jsonl \
  --queries queries.jsonl \
  --output opt_results.json
```

Sample stderr output:

```
Optimizing over 24 configurations...
Best config: strategy=sentence size=512 overlap=64 retrieval=dense  improvement=18.3%
```

---

### `aumai-ragoptimizer chunk`

Split a text file using any strategy and emit the chunks as JSONL for inspection.

```
Usage: aumai-ragoptimizer chunk [OPTIONS]

Options:
  --strategy TEXT       Chunking strategy (required).
  --input PATH          Input text file to chunk.           [required]
  --chunk-size INT      Maximum characters per chunk.       [default: 512]
  --chunk-overlap INT   Character overlap between chunks.   [default: 64]
  --output PATH         Output JSONL file. '-' for stdout.  [default: -]
  --help                Show this message and exit.
```

**Example:**

```bash
# Inspect how recursive chunking would split a document
aumai-ragoptimizer chunk \
  --strategy recursive \
  --input my_document.txt \
  --chunk-size 256 \
  --chunk-overlap 32

# Save chunks to a file for further inspection
aumai-ragoptimizer chunk \
  --strategy paragraph \
  --input my_document.txt \
  --output chunks.jsonl
```

Each output line is a JSON object:

```json
{"index": 0, "chunk": "First chunk text...", "length": 247}
{"index": 1, "chunk": "Second chunk text...", "length": 198}
```

---

## Python API

### Benchmark a single configuration

```python
from aumai_ragoptimizer import (
    RAGBenchmark, RAGConfig, ChunkingStrategy, RetrievalMethod
)

corpus = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Chunking splits documents into smaller pieces for indexing.",
    "BM25 is a classical sparse retrieval algorithm based on term frequency.",
]
queries = ["What does RAG stand for?", "How do you split documents?"]
ground_truth = [[0], [1]]

config = RAGConfig(
    chunking_strategy=ChunkingStrategy.sentence,
    chunk_size=512,
    chunk_overlap=64,
    retrieval_method=RetrievalMethod.dense,
    top_k=3,
)

benchmark = RAGBenchmark()
result = benchmark.run(config, corpus, queries, ground_truth)

print(f"Precision@3: {result.precision_at_k}")
print(f"Recall@3:    {result.recall_at_k}")
print(f"MRR:         {result.mrr}")
print(f"Latency:     {result.latency_ms:.2f} ms")
print(f"Cost/query:  ${result.cost_per_query:.6f}")
```

### Run a grid search

```python
from aumai_ragoptimizer import RAGOptimizer, RAGConfig, ChunkingStrategy

search_space = [
    RAGConfig(chunking_strategy=ChunkingStrategy.fixed_size, chunk_size=256, top_k=5),
    RAGConfig(chunking_strategy=ChunkingStrategy.sentence,   chunk_size=512, top_k=5),
    RAGConfig(chunking_strategy=ChunkingStrategy.paragraph,  chunk_size=1024, top_k=3),
    RAGConfig(
        chunking_strategy=ChunkingStrategy.recursive,
        chunk_size=512,
        chunk_overlap=64,
        top_k=5,
    ),
]

optimizer = RAGOptimizer()
result = optimizer.optimize(corpus, queries, ground_truth, search_space)

print(f"Best strategy:   {result.best_config.chunking_strategy.value}")
print(f"Best chunk size: {result.best_config.chunk_size}")
print(f"Improvement:     {result.improvement_pct:.1f}%")

# Inspect all results
for r in result.all_results:
    score = 0.4 * r.precision_at_k + 0.4 * r.recall_at_k + 0.2 * r.mrr
    print(f"  {r.config.chunking_strategy.value:12s} size={r.config.chunk_size} "
          f"score={score:.4f}")
```

### Use a chunker directly

```python
from aumai_ragoptimizer import RecursiveChunker, SentenceChunker, FixedSizeChunker

text = open("my_document.txt").read()

# Recursive splitting — best for mixed-format documents
chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64)
chunks = chunker.chunk(text)
print(f"Recursive: {len(chunks)} chunks")

# Sentence-aware splitting
chunker = SentenceChunker(chunk_size=400, chunk_overlap=0)
chunks = chunker.chunk(text)
print(f"Sentence: {len(chunks)} chunks")

# Simple fixed-size
chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=32)
chunks = chunker.chunk(text)
print(f"Fixed: {len(chunks)} chunks")
```

### Use metrics as standalone functions

```python
from aumai_ragoptimizer import (
    precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k
)

retrieved = [2, 0, 5, 1, 3]
relevant  = [0, 1]

print(precision_at_k(retrieved, relevant, k=5))    # 0.4
print(recall_at_k(retrieved, relevant, k=5))        # 1.0
print(mean_reciprocal_rank(retrieved, relevant))     # 0.5
print(ndcg_at_k(retrieved, relevant, k=5))           # ~0.7653
```

---

## Configuration Options

### `RAGConfig` fields

| Field | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `chunking_strategy` | `ChunkingStrategy` | `fixed_size` | — | How to split documents |
| `chunk_size` | `int` | `512` | `> 0` | Maximum characters per chunk |
| `chunk_overlap` | `int` | `64` | `>= 0` | Overlapping characters between consecutive chunks |
| `retrieval_method` | `RetrievalMethod` | `dense` | — | Retrieval approach |
| `top_k` | `int` | `5` | `> 0` | Number of chunks to retrieve per query |
| `embedding_model` | `str \| None` | `None` | — | Embedding model identifier (metadata) |

### `ChunkingStrategy` values

| Value | Behaviour |
|---|---|
| `fixed_size` | Slide a fixed character window with configurable overlap |
| `sentence` | Group sentences up to `chunk_size` chars; carry last sentence as overlap |
| `paragraph` | Split on blank lines; merge small paragraphs up to `chunk_size` |
| `recursive` | Try `\n\n`, then `\n`, then `. `, then space, recursively until chunks fit |
| `semantic` | Currently aliased to `sentence`; reserved for embedding-based splitting |

### `RetrievalMethod` values

| Value | Description |
|---|---|
| `dense` | Embedding-similarity retrieval (vector search) |
| `sparse` | Term-frequency retrieval (BM25-style) |
| `hybrid` | Combination of dense and sparse signals |
| `rerank` | Dense retrieval followed by a cross-encoder reranker |

---

## How it works — technical deep dive

### Chunking

Each chunker implements the `TextChunker` protocol — a `chunk(text: str) -> list[str]`
method. `ChunkerFactory.create(config)` instantiates the correct implementation.

`RecursiveChunker` is the most sophisticated: it walks a hierarchy of separators
(`\n\n` → `\n` → `. ` → ` ` → individual characters), recursively splitting with a
smaller separator whenever a piece still exceeds `chunk_size`. This produces
naturally-sized chunks that respect document structure.

### Retrieval simulation

`_simulate_retrieval` uses token-overlap scoring — a lightweight TF-style approach that
computes the fraction of unique query tokens appearing in each chunk and ranks by that
score. In production you would replace this with a real embedding model or BM25 index;
the interface is identical (`query + chunks → ranked chunk indices`).

### Scoring formula in `RAGOptimizer`

```
composite_score = 0.4 × Precision@k + 0.4 × Recall@k + 0.2 × MRR
```

Precision and Recall get equal weight (0.4 each) because both over-retrieval and
under-retrieval hurt downstream generation quality. MRR gets 0.2 because ranking order
matters but is less critical when multiple relevant documents exist.

### Cost model

```
cost_per_query = $0.0001 + $0.00001 × len(all_chunks)
```

This placeholder models a flat per-query API cost plus a per-chunk search cost. Replace
it with your actual embedding and vector-search pricing for accurate production estimates.

---

## Integration with other AumAI projects

- **aumai-contextweaver** — feed the retrieved chunks from `RAGBenchmark` into a
  `ContextManager` to prioritise and fit them within a model's token limit before
  calling the LLM.
- **aumai-benchmarkhub** — embed RAG retrieval quality as a capability dimension in an
  agent benchmark suite; use `RAGBenchmarkResult` metrics as features in agent evaluation.
- **aumai-specs** — use the `RAGConfig` Pydantic model as a validated specification
  object when declaring RAG pipeline requirements in a system design.

---

## Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` first.

```bash
git clone https://github.com/aumai/aumai-ragoptimizer
cd aumai-ragoptimizer
pip install -e ".[dev]"
make test     # pytest with coverage
make lint     # ruff + mypy strict
```

Branch naming: `feature/`, `fix/`, `docs/`. Conventional commits required.
Squash-merge PRs to keep history linear.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
