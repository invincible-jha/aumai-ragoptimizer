# API Reference — aumai-ragoptimizer

Full reference for every public class, function, and model in `aumai-ragoptimizer 0.1.0`.

---

## Module: `aumai_ragoptimizer.models`

### `ChunkingStrategy`

```python
class ChunkingStrategy(str, Enum)
```

Enumeration of supported text-chunking strategies. Inherits from `str` so values are
usable directly as strings in JSON serialisation and CLI arguments.

| Member | Value | Description |
|---|---|---|
| `fixed_size` | `"fixed_size"` | Slide a fixed character window over the text |
| `sentence` | `"sentence"` | Group sentences up to the chunk size |
| `paragraph` | `"paragraph"` | Split on blank lines; merge short paragraphs |
| `semantic` | `"semantic"` | Reserved; currently aliased to `sentence` |
| `recursive` | `"recursive"` | Hierarchical splitting through a separator list |

---

### `RetrievalMethod`

```python
class RetrievalMethod(str, Enum)
```

Supported retrieval approaches.

| Member | Value | Description |
|---|---|---|
| `dense` | `"dense"` | Embedding-similarity vector search |
| `sparse` | `"sparse"` | Term-frequency (BM25-style) search |
| `hybrid` | `"hybrid"` | Combination of dense and sparse |
| `rerank` | `"rerank"` | Dense retrieval + cross-encoder reranking |

---

### `RAGConfig`

```python
class RAGConfig(BaseModel)
```

Full configuration for a RAG pipeline. Used as input to `RAGBenchmark.run()` and
`RAGOptimizer.optimize()`.

**Fields:**

| Field | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `chunking_strategy` | `ChunkingStrategy` | `ChunkingStrategy.fixed_size` | — | Text splitting strategy |
| `chunk_size` | `int` | `512` | `> 0` | Maximum characters per chunk |
| `chunk_overlap` | `int` | `64` | `>= 0` | Characters of overlap between consecutive chunks |
| `retrieval_method` | `RetrievalMethod` | `RetrievalMethod.dense` | — | Retrieval approach |
| `top_k` | `int` | `5` | `> 0` | Number of chunks to retrieve per query |
| `embedding_model` | `str \| None` | `None` | — | Embedding model identifier (metadata only) |

**Example:**

```python
from aumai_ragoptimizer.models import RAGConfig, ChunkingStrategy, RetrievalMethod

config = RAGConfig(
    chunking_strategy=ChunkingStrategy.recursive,
    chunk_size=512,
    chunk_overlap=64,
    retrieval_method=RetrievalMethod.dense,
    top_k=5,
    embedding_model="text-embedding-3-small",
)

# Serialise to dict or JSON
print(config.model_dump())
print(config.model_dump_json(indent=2))

# Deserialise from dict
config2 = RAGConfig.model_validate({
    "chunking_strategy": "sentence",
    "chunk_size": 256,
    "top_k": 3,
})
```

---

### `RAGBenchmarkResult`

```python
class RAGBenchmarkResult(BaseModel)
```

Metrics produced by a single call to `RAGBenchmark.run()`.

**Fields:**

| Field | Type | Constraint | Description |
|---|---|---|---|
| `config` | `RAGConfig` | — | The configuration that was evaluated |
| `precision_at_k` | `float` | `[0.0, 1.0]` | Mean Precision@k across all queries |
| `recall_at_k` | `float` | `[0.0, 1.0]` | Mean Recall@k across all queries |
| `mrr` | `float` | `[0.0, 1.0]` | Mean Reciprocal Rank across all queries |
| `latency_ms` | `float` | `>= 0.0` | Average retrieval latency per query in milliseconds |
| `cost_per_query` | `float` | `>= 0.0` | Estimated cost per query in USD |

**Example:**

```python
result = benchmark.run(config, corpus, queries, ground_truth)
print(f"Precision@{config.top_k}: {result.precision_at_k:.4f}")
print(f"Cost/query: ${result.cost_per_query:.6f}")

# Serialise for storage / comparison
import json
with open("result.json", "w") as f:
    json.dump(result.model_dump(), f, indent=2)
```

---

### `OptimizationResult`

```python
class OptimizationResult(BaseModel)
```

Result of a grid-search optimisation run from `RAGOptimizer.optimize()`.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `best_config` | `RAGConfig` | The configuration with the highest composite score |
| `all_results` | `list[RAGBenchmarkResult]` | One result per config in the search space, in input order |
| `improvement_pct` | `float` | Percentage improvement of best config over the first config in the search space |

**Example:**

```python
opt_result = optimizer.optimize(corpus, queries, ground_truth, search_space)

print(opt_result.best_config.chunking_strategy.value)
print(f"Improvement: {opt_result.improvement_pct:.1f}%")

# Find the worst config
worst = min(opt_result.all_results,
            key=lambda r: 0.4*r.precision_at_k + 0.4*r.recall_at_k + 0.2*r.mrr)
print(f"Worst strategy: {worst.config.chunking_strategy.value}")
```

---

## Module: `aumai_ragoptimizer.core`

### `TextChunker` (Protocol)

```python
@runtime_checkable
class TextChunker(Protocol)
```

Structural protocol for all text-chunking strategies. Any class that implements
`chunk(text: str) -> list[str]` satisfies this protocol — no inheritance needed.

**Methods:**

#### `chunk`

```python
def chunk(self, text: str) -> list[str]
```

Split `text` into a list of string chunks.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to split |

**Returns:** `list[str]` — the list of non-empty text chunks.

---

### `FixedSizeChunker`

```python
class FixedSizeChunker
```

Split text into fixed-size character windows with optional overlap. The simplest and
fastest strategy; works well when document structure is uniform.

#### `__init__`

```python
def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `512` | Maximum characters per chunk |
| `chunk_overlap` | `int` | `64` | Number of trailing characters from the previous chunk to include at the start of the next chunk |

#### `chunk`

```python
def chunk(self, text: str) -> list[str]
```

Returns fixed-size chunks with overlap. Empty strings and whitespace-only chunks are
excluded from the result.

**Example:**

```python
from aumai_ragoptimizer import FixedSizeChunker

chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
chunks = chunker.chunk("A" * 250)
# Produces 3 chunks: [0:100], [80:180], [160:250]
print(len(chunks))  # 3
```

---

### `SentenceChunker`

```python
class SentenceChunker
```

Split text into sentence-based chunks by grouping sentences up to `chunk_size`
characters. Sentence boundaries are detected by the pattern `(?<=[.!?])\s+`.

#### `__init__`

```python
def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `512` | Soft character limit per chunk |
| `chunk_overlap` | `int` | `0` | If `> 0`, the last sentence of the previous chunk seeds the next chunk |

#### `chunk`

```python
def chunk(self, text: str) -> list[str]
```

**Example:**

```python
from aumai_ragoptimizer import SentenceChunker

text = "First sentence. Second sentence. Third sentence. Fourth sentence."
chunker = SentenceChunker(chunk_size=40, chunk_overlap=0)
print(chunker.chunk(text))
# ['First sentence. Second sentence.', 'Third sentence. Fourth sentence.']
```

---

### `ParagraphChunker`

```python
class ParagraphChunker
```

Split text on blank lines (`\n\n` or more), then merge consecutive short paragraphs
until the chunk would exceed `chunk_size`.

#### `__init__`

```python
def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 0) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `1024` | Maximum characters per chunk |
| `chunk_overlap` | `int` | `0` | Not used for paragraph strategy (reserved for future use) |

#### `chunk`

```python
def chunk(self, text: str) -> list[str]
```

**Example:**

```python
from aumai_ragoptimizer import ParagraphChunker

text = "Para one.\n\nPara two.\n\nPara three that is quite a bit longer than the rest."
chunker = ParagraphChunker(chunk_size=20)
print(chunker.chunk(text))
# ['Para one.', 'Para two.', 'Para three that is quite a bit longer than the rest.']
```

---

### `RecursiveChunker`

```python
class RecursiveChunker
```

Recursively split text using a hierarchy of separators until all pieces are within
`chunk_size`. Separator order: `"\n\n"` → `"\n"` → `". "` → `" "` → `""` (character).
This is the recommended default for mixed-format documents.

#### `__init__`

```python
def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | `int` | `512` | Maximum characters per chunk |
| `chunk_overlap` | `int` | `64` | Overlap carried from the tail of each group before the next split |

#### `chunk`

```python
def chunk(self, text: str) -> list[str]
```

**Example:**

```python
from aumai_ragoptimizer import RecursiveChunker

chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
text = "First paragraph.\n\nSecond paragraph with more text. And another sentence."
chunks = chunker.chunk(text)
for c in chunks:
    print(repr(c))
```

---

### `ChunkerFactory`

```python
class ChunkerFactory
```

Factory that instantiates the correct `TextChunker` for a given `RAGConfig`.

#### `create` (static method)

```python
@staticmethod
def create(config: RAGConfig) -> TextChunker
```

| Parameter | Type | Description |
|---|---|---|
| `config` | `RAGConfig` | Configuration object specifying strategy, size, and overlap |

**Returns:** A `TextChunker` instance configured according to `config`.

**Mapping:**

| `config.chunking_strategy` | Returned class |
|---|---|
| `fixed_size` | `FixedSizeChunker` |
| `sentence` | `SentenceChunker` |
| `paragraph` | `ParagraphChunker` |
| `recursive` | `RecursiveChunker` |
| `semantic` | `SentenceChunker` (fallback) |

**Example:**

```python
from aumai_ragoptimizer import ChunkerFactory, RAGConfig, ChunkingStrategy

config  = RAGConfig(chunking_strategy=ChunkingStrategy.recursive, chunk_size=256)
chunker = ChunkerFactory.create(config)
chunks  = chunker.chunk("Your document text here.")
```

---

### `RAGBenchmark`

```python
class RAGBenchmark
```

Evaluate a `RAGConfig` against a labelled corpus and compute retrieval metrics.

#### `run`

```python
def run(
    self,
    config: RAGConfig,
    corpus: list[str],
    queries: list[str],
    ground_truth: list[list[int]],
) -> RAGBenchmarkResult
```

Benchmark `config` against the provided corpus and query set.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `RAGConfig` | The RAG pipeline configuration to evaluate |
| `corpus` | `list[str]` | List of source documents (strings) |
| `queries` | `list[str]` | List of query strings, one per evaluation question |
| `ground_truth` | `list[list[int]]` | Per-query list of relevant corpus indices (zero-based) |

**Returns:** `RAGBenchmarkResult` with aggregated metrics.

**Algorithm:**
1. Build a chunk index using `ChunkerFactory.create(config)`.
2. Track `chunk_to_doc` mapping so chunk indices can be resolved back to document indices.
3. For each query, simulate retrieval and compute Precision@k, Recall@k, MRR, and latency.
4. Average metrics across all queries; estimate per-query cost.

**Example:**

```python
from aumai_ragoptimizer import RAGBenchmark, RAGConfig

benchmark = RAGBenchmark()
result = benchmark.run(
    config=RAGConfig(top_k=5),
    corpus=["doc one", "doc two", "doc three"],
    queries=["query about doc one"],
    ground_truth=[[0]],
)
print(result.precision_at_k)
```

---

### `RAGOptimizer`

```python
class RAGOptimizer
```

Grid-search over a list of `RAGConfig` candidates to find the one with the best
composite score.

#### `__init__`

```python
def __init__(self, benchmark: RAGBenchmark | None = None) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `benchmark` | `RAGBenchmark \| None` | `None` | Custom benchmark instance; a default `RAGBenchmark()` is created if `None` |

#### `optimize`

```python
def optimize(
    self,
    corpus: list[str],
    queries: list[str],
    ground_truth: list[list[int]],
    search_space: list[RAGConfig],
) -> OptimizationResult
```

Run all configs in `search_space` and return the best.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `corpus` | `list[str]` | Source documents |
| `queries` | `list[str]` | Evaluation queries |
| `ground_truth` | `list[list[int]]` | Per-query lists of relevant document indices |
| `search_space` | `list[RAGConfig]` | Candidate configurations to evaluate |

**Returns:** `OptimizationResult`.

**Raises:** `ValueError` if `search_space` is empty.

**Scoring formula:**

```
score = 0.4 × precision_at_k + 0.4 × recall_at_k + 0.2 × mrr
```

**Improvement calculation:**

```
improvement_pct = (score(best) - score(search_space[0])) / (score(search_space[0]) + 1e-9) × 100
```

---

## Module: `aumai_ragoptimizer.metrics`

Pure functions for computing information-retrieval metrics. All functions treat
relevance as binary (a document index is either in the `relevant` set or it is not).

---

### `precision_at_k`

```python
def precision_at_k(retrieved: list[int], relevant: list[int], k: int) -> float
```

Fraction of the top-k retrieved documents that are relevant.

| Parameter | Type | Description |
|---|---|---|
| `retrieved` | `list[int]` | Ordered list of retrieved document indices (most relevant first) |
| `relevant` | `list[int]` | Ground-truth relevant document indices |
| `k` | `int` | Cut-off rank |

**Returns:** `float` in `[0.0, 1.0]`. Returns `0.0` if `k <= 0`.

```python
precision_at_k([0, 2, 1, 3, 4], [0, 1], k=3)  # 2/3 ≈ 0.6667
```

---

### `recall_at_k`

```python
def recall_at_k(retrieved: list[int], relevant: list[int], k: int) -> float
```

Fraction of all relevant documents found in the top-k retrieved results.

| Parameter | Type | Description |
|---|---|---|
| `retrieved` | `list[int]` | Ordered list of retrieved document indices |
| `relevant` | `list[int]` | Ground-truth relevant document indices |
| `k` | `int` | Cut-off rank |

**Returns:** `float` in `[0.0, 1.0]`. Returns `0.0` if `relevant` is empty or `k <= 0`.

```python
recall_at_k([0, 2, 1, 3, 4], [0, 1], k=3)  # 2/2 = 1.0
```

---

### `mean_reciprocal_rank`

```python
def mean_reciprocal_rank(retrieved: list[int], relevant: list[int]) -> float
```

Reciprocal of the rank of the first relevant document in the retrieved list.

| Parameter | Type | Description |
|---|---|---|
| `retrieved` | `list[int]` | Ordered list of retrieved document indices |
| `relevant` | `list[int]` | Ground-truth relevant document indices |

**Returns:** `float` in `[0.0, 1.0]`. Returns `0.0` if no relevant document is found.

```python
mean_reciprocal_rank([2, 0, 1], [0, 1])  # 1/2 = 0.5 (first relevant at rank 2)
mean_reciprocal_rank([0, 1, 2], [0, 1])  # 1/1 = 1.0 (first relevant at rank 1)
```

---

### `ndcg_at_k`

```python
def ndcg_at_k(retrieved: list[int], relevant: list[int], k: int) -> float
```

Normalised Discounted Cumulative Gain at k using binary relevance.

Higher-ranked relevant documents contribute more to the score. The result is normalised
by the ideal DCG (all relevant documents ranked first).

| Parameter | Type | Description |
|---|---|---|
| `retrieved` | `list[int]` | Ordered list of retrieved document indices |
| `relevant` | `list[int]` | Ground-truth relevant document indices |
| `k` | `int` | Cut-off rank |

**Returns:** `float` in `[0.0, 1.0]`. Returns `0.0` if `relevant` is empty or `k <= 0`.

```python
ndcg_at_k([0, 1, 2, 3], [0, 3], k=4)  # ≈ 0.8154
```
