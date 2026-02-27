# Getting Started with aumai-ragoptimizer

This guide takes you from zero to a working RAG evaluation pipeline in under ten minutes.

---

## Prerequisites

- Python 3.11 or later
- pip or a virtual environment manager (venv, uv, conda)
- Basic familiarity with the command line

Optional but recommended:
- `PyYAML` — required only if you want to use YAML config files
- `tiktoken` — used by the sister library `aumai-contextweaver` for accurate token counts

---

## Installation

### From PyPI (recommended)

```bash
pip install aumai-ragoptimizer
```

Verify the install:

```bash
aumai-ragoptimizer --version
```

### From source

```bash
git clone https://github.com/aumai/aumai-ragoptimizer
cd aumai-ragoptimizer
pip install -e .
```

### Development mode (with test and lint dependencies)

```bash
pip install -e ".[dev]"
make test   # should show all tests passing
make lint   # ruff + mypy strict
```

---

## Your First RAG Benchmark

We will build a minimal evaluation corpus, run a benchmark, and inspect the metrics —
all in five steps.

### Step 1 — Create a corpus file

Create `corpus.jsonl` with a few short documents. Each line must be valid JSON with a
`text` field.

```json
{"text": "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a large language model to answer questions from external documents."}
{"text": "Text chunking is the process of splitting long documents into smaller segments before embedding them into a vector store. The choice of chunking strategy affects retrieval quality significantly."}
{"text": "Precision at k measures what fraction of the top-k retrieved results are actually relevant to the query. Higher is better."}
{"text": "Recall at k measures what fraction of all relevant documents appear in the top-k results. High recall means fewer relevant documents are missed."}
{"text": "Mean Reciprocal Rank (MRR) rewards systems that place the first relevant result as high as possible in the ranked list. A perfect score of 1.0 means the top result is always relevant."}
```

### Step 2 — Create a queries file

Create `queries.jsonl`. Each line has a `query` field and a `relevant_docs` field
containing a list of zero-based indices into the corpus above.

```json
{"query": "What is RAG?", "relevant_docs": [0]}
{"query": "How does text chunking affect retrieval?", "relevant_docs": [1]}
{"query": "How do I measure retrieval precision?", "relevant_docs": [2]}
{"query": "What is recall in information retrieval?", "relevant_docs": [3]}
{"query": "What is MRR?", "relevant_docs": [4]}
```

### Step 3 — Run the benchmark

```bash
aumai-ragoptimizer benchmark \
  --corpus corpus.jsonl \
  --queries queries.jsonl \
  --strategy sentence \
  --top-k 3
```

You will see progress on stderr and the JSON result on stdout:

```
Benchmarking sentence strategy on 5 docs / 5 queries...
Precision@3: 0.4667  Recall@3: 1.0000  MRR: 0.7000  Latency: 0.18ms
```

The JSON written to stdout contains the full `RAGBenchmarkResult` including the config
that was used, so results are always reproducible.

### Step 4 — Run the optimizer

Instead of testing one config at a time, let the optimizer grid-search for you:

```bash
aumai-ragoptimizer optimize \
  --corpus corpus.jsonl \
  --queries queries.jsonl \
  --output optimization_results.json
```

The optimizer tests 24 configurations (3 strategies × 2 chunk sizes × 2 overlap
settings × 2 retrieval methods) and reports the winner:

```
Optimizing over 24 configurations...
Best config: strategy=sentence size=512 overlap=64 retrieval=dense  improvement=14.2%
```

### Step 5 — Inspect the full results

```bash
python - <<'EOF'
import json

with open("optimization_results.json") as f:
    result = json.load(f)

best = result["best_config"]
print(f"Best strategy:  {best['chunking_strategy']}")
print(f"Best chunk size: {best['chunk_size']}")
print(f"Improvement:    {result['improvement_pct']:.1f}%")
print()
print("All results (composite score):")
for r in result["all_results"]:
    score = 0.4 * r["precision_at_k"] + 0.4 * r["recall_at_k"] + 0.2 * r["mrr"]
    cfg = r["config"]
    print(f"  {cfg['chunking_strategy']:12s}  size={cfg['chunk_size']:5d}  "
          f"overlap={cfg['chunk_overlap']:3d}  score={score:.4f}")
EOF
```

---

## Common Patterns

### Pattern 1 — Test every strategy on your own document

```python
from aumai_ragoptimizer import (
    RAGBenchmark, RAGConfig, ChunkingStrategy, RAGOptimizer
)

# Load your own data
corpus      = [line.strip() for line in open("my_docs.txt") if line.strip()]
queries     = ["your question 1", "your question 2"]
ground_truth = [[0], [1]]  # which corpus indices are relevant for each query

# Sweep all strategies
search_space = [
    RAGConfig(chunking_strategy=s, chunk_size=512, top_k=5)
    for s in ChunkingStrategy
]

opt = RAGOptimizer()
result = opt.optimize(corpus, queries, ground_truth, search_space)
print(result.best_config.model_dump_json(indent=2))
```

### Pattern 2 — Reuse a saved config for reproducible benchmarks

Save your best config once:

```bash
cat > best_config.json <<EOF
{
  "chunking_strategy": "recursive",
  "chunk_size": 512,
  "chunk_overlap": 64,
  "retrieval_method": "dense",
  "top_k": 5
}
EOF
```

Then always benchmark against it:

```bash
aumai-ragoptimizer benchmark \
  --config best_config.json \
  --corpus new_corpus.jsonl \
  --queries new_queries.jsonl \
  --output regression_result.json
```

This is ideal for CI/CD: add a step that asserts `precision_at_k >= 0.60` on your
evaluation set before merging a change to your document pipeline.

### Pattern 3 — Inspect chunks before indexing

Use the `chunk` CLI command to preview exactly what your chunker will produce before you
commit to a strategy. This catches unexpected splitting behaviour early.

```bash
aumai-ragoptimizer chunk \
  --strategy recursive \
  --input documentation.txt \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --output chunks.jsonl

# Count chunks and inspect the first few
head -5 chunks.jsonl | python -c "
import sys, json
for line in sys.stdin:
    obj = json.loads(line)
    print(f\"[{obj['index']:3d}] len={obj['length']:4d}  {obj['chunk'][:80]}...\")
"
```

### Pattern 4 — Write a custom chunker

The `TextChunker` protocol is a structural protocol (no base class required). Any class
with a `chunk(text: str) -> list[str]` method satisfies it.

```python
from aumai_ragoptimizer import RAGBenchmark, RAGConfig, ChunkingStrategy, TextChunker

class MarkdownHeaderChunker:
    """Split on H2/H3 headers in Markdown documents."""

    def chunk(self, text: str) -> list[str]:
        import re
        parts = re.split(r"(?=^#{2,3} )", text, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]

chunker = MarkdownHeaderChunker()
config  = RAGConfig(chunking_strategy=ChunkingStrategy.sentence, top_k=5)

# Use the chunker directly — bypass ChunkerFactory
all_chunks: list[str] = []
for doc in corpus:
    all_chunks.extend(chunker.chunk(doc))
```

### Pattern 5 — Use metrics as standalone functions in your own evaluator

```python
from aumai_ragoptimizer import (
    precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k
)

# Scores from your own retrieval system
retrieved = [3, 0, 7, 1, 5]
relevant  = [0, 1, 3]

k = 5
print(f"P@{k}:  {precision_at_k(retrieved, relevant, k):.4f}")
print(f"R@{k}:  {recall_at_k(retrieved, relevant, k):.4f}")
print(f"MRR:   {mean_reciprocal_rank(retrieved, relevant):.4f}")
print(f"NDCG@{k}: {ndcg_at_k(retrieved, relevant, k):.4f}")
```

---

## Troubleshooting FAQ

**Q: `aumai-ragoptimizer` command not found after `pip install`.**

Ensure the Python `bin` or `Scripts` directory is on your `PATH`. With virtual
environments, activate the environment first: `source .venv/bin/activate` (Linux/macOS)
or `.venv\Scripts\activate` (Windows).

---

**Q: `ImportError: No module named 'yaml'` when using `--config` with a YAML file.**

Install PyYAML:

```bash
pip install pyyaml
```

---

**Q: My optimization results look identical for all strategies.**

This usually means your evaluation set is too small or your `ground_truth` indices are
wrong. Double-check that:
1. `relevant_docs` indices are zero-based and correspond to positions in your corpus.
2. Your corpus has enough documents to make retrieval non-trivial (aim for 20+).
3. Your queries actually appear somewhere in the corpus text.

---

**Q: The improvement percentage is negative.**

The improvement is calculated relative to the *first* config in the search space, not
the worst config. If your first config happens to be good, improvement vs. the best can
be small or negative. Look at the absolute scores to find the best configuration.

---

**Q: Latency numbers seem constant or near zero.**

The built-in retrieval simulation is CPU-only and very fast for small corpora. Latency
numbers will be more meaningful when you replace `_simulate_retrieval` with a real
embedding model or vector database call.

---

**Q: How do I add `top_k` to the optimizer search space?**

The built-in grid in the CLI uses `top_k=5` for all configs. Using the Python API you
can include any `top_k` values you want:

```python
from aumai_ragoptimizer import RAGConfig, ChunkingStrategy, RAGOptimizer

search_space = [
    RAGConfig(chunking_strategy=ChunkingStrategy.sentence, chunk_size=512, top_k=k)
    for k in [3, 5, 10]
]
opt = RAGOptimizer()
result = opt.optimize(corpus, queries, ground_truth, search_space)
print(f"Best top_k: {result.best_config.top_k}")
```
