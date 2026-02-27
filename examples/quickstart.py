"""aumai-ragoptimizer quickstart examples.

Demonstrates five progressive use cases:
  1. Chunking a document with each built-in strategy.
  2. Running a single benchmark against a labelled corpus.
  3. Grid-searching to find the best configuration.
  4. Using the four IR metric functions directly.
  5. Writing a custom chunker that satisfies the TextChunker protocol.

Run directly to verify your installation:

    python examples/quickstart.py
"""

from __future__ import annotations

import re

from aumai_ragoptimizer import (
    ChunkerFactory,
    ChunkingStrategy,
    FixedSizeChunker,
    ParagraphChunker,
    RAGBenchmark,
    RAGConfig,
    RAGOptimizer,
    RecursiveChunker,
    RetrievalMethod,
    SentenceChunker,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Shared demo data
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    (
        "Retrieval-Augmented Generation (RAG) is a technique that combines a "
        "retrieval step with a large language model. Given a user question, the "
        "system first retrieves the most relevant document chunks, then passes "
        "them as context to the LLM so it can answer accurately."
    ),
    (
        "Text chunking is the process of splitting long documents into smaller "
        "segments before embedding them. The choice of chunking strategy — "
        "fixed-size, sentence-based, paragraph-based, or recursive — affects "
        "retrieval quality significantly. Smaller chunks improve precision; "
        "larger chunks improve recall."
    ),
    (
        "Precision at k (P@k) measures what fraction of the top-k retrieved "
        "results are actually relevant. Recall at k (R@k) measures what "
        "fraction of all relevant documents appear in the top-k. Both metrics "
        "are in [0, 1] and higher is always better."
    ),
    (
        "Mean Reciprocal Rank (MRR) rewards retrieval systems that place the "
        "first relevant result as high as possible in the ranked list. A perfect "
        "MRR of 1.0 means the top result is always the most relevant document."
    ),
    (
        "Dense retrieval uses an embedding model to convert queries and documents "
        "into vectors, then finds the nearest neighbours by cosine similarity. "
        "Sparse retrieval (e.g. BM25) uses term-frequency statistics. Hybrid "
        "retrieval combines both signals."
    ),
]

QUERIES: list[str] = [
    "What is Retrieval-Augmented Generation?",
    "How does chunking affect retrieval quality?",
    "What do precision and recall measure?",
    "What is MRR?",
    "dense vs sparse retrieval",
]

# Ground truth: for each query, which corpus indices are relevant?
GROUND_TRUTH: list[list[int]] = [
    [0],  # query 0 -> document 0
    [1],  # query 1 -> document 1
    [2],  # query 2 -> document 2
    [3],  # query 3 -> document 3
    [4],  # query 4 -> document 4
]


# ---------------------------------------------------------------------------
# Demo 1 — Chunking strategies compared
# ---------------------------------------------------------------------------


def demo_chunking_strategies() -> None:
    """Show how the four built-in strategies chunk the same text."""
    print("=" * 60)
    print("Demo 1: Chunking strategies")
    print("=" * 60)

    sample_text = CORPUS[1]  # The paragraph about chunking
    strategies = [
        ("FixedSizeChunker",  FixedSizeChunker(chunk_size=120, chunk_overlap=20)),
        ("SentenceChunker",   SentenceChunker(chunk_size=120, chunk_overlap=0)),
        ("ParagraphChunker",  ParagraphChunker(chunk_size=200, chunk_overlap=0)),
        ("RecursiveChunker",  RecursiveChunker(chunk_size=120, chunk_overlap=20)),
    ]

    for name, chunker in strategies:
        chunks = chunker.chunk(sample_text)
        print(f"\n{name} ({len(chunks)} chunks):")
        for idx, chunk in enumerate(chunks):
            # Print a short preview of each chunk
            preview = chunk[:70].replace("\n", " ")
            print(f"  [{idx}] ({len(chunk):3d} chars)  {preview}...")

    print()


# ---------------------------------------------------------------------------
# Demo 2 — Single-config benchmark
# ---------------------------------------------------------------------------


def demo_single_benchmark() -> None:
    """Run a single RAGConfig through RAGBenchmark and print all metrics."""
    print("=" * 60)
    print("Demo 2: Single-config benchmark")
    print("=" * 60)

    config = RAGConfig(
        chunking_strategy=ChunkingStrategy.sentence,
        chunk_size=200,
        chunk_overlap=0,
        retrieval_method=RetrievalMethod.dense,
        top_k=3,
    )

    benchmark = RAGBenchmark()
    result = benchmark.run(config, CORPUS, QUERIES, GROUND_TRUTH)

    print(f"\nConfig:        {config.chunking_strategy.value}, size={config.chunk_size}")
    print(f"Precision@{config.top_k}:  {result.precision_at_k:.4f}")
    print(f"Recall@{config.top_k}:     {result.recall_at_k:.4f}")
    print(f"MRR:           {result.mrr:.4f}")
    print(f"Latency:       {result.latency_ms:.3f} ms")
    print(f"Cost/query:    ${result.cost_per_query:.6f}")
    print()


# ---------------------------------------------------------------------------
# Demo 3 — Grid search optimisation
# ---------------------------------------------------------------------------


def demo_optimizer() -> None:
    """Grid-search a custom search space and print ranked results."""
    print("=" * 60)
    print("Demo 3: Grid-search optimisation")
    print("=" * 60)

    # Build a focused search space: four strategies × two chunk sizes.
    search_space: list[RAGConfig] = [
        RAGConfig(
            chunking_strategy=strategy,
            chunk_size=size,
            chunk_overlap=overlap,
            top_k=3,
        )
        for strategy in [
            ChunkingStrategy.fixed_size,
            ChunkingStrategy.sentence,
            ChunkingStrategy.paragraph,
            ChunkingStrategy.recursive,
        ]
        for size in [100, 200]
        for overlap in [0, 20]
    ]

    optimizer = RAGOptimizer()
    opt_result = optimizer.optimize(CORPUS, QUERIES, GROUND_TRUTH, search_space)

    print(f"\nSearch space size: {len(search_space)} configs")
    print(f"Best strategy:     {opt_result.best_config.chunking_strategy.value}")
    print(f"Best chunk size:   {opt_result.best_config.chunk_size}")
    print(f"Best overlap:      {opt_result.best_config.chunk_overlap}")
    print(f"Improvement:       {opt_result.improvement_pct:.1f}%")

    # Print ranked leaderboard
    def composite_score(r: object) -> float:
        # r is a RAGBenchmarkResult
        return 0.4 * r.precision_at_k + 0.4 * r.recall_at_k + 0.2 * r.mrr  # type: ignore[attr-defined]

    ranked = sorted(opt_result.all_results, key=composite_score, reverse=True)
    print("\nTop 5 configurations:")
    for rank, r in enumerate(ranked[:5], start=1):
        cfg = r.config
        score = composite_score(r)
        print(
            f"  #{rank}  {cfg.chunking_strategy.value:12s}"
            f"  size={cfg.chunk_size:4d}  overlap={cfg.chunk_overlap:3d}"
            f"  score={score:.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Demo 4 — Standalone IR metrics
# ---------------------------------------------------------------------------


def demo_metrics() -> None:
    """Use the four IR metric functions independently."""
    print("=" * 60)
    print("Demo 4: Standalone IR metrics")
    print("=" * 60)

    # Simulate a retrieval system's output for a single query.
    retrieved = [2, 0, 4, 1, 3]   # ranked list of retrieved document indices
    relevant  = [0, 1]             # ground-truth relevant indices

    k = 5
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    mrr = mean_reciprocal_rank(retrieved, relevant)
    ndcg = ndcg_at_k(retrieved, relevant, k)

    print(f"\nRetrieved:  {retrieved}")
    print(f"Relevant:   {relevant}")
    print(f"\nPrecision@{k}: {p:.4f}   (2 of 5 retrieved are relevant)")
    print(f"Recall@{k}:    {r:.4f}   (both relevant docs found)")
    print(f"MRR:          {mrr:.4f}   (first relevant at rank 2)")
    print(f"NDCG@{k}:      {ndcg:.4f}   (graded rank quality)")
    print()


# ---------------------------------------------------------------------------
# Demo 5 — Custom chunker via the TextChunker protocol
# ---------------------------------------------------------------------------


def demo_custom_chunker() -> None:
    """Implement a custom Markdown header chunker and plug it in."""
    print("=" * 60)
    print("Demo 5: Custom chunker (Markdown headers)")
    print("=" * 60)

    class MarkdownHeaderChunker:
        """Split a Markdown document on H2/H3 header boundaries.

        Any class with a `chunk(text: str) -> list[str]` method satisfies
        the TextChunker protocol and can be used wherever a chunker is expected.
        """

        _HEADER_RE = re.compile(r"(?=^#{2,3} )", re.MULTILINE)

        def chunk(self, text: str) -> list[str]:
            parts = self._HEADER_RE.split(text)
            return [p.strip() for p in parts if p.strip()]

    markdown_doc = """
# Introduction
This document covers chunking strategies in aumai-ragoptimizer.

## Fixed-Size Chunking
Fixed-size chunking slices text into windows of a fixed character count.
It is fast and predictable but ignores document structure.

## Sentence Chunking
Sentence chunking groups complete sentences together up to a size limit.
It preserves grammatical meaning better than fixed-size chunking.

## Recursive Chunking
Recursive chunking tries larger separators first, falling back to smaller
ones when chunks are still too big. It balances structure and size.
    """

    chunker = MarkdownHeaderChunker()
    chunks = chunker.chunk(markdown_doc)

    print(f"\nMarkdown document split into {len(chunks)} header sections:")
    for idx, chunk in enumerate(chunks):
        first_line = chunk.split("\n")[0]
        print(f"  [{idx}] {first_line}")

    # The custom chunker also satisfies the TextChunker protocol at runtime.
    from aumai_ragoptimizer import TextChunker
    assert isinstance(chunker, TextChunker), "Custom chunker satisfies the protocol"
    print("\nProtocol check passed: MarkdownHeaderChunker is a valid TextChunker.")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all five demos in sequence."""
    print("\naumai-ragoptimizer quickstart\n")
    demo_chunking_strategies()
    demo_single_benchmark()
    demo_optimizer()
    demo_metrics()
    demo_custom_chunker()
    print("All demos completed successfully.")


if __name__ == "__main__":
    main()
