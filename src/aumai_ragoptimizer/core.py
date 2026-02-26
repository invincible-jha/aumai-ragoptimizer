"""Core chunking, benchmarking, and optimisation logic for aumai-ragoptimizer."""

from __future__ import annotations

import re
import time
from typing import Protocol, runtime_checkable

from aumai_ragoptimizer.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from aumai_ragoptimizer.models import (
    ChunkingStrategy,
    OptimizationResult,
    RAGBenchmarkResult,
    RAGConfig,
)

# ---------------------------------------------------------------------------
# Chunker protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class TextChunker(Protocol):
    """Protocol for all text-chunking strategies."""

    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of string chunks."""
        ...


class FixedSizeChunker:
    """Split text into fixed-size character windows with optional overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Return fixed-size chunks with overlap."""
        if not text:
            return []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks


class SentenceChunker:
    """Split text into sentence-based chunks, grouping up to *chunk_size* chars."""

    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Return sentence-grouped chunks."""
        sentences = self._SENTENCE_BOUNDARY.split(text.strip())
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_len + len(sentence) > self.chunk_size and current_parts:
                chunks.append(" ".join(current_parts))
                # Overlap: keep last sentence as seed for the next chunk.
                if self.chunk_overlap > 0:
                    current_parts = [current_parts[-1]]
                    current_len = len(current_parts[0])
                else:
                    current_parts = []
                    current_len = 0
            current_parts.append(sentence)
            current_len += len(sentence) + 1

        if current_parts:
            chunks.append(" ".join(current_parts))
        return chunks


class ParagraphChunker:
    """Split text on blank lines, merging small paragraphs up to *chunk_size*."""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 0) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Return paragraph-based chunks."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > self.chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            current_parts.append(para)
            current_len += len(para)

        if current_parts:
            chunks.append("\n\n".join(current_parts))
        return chunks


class RecursiveChunker:
    """Recursively split using a hierarchy of separators until chunks are small enough."""

    _SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Return recursively-split chunks."""
        return self._split(text.strip(), self._SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]
        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(sep)
            if current_len + part_len > self.chunk_size and current_parts:
                merged = sep.join(current_parts)
                if len(merged) > self.chunk_size:
                    chunks.extend(self._split(merged, remaining_seps))
                else:
                    chunks.append(merged)
                # Overlap.
                if self.chunk_overlap > 0 and current_parts:
                    overlap_parts = [current_parts[-1]]
                    overlap_len = len(current_parts[-1])
                    idx = -2
                    while (
                        idx >= -len(current_parts)
                        and overlap_len + len(current_parts[idx]) < self.chunk_overlap
                    ):
                        overlap_parts.insert(0, current_parts[idx])
                        overlap_len += len(current_parts[idx])
                        idx -= 1
                    current_parts = overlap_parts
                    current_len = overlap_len
                else:
                    current_parts = []
                    current_len = 0
            current_parts.append(part)
            current_len += part_len

        if current_parts:
            merged = sep.join(current_parts)
            if len(merged) > self.chunk_size:
                chunks.extend(self._split(merged, remaining_seps))
            else:
                chunks.append(merged)
        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# ChunkerFactory
# ---------------------------------------------------------------------------


class ChunkerFactory:
    """Instantiate the correct TextChunker for a given strategy."""

    @staticmethod
    def create(config: RAGConfig) -> TextChunker:
        """Return a chunker configured by *config*."""
        strategy = config.chunking_strategy
        size = config.chunk_size
        overlap = config.chunk_overlap

        if strategy == ChunkingStrategy.fixed_size:
            return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
        if strategy == ChunkingStrategy.sentence:
            return SentenceChunker(chunk_size=size, chunk_overlap=overlap)
        if strategy == ChunkingStrategy.paragraph:
            return ParagraphChunker(chunk_size=size, chunk_overlap=overlap)
        if strategy == ChunkingStrategy.recursive:
            return RecursiveChunker(chunk_size=size, chunk_overlap=overlap)
        # semantic: fall back to sentence chunking (embedding model not wired here)
        return SentenceChunker(chunk_size=size, chunk_overlap=overlap)


# ---------------------------------------------------------------------------
# Retrieval simulation
# ---------------------------------------------------------------------------


def _simulate_retrieval(
    query: str,
    chunks: list[str],
    top_k: int,
) -> list[int]:
    """Return indices of the top-k chunks most relevant to *query*.

    This is a lightweight TF-style simulation using token overlap.  In a real
    pipeline this would call an embedding model or a BM25 index.
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return list(range(min(top_k, len(chunks))))

    scores: list[tuple[float, int]] = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(query_tokens & chunk_tokens)
        score = overlap / (len(query_tokens) + 1e-9)
        scores.append((score, idx))

    scores.sort(key=lambda pair: pair[0], reverse=True)
    return [idx for _, idx in scores[:top_k]]


# ---------------------------------------------------------------------------
# RAGBenchmark
# ---------------------------------------------------------------------------


class RAGBenchmark:
    """Evaluate a RAGConfig against a labelled corpus."""

    def run(
        self,
        config: RAGConfig,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
    ) -> RAGBenchmarkResult:
        """Benchmark *config* and return metrics.

        Args:
            config:       RAG pipeline configuration to evaluate.
            corpus:       List of source documents (strings).
            queries:      List of query strings.
            ground_truth: Per-query list of relevant *corpus* indices.

        Returns:
            A RAGBenchmarkResult with aggregated metrics.
        """
        chunker = ChunkerFactory.create(config)

        # Build chunk index: list of chunks, each annotated with its source doc index.
        all_chunks: list[str] = []
        chunk_to_doc: list[int] = []
        for doc_idx, doc in enumerate(corpus):
            for chunk in chunker.chunk(doc):
                all_chunks.append(chunk)
                chunk_to_doc.append(doc_idx)

        precisions: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        latencies: list[float] = []

        for query, relevant_docs in zip(queries, ground_truth):
            t_start = time.perf_counter()
            retrieved_chunk_idxs = _simulate_retrieval(query, all_chunks, config.top_k)
            latency_ms = (time.perf_counter() - t_start) * 1000

            # Map chunk indices back to document indices (deduplicated, order preserved).
            seen: set[int] = set()
            retrieved_docs: list[int] = []
            for chunk_idx in retrieved_chunk_idxs:
                doc_idx = chunk_to_doc[chunk_idx]
                if doc_idx not in seen:
                    seen.add(doc_idx)
                    retrieved_docs.append(doc_idx)

            precisions.append(precision_at_k(retrieved_docs, relevant_docs, config.top_k))
            recalls.append(recall_at_k(retrieved_docs, relevant_docs, config.top_k))
            mrrs.append(mean_reciprocal_rank(retrieved_docs, relevant_docs))
            latencies.append(latency_ms)

        n = max(len(queries), 1)
        avg_precision = sum(precisions) / n
        avg_recall = sum(recalls) / n
        avg_mrr = sum(mrrs) / n
        avg_latency = sum(latencies) / n

        # Rough cost model: $0.0001 per query + $0.00001 per chunk searched.
        cost_per_query = 0.0001 + 0.00001 * len(all_chunks)

        return RAGBenchmarkResult(
            config=config,
            precision_at_k=round(avg_precision, 4),
            recall_at_k=round(avg_recall, 4),
            mrr=round(avg_mrr, 4),
            latency_ms=round(avg_latency, 4),
            cost_per_query=round(cost_per_query, 6),
        )


# ---------------------------------------------------------------------------
# RAGOptimizer
# ---------------------------------------------------------------------------


class RAGOptimizer:
    """Grid-search over a list of RAGConfig candidates to find the best one."""

    def __init__(self, benchmark: RAGBenchmark | None = None) -> None:
        self._benchmark = benchmark or RAGBenchmark()

    def optimize(
        self,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
        search_space: list[RAGConfig],
    ) -> OptimizationResult:
        """Run all configs in *search_space* and return the best.

        The scoring function is: 0.4 * Precision@k + 0.4 * Recall@k + 0.2 * MRR.

        Args:
            corpus:       Source documents.
            queries:      Evaluation queries.
            ground_truth: Per-query list of relevant document indices.
            search_space: Candidate RAGConfig objects to evaluate.

        Returns:
            OptimizationResult with the best config and all benchmark scores.
        """
        if not search_space:
            raise ValueError("search_space must contain at least one RAGConfig.")

        all_results: list[RAGBenchmarkResult] = []
        for config in search_space:
            result = self._benchmark.run(config, corpus, queries, ground_truth)
            all_results.append(result)

        def score(result: RAGBenchmarkResult) -> float:
            return (
                0.4 * result.precision_at_k
                + 0.4 * result.recall_at_k
                + 0.2 * result.mrr
            )

        best = max(all_results, key=score)
        baseline_score = score(all_results[0])
        best_score = score(best)
        improvement_pct = (
            ((best_score - baseline_score) / (baseline_score + 1e-9)) * 100
        )

        return OptimizationResult(
            best_config=best.config,
            all_results=all_results,
            improvement_pct=round(improvement_pct, 2),
        )


__all__ = [
    "TextChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "ChunkerFactory",
    "RAGBenchmark",
    "RAGOptimizer",
]
