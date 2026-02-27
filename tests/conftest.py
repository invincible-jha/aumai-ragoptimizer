"""Shared test fixtures for aumai-ragoptimizer."""
from __future__ import annotations

import pytest

from aumai_ragoptimizer.core import (
    FixedSizeChunker,
    ParagraphChunker,
    RAGBenchmark,
    RecursiveChunker,
    SentenceChunker,
)
from aumai_ragoptimizer.models import ChunkingStrategy, RAGConfig, RetrievalMethod

SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog. This is a classic pangram.",
    "Python is a high-level programming language. It emphasizes code readability.",
    "Machine learning models require large datasets for training. Data quality matters.",
    "Retrieval Augmented Generation combines search with language model generation.",
    "Vector databases store dense embeddings for efficient similarity search.",
]

SAMPLE_QUERIES = [
    "Python programming language",
    "machine learning data",
    "vector search embeddings",
]

SAMPLE_GROUND_TRUTH = [
    [1],   # query 0 -> doc 1 (Python)
    [2],   # query 1 -> doc 2 (ML)
    [4],   # query 2 -> doc 4 (vector)
]


@pytest.fixture()
def corpus() -> list[str]:
    return list(SAMPLE_CORPUS)


@pytest.fixture()
def queries() -> list[str]:
    return list(SAMPLE_QUERIES)


@pytest.fixture()
def ground_truth() -> list[list[int]]:
    return [list(g) for g in SAMPLE_GROUND_TRUTH]


@pytest.fixture()
def default_config() -> RAGConfig:
    return RAGConfig()


@pytest.fixture()
def benchmark() -> RAGBenchmark:
    return RAGBenchmark()


@pytest.fixture()
def long_text() -> str:
    return (
        "This is a long document. " * 50
        + "\n\nSecond paragraph. " * 30
        + "\n\nThird paragraph with unique terms like quantum entanglement. " * 20
    )
