"""Pydantic v2 models for aumai-ragoptimizer."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Supported text-chunking strategies."""

    fixed_size = "fixed_size"
    sentence = "sentence"
    paragraph = "paragraph"
    semantic = "semantic"
    recursive = "recursive"


class RetrievalMethod(str, Enum):
    """Supported retrieval methods."""

    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"
    rerank = "rerank"


class RAGConfig(BaseModel):
    """Full configuration for a RAG pipeline."""

    chunking_strategy: ChunkingStrategy = ChunkingStrategy.fixed_size
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    retrieval_method: RetrievalMethod = RetrievalMethod.dense
    top_k: int = Field(default=5, gt=0)
    embedding_model: str | None = None


class RAGBenchmarkResult(BaseModel):
    """Benchmark metrics for a single RAGConfig evaluation."""

    config: RAGConfig
    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    cost_per_query: float = Field(ge=0.0)


class OptimizationResult(BaseModel):
    """Result of a grid-search optimisation run."""

    best_config: RAGConfig
    all_results: list[RAGBenchmarkResult]
    improvement_pct: float


__all__ = [
    "ChunkingStrategy",
    "RetrievalMethod",
    "RAGConfig",
    "RAGBenchmarkResult",
    "OptimizationResult",
]
