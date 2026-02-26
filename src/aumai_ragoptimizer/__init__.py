"""aumai-ragoptimizer — benchmark and optimize RAG pipeline configurations."""

from aumai_ragoptimizer.core import (
    ChunkerFactory,
    FixedSizeChunker,
    ParagraphChunker,
    RAGBenchmark,
    RAGOptimizer,
    RecursiveChunker,
    SentenceChunker,
    TextChunker,
)
from aumai_ragoptimizer.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from aumai_ragoptimizer.models import (
    ChunkingStrategy,
    OptimizationResult,
    RAGBenchmarkResult,
    RAGConfig,
    RetrievalMethod,
)

__version__ = "0.1.0"

__all__ = [
    "ChunkerFactory",
    "FixedSizeChunker",
    "ParagraphChunker",
    "RAGBenchmark",
    "RAGOptimizer",
    "RecursiveChunker",
    "SentenceChunker",
    "TextChunker",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "ChunkingStrategy",
    "OptimizationResult",
    "RAGBenchmarkResult",
    "RAGConfig",
    "RetrievalMethod",
]
