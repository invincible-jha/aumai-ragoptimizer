"""Comprehensive tests for aumai_ragoptimizer core, metrics, and models."""
from __future__ import annotations

import pytest

from aumai_ragoptimizer.core import (
    ChunkerFactory,
    FixedSizeChunker,
    ParagraphChunker,
    RAGBenchmark,
    RAGOptimizer,
    RecursiveChunker,
    SentenceChunker,
    TextChunker,
    _simulate_retrieval,
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


# ---------------------------------------------------------------------------
# Tests for FixedSizeChunker
# ---------------------------------------------------------------------------


class TestFixedSizeChunker:
    def test_empty_text_returns_empty(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100)
        assert chunker.chunk("") == []

    def test_single_chunk_small_text(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_multiple_chunks_on_large_text(self) -> None:
        text = "a" * 300
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_overlap_creates_overlapping_chunks(self) -> None:
        text = "a" * 200
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=50)
        chunks = chunker.chunk(text)
        # With overlap=50, step=50; should produce more chunks
        assert len(chunks) > 2

    def test_chunk_size_respected(self) -> None:
        text = "word " * 200
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_all_content_preserved_no_overlap(self) -> None:
        text = "abcdefghij"
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk(text)
        combined = "".join(chunks)
        assert combined == text

    def test_whitespace_only_chunk_skipped(self) -> None:
        text = "    "
        chunker = FixedSizeChunker(chunk_size=2, chunk_overlap=0)
        chunks = chunker.chunk(text)
        assert chunks == []

    def test_exact_size_text_single_chunk(self) -> None:
        text = "a" * 100
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Tests for SentenceChunker
# ---------------------------------------------------------------------------


class TestSentenceChunker:
    def test_single_sentence_single_chunk(self) -> None:
        chunker = SentenceChunker(chunk_size=500)
        chunks = chunker.chunk("This is one sentence.")
        assert len(chunks) == 1

    def test_multiple_sentences_within_budget(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunker = SentenceChunker(chunk_size=500)
        chunks = chunker.chunk(text)
        # All fit in one chunk
        assert len(chunks) == 1

    def test_sentences_split_when_over_budget(self) -> None:
        # Make each sentence long enough to force splits
        long_sentence = "This is a very long sentence that is long. "
        text = long_sentence * 10
        chunker = SentenceChunker(chunk_size=50)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_empty_text_returns_empty(self) -> None:
        chunker = SentenceChunker()
        assert chunker.chunk("") == []

    def test_overlap_retains_last_sentence(self) -> None:
        sentences = [f"Sentence {i} about topic." for i in range(20)]
        text = " ".join(sentences)
        chunker = SentenceChunker(chunk_size=60, chunk_overlap=1)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_no_empty_chunks(self) -> None:
        text = "Hello world. Foo bar."
        chunker = SentenceChunker(chunk_size=100)
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert chunk.strip() != ""


# ---------------------------------------------------------------------------
# Tests for ParagraphChunker
# ---------------------------------------------------------------------------


class TestParagraphChunker:
    def test_single_paragraph_single_chunk(self) -> None:
        chunker = ParagraphChunker(chunk_size=1000)
        chunks = chunker.chunk("This is a single paragraph.")
        assert len(chunks) == 1

    def test_multiple_paragraphs_merged_when_small(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        chunker = ParagraphChunker(chunk_size=500)
        chunks = chunker.chunk(text)
        # Small paragraphs merged into one
        assert len(chunks) == 1

    def test_large_paragraphs_split(self) -> None:
        para = "word " * 300
        text = para + "\n\n" + para
        chunker = ParagraphChunker(chunk_size=500)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_empty_text(self) -> None:
        chunker = ParagraphChunker()
        assert chunker.chunk("") == []

    def test_ignores_blank_lines_as_separators(self) -> None:
        text = "First.\n\n\n\nSecond."
        chunker = ParagraphChunker(chunk_size=500)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1  # Both fit in one chunk

    def test_no_empty_chunks(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        chunker = ParagraphChunker()
        for chunk in chunker.chunk(text):
            assert chunk.strip() != ""


# ---------------------------------------------------------------------------
# Tests for RecursiveChunker
# ---------------------------------------------------------------------------


class TestRecursiveChunker:
    def test_short_text_single_chunk(self) -> None:
        chunker = RecursiveChunker(chunk_size=500)
        chunks = chunker.chunk("Short text here.")
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self) -> None:
        text = "word " * 200
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_empty_text_returns_empty(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []

    def test_no_empty_chunks(self, long_text: str) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(long_text)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_chunks_cover_content(self) -> None:
        text = "The quick brown fox."
        chunker = RecursiveChunker(chunk_size=100)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Original content should appear in one of the chunks
        assert any("quick" in c for c in chunks)

    def test_overlap_produces_more_chunks(self) -> None:
        text = "word " * 100
        chunker_no_overlap = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunker_overlap = RecursiveChunker(chunk_size=50, chunk_overlap=25)
        chunks_no_overlap = chunker_no_overlap.chunk(text)
        chunks_overlap = chunker_overlap.chunk(text)
        # With overlap, there may be more or equal chunks
        assert len(chunks_overlap) >= len(chunks_no_overlap)


# ---------------------------------------------------------------------------
# Tests for ChunkerFactory
# ---------------------------------------------------------------------------


class TestChunkerFactory:
    def test_fixed_size_strategy(self) -> None:
        config = RAGConfig(chunking_strategy=ChunkingStrategy.fixed_size)
        chunker = ChunkerFactory.create(config)
        assert isinstance(chunker, FixedSizeChunker)

    def test_sentence_strategy(self) -> None:
        config = RAGConfig(chunking_strategy=ChunkingStrategy.sentence)
        chunker = ChunkerFactory.create(config)
        assert isinstance(chunker, SentenceChunker)

    def test_paragraph_strategy(self) -> None:
        config = RAGConfig(chunking_strategy=ChunkingStrategy.paragraph)
        chunker = ChunkerFactory.create(config)
        assert isinstance(chunker, ParagraphChunker)

    def test_recursive_strategy(self) -> None:
        config = RAGConfig(chunking_strategy=ChunkingStrategy.recursive)
        chunker = ChunkerFactory.create(config)
        assert isinstance(chunker, RecursiveChunker)

    def test_semantic_falls_back_to_sentence(self) -> None:
        config = RAGConfig(chunking_strategy=ChunkingStrategy.semantic)
        chunker = ChunkerFactory.create(config)
        assert isinstance(chunker, SentenceChunker)

    def test_chunker_protocol(self) -> None:
        config = RAGConfig(chunking_strategy=ChunkingStrategy.fixed_size)
        chunker = ChunkerFactory.create(config)
        assert isinstance(chunker, TextChunker)

    @pytest.mark.parametrize("strategy", list(ChunkingStrategy))
    def test_all_strategies_create_chunker(self, strategy: ChunkingStrategy) -> None:
        config = RAGConfig(chunking_strategy=strategy)
        chunker = ChunkerFactory.create(config)
        assert hasattr(chunker, "chunk")


# ---------------------------------------------------------------------------
# Tests for _simulate_retrieval
# ---------------------------------------------------------------------------


class TestSimulateRetrieval:
    def test_returns_correct_count(self) -> None:
        chunks = ["python code", "sql database", "machine learning"]
        result = _simulate_retrieval("python", chunks, top_k=2)
        assert len(result) == 2

    def test_relevant_chunk_ranked_first(self) -> None:
        chunks = ["completely unrelated", "python programming language", "sql queries"]
        result = _simulate_retrieval("python programming", chunks, top_k=3)
        assert result[0] == 1  # "python programming language" should be first

    def test_empty_query_returns_top_k(self) -> None:
        chunks = ["a", "b", "c", "d"]
        result = _simulate_retrieval("", chunks, top_k=2)
        assert len(result) == 2

    def test_no_chunks(self) -> None:
        result = _simulate_retrieval("query", [], top_k=5)
        assert result == []

    def test_top_k_larger_than_chunks(self) -> None:
        chunks = ["chunk1", "chunk2"]
        result = _simulate_retrieval("query", chunks, top_k=10)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Tests for metrics functions
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_precision_at_k_perfect(self) -> None:
        assert precision_at_k([0, 1, 2], [0, 1, 2], k=3) == 1.0

    def test_precision_at_k_zero(self) -> None:
        assert precision_at_k([3, 4, 5], [0, 1, 2], k=3) == 0.0

    def test_precision_at_k_partial(self) -> None:
        score = precision_at_k([0, 3, 1], [0, 1], k=3)
        assert abs(score - 2 / 3) < 1e-9

    def test_precision_at_k_zero_k(self) -> None:
        assert precision_at_k([0, 1], [0, 1], k=0) == 0.0

    def test_recall_at_k_perfect(self) -> None:
        assert recall_at_k([0, 1, 2], [0, 1, 2], k=3) == 1.0

    def test_recall_at_k_zero(self) -> None:
        assert recall_at_k([3, 4, 5], [0, 1, 2], k=3) == 0.0

    def test_recall_at_k_partial(self) -> None:
        score = recall_at_k([0, 3], [0, 1], k=2)
        assert abs(score - 0.5) < 1e-9

    def test_recall_at_k_empty_relevant(self) -> None:
        assert recall_at_k([0, 1], [], k=2) == 0.0

    def test_recall_at_k_zero_k(self) -> None:
        assert recall_at_k([0, 1], [0, 1], k=0) == 0.0

    def test_mrr_first_hit_rank_1(self) -> None:
        assert mean_reciprocal_rank([0, 1, 2], [0]) == 1.0

    def test_mrr_first_hit_rank_2(self) -> None:
        score = mean_reciprocal_rank([3, 0, 1], [0])
        assert abs(score - 0.5) < 1e-9

    def test_mrr_no_hit(self) -> None:
        assert mean_reciprocal_rank([3, 4, 5], [0, 1]) == 0.0

    def test_mrr_empty_relevant(self) -> None:
        assert mean_reciprocal_rank([0, 1], []) == 0.0

    def test_ndcg_at_k_perfect(self) -> None:
        score = ndcg_at_k([0, 1, 2], [0, 1, 2], k=3)
        assert abs(score - 1.0) < 1e-9

    def test_ndcg_at_k_zero(self) -> None:
        score = ndcg_at_k([3, 4], [0, 1], k=2)
        assert score == 0.0

    def test_ndcg_at_k_zero_k(self) -> None:
        assert ndcg_at_k([0, 1], [0, 1], k=0) == 0.0

    def test_ndcg_at_k_empty_relevant(self) -> None:
        assert ndcg_at_k([0, 1], [], k=2) == 0.0

    def test_ndcg_at_k_partial(self) -> None:
        score = ndcg_at_k([0, 3, 1], [0, 1], k=3)
        assert 0.0 < score <= 1.0

    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_precision_in_range(self, k: int) -> None:
        score = precision_at_k([0, 1, 2, 3], [0, 2], k=k)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_recall_in_range(self, k: int) -> None:
        score = recall_at_k([0, 1, 2, 3], [0, 2], k=k)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests for RAGBenchmark
# ---------------------------------------------------------------------------


class TestRAGBenchmark:
    def test_run_returns_result(
        self,
        benchmark: RAGBenchmark,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
        default_config: RAGConfig,
    ) -> None:
        result = benchmark.run(default_config, corpus, queries, ground_truth)
        assert isinstance(result, RAGBenchmarkResult)

    def test_metrics_in_range(
        self,
        benchmark: RAGBenchmark,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
        default_config: RAGConfig,
    ) -> None:
        result = benchmark.run(default_config, corpus, queries, ground_truth)
        assert 0.0 <= result.precision_at_k <= 1.0
        assert 0.0 <= result.recall_at_k <= 1.0
        assert 0.0 <= result.mrr <= 1.0

    def test_latency_positive(
        self,
        benchmark: RAGBenchmark,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
        default_config: RAGConfig,
    ) -> None:
        result = benchmark.run(default_config, corpus, queries, ground_truth)
        assert result.latency_ms >= 0.0

    def test_cost_per_query_positive(
        self,
        benchmark: RAGBenchmark,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
        default_config: RAGConfig,
    ) -> None:
        result = benchmark.run(default_config, corpus, queries, ground_truth)
        assert result.cost_per_query > 0.0

    def test_config_preserved_in_result(
        self,
        benchmark: RAGBenchmark,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
        default_config: RAGConfig,
    ) -> None:
        result = benchmark.run(default_config, corpus, queries, ground_truth)
        assert result.config == default_config

    @pytest.mark.parametrize("strategy", [
        ChunkingStrategy.fixed_size,
        ChunkingStrategy.sentence,
        ChunkingStrategy.paragraph,
        ChunkingStrategy.recursive,
    ])
    def test_all_strategies_run(
        self,
        benchmark: RAGBenchmark,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
        strategy: ChunkingStrategy,
    ) -> None:
        config = RAGConfig(chunking_strategy=strategy, chunk_size=200)
        result = benchmark.run(config, corpus, queries, ground_truth)
        assert isinstance(result, RAGBenchmarkResult)

    def test_empty_queries_single_entry(
        self,
        benchmark: RAGBenchmark,
        corpus: list[str],
    ) -> None:
        result = benchmark.run(RAGConfig(), corpus, ["query"], [[0]])
        assert isinstance(result, RAGBenchmarkResult)


# ---------------------------------------------------------------------------
# Tests for RAGOptimizer
# ---------------------------------------------------------------------------


class TestRAGOptimizer:
    def test_optimize_returns_result(
        self,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
    ) -> None:
        optimizer = RAGOptimizer()
        search_space = [
            RAGConfig(chunking_strategy=ChunkingStrategy.fixed_size, chunk_size=100),
            RAGConfig(chunking_strategy=ChunkingStrategy.sentence, chunk_size=100),
        ]
        result = optimizer.optimize(corpus, queries, ground_truth, search_space)
        assert isinstance(result, OptimizationResult)

    def test_optimize_empty_search_space_raises(
        self, corpus: list[str], queries: list[str], ground_truth: list[list[int]]
    ) -> None:
        optimizer = RAGOptimizer()
        with pytest.raises(ValueError, match="search_space"):
            optimizer.optimize(corpus, queries, ground_truth, [])

    def test_optimize_best_config_is_in_search_space(
        self,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
    ) -> None:
        search_space = [
            RAGConfig(chunking_strategy=ChunkingStrategy.fixed_size, chunk_size=100),
            RAGConfig(chunking_strategy=ChunkingStrategy.sentence, chunk_size=200),
        ]
        optimizer = RAGOptimizer()
        result = optimizer.optimize(corpus, queries, ground_truth, search_space)
        assert result.best_config in search_space

    def test_optimize_all_results_present(
        self,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
    ) -> None:
        search_space = [
            RAGConfig(chunking_strategy=ChunkingStrategy.fixed_size, chunk_size=100),
            RAGConfig(chunking_strategy=ChunkingStrategy.sentence, chunk_size=200),
            RAGConfig(chunking_strategy=ChunkingStrategy.paragraph, chunk_size=300),
        ]
        optimizer = RAGOptimizer()
        result = optimizer.optimize(corpus, queries, ground_truth, search_space)
        assert len(result.all_results) == 3

    def test_optimize_single_config_improvement_zero_or_more(
        self,
        corpus: list[str],
        queries: list[str],
        ground_truth: list[list[int]],
    ) -> None:
        search_space = [RAGConfig()]
        optimizer = RAGOptimizer()
        result = optimizer.optimize(corpus, queries, ground_truth, search_space)
        # With one config, best == baseline -> improvement = 0
        assert isinstance(result.improvement_pct, float)


# ---------------------------------------------------------------------------
# Tests for models validation
# ---------------------------------------------------------------------------


class TestModels:
    def test_rag_config_defaults(self) -> None:
        config = RAGConfig()
        assert config.chunking_strategy == ChunkingStrategy.fixed_size
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.top_k == 5

    def test_rag_config_chunk_size_min_1(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            RAGConfig(chunk_size=0)

    def test_rag_config_top_k_min_1(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            RAGConfig(top_k=0)

    def test_rag_config_chunk_overlap_min_0(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            RAGConfig(chunk_overlap=-1)

    def test_chunking_strategy_values(self) -> None:
        assert ChunkingStrategy.fixed_size.value == "fixed_size"
        assert ChunkingStrategy.sentence.value == "sentence"
        assert ChunkingStrategy.paragraph.value == "paragraph"
        assert ChunkingStrategy.recursive.value == "recursive"
        assert ChunkingStrategy.semantic.value == "semantic"

    def test_retrieval_method_values(self) -> None:
        assert RetrievalMethod.dense.value == "dense"
        assert RetrievalMethod.sparse.value == "sparse"
        assert RetrievalMethod.hybrid.value == "hybrid"
        assert RetrievalMethod.rerank.value == "rerank"

    def test_rag_benchmark_result_score_range(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            RAGBenchmarkResult(
                config=RAGConfig(),
                precision_at_k=1.5,  # > 1.0
                recall_at_k=0.5,
                mrr=0.5,
                latency_ms=1.0,
                cost_per_query=0.001,
            )
