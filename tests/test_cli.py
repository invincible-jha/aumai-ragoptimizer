"""Comprehensive CLI tests for aumai-ragoptimizer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_ragoptimizer.cli import main


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from mixed CLI output."""
    start = text.index("{")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("No JSON object found in output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SAMPLE_CORPUS = [
    {"text": "Python is a high-level programming language."},
    {"text": "Machine learning requires large datasets for training."},
    {"text": "Vector databases enable efficient similarity search."},
]

SAMPLE_QUERIES = [
    {"query": "python programming", "relevant_docs": [0]},
    {"query": "machine learning data", "relevant_docs": [1]},
    {"query": "vector search", "relevant_docs": [2]},
]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# version / help
# ---------------------------------------------------------------------------


class TestCliMeta:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ragoptimizer" in result.output.lower() or "RAG" in result.output


# ---------------------------------------------------------------------------
# `benchmark` command
# ---------------------------------------------------------------------------


class TestBenchmarkCommand:
    def test_benchmark_requires_corpus(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(main, ["benchmark", "--queries", "queries.jsonl"])
            assert result.exit_code != 0

    def test_benchmark_requires_queries(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            result = runner.invoke(main, ["benchmark", "--corpus", "corpus.jsonl"])
            assert result.exit_code != 0

    def test_benchmark_basic_run(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(
                main,
                [
                    "benchmark",
                    "--corpus", "corpus.jsonl",
                    "--queries", "queries.jsonl",
                    "--output", "-",
                ],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert "precision_at_k" in data
            assert "recall_at_k" in data
            assert "mrr" in data

    def test_benchmark_to_file(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(
                main,
                [
                    "benchmark",
                    "--corpus", "corpus.jsonl",
                    "--queries", "queries.jsonl",
                    "--output", "result.json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(Path("result.json").read_text(encoding="utf-8"))
            assert "precision_at_k" in data

    def test_benchmark_with_json_config_file(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            config = {
                "chunking_strategy": "sentence",
                "chunk_size": 100,
                "chunk_overlap": 0,
                "top_k": 3,
            }
            Path("config.json").write_text(json.dumps(config), encoding="utf-8")
            result = runner.invoke(
                main,
                [
                    "benchmark",
                    "--corpus", "corpus.jsonl",
                    "--queries", "queries.jsonl",
                    "--config", "config.json",
                ],
            )
            assert result.exit_code == 0

    @pytest.mark.parametrize("strategy", ["fixed_size", "sentence", "paragraph", "recursive"])
    def test_benchmark_all_strategies(self, strategy: str) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(
                main,
                [
                    "benchmark",
                    "--corpus", "corpus.jsonl",
                    "--queries", "queries.jsonl",
                    "--strategy", strategy,
                ],
            )
            assert result.exit_code == 0

    def test_benchmark_top_k_option(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(
                main,
                [
                    "benchmark",
                    "--corpus", "corpus.jsonl",
                    "--queries", "queries.jsonl",
                    "--top-k", "3",
                ],
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# `optimize` command
# ---------------------------------------------------------------------------


class TestOptimizeCommand:
    def test_optimize_requires_corpus(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(main, ["optimize", "--queries", "queries.jsonl"])
            assert result.exit_code != 0

    def test_optimize_basic_run(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(
                main,
                [
                    "optimize",
                    "--corpus", "corpus.jsonl",
                    "--queries", "queries.jsonl",
                ],
            )
            assert result.exit_code == 0
            data = _extract_json(result.output)
            assert "best_config" in data
            assert "all_results" in data
            assert "improvement_pct" in data

    def test_optimize_to_file(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_jsonl(Path("corpus.jsonl"), SAMPLE_CORPUS)
            _write_jsonl(Path("queries.jsonl"), SAMPLE_QUERIES)
            result = runner.invoke(
                main,
                [
                    "optimize",
                    "--corpus", "corpus.jsonl",
                    "--queries", "queries.jsonl",
                    "--output", "opt.json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(Path("opt.json").read_text(encoding="utf-8"))
            assert "best_config" in data


# ---------------------------------------------------------------------------
# `chunk` command
# ---------------------------------------------------------------------------


class TestChunkCommand:
    def test_chunk_requires_strategy_and_input(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["chunk"])
        assert result.exit_code != 0

    def test_chunk_fixed_size_stdout(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("input.txt").write_text("Hello world. " * 100, encoding="utf-8")
            result = runner.invoke(
                main,
                [
                    "chunk",
                    "--strategy", "fixed_size",
                    "--input", "input.txt",
                    "--chunk-size", "50",
                    "--chunk-overlap", "0",
                ],
            )
            assert result.exit_code == 0
            lines = [ln for ln in result.output.strip().split("\n") if ln.strip()]
            assert len(lines) > 0
            json_lines = [ln for ln in lines if ln.strip().startswith("{")]
            assert len(json_lines) > 0
            for line in json_lines:
                obj = json.loads(line)
                assert "chunk" in obj
                assert "index" in obj
                assert "length" in obj

    def test_chunk_to_file(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("input.txt").write_text("Some text. " * 50, encoding="utf-8")
            result = runner.invoke(
                main,
                [
                    "chunk",
                    "--strategy", "sentence",
                    "--input", "input.txt",
                    "--output", "chunks.jsonl",
                ],
            )
            assert result.exit_code == 0
            content = Path("chunks.jsonl").read_text(encoding="utf-8")
            lines = [l for l in content.strip().split("\n") if l.strip()]
            assert len(lines) > 0

    @pytest.mark.parametrize("strategy", ["fixed_size", "sentence", "paragraph", "recursive"])
    def test_chunk_all_strategies(self, strategy: str) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("input.txt").write_text(
                "First paragraph.\n\nSecond paragraph. With more text. " * 5,
                encoding="utf-8",
            )
            result = runner.invoke(
                main,
                [
                    "chunk",
                    "--strategy", strategy,
                    "--input", "input.txt",
                ],
            )
            assert result.exit_code == 0
