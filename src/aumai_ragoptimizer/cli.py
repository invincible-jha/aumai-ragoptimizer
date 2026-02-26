"""CLI entry point for aumai-ragoptimizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from aumai_ragoptimizer.core import ChunkerFactory, RAGBenchmark, RAGOptimizer
from aumai_ragoptimizer.models import (
    ChunkingStrategy,
    RAGConfig,
    RetrievalMethod,
)


def _load_jsonl(path: str) -> list[dict[str, object]]:
    """Load a .jsonl file into a list of dicts."""
    records: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_corpus(path: str) -> list[str]:
    """Load corpus JSONL where each record has a 'text' field."""
    records = _load_jsonl(path)
    return [str(r.get("text", r.get("content", ""))) for r in records]


def _load_queries(path: str) -> tuple[list[str], list[list[int]]]:
    """Load queries JSONL: each record has 'query' and 'relevant_docs' fields."""
    records = _load_jsonl(path)
    queries: list[str] = []
    ground_truth: list[list[int]] = []
    for record in records:
        queries.append(str(record.get("query", "")))
        relevant_raw = record.get("relevant_docs", [])
        if isinstance(relevant_raw, list):
            ground_truth.append([int(x) for x in relevant_raw])
        else:
            ground_truth.append([])
    return queries, ground_truth


@click.group()
@click.version_option()
def main() -> None:
    """AumAI RAGOptimizer — benchmark and optimize RAG pipeline configurations."""


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------


@main.command("benchmark")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, readable=True),
    default=None,
    help="YAML or JSON file with a RAGConfig definition.",
)
@click.option(
    "--corpus",
    "corpus_path",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="JSONL file with documents (field: 'text').",
)
@click.option(
    "--queries",
    "queries_path",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="JSONL file with queries (fields: 'query', 'relevant_docs').",
)
@click.option("--top-k", default=5, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--strategy",
    default="fixed_size",
    show_default=True,
    type=click.Choice([s.value for s in ChunkingStrategy], case_sensitive=False),
)
@click.option(
    "--output",
    default="-",
    type=click.Path(allow_dash=True),
    show_default=True,
    help="Output file for results JSON. '-' for stdout.",
)
def benchmark_cmd(
    config_path: str | None,
    corpus_path: str,
    queries_path: str,
    top_k: int,
    strategy: str,
    output: str,
) -> None:
    """Run a RAG benchmark against a corpus and query set."""
    if config_path is not None:
        raw = Path(config_path).read_text(encoding="utf-8")
        if config_path.endswith(".json"):
            rag_config = RAGConfig.model_validate(json.loads(raw))
        else:
            try:
                import yaml  # type: ignore[import-untyped]

                rag_config = RAGConfig.model_validate(yaml.safe_load(raw))
            except ImportError:
                raise click.ClickException("Install PyYAML to use YAML config files.")
    else:
        rag_config = RAGConfig(
            chunking_strategy=ChunkingStrategy(strategy),
            top_k=top_k,
        )

    corpus = _load_corpus(corpus_path)
    queries, ground_truth = _load_queries(queries_path)

    click.echo(
        f"Benchmarking {rag_config.chunking_strategy.value} strategy on "
        f"{len(corpus)} docs / {len(queries)} queries...",
        err=True,
    )

    benchmark = RAGBenchmark()
    result = benchmark.run(rag_config, corpus, queries, ground_truth)

    output_data = result.model_dump()

    out_text = json.dumps(output_data, indent=2)
    if output == "-":
        click.echo(out_text)
    else:
        Path(output).write_text(out_text, encoding="utf-8")
        click.echo(f"Results written to: {output}", err=True)

    click.echo(
        f"\nPrecision@{rag_config.top_k}: {result.precision_at_k:.4f}  "
        f"Recall@{rag_config.top_k}: {result.recall_at_k:.4f}  "
        f"MRR: {result.mrr:.4f}  "
        f"Latency: {result.latency_ms:.2f}ms",
        err=True,
    )


# ---------------------------------------------------------------------------
# optimize command
# ---------------------------------------------------------------------------


@main.command("optimize")
@click.option(
    "--corpus",
    "corpus_path",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="JSONL file with documents.",
)
@click.option(
    "--queries",
    "queries_path",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="JSONL file with queries and ground truth.",
)
@click.option(
    "--output",
    default="-",
    type=click.Path(allow_dash=True),
    show_default=True,
)
def optimize_cmd(corpus_path: str, queries_path: str, output: str) -> None:
    """Grid-search over RAG configurations to find the optimal one."""
    corpus = _load_corpus(corpus_path)
    queries, ground_truth = _load_queries(queries_path)

    # Build a compact search space covering key knobs.
    search_space: list[RAGConfig] = [
        RAGConfig(
            chunking_strategy=strategy,
            chunk_size=size,
            chunk_overlap=overlap,
            retrieval_method=retrieval,
            top_k=5,
        )
        for strategy in [
            ChunkingStrategy.fixed_size,
            ChunkingStrategy.sentence,
            ChunkingStrategy.paragraph,
        ]
        for size in [256, 512]
        for overlap in [0, 64]
        for retrieval in [RetrievalMethod.dense, RetrievalMethod.sparse]
    ]

    click.echo(
        f"Optimizing over {len(search_space)} configurations...", err=True
    )

    optimizer = RAGOptimizer()
    opt_result = optimizer.optimize(corpus, queries, ground_truth, search_space)

    output_data = opt_result.model_dump()
    out_text = json.dumps(output_data, indent=2)
    if output == "-":
        click.echo(out_text)
    else:
        Path(output).write_text(out_text, encoding="utf-8")
        click.echo(f"Results written to: {output}", err=True)

    best = opt_result.best_config
    click.echo(
        f"\nBest config: strategy={best.chunking_strategy.value} "
        f"size={best.chunk_size} overlap={best.chunk_overlap} "
        f"retrieval={best.retrieval_method.value}  "
        f"improvement={opt_result.improvement_pct:.1f}%",
        err=True,
    )


# ---------------------------------------------------------------------------
# chunk command
# ---------------------------------------------------------------------------


@main.command("chunk")
@click.option(
    "--strategy",
    required=True,
    type=click.Choice([s.value for s in ChunkingStrategy], case_sensitive=False),
    help="Chunking strategy to apply.",
)
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Input text file to chunk.",
)
@click.option(
    "--chunk-size",
    default=512,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option(
    "--chunk-overlap",
    default=64,
    show_default=True,
    type=click.IntRange(min=0),
)
@click.option(
    "--output",
    default="-",
    type=click.Path(allow_dash=True),
    show_default=True,
    help="Output file (.jsonl). '-' for stdout.",
)
def chunk_cmd(
    strategy: str,
    input_path: str,
    chunk_size: int,
    chunk_overlap: int,
    output: str,
) -> None:
    """Chunk a text file using the specified strategy and print the results."""
    text = Path(input_path).read_text(encoding="utf-8")

    config = RAGConfig(
        chunking_strategy=ChunkingStrategy(strategy),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunker = ChunkerFactory.create(config)
    chunks = chunker.chunk(text)

    click.echo(f"Produced {len(chunks)} chunks.", err=True)

    if output == "-":
        out_fh = sys.stdout
    else:
        out_fh = open(output, "w", encoding="utf-8")  # noqa: SIM115

    try:
        for idx, chunk in enumerate(chunks):
            out_fh.write(
                json.dumps({"index": idx, "chunk": chunk, "length": len(chunk)}) + "\n"
            )
    finally:
        if output != "-":
            out_fh.close()


if __name__ == "__main__":
    main()
