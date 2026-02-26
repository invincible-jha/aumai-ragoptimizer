"""Retrieval evaluation metrics for aumai-ragoptimizer."""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant.

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant:  Set of relevant document indices (ground truth).
        k:         Cut-off position.

    Returns:
        Precision@k in [0, 1].
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved results.

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant:  Set of relevant document indices (ground truth).
        k:         Cut-off position.

    Returns:
        Recall@k in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved: list[int], relevant: list[int]) -> float:
    """Reciprocal rank of the first relevant document in the retrieved list.

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant:  Set of relevant document indices (ground truth).

    Returns:
        MRR in [0, 1].
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Assumes binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant:  Set of relevant document indices (ground truth).
        k:         Cut-off position.

    Returns:
        NDCG@k in [0, 1].
    """
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]

    def dcg(items: list[int]) -> float:
        return sum(
            (1.0 if doc_id in relevant_set else 0.0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(items, start=1)
        )

    actual_dcg = dcg(top_k)
    # Ideal DCG: place all relevant docs first.
    ideal_top_k = [doc_id for doc_id in relevant if doc_id in relevant_set][:k]
    ideal_dcg = dcg(ideal_top_k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
]
