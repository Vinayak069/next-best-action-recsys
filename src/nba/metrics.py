"""Ranking metrics for next-basket recommendation (binary relevance: the item is in the basket)."""
from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(sum(gains) / ideal) if ideal else 0.0


def evaluate(recs: dict[int, list[str]], targets: dict[int, set[str]], ks=(10, 20),
             history: dict[int, set[str]] | None = None) -> dict[str, float]:
    """Mean metrics over customers with a target basket.

    hit@k: share of customers with at least one recommended item in their next basket (the
    "would they act on the offer" view). recall@k: share of the basket covered. If `history` is
    given, also scores only the items the customer has never bought (discovery).
    """
    out: dict[str, list[float]] = {}
    for cust, rel in targets.items():
        ranked = recs.get(cust, [])
        for k in ks:
            top = ranked[:k]
            hits = sum(1 for i in top if i in rel)
            out.setdefault(f"hit@{k}", []).append(float(hits > 0))
            out.setdefault(f"recall@{k}", []).append(hits / len(rel))
            out.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(ranked, rel, k))
        if history is not None:
            new_rel = rel - history.get(cust, set())
            if new_rel:
                new_ranked = [i for i in ranked if i not in history.get(cust, set())]
                out.setdefault("new_item_hit@10", []).append(float(any(i in new_rel for i in new_ranked[:10])))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["customers"] = len(targets)
    return res


def coverage(recs: dict[int, list[str]], n_items: int, k: int = 10) -> float:
    """Share of the catalogue that appears in at least one top-k list."""
    return len({i for r in recs.values() for i in r[:k]}) / n_items
