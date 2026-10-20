"""Two-stage next-best-action: candidates from every model, then a LightGBM LambdaRank model
that learns how to weigh them (and simple customer/item facts) for each customer."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["repeat", "repeat_rank", "knn", "knn_rank", "als", "als_rank", "sasrec", "sasrec_rank", "basket_tf", "basket_tf_rank", "pop",
            "pop_rank", "bought_before", "days_since_item", "item_buyers_30d", "cust_baskets", "cust_days_since"]


def _rank(scores: dict[str, float]) -> dict[str, int]:
    return {i: r for r, (i, _) in enumerate(sorted(scores.items(), key=lambda t: (-t[1], t[0])))}


def candidates(customers, model_scores: dict[str, dict[int, dict[str, float]]], pop_scores: dict[str, float],
               train: pd.DataFrame, cutoff: pd.Timestamp, per_model: int = 50) -> pd.DataFrame:
    """One row per (customer, candidate item) with every model's score and rank as features."""
    last_item = train.groupby(["customer_id", "stock_code"])["invoice_date"].max()
    last_cust = train.groupby("customer_id")["invoice_date"].max()
    baskets = train.groupby("customer_id")["invoice"].nunique()
    recent = train[train["invoice_date"] >= cutoff - pd.Timedelta(days=30)]
    buyers30 = recent.groupby("stock_code")["customer_id"].nunique()
    pop_rank = _rank(pop_scores)
    pop_top = [i for i, _ in sorted(pop_scores.items(), key=lambda t: -t[1])[:per_model]]
    rows = []
    for c in customers:
        per = {m: s.get(c, {}) for m, s in model_scores.items()}
        ranks = {m: _rank(s) for m, s in per.items()}
        cand = set(pop_top)
        for m, r in ranks.items():
            cand.update(i for i, k in r.items() if k < per_model)
        for i in cand:
            row = {"customer_id": c, "item": i}
            for m in ("repeat", "knn", "als", "sasrec", "basket_tf"):
                row[m] = per.get(m, {}).get(i, 0.0)
                row[f"{m}_rank"] = ranks.get(m, {}).get(i, 999)
            row["pop"] = pop_scores.get(i, 0.0)
            row["pop_rank"] = pop_rank.get(i, 9999)
            li = last_item.get((c, i))
            row["bought_before"] = float(li is not None)
            row["days_since_item"] = (cutoff - li).days if li is not None else 9999
            row["item_buyers_30d"] = float(buyers30.get(i, 0))
            row["cust_baskets"] = float(baskets.get(c, 0))
            lc = last_cust.get(c)
            row["cust_days_since"] = (cutoff - lc).days if lc is not None else 9999
            rows.append(row)
    return pd.DataFrame(rows)


class Reranker:
    def __init__(self, seed: int = 0):
        self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                    random_state=seed, verbose=-1)

    def fit(self, cand: pd.DataFrame, targets: dict[int, set[str]]):
        df = cand[cand["customer_id"].isin(targets.keys())].sort_values("customer_id", kind="stable")
        y = np.array([i in targets[c] for c, i in zip(df["customer_id"], df["item"])], dtype=int)
        groups = df.groupby("customer_id", sort=False).size().values
        self.model.fit(df[FEATURES], y, group=groups, eval_at=[10])
        return self

    def recommend(self, cand: pd.DataFrame, k: int = 20) -> dict[int, list[str]]:
        df = cand.copy()
        df["score"] = self.model.predict(df[FEATURES])
        df = df.sort_values(["customer_id", "score", "item"], ascending=[True, False, True])
        return df.groupby("customer_id")["item"].apply(lambda s: list(s[:k])).to_dict()

    def importance(self) -> dict[str, float]:
        gain = self.model.booster_.feature_importance("gain")
        return {f: float(g) for f, g in sorted(zip(FEATURES, gain), key=lambda t: -t[1])}
