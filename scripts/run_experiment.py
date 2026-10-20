"""Runs every model on the validation and test splits, the re-ranker, and the replay A/B test.
Writes docs/results/*.json and docs/figures/*.png.

    python scripts/run_experiment.py --data data/online_retail_ii.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nba import abtest, baselines, basket_transformer, data, metrics, rerank, sasrec

VAL_CUTOFF, TEST_CUTOFF = "2011-06-01", "2011-09-01"


def fit_all(split: data.Split, cfg: sasrec.SASRecConfig, bcfg: basket_transformer.BasketConfig, log):
    customers = sorted(split.targets)
    out, timings = {}, {}
    t = time.time()
    pop = baselines.Popularity().fit(split.train, split.cutoff)
    timings["popularity"] = time.time() - t
    models = {"repeat": baselines.Repeat(), "knn": baselines.ItemKNN(), "als": baselines.ImplicitALS()}
    for name, m in models.items():
        t = time.time()
        m.fit(split.train, split.cutoff)
        out[name] = {c: m.scores(c) for c in customers}
        timings[name] = time.time() - t
    t = time.time()
    sr = sasrec.SASRec(cfg).fit(data.sequences(split.train), log=log)
    out["sasrec"] = sr.scores_batch(customers)
    timings["sasrec"] = time.time() - t
    t = time.time()
    bt = basket_transformer.BasketTransformer(bcfg).fit(split.train, log=log)
    out["basket_tf"] = bt.scores_batch(customers)
    timings["basket_tf"] = time.time() - t
    return pop, out, timings, {"sasrec": sr.history, "basket_tf": bt.history}


def ranked(scores: dict[int, dict[str, float]], k=20, fallback: list[str] | None = None):
    recs = {}
    for c, s in scores.items():
        r = baselines.top_k(s, k)
        if fallback and len(r) < k:  # fill short lists (e.g. new-ish customers) with best sellers
            r += [i for i in fallback if i not in r][: k - len(r)]
        recs[c] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/online_retail_ii.parquet")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--basket-epochs", type=int, default=15)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = data.clean(data.load(args.data))
    summary = {"rows": len(df), "customers": int(df["customer_id"].nunique()), "items": int(df["stock_code"].nunique()),
               "baskets": int(df["invoice"].nunique()), "first": str(df["invoice_date"].min()),
               "last": str(df["invoice_date"].max())}
    print("data", summary)
    val = data.temporal_split(df[df["invoice_date"] < pd.Timestamp(TEST_CUTOFF)], VAL_CUTOFF)
    test = data.temporal_split(df, TEST_CUTOFF)
    summary.update({"val_customers": len(val.targets), "test_customers": len(test.targets),
                    "test_basket_size_mean": float(pd.Series([len(v) for v in test.targets.values()]).mean())})
    cfg = sasrec.SASRecConfig(epochs=args.epochs)
    bcfg = basket_transformer.BasketConfig(epochs=args.basket_epochs)

    results = {}
    for name, split in (("val", val), ("test", test)):
        pop, scores, timings, hist = fit_all(split, cfg, bcfg, log=lambda m: print(name, m, flush=True))
        history = split.train.groupby("customer_id")["stock_code"].agg(set).to_dict()
        recs = {"popularity": {c: pop.recommend(c, 20) for c in split.targets}}
        for m, s in scores.items():
            recs[m] = ranked(s, fallback=pop.ranked)
        cand = rerank.candidates(sorted(split.targets), scores, pop.score, split.train, split.cutoff)
        results[name] = {"split": split, "recs": recs, "cand": cand, "history": history, "timings": timings,
                         "sasrec_loss": hist, "pop": pop}

    rr = rerank.Reranker().fit(results["val"]["cand"], val.targets)
    results["test"]["recs"]["rerank"] = rr.recommend(results["test"]["cand"])
    n_items = summary["items"]
    table = {}
    for m, r in results["test"]["recs"].items():
        table[m] = metrics.evaluate(r, test.targets, history=results["test"]["history"])
        table[m]["coverage@10"] = metrics.coverage(r, n_items)
    val_table = {m: metrics.evaluate(r, val.targets) for m, r in results["val"]["recs"].items()}
    for m, v in table.items():
        print(f"{m:11s} " + " ".join(f"{k}={v[k]:.4f}" for k in ("hit@10", "recall@10", "ndcg@10", "new_item_hit@10",
                                                                  "coverage@10")))

    ab = {
        "rerank_vs_popularity": abtest.replay_ab(results["test"]["recs"]["popularity"], results["test"]["recs"]["rerank"],
                                                 test.targets, k=5),
        "rerank_vs_repeat": abtest.replay_ab(results["test"]["recs"]["repeat"], results["test"]["recs"]["rerank"],
                                             test.targets, k=5),
        "basket_tf_vs_repeat": abtest.replay_ab(results["test"]["recs"]["repeat"], results["test"]["recs"]["basket_tf"],
                                                test.targets, k=5),
        "rerank_vs_basket_tf": abtest.replay_ab(results["test"]["recs"]["basket_tf"], results["test"]["recs"]["rerank"],
                                                test.targets, k=5),
    }
    print("ab", json.dumps(ab, default=str)[:1500])

    json.dump({"data": summary, "test": table, "val": val_table, "timings_s": results["test"]["timings"],
               "train_loss": results["test"]["sasrec_loss"], "reranker_gain": rr.importance(),
               "config": {"val_cutoff": VAL_CUTOFF, "test_cutoff": TEST_CUTOFF, "sasrec": vars(cfg), "basket_tf": vars(bcfg)}},
              open(out / "results" / "metrics.json", "w"), indent=2, default=str)
    json.dump(ab, open(out / "results" / "ab_test.json", "w"), indent=2, default=str)

    names = list(table)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(names, [table[m]["ndcg@10"] for m in names], color="#4C72B0")
    ax.set_ylabel("NDCG@10 (test)")
    ax.set_title("Next-basket ranking quality, Sep-Dec 2011")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out / "figures" / "ndcg_by_model.png", dpi=150)


if __name__ == "__main__":
    main()
