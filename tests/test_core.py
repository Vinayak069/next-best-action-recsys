import numpy as np
import pandas as pd
import pytest
import torch

from nba import abtest, baselines, basket_transformer as bt, data, metrics


def raw():
    rows = [  # customer, invoice, date, code, qty, price
        (1, "100", "2011-01-01", "10001", 2, 1.0), (1, "100", "2011-01-01", "10002", 1, 2.0),
        (1, "101", "2011-02-01", "10001", 1, 1.0), (1, "C102", "2011-02-02", "10001", -1, 1.0),
        (1, "103", "2011-03-05", "10001", 1, 1.0), (1, "103", "2011-03-05", "10003", 1, 1.0),
        (2, "200", "2011-01-15", "10002", 1, 2.0), (2, "201", "2011-03-10", "10002", 1, 2.0),
        (2, "200", "2011-01-15", "POST", 1, 18.0), (None, "300", "2011-01-20", "10001", 1, 1.0),
        (3, "400", "2011-03-20", "10003", 1, 1.0),
    ]
    return pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "stock_code", "quantity", "price"]
                        ).assign(country="UK")


def test_clean_drops_cancellations_fees_and_guests():
    df = data.clean(raw())
    assert not df.invoice.str.startswith("C").any() and "POST" not in set(df.stock_code)
    assert df.customer_id.notna().all() and len(df) == 8


def test_temporal_split_is_warm_start_next_basket():
    s = data.temporal_split(data.clean(raw()), "2011-03-01")
    assert s.train.invoice_date.max() < pd.Timestamp("2011-03-01")
    assert s.targets == {1: {"10001", "10003"}, 2: {"10002"}}  # customer 3 has no history


def test_metrics():
    recs = {1: ["a", "b", "c"], 2: ["x", "y"]}
    tg = {1: {"a", "c"}, 2: {"z"}}
    m = metrics.evaluate(recs, tg, ks=(2,), history={1: {"a"}})
    assert m["hit@2"] == 0.5 and m["recall@2"] == 0.25
    assert m["ndcg@2"] == pytest.approx((1 / (1 + 1 / np.log2(3))) / 2)
    assert m["new_item_hit@10"] == pytest.approx(0.5)  # customer 1 finds c; customer 2 misses z
    assert metrics.coverage(recs, 10, k=2) == 0.4


def test_repeat_scores_bought_items():
    tr = data.temporal_split(data.clean(raw()), "2011-03-01")
    rep = baselines.Repeat().fit(tr.train, tr.cutoff)
    assert rep.recommend(1, 1) == ["10001"]  # bought twice, most recently


def test_history_features():
    items = torch.tensor([[[1, 2], [1, 0], [0, 0]]])  # baskets {1,2}, {1}, padding
    freq, rec, inb = bt.history_features(items, 4)
    assert freq[0, 1, 1] == pytest.approx(np.log1p(2)) and freq[0, 1, 2] == pytest.approx(np.log1p(1))
    assert rec[0, 1, 1] == 1.0 and rec[0, 1, 2] == pytest.approx(0.5) and rec[0, 1, 3] == 0
    assert inb[0, 1].tolist() == [0, 1, 0, 0] and freq[0, :, 0].sum() == 0


def test_attention_mask_never_gives_nan():
    net = bt.BasketNet(5, bt.BasketConfig(dim=8, heads=2, layers=1))
    items = torch.tensor([[[0, 0], [0, 0], [1, 2]], [[3, 0], [4, 1], [2, 0]]])  # left padding in row 0
    out = net.logits(net(items, torch.zeros(2, 3, dtype=torch.long)), items)
    assert torch.isfinite(out).all()


def test_abtest():
    rng = np.random.default_rng(0)
    a, b = (rng.random(4000) < 0.30).astype(float), (rng.random(4000) < 0.35).astype(float)
    r = abtest.two_proportion_test(a, b)
    assert r["p_value"] < 0.01 and r["ci95"][0] < r["abs_lift"] < r["ci95"][1]
    assert 1300 < abtest.sample_size_per_arm(0.30, 0.05) < 1450
