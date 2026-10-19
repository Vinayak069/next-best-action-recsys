"""Classical recommenders. Each exposes fit(train_df) and scores(customer) -> dict item -> score
(or recommend for a ranked list), so the re-ranker can use their scores as features."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


def top_k(scores: dict[str, float], k: int, exclude: set[str] | None = None) -> list[str]:
    items = [(s, i) for i, s in scores.items() if not exclude or i not in exclude]
    items.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in items[:k]]


class Popularity:
    """Most-bought items (by distinct customers) in the last `window_days` before the cutoff."""

    def __init__(self, window_days: int = 30):
        self.window_days = window_days

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp):
        recent = train[train["invoice_date"] >= cutoff - pd.Timedelta(days=self.window_days)]
        counts = recent.groupby("stock_code")["customer_id"].nunique()
        self.ranked = counts.sort_values(ascending=False, kind="stable").index.tolist()
        self.score = {i: float(c) / counts.max() for i, c in counts.items()}
        return self

    def scores(self, cust) -> dict[str, float]:
        return self.score

    def recommend(self, cust, k):
        return self.ranked[:k]


class Repeat:
    """The customer's own past items, scored by how often and how recently they were bought.
    Retail baskets are dominated by repurchases, so this is the baseline to beat."""

    def __init__(self, half_life_days: float = 60.0):
        self.half_life_days = half_life_days

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp):
        d = train.drop_duplicates(["customer_id", "invoice", "stock_code"])
        age = (cutoff - d["invoice_date"]).dt.total_seconds() / 86400.0
        w = np.power(0.5, age / self.half_life_days)
        s = pd.DataFrame({"c": d["customer_id"].values, "i": d["stock_code"].values, "w": w.values})
        g = s.groupby(["c", "i"])["w"].sum()
        self.score = defaultdict(dict)
        for (c, i), v in g.items():
            self.score[c][i] = float(v)
        return self

    def scores(self, cust):
        return self.score.get(cust, {})

    def recommend(self, cust, k):
        return top_k(self.scores(cust), k)


def interaction_matrix(train: pd.DataFrame):
    customers = np.sort(train["customer_id"].unique())
    items = np.sort(train["stock_code"].unique())
    ci = {c: n for n, c in enumerate(customers)}
    ii = {c: n for n, c in enumerate(items)}
    d = train.drop_duplicates(["customer_id", "invoice", "stock_code"])
    rows = d["customer_id"].map(ci).values
    cols = d["stock_code"].map(ii).values
    m = sp.csr_matrix((np.ones(len(d)), (rows, cols)), shape=(len(customers), len(items)))
    m.sum_duplicates()  # entries = number of baskets containing the item
    return m, customers, items, ci


class ItemKNN:
    """Item-item cosine similarity on who bought what; a customer's score for an item is the
    similarity to everything they bought."""

    def __init__(self, neighbours: int = 100):
        self.neighbours = neighbours

    def fit(self, train: pd.DataFrame, cutoff=None):
        m, self.customers, self.items, self.ci = interaction_matrix(train)
        b = (m > 0).astype(np.float32)
        norms = np.sqrt(np.asarray(b.sum(axis=0)).ravel()) + 1e-9
        sim = (b.T @ b).toarray() / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < sim.shape[0]:  # keep each item's strongest neighbours only
            kth = np.partition(sim, -self.neighbours, axis=1)[:, -self.neighbours][:, None]
            sim[sim < kth] = 0.0
        self.sim = sim.astype(np.float32)
        self.user_items = b.tocsr()
        return self

    def scores(self, cust):
        r = self.ci.get(cust)
        if r is None:
            return {}
        v = np.asarray(self.user_items[r] @ self.sim).ravel()
        idx = np.argpartition(-v, min(200, len(v) - 1))[:200]
        return {self.items[j]: float(v[j]) for j in idx if v[j] > 0}

    def recommend(self, cust, k):
        return top_k(self.scores(cust), k)


class ImplicitALS:
    """Matrix factorisation for implicit feedback (Hu, Koren and Volinsky 2008): confidence
    1 + alpha * log(1 + baskets), solved by alternating least squares."""

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 20.0, iters: int = 12, seed: int = 0):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, conf: sp.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        f = fixed.shape[1]
        gram = fixed.T @ fixed + self.reg * np.eye(f)
        out = np.zeros((conf.shape[0], f))
        for u in range(conf.shape[0]):
            lo, hi = conf.indptr[u], conf.indptr[u + 1]
            idx, c = conf.indices[lo:hi], conf.data[lo:hi]
            if len(idx) == 0:
                continue
            y = fixed[idx]
            a = gram + (y.T * (c - 1.0)) @ y
            b = (y.T * c).sum(axis=1)
            out[u] = np.linalg.solve(a, b)
        return out

    def fit(self, train: pd.DataFrame, cutoff=None):
        m, self.customers, self.items, self.ci = interaction_matrix(train)
        conf = m.copy().astype(np.float64)
        conf.data = 1.0 + self.alpha * np.log1p(conf.data)
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.01, (m.shape[0], self.f))
        self.V = rng.normal(0, 0.01, (m.shape[1], self.f))
        conf_t = conf.T.tocsr()
        for _ in range(self.iters):
            self.U = self._solve(conf, self.V)
            self.V = self._solve(conf_t, self.U)
        return self

    def scores(self, cust):
        r = self.ci.get(cust)
        if r is None:
            return {}
        v = self.V @ self.U[r]
        idx = np.argpartition(-v, 200)[:200]
        return {self.items[j]: float(v[j]) for j in idx}

    def recommend(self, cust, k):
        return top_k(self.scores(cust), k)
