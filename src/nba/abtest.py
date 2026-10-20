"""Replay A/B test: what a live experiment comparing two recommenders would conclude.

Customers are randomly split between control and treatment. Each arm shows its top-k offers,
and a customer "converts" if their next basket contains any of them. This is an offline replay
on observed baskets, not a live test: it cannot capture how showing an offer changes behaviour.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def conversions(recs: dict[int, list[str]], targets: dict[int, set[str]], customers, k: int) -> np.ndarray:
    return np.array([float(any(i in targets[c] for i in recs.get(c, [])[:k])) for c in customers])


def two_proportion_test(x_c: np.ndarray, x_t: np.ndarray) -> dict:
    n_c, n_t = len(x_c), len(x_t)
    p_c, p_t = x_c.mean(), x_t.mean()
    pooled = (x_c.sum() + x_t.sum()) / (n_c + n_t)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se0 if se0 > 0 else 0.0
    se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    return {"n_control": n_c, "n_treatment": n_t, "control_rate": p_c, "treatment_rate": p_t,
            "abs_lift": p_t - p_c, "rel_lift": (p_t - p_c) / p_c if p_c else float("nan"),
            "z": z, "p_value": 2 * (1 - stats.norm.cdf(abs(z))),
            "ci95": (p_t - p_c - 1.96 * se, p_t - p_c + 1.96 * se)}


def bootstrap_ci(x_c: np.ndarray, x_t: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = [rng.choice(x_t, len(x_t)).mean() - rng.choice(x_c, len(x_c)).mean() for _ in range(n_boot)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def sample_size_per_arm(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers per arm to detect an absolute lift of `mde_abs` (two-sided z-test)."""
    p1, p2 = p_base, p_base + mde_abs
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde_abs ** 2
    return int(math.ceil(n))


def replay_ab(control: dict[int, list[str]], treatment: dict[int, list[str]], targets: dict[int, set[str]],
              k: int = 5, seed: int = 0) -> dict:
    customers = np.array(sorted(targets))
    rng = np.random.default_rng(seed)
    arm = rng.random(len(customers)) < 0.5
    x_c = conversions(control, targets, customers[~arm], k)
    x_t = conversions(treatment, targets, customers[arm], k)
    res = two_proportion_test(x_c, x_t)
    res["bootstrap_ci95"] = bootstrap_ci(x_c, x_t, seed=seed)
    # Offline we can also score both policies on every customer: a paired comparison (McNemar).
    a = conversions(control, targets, customers, k)
    b = conversions(treatment, targets, customers, k)
    only_t, only_c = int(((b == 1) & (a == 0)).sum()), int(((a == 1) & (b == 0)).sum())
    res["paired"] = {"control_rate": float(a.mean()), "treatment_rate": float(b.mean()), "only_treatment": only_t,
                     "only_control": only_c,
                     "mcnemar_p": float(stats.binomtest(only_t, only_t + only_c).pvalue) if only_t + only_c else 1.0}
    res["n_per_arm_for_observed_lift"] = (sample_size_per_arm(res["paired"]["control_rate"],
                                                              res["paired"]["treatment_rate"] - res["paired"]["control_rate"])
                                          if res["paired"]["treatment_rate"] > res["paired"]["control_rate"] else None)
    res["k"] = k
    return res
