"""Load Online Retail II and turn it into time-ordered customer purchase histories.

The raw file mixes sales with cancellations (invoice numbers starting with "C"), returns
(negative quantities), postage and fee lines, and guest checkouts with no customer id. Only
real product purchases by identified customers are kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Stock codes that are charges or adjustments, not products.
NON_PRODUCT = {"POST", "DOT", "M", "C2", "BANK CHARGES", "PADS", "D", "CRUK", "S", "B", "AMAZONFEE", "ADJUST",
               "ADJUST2", "TEST001", "TEST002", "SP1002", "gift_0001_10", "gift_0001_20", "gift_0001_30",
               "gift_0001_40", "gift_0001_50"}


def load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=["customer_id"]).copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    keep = (
        ~df["invoice"].str.startswith("C")
        & (df["quantity"] > 0)
        & (df["price"] > 0)
        & ~df["stock_code"].isin({c.upper() for c in NON_PRODUCT})
        & df["stock_code"].str.match(r"^\d{5}")  # product codes start with five digits
    )
    df = df.loc[keep, ["customer_id", "invoice", "invoice_date", "stock_code", "quantity", "price", "country"]]
    df["customer_id"] = df["customer_id"].astype(np.int64)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df.sort_values(["invoice_date", "invoice", "stock_code"]).reset_index(drop=True)


@dataclass
class Split:
    """Everything before `cutoff` is history; each customer's first basket after it is the target."""

    cutoff: pd.Timestamp
    train: pd.DataFrame                 # purchases before the cutoff
    targets: dict[int, set[str]]        # customer -> items in their first basket after the cutoff
    target_dates: dict[int, pd.Timestamp]


def temporal_split(df: pd.DataFrame, cutoff: str, horizon_days: int | None = None) -> Split:
    """Next-basket split. Only customers with at least one purchase before the cutoff are
    evaluated (warm start), because the task is recommending to existing customers."""
    t = pd.Timestamp(cutoff)
    train = df[df["invoice_date"] < t]
    future = df[df["invoice_date"] >= t]
    if horizon_days is not None:
        future = future[future["invoice_date"] < t + pd.Timedelta(days=horizon_days)]
    warm = set(train["customer_id"].unique())
    future = future[future["customer_id"].isin(warm)]
    first_inv = future.sort_values("invoice_date").groupby("customer_id")["invoice"].first()
    basket = future[future["invoice"].isin(set(first_inv.values))]
    targets = basket.groupby("customer_id")["stock_code"].agg(set).to_dict()
    dates = basket.groupby("customer_id")["invoice_date"].min().to_dict()
    return Split(t, train, targets, dates)


def sequences(train: pd.DataFrame) -> dict[int, list[str]]:
    """Each customer's items in purchase order; repeats within one basket appear once."""
    dedup = train.drop_duplicates(["customer_id", "invoice", "stock_code"])
    return dedup.groupby("customer_id")["stock_code"].agg(list).to_dict()


class Vocab:
    """Item code <-> integer id. Id 0 is reserved for padding."""

    def __init__(self, items):
        self.items = ["<pad>"] + sorted(set(items))
        self.index = {c: i for i, c in enumerate(self.items)}

    def __len__(self):
        return len(self.items)

    def encode(self, codes):
        return [self.index[c] for c in codes if c in self.index]
