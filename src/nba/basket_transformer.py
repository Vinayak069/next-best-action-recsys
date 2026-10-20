"""Next-basket Transformer: SASRec-style causal self-attention over a customer's baskets.

Item-level SASRec flattens each basket into a sequence, but the order of items inside a basket
is arbitrary, so "predict the next item" mostly asks the model to guess an ordering that does
not exist. Here each step is a whole basket: its items are mean-pooled into one vector, the
time since the previous basket is added as an embedding, a causal Transformer reads the basket
sequence, and the output at step t is trained to put probability on every item of basket t+1
(listwise softmax). Output item embeddings are tied to input ones.

Most of a retail customer's next basket is items they bought before, which a pure
"explore" scorer underweights. So each item's score also gets a repeat term: the customer's
history with that item (how often, how recently, whether it was in the latest basket) through
three learned weights, scaled by a per-customer repeat propensity read from the Transformer state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nba.data import Vocab


@dataclass
class BasketConfig:
    max_baskets: int = 32
    max_items: int = 48       # items kept per basket (a random sample if larger)
    dim: int = 64
    heads: int = 2
    layers: int = 2
    dropout: float = 0.2
    lr: float = 2e-3
    epochs: int = 40
    batch: int = 64
    weight_decay: float = 1e-4
    seed: int = 0


def gap_bucket(days: float) -> int:
    """0 for the first basket, then log2-spaced buckets of days since the previous one."""
    return 0 if days is None else int(min(14, 1 + np.floor(np.log2(1 + max(days, 0.0)))))


def baskets_by_customer(train: pd.DataFrame) -> dict[int, list[tuple[pd.Timestamp, list[str]]]]:
    d = train.drop_duplicates(["customer_id", "invoice", "stock_code"])
    g = d.groupby(["customer_id", "invoice"]).agg(t=("invoice_date", "min"), items=("stock_code", list))
    out: dict[int, list] = {}
    for (c, _), row in g.sort_values("t").iterrows():
        out.setdefault(c, []).append((row["t"], row["items"]))
    return out


class BasketNet(nn.Module):
    def __init__(self, n_items: int, cfg: BasketConfig):
        super().__init__()
        self.item = nn.Embedding(n_items, cfg.dim, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(n_items))
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.gap = nn.Embedding(16, cfg.dim)
        self.pos = nn.Embedding(cfg.max_baskets + 1, cfg.dim)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * 4, cfg.dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.heads = cfg.heads
        self.rep_w = nn.Parameter(torch.tensor([1.0, 1.0, 0.5]))   # log-frequency, recency, in latest basket
        self.rep_gate = nn.Linear(cfg.dim, 1)
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, items: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        """items: (B, T, M) item ids (0 = padding), gaps: (B, T). Returns (B, T, D)."""
        B, T, _ = items.shape
        emb = self.item(items)                                    # (B, T, M, D)
        cnt = (items > 0).sum(-1, keepdim=True).clamp(min=1)
        basket = self.proj(emb.sum(2) / cnt)                      # mean of the basket's items
        pos = torch.arange(T, device=items.device).unsqueeze(0)
        x = self.drop(basket + self.gap(gaps) + self.pos(pos))
        # Causal mask plus "ignore padding baskets", but every position may always see itself: a
        # row with every key masked gives NaN attention, which then leaks into real positions.
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=items.device), diagonal=1)
        pad = (items > 0).sum(-1) == 0                            # (B, T)
        mask = causal.unsqueeze(0) | pad.unsqueeze(1)             # (B, T, T): True = blocked
        mask = mask & ~torch.eye(T, dtype=torch.bool, device=items.device).unsqueeze(0)
        mask = mask.repeat_interleave(self.heads, dim=0)
        h = self.encoder(x, mask=mask)
        return self.norm(h)

    def logits(self, h: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Scores for every item after each step: explore (h . item embedding) + gated repeat term."""
        f = history_features(items, self.item.num_embeddings)        # 3 x (B, T, V)
        rep = self.rep_w[0] * f[0] + self.rep_w[1] * f[1] + self.rep_w[2] * f[2]
        gate = nn.functional.softplus(self.rep_gate(h))              # (B, T, 1)
        return h @ self.item.weight.T + self.bias + gate * rep


def history_features(items: torch.Tensor, n_items: int) -> tuple[torch.Tensor, ...]:
    """For baskets up to and including step t: log(1 + baskets containing the item), 1/(1 + baskets
    since it was last bought) (0 if never), and whether it is in basket t. Shapes (B, T, V)."""
    B, T, _ = items.shape
    mh = torch.zeros(B, T, n_items, device=items.device)
    mh.scatter_(2, items, 1.0)
    mh[:, :, 0] = 0
    freq = torch.log1p(mh.cumsum(1))
    t = torch.arange(T, device=items.device).view(1, T, 1).float()
    last = torch.where(mh > 0, t, torch.full_like(mh, -1.0)).cummax(1).values
    rec = torch.where(last >= 0, 1.0 / (1.0 + t - last), torch.zeros_like(mh))
    return freq, rec, mh


class BasketTransformer:
    def __init__(self, cfg: BasketConfig | None = None):
        self.cfg = cfg or BasketConfig()

    def _encode(self, baskets, rng) -> tuple[np.ndarray, np.ndarray]:
        """Left-pads the last max_baskets+1 baskets into (T+1, M) ids and (T+1,) gap buckets."""
        cfg = self.cfg
        T1 = cfg.max_baskets + 1
        ids = np.zeros((T1, cfg.max_items), dtype=np.int64)
        gaps = np.zeros(T1, dtype=np.int64)
        seq = baskets[-T1:]
        off = T1 - len(seq)
        prev = None
        for j, (t, items) in enumerate(seq):
            enc = self.vocab.encode(items)
            if len(enc) > cfg.max_items:
                enc = list(rng.choice(enc, cfg.max_items, replace=False))
            ids[off + j, : len(enc)] = enc
            gaps[off + j] = gap_bucket(None if prev is None else (t - prev).total_seconds() / 86400)
            prev = t
        return ids, gaps

    def fit(self, train: pd.DataFrame, log=print):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.baskets = baskets_by_customer(train)
        self.vocab = Vocab(train["stock_code"].unique())
        xs, gs = [], []
        for bs in self.baskets.values():
            if len(bs) < 2:
                continue
            # one training window ending at each multiple of max_baskets, so long histories are all used
            for end in range(len(bs), 1, -cfg.max_baskets):
                ids, gaps = self._encode(bs[:end], rng)
                xs.append(ids)
                gs.append(gaps)
        X, G = torch.tensor(np.stack(xs)), torch.tensor(np.stack(gs))
        self.net = BasketNet(len(self.vocab), cfg)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history = []
        for epoch in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            total, n = 0.0, 0
            for b in range(0, len(X), cfg.batch):
                idx = perm[b:b + cfg.batch]
                items, gaps = X[idx], G[idx]
                inp, tgt = items[:, :-1], items[:, 1:]
                logp = torch.log_softmax(self.net.logits(self.net(inp, gaps[:, :-1]), inp), dim=-1)  # (B, T, V)
                tmask = (tgt > 0).float()
                ll = torch.gather(logp, 2, tgt) * tmask                     # log-prob of each target item
                per_step = ll.sum(-1) / tmask.sum(-1).clamp(min=1)
                valid = (tmask.sum(-1) > 0) & ((inp > 0).sum(-1) > 0)
                loss = -(per_step * valid).sum() / valid.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
                n += len(idx)
            self.history.append(total / n)
            if log:
                log(f"basket-transformer epoch {epoch + 1}/{cfg.epochs} loss {total / n:.4f}")
        self.net.eval()
        return self

    @torch.no_grad()
    def scores_batch(self, customers: list[int], top: int = 200, at: pd.Timestamp | None = None
                     ) -> dict[int, dict[str, float]]:
        rng = np.random.default_rng(0)
        out = {}
        for b in range(0, len(customers), 128):
            cs = customers[b:b + 128]
            ids, gaps = zip(*[self._encode(self.baskets.get(c, [])[-self.cfg.max_baskets:], rng) for c in cs])
            items = torch.tensor(np.stack(ids))[:, 1:]  # drop the extra slot used for training targets
            g = torch.tensor(np.stack(gaps))[:, 1:]
            logits = self.net.logits(self.net(items, g), items)[:, -1, :]
            logits[:, 0] = -1e9
            val, idx = torch.topk(logits, min(top, logits.shape[1] - 1), dim=1)
            for c, v, i in zip(cs, val.numpy(), idx.numpy()):
                out[c] = {self.vocab.items[j]: float(x) for j, x in zip(i, v)} if c in self.baskets else {}
        return out
