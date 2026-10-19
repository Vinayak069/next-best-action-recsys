"""SASRec (Kang and McAuley 2018): a causal Transformer over a customer's purchase sequence that
predicts the next item. Trained with full softmax cross-entropy over the catalogue (a few
thousand items, so sampling negatives is unnecessary)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from nba.data import Vocab


@dataclass
class SASRecConfig:
    max_len: int = 100
    dim: int = 64
    heads: int = 2
    layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 30
    batch: int = 128
    weight_decay: float = 0.0
    seed: int = 0


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, cfg: SASRecConfig):
        super().__init__()
        self.item = nn.Embedding(n_items, cfg.dim, padding_idx=0)
        self.pos = nn.Embedding(cfg.max_len, cfg.dim)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * 4, cfg.dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers)
        self.norm = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.max_len = cfg.max_len

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item ids, left-padded with 0. Returns hidden states (B, L, D)."""
        L = seq.shape[1]
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        x = self.drop(self.item(seq) * (self.item.embedding_dim ** 0.5) + self.pos(pos))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        h = self.encoder(x, mask=causal, src_key_padding_mask=(seq == 0))
        return self.norm(h)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item.weight.T  # tied input/output embeddings


def _left_pad(ids: list[int], L: int) -> list[int]:
    ids = ids[-L:]
    return [0] * (L - len(ids)) + ids


class SASRec:
    def __init__(self, cfg: SASRecConfig | None = None):
        self.cfg = cfg or SASRecConfig()

    def fit(self, seqs: dict[int, list[str]], log=print):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.vocab = Vocab(i for s in seqs.values() for i in s)
        self.seqs = {c: self.vocab.encode(s) for c, s in seqs.items()}
        L = cfg.max_len + 1
        # Long histories are cut into windows so every purchase is a training target once.
        rows = []
        for s in self.seqs.values():
            if len(s) < 2:
                continue
            for end in range(len(s), 1, -cfg.max_len):
                rows.append(_left_pad(s[max(0, end - L):end], L))
        data = torch.tensor(rows, dtype=torch.long)
        self.net = SASRecNet(len(self.vocab), cfg)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.history = []
        for epoch in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(len(data))
            total, n = 0.0, 0
            for b in range(0, len(data), cfg.batch):
                batch = data[perm[b:b + cfg.batch]]
                inp, tgt = batch[:, :-1], batch[:, 1:]
                tgt = tgt.masked_fill(inp == 0, 0)  # no target where the input is padding
                logits = self.net.logits(self.net(inp))
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(batch)
                n += len(batch)
            self.history.append(total / n)
            if log:
                log(f"sasrec epoch {epoch + 1}/{cfg.epochs} loss {total / n:.4f}")
        self.net.eval()
        return self

    @torch.no_grad()
    def scores_batch(self, customers: list[int], top: int = 200) -> dict[int, dict[str, float]]:
        out = {}
        for b in range(0, len(customers), 256):
            cs = [c for c in customers[b:b + 256]]
            seq = torch.tensor([_left_pad(self.seqs.get(c, []), self.cfg.max_len) for c in cs], dtype=torch.long)
            logits = self.net.logits(self.net(seq)[:, -1, :])
            logits[:, 0] = -1e9
            val, idx = torch.topk(logits, min(top, logits.shape[1] - 1), dim=1)
            for c, v, i in zip(cs, val.numpy(), idx.numpy()):
                out[c] = {self.vocab.items[j]: float(x) for j, x in zip(i, v)} if self.seqs.get(c) else {}
        return out
