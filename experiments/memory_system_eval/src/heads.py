"""Small learned heads on frozen J-BRAIN features (MLP: one hidden layer of 512, GELU, dropout 0.1), feature packing and pooling."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jbrain_frozen as J

LAYERS = (5, 7)
D = 768 * len(LAYERS)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=512, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def pack_features(model, seqs: list[list[int]], chunk: int = 4096, norm=None, truncate: bool = False, max_tokens: int = 16384):
    """Frozen features of many independent sequences packed into one fp16 array [tokens, D]; optional (mu, sd) normalisation."""
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    X = np.empty((int(lengths.sum()), D), dtype=np.float16)
    for s in range(0, len(seqs), chunk):
        enc = J.encode_batch(model, seqs[s:s + chunk], capture=LAYERS, out_dtype=torch.float32, truncate=truncate, max_tokens=max_tokens)
        for k, f in enumerate(enc):
            x = torch.cat([f[c] for c in LAYERS], 1).numpy()
            if norm is not None:
                x = (x - norm[0]) / norm[1]
            o = offsets[s + k]
            X[o:o + lengths[s + k]] = x.astype(np.float16)
    return X, offsets, lengths


def norm_stats(X: np.ndarray, n: int = 200000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    x = X[idx].astype(np.float32)
    return x.mean(0), x.std(0) + 1e-4


def line_pool(X, off, ln):
    out = np.empty((len(off), 2 * D), dtype=np.float32)
    for i, (o, n) in enumerate(zip(off, ln)):
        x = X[o:o + n].astype(np.float32)
        out[i] = np.concatenate([x[-1], x.mean(0)])
    return out


def span_feat(X, o, n, a, b):
    x = X[o:o + n].astype(np.float32)
    return np.concatenate([x[a:b].mean(0), x[b - 1], x[-1], x.mean(0)])


def q_pool(X, off, ln, q_start):
    out = np.empty((len(off), 3 * D), dtype=np.float32)
    for i, (o, n, q0) in enumerate(zip(off, ln, q_start)):
        x = X[o:o + n].astype(np.float32)
        out[i] = np.concatenate([x[q0:].mean(0), x[-1], x.mean(0)])
    return out


def window_index(off, ln, W=(-1, 0, 1, 2)):
    """Global row indices of the tagger window for every token (-1 where the window leaves the sequence)."""
    rows = []
    for o, n in zip(off, ln):
        t = np.arange(n)[:, None] + np.array(W)[None, :]
        r = o + t
        r[(t < 0) | (t >= n)] = -1
        rows.append(r)
    return np.concatenate(rows).astype(np.int64)


def gather_window(Xg, rows):
    """Xg: [N, D] tensor (fp16, device); rows: [B, W] long tensor -> [B, W*D] float32 with zeros outside the sequence."""
    m = (rows >= 0)
    x = Xg[rows.clamp(min=0)].float() * m[..., None]
    return x.reshape(rows.shape[0], -1)


def train_mlp(make_batch, n_train, val_fn, d_in, d_out, lr, seed, epochs=8, batch=256, dev="cuda", loss_fn=None):
    """Generic Adam training with best-validation checkpointing. make_batch(idx) -> (x, y); val_fn(model) -> (loss, metric)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = MLP(d_in, d_out).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = loss_fn or F.cross_entropy
    best, hist = (math.inf, None, -1), []
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(n_train)
        tot, cnt = 0.0, 0
        for s in range(0, n_train, batch):
            x, y = make_batch(perm[s:s + batch])
            loss = loss_fn(m(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            cnt += len(y)
        m.eval()
        with torch.no_grad():
            vl, vm = val_fn(m)
        hist.append({"epoch": ep, "train_loss": tot / cnt, "val_loss": vl, "val_metric": vm})
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep)
    final = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return {"best_state": best[1], "best_epoch": best[2], "best_val_loss": best[0], "final_state": final, "history": hist}


class BiEncoder(nn.Module):
    def __init__(self, d_q, d_r, d_out=256):
        super().__init__()
        self.q, self.r = MLP(d_q, d_out), MLP(d_r, d_out)

    def forward_q(self, x):
        return F.normalize(self.q(x), dim=-1)

    def forward_r(self, x):
        return F.normalize(self.r(x), dim=-1)
