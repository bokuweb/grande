"""Train a pointer head on hidden states `grande features` extracted, and
write head.safetensors (with `layout` metadata so `--head` picks the same
layout at inference).

The backbone never moves: the rows came out of the served, quantized model,
so what this head sees in training is exactly what it sees in production.
Training takes seconds; extraction is the slow part.

    python tools/train_head.py --features runs/feat/e2b-jnli.safetensors [more.safetensors] \
        --out runs/head-e2b-jnli --dp 256 --epochs 30

Held-out: records are split by item id (all orders of one record on one
side), the last 10% by id are validation, the best epoch is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


def load(paths):
    decide, opts, n_opts, gold, item, meta = [], [], [], [], [], None
    base = 0
    for p in paths:
        with safe_open(p, "pt") as f:
            m = f.metadata() or {}
            if meta is None:
                meta = m
            elif m.get("layout") != meta.get("layout"):
                raise SystemExit(f"{p}: layout {m.get('layout')} != {meta.get('layout')}")
            decide.append(f.get_tensor("decide").float())
            o = f.get_tensor("opts").float()
            opts.append(o)
            n_opts.append(f.get_tensor("n_opts"))
            gold.append(f.get_tensor("gold"))
            item.append(f.get_tensor("item") + base)
            base += int(f.get_tensor("item").max()) + 1
    k = max(o.shape[1] for o in opts)
    opts = [F.pad(o, (0, 0, 0, k - o.shape[1])) for o in opts]
    return (
        torch.cat(decide),
        torch.cat(opts),
        torch.cat(n_opts).long(),
        torch.cat(gold).long(),
        torch.cat(item).long(),
        meta,
    )


class Head(torch.nn.Module):
    def __init__(self, d, dp):
        super().__init__()
        self.q, self.k = torch.nn.Linear(d, dp), torch.nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, decide, opts, n_opts):
        # decide [B, d], opts [B, K, d] -> logits [B, K], padding masked
        q = self.q(decide)
        k = self.k(opts)
        z = torch.einsum("bkd,bd->bk", k, q) * self.scale
        mask = torch.arange(opts.shape[1])[None, :] >= n_opts[:, None]
        return z.masked_fill(mask, -1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dp", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.1, help="share of items (by id) held out for model selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false", help="do not z-score the hidden features (default: on; folded into the head at export)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    decide, opts, n_opts, gold, item, meta = load(a.features)
    n, k, d = opts.shape
    ids = item.unique()
    n_val = max(1, int(len(ids) * a.holdout))
    val_ids = ids[-n_val:]
    is_val = torch.isin(item, val_ids)
    tr, va = (~is_val).nonzero().squeeze(1), is_val.nonzero().squeeze(1)
    print(f"{n} rows ({len(ids)} items), d={d}, K={k}; train {len(tr)} / val {len(va)}; layout {meta.get('layout')}")

    mu = torch.zeros(d)
    sd = torch.ones(d)
    if a.standardize:
        allrows = torch.cat([decide[tr], opts[tr].reshape(-1, d)[(torch.arange(k)[None, :] < n_opts[tr, None]).reshape(-1)]])
        mu, sd = allrows.mean(0), allrows.std(0).clamp_min(1e-3)
    def norm(x):
        return (x - mu) / sd

    head = Head(d, a.dp)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best, best_state, t0 = (-1.0, 1e9), None, time.time()
    log = []
    for epoch in range(a.epochs):
        head.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            b = perm[i : i + a.batch]
            z = head(norm(decide[b]), norm(opts[b]), n_opts[b])
            loss = F.cross_entropy(z, gold[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        head.eval()
        with torch.no_grad():
            zv = head(norm(decide[va]), norm(opts[va]), n_opts[va])
            vloss = F.cross_entropy(zv, gold[va]).item()
            vacc = (zv.argmax(1) == gold[va]).float().mean().item()
            zt = head(norm(decide[tr]), norm(opts[tr]), n_opts[tr])
            tacc = (zt.argmax(1) == gold[tr]).float().mean().item()
        msg = f"epoch {epoch + 1:3d}  train loss {tot / len(tr):.4f} acc {tacc:.3f}  val loss {vloss:.4f} acc {vacc:.3f}  {time.time() - t0:.0f}s"
        print(msg)
        log.append(msg)
        if (vacc, -vloss) > best:
            best = (vacc, -vloss)
            best_state = {k_: v.detach().clone() for k_, v in head.state_dict().items()}
    head.load_state_dict(best_state)

    # Fold the standardization into the affine maps: W (x - mu) / sd + b = (W / sd) x + (b - W mu / sd).
    sdict = {}
    for name in ("q", "k"):
        w = head.state_dict()[f"{name}.weight"] / sd[None, :]
        b = head.state_dict()[f"{name}.bias"] - w @ mu
        sdict[f"{name}.weight"] = w.contiguous()
        sdict[f"{name}.bias"] = b.contiguous()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(sdict, str(out / "head.safetensors"), metadata={"layout": meta.get("layout", ""), "model": meta.get("model", ""), "task": meta.get("task", ""), "best_val_acc": f"{best[0]:.4f}"})
    (out / "train.log").write_text("\n".join(log) + "\n")
    json.dump({**vars(a), "best_val_acc": best[0], "best_val_loss": -best[1], "n_rows": n, "n_items": len(ids)}, open(out / "config.json", "w"), indent=2, ensure_ascii=False)
    print(f"best val acc {best[0]:.3f}; wrote {out / 'head.safetensors'}")


if __name__ == "__main__":
    main()
