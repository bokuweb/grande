"""Train small heads on the frozen embeddings `tools/e5_features.py` dumped
and score them on JGLUE valid with the protocol of docs/comparison.md /
docs/laya.md: one temperature fitted on the even-indexed valid records,
accuracy / ECE / NLL reported on the odd-indexed ones, plus the first 400
rows raw (the rows the trained-head numbers in the README use).

    python tools/e5_head.py --feat runs/e5 --out runs/e5/summary.json

Heads (all take seconds on the CPU):
  JNLI  cos       zero-shot: two thresholds on cos(premise, hypothesis), fit on the even half
        sep-lr    logistic regression on [u, v, |u-v|, u*v]
        sep-mlp   one hidden layer on the same
        pair-lr / pair-mlp   the pair embedded as one sequence
        both-mlp  pair + separate features
  JCQA  cos       zero-shot: argmax cos(question, choice), softmax(cos / tau)
        sep-mlp   per-choice scorer on [q, c, |q-c|, q*c]
        joint-mlp per-choice scorer on the (question, choice) sequence embedding
        both-mlp  joint + separate
Model selection is on the last 10% of train (by record), never on valid.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ---- metrics ---------------------------------------------------------------

def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf), np.asarray(ok)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(tot)


def fit_temperature(logits, gold):
    """Minimise NLL over T on a log grid (the same one-scalar fit omg uses)."""
    best = (1e9, 1.0)
    for lt in np.linspace(-2.5, 2.5, 501):
        T = math.exp(lt)
        nll = F.cross_entropy(torch.as_tensor(logits) / T, torch.as_tensor(gold)).item()
        if nll < best[0]:
            best = (nll, T)
    return best[1]


def score(logits, gold, T=1.0):
    p = F.softmax(torch.as_tensor(logits, dtype=torch.float32) / T, -1).numpy()
    pred = p.argmax(-1)
    ok = (pred == gold).astype(float)
    conf = p.max(-1)
    nll = -np.log(np.maximum(p[np.arange(len(gold)), gold], 1e-12))
    return {"acc": float(ok.mean()), "ece": ece(conf, ok), "nll": float(nll.mean())}


def report(name, logits, gold, ms=None):
    """Even half fits T, odd half is reported; first 400 raw."""
    n = len(gold)
    ev, od = np.arange(0, n, 2), np.arange(1, n, 2)
    T = fit_temperature(logits[ev], gold[ev])
    raw, sc = score(logits[od], gold[od]), score(logits[od], gold[od], T)
    f400 = score(logits[:400], gold[:400])
    row = {"head": name, "n": len(od), "acc": raw["acc"], "ece": raw["ece"], "ece_T": sc["ece"], "nll": raw["nll"], "nll_T": sc["nll"], "T": T, "acc_400": f400["acc"], "ece_400": f400["ece"], "nll_400": f400["nll"]}
    if ms is not None:
        row["ms"] = ms
    print(f"  {name:10s} n={len(od)} acc {raw['acc']:.3f}  ECE {raw['ece']:.3f} -> {sc['ece']:.3f}  NLL {raw['nll']:.3f} -> {sc['nll']:.3f}  T {T:.2f} | first400 acc {f400['acc']:.3f} ECE {f400['ece']:.3f}", flush=True)
    return row


# ---- features --------------------------------------------------------------

def pair_feats(u, v):
    return np.concatenate([u, v, np.abs(u - v), u * v], -1)


def jnli_inputs(d, kind):
    u, v, p = d["a"], d["b"], d["pair"]
    if kind == "sep":
        return pair_feats(u, v)
    if kind == "pair":
        return p
    return np.concatenate([p, pair_feats(u, v)], -1)


def jcqa_inputs(d, kind):
    q = d["q"]
    per = []
    for i in range(5):
        c, qc = d[f"c{i}"], d[f"qc{i}"]
        if kind == "sep":
            per.append(pair_feats(q, c))
        elif kind == "joint":
            per.append(qc)
        else:
            per.append(np.concatenate([qc, pair_feats(q, c)], -1))
    return np.stack(per, 1)  # [N, 5, d]


# ---- heads -----------------------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, d, out, hidden, drop):
        super().__init__()
        if hidden:
            self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(hidden, out))
        else:
            self.net = torch.nn.Linear(d, out)

    def forward(self, x):
        return self.net(x)


def train(x, y, out_dim, hidden, epochs, lr, wd, drop, batch=256, per_choice=False, n_opts=None):
    """x [N, d] -> logits [N, out_dim], or per_choice: x [N, K, d] -> [N, K] (rows past
    n_opts[i] are padding and masked). Last 10% of records are the selection split;
    the best epoch by its loss is returned. `predict.state` holds what a server needs."""
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    mu, sd = x.reshape(-1, x.shape[-1]).mean(0), x.reshape(-1, x.shape[-1]).std(0) + 1e-6
    x = (x - mu) / sd
    n = len(y)
    cut = int(n * 0.9)
    xt, yt, xv, yv = x[:cut], y[:cut], x[cut:], y[cut:]
    m = MLP(x.shape[-1], 1 if per_choice else out_dim, hidden, drop)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    mask = None if n_opts is None else torch.arange(x.shape[1])[None, :] >= torch.as_tensor(n_opts)[:, None]

    def fwd(z, mk=None):
        out = m(z).squeeze(-1) if per_choice else m(z)
        return out if mk is None else out.masked_fill(mk, -1e9)

    best, state = 1e9, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(cut)
        for i in range(0, cut, batch):
            idx = perm[i : i + batch]
            loss = F.cross_entropy(fwd(xt[idx], None if mask is None else mask[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            zv = fwd(xv, None if mask is None else mask[cut:])
            vl = F.cross_entropy(zv, yv).item()
            va = (zv.argmax(-1) == yv).float().mean().item()
        if vl < best:
            best, state, best_ep = vl, {k: v.clone() for k, v in m.state_dict().items()}, (ep, va)
    m.load_state_dict(state)
    m.eval()

    def predict(xx):
        xx = (torch.as_tensor(xx, dtype=torch.float32) - mu) / sd
        with torch.no_grad():
            return fwd(xx).numpy()

    predict.state = {"model": state, "mu": mu, "sd": sd, "hidden": hidden, "d": x.shape[-1], "per_choice": per_choice, "out": 1 if per_choice else out_dim}
    return predict, {"val_loss": best, "epoch": best_ep[0], "val_acc": best_ep[1], "params": sum(p.numel() for p in m.parameters())}


# ---- zero-shot -------------------------------------------------------------

def jnli_cos(d):
    return (d["a"] * d["b"]).sum(-1)


def jnli_cos_thresholds(cos, gold):
    """Two cut points on cos, entailment above, neutral in the middle, contradiction below
    (and every other assignment of the three labels to the three bands); best on the even half."""
    qs = np.quantile(cos, np.linspace(0.02, 0.98, 49))
    best = (0, None)
    import itertools
    for lo, hi in itertools.combinations(qs, 2):
        band = (cos > hi).astype(int) * 2 + ((cos > lo) & (cos <= hi)).astype(int)  # 0 low, 1 mid, 2 high
        for perm in itertools.permutations(range(3)):
            pred = np.array(perm)[band]
            acc = (pred == gold).mean()
            if acc > best[0]:
                best = (acc, (lo, hi, perm))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="runs/e5")
    ap.add_argument("--out", default="runs/e5/summary.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--drop", type=float, default=0.2)
    a = ap.parse_args()
    feat = Path(a.feat)
    rows = {}

    def lat(task, key):
        p = feat / f"{task}-valid-latency.json"
        return json.load(open(p))["latency"][key]["mean_ms"] if p.exists() else None

    # ---- JNLI
    tr, va = np.load(feat / "jnli-train.npz"), np.load(feat / "jnli-valid.npz")
    gold_tr, gold_va = tr["gold"], va["gold"]
    print(f"JNLI train {len(gold_tr)} valid {len(gold_va)}  label dist valid {np.bincount(gold_va) / len(gold_va)}")
    rows["jnli"] = []
    cos_tr, cos_va = jnli_cos(tr), jnli_cos(va)
    n = len(gold_va)
    ev, od = np.arange(0, n, 2), np.arange(1, n, 2)
    lo, hi, perm = jnli_cos_thresholds(cos_va[ev], gold_va[ev])
    band = (cos_va > hi).astype(int) * 2 + ((cos_va > lo) & (cos_va <= hi)).astype(int)
    pred = np.array(perm)[band]
    acc_od, acc_400 = float((pred[od] == gold_va[od]).mean()), float((pred[:400] == gold_va[:400]).mean())
    print(f"  cos        n={len(od)} acc {acc_od:.3f} (thresholds {lo:.3f}/{hi:.3f}, bands {perm}) | first400 acc {acc_400:.3f}")
    rows["jnli"].append({"head": "cos", "n": len(od), "acc": acc_od, "acc_400": acc_400, "thresholds": [float(lo), float(hi)], "bands": list(perm), "ms": lat("jnli", "separate")})
    for kind, hidden in [("sep", 0), ("sep", a.hidden), ("pair", 0), ("pair", a.hidden), ("both", a.hidden)]:
        name = f"{kind}-{'mlp' if hidden else 'lr'}"
        predict, info = train(jnli_inputs(tr, kind), gold_tr, 3, hidden, a.epochs, a.lr, a.wd, a.drop)
        row = report(name, predict(jnli_inputs(va, kind)), gold_va, lat("jnli", "pair" if kind == "pair" else "separate"))
        row.update(info)
        rows["jnli"].append(row)

    # ---- JCQA
    tr, va = np.load(feat / "jcqa-train.npz"), np.load(feat / "jcqa-valid.npz")
    gold_tr, gold_va = tr["gold"], va["gold"]
    print(f"JCQA train {len(gold_tr)} valid {len(gold_va)}")
    rows["jcqa"] = []
    cos_va = np.stack([(va["q"] * va[f"c{i}"]).sum(-1) for i in range(5)], 1)
    rows["jcqa"].append(report("cos", cos_va, gold_va, lat("jcqa", "separate")))
    for kind, hidden in [("sep", a.hidden), ("joint", a.hidden), ("both", a.hidden)]:
        name = f"{kind}-mlp"
        predict, info = train(jcqa_inputs(tr, kind), gold_tr, 5, hidden, a.epochs, a.lr, a.wd, a.drop, per_choice=True)
        row = report(name, predict(jcqa_inputs(va, kind)), gold_va, lat("jcqa", "separate" if kind == "sep" else "joint"))
        row.update(info)
        rows["jcqa"].append(row)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(a), "results": rows}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
