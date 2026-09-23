"""Train a Laya-shaped cross-encoder on Ruri v3 (ModernBERT-Ja) and measure
whether it reads a question's instructions (docs/cross.md).

One question = one sequence, Laya's layout (crates/omg-wgpu/src/laya/prompt.rs):

    <s> <type> question: <instructions> </s> <mask> opt0 <mask> opt1 … </s> <state json> </s>

The encoder reads instructions, options and state together; a scorer
(LayerNorm → Linear → GELU → Linear(1)) turns the hidden row at each
`<mask>` into the option's logit, a softmax over them is the answer. Unlike
the (state, option) head of docs/e5.md / docs/ruri.md, two questions over
one state are two different sequences.

Training data:
  - .cache/distill/labeled.jsonl: synthetic Japanese states with typed
    questions and Gemma 4 E4B's probabilities (soft targets); only the
    states in train.jsonl, and never the held-out instructions below.
  - JGLUE JNLI / JCommonsenseQA train, in tools/http_eval.py's request
    shape (hard targets).
Evaluation:
  - seen instructions on the 200 held-out states: agreement with E4B;
  - HELD-OUT instructions (never trained on) on all states: agreement with
    E4B — whether it reads a question it has not seen;
  - JGLUE valid, odd half, with tools/e5_head.py's protocol;
  - ticket-ja / contract: do the noul answers differ from one another.

    python tools/cross_train.py --model cl-nagoya/ruri-v3-70m --out runs/cross/70m
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from e5_head import report  # noqa: E402
from http_eval import payload  # noqa: E402

QTYPES = {"choice": "choice", "score": "score", "noul": "noul"}
# Instructions never trained on: one to three per family, every type.
HELD_OUT = {
    "セキュリティに関わる問い合わせか",
    "顧客が求めている対応はどれか",
    "レビュアーは製品を他人に勧めているか",
    "このレビューの評価はどれか",
    "この条項は契約終了後も効力が続く義務を定めているか",
    "乙（受託者）から見たこの条項のリスクはどの程度か",
    "会議の場所が明記されているか",
    "この文章に数値が含まれているか",
}
BUCKETS = (128, 256, 384, 512)


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def criterion_text(v):
    if v is None or v == "":
        return None
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def options(q):
    """Option texts in omg's key order (noul: true, false) and the keys."""
    t = q["type"]
    if t == "choice":
        keys = list(q["criteria"])
        return keys, [k if criterion_text(v) is None else f"{k}: {criterion_text(v)}" for k, v in q["criteria"].items()]
    if t == "score":
        c = q["criteria"]
        return [str(i) for i in range(len(c))], [f"level {i}: {criterion_text(v)}" for i, v in enumerate(c)]
    crit = q.get("criteria") or {}
    tt = criterion_text(crit.get("true")) or "yes, the statement holds"
    ff = criterion_text(crit.get("false")) or "no, the statement does not hold"
    return ["true", "false"], [f"true: {tt}", f"false: {ff}"]


class Builder:
    def __init__(self, tok, max_len=512, head_max_len=192):
        self.tok, self.max_len, self.head_max_len = tok, max_len, head_max_len
        self.cls, self.sep, self.mask = tok.convert_tokens_to_ids(["<s>", "</s>", "<mask>"])
        self.pad = tok.pad_token_id

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def state_ids(self, state):
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return self.enc(s)

    def build(self, q, state_ids, order):
        """ids and marker positions; `order` lists option indices (omg order) as laid out."""
        keys, opts = options(q)
        head = self.enc(f"{QTYPES[q['type']]} question: {q.get('instructions') or ''}")
        opt_ids = [[self.mask] + self.enc(" " + opts[i])[:48] for i in order]
        used = sum(map(len, opt_ids))
        if self.head_max_len - used < 16:
            per = max((self.head_max_len - 16) // len(opt_ids), 4)
            opt_ids = [o[:per] for o in opt_ids]
            used = sum(map(len, opt_ids))
        head = head[: max(self.head_max_len - used, 8)]
        ids = [self.cls] + head + [self.sep]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids += o
        ids.append(self.sep)
        room = self.max_len - len(ids) - 1
        ids += state_ids[: max(room, 0)] + [self.sep]
        return ids[: self.max_len], markers


class Scorer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.l1 = torch.nn.Linear(d, d)
        self.l2 = torch.nn.Linear(d, 1)

    def forward(self, h):
        return self.l2(F.gelu(self.l1(self.norm(h)))).squeeze(-1)


class Cross(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.scorer = Scorer(self.enc.config.hidden_size)

    def forward(self, ids, attn, markers, kmask):
        h = self.enc(input_ids=ids, attention_mask=attn).last_hidden_state
        rows = torch.gather(h, 1, markers.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        z = self.scorer(rows)
        return z.masked_fill(~kmask, -1e4)


def examples_distill(b, recs, keep, rng, shuffle):
    """keep(instruction) -> bool. Soft targets from E4B."""
    out = []
    for ri, r in enumerate(recs):
        sid = b.state_ids(r["state"])
        for qid, q in r["questions"].items():
            if not keep(q.get("instructions") or ""):
                continue
            k = len(options(q)[0])
            order = list(range(k))
            if shuffle and q["type"] == "choice":
                rng.shuffle(order)
            ids, markers = b.build(q, sid, order)
            p = np.asarray(r["probs"][qid], dtype=np.float32)
            out.append({"ids": ids, "markers": markers, "order": order, "target": p[order].tolist(),
                        "src": "distill", "type": q["type"], "ins": q.get("instructions"), "family": r["family"], "rec": ri, "qid": qid})
    return out


def examples_jglue(b, task, recs, rng, shuffle):
    out = []
    for i, r in enumerate(recs):
        body, gold = payload(task, r, "x")
        q = body["questions"]["answer"]
        keys = list(q["criteria"])
        order = list(range(len(keys)))
        if shuffle:
            rng.shuffle(order)
        ids, markers = b.build(q, b.state_ids(body["state"]), order)
        target = [0.0] * len(keys)
        target[order.index(keys.index(gold))] = 1.0
        out.append({"ids": ids, "markers": markers, "order": order, "target": target, "src": task, "gold": keys.index(gold), "rec": i})
    return out


def batches(exs, bs, pad, shuffle, rng):
    by = {}
    for e in exs:
        L = next(x for x in BUCKETS if x >= len(e["ids"]))
        by.setdefault(L, []).append(e)
    chunks = []
    for L, es in by.items():
        if shuffle:
            rng.shuffle(es)
        step = max(1, bs * 256 // L)  # constant tokens per batch
        chunks += [(L, es[i : i + step]) for i in range(0, len(es), step)]
    if shuffle:
        rng.shuffle(chunks)
    for L, es in chunks:
        K = 8  # fixed option slots: one graph per bucket
        ids = torch.full((len(es), L), pad, dtype=torch.long)
        attn = torch.zeros((len(es), L), dtype=torch.long)
        mk = torch.zeros((len(es), K), dtype=torch.long)
        km = torch.zeros((len(es), K), dtype=torch.bool)
        tg = torch.zeros((len(es), K))
        for j, e in enumerate(es):
            n = len(e["ids"])
            ids[j, :n] = torch.tensor(e["ids"])
            attn[j, :n] = 1
            m = e["markers"][:K]
            mk[j, : len(m)] = torch.tensor(m)
            km[j, : len(m)] = True
            tg[j, : len(m)] = torch.tensor(e["target"][:K])
        yield es, ids, attn, mk, km, tg


@torch.no_grad()
def predict(model, exs, pad, device, bs=32):
    model.eval()
    for es, ids, attn, mk, km, _ in batches(exs, bs, pad, False, None):
        z = model(ids.to(device), attn.to(device), mk.to(device), km.to(device)).float().cpu()
        for j, e in enumerate(es):
            k = len(e["markers"])
            zz = np.zeros(k, np.float32)
            zz[e["order"]] = z[j, :k].numpy()  # back to omg order
            e["logits"] = zz


def distill_metrics(exs):
    """Agreement with E4B's argmax, soft cross-entropy and ECE vs E4B's label, by type."""
    res = {}
    for t in ("all", "choice", "score", "noul"):
        es = [e for e in exs if t == "all" or e["type"] == t]
        if not es:
            continue
        agree, ce, conf = [], [], []
        for e in es:
            p = np.exp(e["logits"] - e["logits"].max())
            p /= p.sum()
            tgt = np.zeros_like(p)
            tgt[e["order"]] = e["target"]
            agree.append(float(p.argmax() == tgt.argmax()))
            ce.append(float(-(tgt * np.log(np.maximum(p, 1e-9))).sum()))
            conf.append(float(p.max()))
        from e5_head import ece
        res[t] = {"n": len(es), "agree": float(np.mean(agree)), "soft_ce": float(np.mean(ce)), "ece": ece(conf, agree)}
    return res


def train(args):
    rng = random.Random(0)
    torch.manual_seed(0)
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    b = Builder(tok)
    lab = rows(".cache/distill/labeled.jsonl")
    train_states = {json.dumps(r["state"], ensure_ascii=False) for r in rows(".cache/distill/train.jsonl")}
    tr = [r for r in lab if json.dumps(r["state"], ensure_ascii=False) in train_states]
    te = [r for r in lab if json.dumps(r["state"], ensure_ascii=False) not in train_states]
    print(f"distill: {len(tr)} train states, {len(te)} held-out states; {len(HELD_OUT)} held-out instructions", flush=True)
    jnli_tr, jcqa_tr = rows(".cache/jglue/jnli-train.jsonl"), rows(".cache/jglue/jcommonsenseqa-train.jsonl")
    if args.jglue_limit:
        jnli_tr, jcqa_tr = jnli_tr[: args.jglue_limit], jcqa_tr[: args.jglue_limit]

    def build_train():
        ex = examples_distill(b, tr, lambda s: s not in HELD_OUT, rng, True)
        ex = ex * args.distill_repeat
        ex += examples_jglue(b, "jnli", jnli_tr, rng, True) + examples_jglue(b, "jcqa", jcqa_tr, rng, True)
        return ex

    ex_train = build_train()
    n_d = sum(e["src"] == "distill" for e in ex_train)
    print(f"train sequences: {len(ex_train)} (distill {n_d}, JGLUE {len(ex_train) - n_d}); "
          f"mean {np.mean([len(e['ids']) for e in ex_train]):.0f} tokens", flush=True)

    model = Cross(args.model).to(dev)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": args.lr},
                             {"params": model.scorer.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * sum(1 for _ in batches(ex_train, args.bs, b.pad, True, random.Random(1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, 0.06 * steps)) * max(0.0, (steps - s) / (steps * 0.94)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        if ep > 0:
            ex_train = build_train()  # fresh option shuffles
        tot, n = 0.0, 0
        for _, ids, attn, mk, km, tg in batches(ex_train, args.bs, b.pad, True, rng):
            z = model(ids.to(dev), attn.to(dev), mk.to(dev), km.to(dev))
            loss = -(tg.to(dev) * F.log_softmax(z, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot += loss.item() * len(ids)
            n += len(ids)
            if step % 200 == 0:
                print(f"  ep {ep} step {step}/{steps} loss {tot / n:.3f} {time.perf_counter() - t0:.0f}s", flush=True)
        print(f"epoch {ep}: loss {tot / n:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    return model, tok, b


def evaluate(args, model, tok, b):
    dev = args.device
    rng = random.Random(0)
    lab = rows(".cache/distill/labeled.jsonl")
    train_states = {json.dumps(r["state"], ensure_ascii=False) for r in rows(".cache/distill/train.jsonl")}
    te = [r for r in lab if json.dumps(r["state"], ensure_ascii=False) not in train_states]
    res = {"args": vars(args)}
    seen = examples_distill(b, te, lambda s: s not in HELD_OUT, rng, False)
    unseen = examples_distill(b, lab, lambda s: s in HELD_OUT, rng, False)
    predict(model, seen + unseen, b.pad, dev)
    res["distill_seen"] = distill_metrics(seen)
    res["distill_unseen"] = distill_metrics(unseen)
    res["distill_unseen_by_instruction"] = {
        ins: distill_metrics([e for e in unseen if e["ins"] == ins])["all"] for ins in sorted(HELD_OUT)}
    # A prior-only baseline: E4B's most frequent answer for the instruction.
    res["prior_baseline_unseen"] = prior_baseline(unseen)
    res["prior_baseline_seen"] = prior_baseline(seen)
    for task, f in (("jnli", "jnli"), ("jcqa", "jcommonsenseqa")):
        va = rows(f".cache/jglue/{f}-valid.jsonl")
        ex = examples_jglue(b, task, va, rng, False)
        predict(model, ex, b.pad, dev)
        res[task] = report(f"cross-{task}", np.stack([e["logits"] for e in ex]), np.array([e["gold"] for e in ex]))
    for name, e in res.items():
        if name.startswith("distill") or name.startswith("prior"):
            print(name, json.dumps(e, ensure_ascii=False)[:600], flush=True)
    json.dump(res, open(Path(args.out) / "summary.json", "w"), indent=1, ensure_ascii=False)
    return res


def prior_baseline(exs):
    from collections import Counter, defaultdict
    by = defaultdict(Counter)
    for e in exs:
        tgt = np.zeros(len(e["markers"]))
        tgt[e["order"]] = e["target"]
        by[e["ins"]][int(tgt.argmax())] += 1
    agree = sum(c.most_common(1)[0][1] for c in by.values()) / max(1, len(exs))
    return {"n": len(exs), "agree": agree}


@torch.no_grad()
def answer(model, b, req, device):
    """omg-shaped answers for one request (every question its own sequence)."""
    sid = b.state_ids(req["state"])
    exs = []
    for qid, q in req["questions"].items():
        k = len(options(q)[0])
        ids, markers = b.build(q, sid, list(range(k)))
        exs.append({"ids": ids, "markers": markers, "order": list(range(k)), "target": [0.0] * k, "qid": qid})
    t = time.perf_counter()
    predict(model, exs, b.pad, device)
    ms = (time.perf_counter() - t) * 1000
    out = {}
    for e in exs:
        q = req["questions"][e["qid"]]
        keys, _ = options(q)
        p = np.exp(e["logits"] - e["logits"].max())
        p /= p.sum()
        if q["type"] == "noul":
            out[e["qid"]] = round(float(p[0]), 3)
        elif q["type"] == "score":
            out[e["qid"]] = round(float((np.arange(len(p)) * p).sum()), 2)
        else:
            out[e["qid"]] = keys[int(p.argmax())]
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cl-nagoya/ruri-v3-70m")
    ap.add_argument("--out", default="runs/cross/70m")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16, help="sequences per batch at 256 tokens (scaled by bucket length)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--distill-repeat", type=int, default=2, help="copies of the distill questions per epoch (fresh option order each)")
    ap.add_argument("--jglue-limit", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--probe", nargs="*", default=[], help="request JSON files to answer after training")
    a = ap.parse_args()
    if a.eval_only:
        tok = AutoTokenizer.from_pretrained(a.model)
        b = Builder(tok)
        model = Cross(a.model).to(a.device)
        model.load_state_dict(torch.load(Path(a.out) / "model.pt", map_location=a.device))
    else:
        model, tok, b = train(a)
    evaluate(a, model, tok, b)
    for p in a.probe:
        req = json.load(open(p))
        out, ms = answer(model, b, req, a.device)
        print(p, f"{ms:.0f} ms", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
