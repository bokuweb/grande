"""Embed JGLUE (JNLI, JCommonsenseQA) with a frozen sentence-embedding model
(default intfloat/multilingual-e5-small) and dump the vectors a head can be
trained on, plus batch-1 latency per record.

    python tools/e5_features.py --out runs/e5              # all four splits
    python tools/e5_features.py --out runs/e5 --latency 200   # also time 200 valid records at batch 1

Per record the script embeds every text a head might read:
  JNLI: premise, hypothesis (separately) and the pair as one sequence
  JCQA: question, each choice (separately) and each (question, choice) as one sequence
e5 wants a "query: " prefix on both sides of a symmetric task; that is what
is used here for everything. Pooling is the model card's (masked mean),
vectors are L2-normalised.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

JNLI_LABELS = ["entailment", "contradiction", "neutral"]


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def jnli_texts(r):
    s1, s2 = r["sentence1"], r["sentence2"]
    return {"a": f"query: {s1}", "b": f"query: {s2}", "pair": f"query: 前提: {s1} 仮説: {s2}"}


def jcqa_texts(r):
    q = r["question"]
    out = {"q": f"query: {q}"}
    for i in range(5):
        c = r[f"choice{i}"]
        out[f"c{i}"] = f"query: {c}"
        out[f"qc{i}"] = f"query: 質問: {q} 答え: {c}"
    return out


class Embedder:
    def __init__(self, name, device, dtype, max_len):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(device).eval()
        self.device, self.max_len = device, max_len

    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


def embed_split(emb, recs, texts_fn, batch):
    keys = list(texts_fn(recs[0]).keys())
    flat = [t for r in recs for t in texts_fn(r).values()]
    out = np.zeros((len(flat), emb.model.config.hidden_size), np.float32)
    t = time.perf_counter()
    for i in range(0, len(flat), batch):
        out[i : i + batch] = emb(flat[i : i + batch])
        if (i // batch) % 50 == 0:
            print(f"  {i}/{len(flat)} {time.perf_counter() - t:.0f}s", flush=True)
    out = out.reshape(len(recs), len(keys), -1)
    return {k: out[:, j] for j, k in enumerate(keys)}


def latency(emb, recs, texts_fn, groups, n, sync):
    """Batch-1 wall time per record: every group of texts is one forward
    (the way a serving head would call it). Returns ms per record per group."""
    res = {}
    for name, keys in groups.items():
        times = []
        for r in recs[:n]:
            tx = texts_fn(r)
            t = time.perf_counter()
            emb([tx[k] for k in keys])
            sync()
            times.append((time.perf_counter() - t) * 1000)
        times = times[5:]  # warm-up
        res[name] = {"mean_ms": float(np.mean(times)), "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
        print(f"  latency {name}: {res[name]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="runs/e5")
    ap.add_argument("--data", default=".cache/jglue")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--splits", nargs="+", default=["jnli-train", "jnli-valid", "jcqa-train", "jcqa-valid"])
    ap.add_argument("--latency", type=int, default=0, help="time this many valid records at batch 1")
    a = ap.parse_args()

    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    emb = Embedder(a.model, a.device, dtype, a.max_len)
    sync = torch.mps.synchronize if a.device == "mps" else (lambda: None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"jnli": "jnli", "jcqa": "jcommonsenseqa"}
    for split in a.splits:
        task, part = split.split("-")
        recs = rows(f"{a.data}/{files[task]}-{part}.jsonl")
        print(f"{split}: {len(recs)} records", flush=True)
        fn = jnli_texts if task == "jnli" else jcqa_texts
        t = time.perf_counter()
        feats = embed_split(emb, recs, fn, a.batch)
        gold = np.array([JNLI_LABELS.index(r["label"]) if task == "jnli" else int(r["label"]) for r in recs])
        np.savez(out / f"{split}.npz", gold=gold, **feats)
        print(f"  wrote {out / f'{split}.npz'} in {time.perf_counter() - t:.0f}s", flush=True)
        if a.latency and part == "valid":
            groups = {"pair": ["pair"], "separate": ["a", "b"]} if task == "jnli" else {"joint": [f"qc{i}" for i in range(5)], "separate": ["q"] + [f"c{i}" for i in range(5)]}
            lat = latency(emb, recs, fn, groups, a.latency, sync)
            json.dump({"model": a.model, "device": a.device, "dtype": a.dtype, "n": a.latency, "latency": lat}, open(out / f"{split}-latency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
