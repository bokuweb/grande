"""Fine-tune the whole embedding encoder (default multilingual-e5-small) as a
cross-encoder on one JGLUE task and score it with the protocol of
tools/e5_head.py. This is the "small model trained on the task" row, the
counterpart of grande's 270M + LoRA + head, not of the frozen-backbone heads.

    python tools/e5_finetune.py --task jnli --out runs/e5/ft-jnli
    python tools/e5_finetune.py --task jcqa --out runs/e5/ft-jcqa

JNLI: "query: 前提: … 仮説: …" -> masked mean -> Linear(384, 3).
JCQA: five "query: 質問: … 答え: …" sequences -> masked mean -> Linear(384, 1), softmax over the five.
The last 10% of train (by record) picks the epoch; valid is only scored.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from e5_features import jcqa_texts, jnli_texts, rows, JNLI_LABELS  # noqa: E402
from e5_head import report  # noqa: E402

torch.manual_seed(0)


class CrossEncoder(torch.nn.Module):
    def __init__(self, name, out):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, out)

    def forward(self, enc):
        h = self.enc(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1)
        return self.head(v)


def texts_of(task, r):
    if task == "jnli":
        return [jnli_texts(r)["pair"]], JNLI_LABELS.index(r["label"])
    t = jcqa_texts(r)
    return [t[f"qc{i}"] for i in range(5)], int(r["label"])


def batches(recs, task, bs, shuffle):
    idx = np.random.permutation(len(recs)) if shuffle else np.arange(len(recs))
    for i in range(0, len(idx), bs):
        chunk = [texts_of(task, recs[j]) for j in idx[i : i + bs]]
        yield [t for ts, _ in chunk for t in ts], torch.tensor([g for _, g in chunk])


def logits_of(model, tok, texts, task, device, max_len):
    # fixed-length padding: MPS compiles and caches one graph per input shape, and
    # dynamic lengths at batch 32 x 5 sequences grew the JCQA run past 13 GB
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt").to(device)
    z = model(enc)
    return z if task == "jnli" else z.view(-1, 5)


@torch.no_grad()
def predict(model, tok, recs, task, device, max_len, bs=64):
    model.eval()
    out = []
    for texts, _ in batches(recs, task, bs, False):
        out.append(logits_of(model, tok, texts, task, device, max_len).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--task", choices=["jnli", "jcqa"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=".cache/jglue")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=80, help="every sequence is padded to this (JNLI pairs max 74 tokens, JCQA 66)")
    ap.add_argument("--limit", type=int, help="train on the first N records only")
    a = ap.parse_args()
    np.random.seed(0)

    files = {"jnli": "jnli", "jcqa": "jcommonsenseqa"}
    train = rows(f"{a.data}/{files[a.task]}-train.jsonl")
    valid = rows(f"{a.data}/{files[a.task]}-valid.jsonl")
    if a.limit:
        train = train[: a.limit]
    cut = int(len(train) * 0.9)
    train, sel = train[:cut], train[cut:]
    gold_sel = np.array([texts_of(a.task, r)[1] for r in sel])
    gold_va = np.array([texts_of(a.task, r)[1] for r in valid])

    tok = AutoTokenizer.from_pretrained(a.model)
    model = CrossEncoder(a.model, 3 if a.task == "jnli" else 1).to(a.device)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr}, {"params": model.head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / (steps * 0.94)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, log = 1e9, []
    for ep in range(a.epochs):
        model.train()
        t, n, tot = time.perf_counter(), 0, 0.0
        for texts, gold in batches(train, a.task, a.bs, True):
            loss = F.cross_entropy(logits_of(model, tok, texts, a.task, a.device, a.max_len), gold.to(a.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(gold)
            n += len(gold)
            if (n // a.bs) % 100 == 0:
                print(f"  ep {ep} {n}/{len(train)} loss {tot / n:.3f} {time.perf_counter() - t:.0f}s", flush=True)
        z = predict(model, tok, sel, a.task, a.device, a.max_len)
        sel_loss = F.cross_entropy(torch.as_tensor(z), torch.as_tensor(gold_sel)).item()
        sel_acc = float((z.argmax(-1) == gold_sel).mean())
        log.append({"epoch": ep, "train_loss": tot / n, "sel_loss": sel_loss, "sel_acc": sel_acc, "s": time.perf_counter() - t})
        print(f"epoch {ep}: train {tot / n:.3f} sel loss {sel_loss:.3f} acc {sel_acc:.3f} ({time.perf_counter() - t:.0f}s)", flush=True)
        if sel_loss < best:
            best = sel_loss
            zv = predict(model, tok, valid, a.task, a.device, a.max_len)
            np.save(out / "valid-logits.npy", zv)
            torch.save(model.state_dict(), out / "model.pt")
    zv = np.load(out / "valid-logits.npy")
    row = report(f"ft-{a.task}", zv, gold_va)
    json.dump({"args": vars(a), "log": log, "result": row}, open(out / "summary.json", "w"), indent=1)
    print(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
