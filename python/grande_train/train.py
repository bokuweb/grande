"""LoRA + pointer head on Gemma 4 base with the packed branch layout.

    python -m grande_train.train --base google/gemma-4-E2B --jnli .cache/jglue/jnli-train.jsonl \
        --jcqa .cache/jglue/jcommonsenseqa-train.jsonl --out runs/grande-e2b --n-per-source 1500

Writes `head.safetensors` (grande-core loads it), the LoRA adapter, and a
training log. Merge the adapter and convert to GGUF with llama.cpp's
`convert_hf_to_gguf.py` to serve it from Rust.

Status: written against kev's recipe and the transformers >= 5 API; not yet
run end to end on Gemma 4. Expect to adjust how the text backbone is reached
(`model.model.language_model` on the multimodal E2B / E4B checkpoints) and the
per-layer-type attention mask.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .data import jcqa, jnli, load_jsonl, rendered_label, shuffled_order
from .model import DecisionModel
from .render import Renderer


def text_backbone(model):
    """Return the decoder stack without lm_head for Gemma 4 checkpoints."""
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/gemma-4-E2B")
    p.add_argument("--jnli")
    p.add_argument("--jcqa")
    p.add_argument("--n-per-source", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    renderer = Renderer(tok)
    cfg = AutoConfig.from_pretrained(a.base)
    text_cfg = getattr(cfg, "text_config", cfg)
    full = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    backbone = text_backbone(full)
    model = DecisionModel(backbone, text_cfg.hidden_size, lora_r=a.lora, sliding_window=getattr(text_cfg, "sliding_window", None)).to(a.device)

    records = []
    for path, conv in ((a.jnli, jnli), (a.jcqa, jcqa)):
        if path:
            rows = load_jsonl(path, conv)
            rng.shuffle(rows)
            records += rows[: a.n_per_source]
    rng.shuffle(records)
    print(f"{len(records)} records, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")
    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        rng.shuffle(records)
        for i in range(0, len(records), a.batch):
            batch = records[i : i + a.batch]
            encs, labels = [], []
            for r in batch:
                # Shuffle option order per example so the head cannot learn positions.
                orders = {qid: shuffled_order(rng, len(q["criteria"]) if q["type"] != "noul" else 2) for qid, q in r["questions"].items()}
                enc = renderer.encode(r, orders)
                encs.append(enc)
                labels.append([rendered_label(r, qid, orders[qid]) for qid in r["questions"]])
            loss = model.loss(encs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0:
                msg = f"epoch {epoch} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s"
                print(msg)
                log.write(msg + "\n")
                log.flush()
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, str(out / "head.safetensors"))
    if a.lora:
        model.lm.save_pretrained(str(out / "adapter"))
    json.dump(vars(a), open(out / "config.json", "w"), indent=2)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
