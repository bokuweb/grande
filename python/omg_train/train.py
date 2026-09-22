"""LoRA + pointer head on Gemma 4 base with the packed branch layout.

    python -m omg_train.train --base google/gemma-4-E2B --jnli .cache/jglue/jnli-train.jsonl \
        --jcqa .cache/jglue/jcommonsenseqa-train.jsonl --out runs/grande-e2b --n-per-source 1500

Writes `head.safetensors` (omg-core loads it), the LoRA adapter, and a
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

from .data import distilled, jcqa, jnli, jsts, load_jsonl, rendered_label, shuffled_order
from .model import DecisionModel
from .render import Renderer


def backbone_path(model) -> list[str]:
    """Attribute path from the CausalLM wrapper to the decoder stack without
    lm_head: `["model"]` for text-only checkpoints, `["model", "language_model"]`
    for the multimodal Gemma 4 E2B / E4B ones."""
    path, m = [], model
    for attr in ("model", "language_model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
            path.append(attr)
    return path


def text_backbone(model):
    m = model
    for attr in backbone_path(model):
        m = getattr(m, attr)
    return m


def truncate_layers(backbone, text_cfg, n: int):
    """Early exit as a smaller model: drop decoder layers after `n`. The final
    norm then reads layer n's output, the head trains on that, and the merged
    checkpoint converts to a GGUF with n layers — no runtime support needed."""
    layers = backbone.layers
    if n >= len(layers):
        return
    backbone.layers = layers[:n]
    text_cfg.num_hidden_layers = n
    if getattr(text_cfg, "layer_types", None):
        text_cfg.layer_types = list(text_cfg.layer_types)[:n]
    backbone.config.num_hidden_layers = n
    if getattr(backbone.config, "layer_types", None):
        backbone.config.layer_types = list(backbone.config.layer_types)[:n]


def save(model, out: Path, lora: int | None):
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, str(out / "head.safetensors"))
    if lora:
        model.lm.save_pretrained(str(out / "adapter"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/gemma-4-E2B")
    p.add_argument("--jnli")
    p.add_argument("--jcqa")
    p.add_argument("--jsts")
    p.add_argument("--distill", help="teacher-labelled records (tools/teacher_label.py); trained with KL to the teacher distribution")
    p.add_argument("--n-distill", type=int, default=100000)
    p.add_argument("--n-per-source", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=500, help="checkpoint the head and adapter every N steps")
    p.add_argument("--keep-layers", type=int, help="early exit: keep only the first N decoder layers (the head reads layer N's output)")
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
    if a.keep_layers:
        truncate_layers(backbone, text_cfg, a.keep_layers)
    model = DecisionModel(backbone, text_cfg.hidden_size, lora_r=a.lora, sliding_window=getattr(text_cfg, "sliding_window", None)).to(a.device)

    records = []
    for path, conv in ((a.jnli, jnli), (a.jcqa, jcqa), (a.jsts, jsts)):
        if path:
            rows = load_jsonl(path, conv)
            rng.shuffle(rows)
            records += rows[: a.n_per_source]
    if a.distill:
        rows = load_jsonl(a.distill, distilled)
        rng.shuffle(rows)
        records += rows[: a.n_distill]
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
            encs, labels, soft = [], [], []
            for r in batch:
                # Shuffle option order per example so the head cannot learn positions.
                orders = {qid: shuffled_order(rng, len(q["criteria"]) if q["type"] != "noul" else 2) for qid, q in r["questions"].items()}
                enc = renderer.encode(r, orders)
                encs.append(enc)
                labels.append([rendered_label(r, qid, orders[qid]) for qid in r["questions"]])
                probs = r.get("probs")
                soft.append([[probs[qid][o] for o in orders[qid]] if probs and probs.get(qid) else None for qid in r["questions"]])
            loss = model.loss(encs, labels, soft if any(any(x is not None for x in s) for s in soft) else None)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0:
                msg = f"epoch {epoch} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s"
                print(msg)
                log.write(msg + "\n")
                log.flush()
            if a.save_every and step % a.save_every == 0:
                save(model, out, a.lora)
    save(model, out, a.lora)
    json.dump(vars(a), open(out / "config.json", "w"), indent=2)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
