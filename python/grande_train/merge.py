"""Merge a trained LoRA adapter into the base checkpoint for GGUF conversion.

    python -m grande_train.merge --base unsloth/gemma-3-270m --run runs/x --out runs/x/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train import backbone_path, text_backbone


def prune_added_tokens(out: str, vocab_size: int | None):
    """Some mirrors register added tokens past the embedding table (e.g. the
    270m mirror's `<image_soft_token>` at 262144). llama.cpp's converter
    asserts on them; they are unused here, so drop them."""
    import json
    import os

    if vocab_size is None:
        return
    path = os.path.join(out, "tokenizer.json")
    t = json.load(open(path, encoding="utf-8"))
    before = len(t.get("added_tokens", []))
    t["added_tokens"] = [x for x in t.get("added_tokens", []) if x["id"] < vocab_size]
    if len(t["added_tokens"]) != before:
        json.dump(t, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        extra = os.path.join(out, "added_tokens.json")
        if os.path.exists(extra):
            os.remove(extra)
        print(f"dropped {before - len(t['added_tokens'])} added token(s) beyond vocab_size={vocab_size}")
    # transformers re-adds tokens named in tokenizer_config's model-specific
    # special token fields; strip references to tokens we just dropped.
    cfg_path = os.path.join(out, "tokenizer_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    kept = {x["content"] for x in t["added_tokens"]}
    changed = False
    for key in list(cfg):
        v = cfg[key]
        if isinstance(v, str) and v.startswith("<") and v.endswith(">") and v not in kept and key.endswith("_token"):
            del cfg[key]
            changed = True
        elif isinstance(v, dict) and key == "model_specific_special_tokens":
            pruned = {k: tok for k, tok in v.items() if tok in kept}
            if pruned != v:
                cfg[key] = pruned
                changed = True
    if changed:
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    full = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    path = backbone_path(full)
    backbone = text_backbone(full)
    merged = PeftModel.from_pretrained(backbone, f"{a.run}/adapter").merge_and_unload()
    parent = full
    for attr in path[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, path[-1], merged)
    full.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    prune_added_tokens(a.out, full.config.vocab_size if hasattr(full.config, "vocab_size") else None)
    print(f"merged model saved to {a.out}")


if __name__ == "__main__":
    main()
