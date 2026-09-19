"""Prune the vocabulary of a merged HF checkpoint (for the browser / ONNX
path). Same policy as tools/prune_vocab.py for GGUF: keep every added /
special / byte token, every token the corpus uses, and the BPE merge closure.

    python tools/prune_hf_vocab.py --model runs/grande-270m-12k/merged \
        --corpus ".cache/jglue/*-train.jsonl" "examples/*.json" ".cache/corpus/wiki-ja.jsonl" \
        --out runs/grande-270m-12k/merged-pruned
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from prune_vocab import corpus_texts  # noqa: E402  (same directory)

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=["user", "model", "State:", "Question:", "Answer with one letter."])
    a = ap.parse_args()
    src, out = Path(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tj = json.load(open(src / "tokenizer.json", encoding="utf-8"))
    vocab: dict[str, int] = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    merge_strs = [m if isinstance(m, str) else " ".join(m) for m in merges]
    id2tok = {i: t for t, i in vocab.items()}
    V = len(vocab)
    print(f"vocab {V}, merges {len(merges)}")

    tok = AutoTokenizer.from_pretrained(src)
    keep: set[int] = set()
    for at in tj.get("added_tokens", []):
        keep.add(at["id"])
    for t, i in vocab.items():
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t) or (t.startswith("<") and t.endswith(">") and i < 300):
            keep.add(i)
    used = set()
    for text in list(corpus_texts(a.corpus)) + a.extra:
        used.update(i for i in tok.encode(text, add_special_tokens=False) if i < V)
    print(f"{len(used)} distinct tokens used")
    keep |= used
    for ch in LABELS:
        if ch in vocab:
            keep.add(vocab[ch])
    produced = {}
    for m in merge_strs:
        x, y = m.split(" ", 1)
        produced.setdefault(x + y, (x, y))
    stack = list(keep)
    while stack:
        i = stack.pop()
        pair = produced.get(id2tok.get(i, ""))
        if not pair:
            continue
        for part in pair:
            j = vocab.get(part)
            if j is not None and j not in keep:
                keep.add(j)
                stack.append(j)
    keep_sorted = sorted(keep)
    new_id = {old: new for new, old in enumerate(keep_sorted)}
    print(f"keeping {len(keep_sorted)} tokens ({100 * len(keep_sorted) / V:.1f}%)")

    # tokenizer.json
    tj["model"]["vocab"] = {id2tok[o]: n for o, n in new_id.items()}
    kept_tok = set(tj["model"]["vocab"])
    new_merges = []
    for m, ms in zip(merges, merge_strs):
        x, y = ms.split(" ", 1)
        if x in kept_tok and y in kept_tok and (x + y) in kept_tok:
            new_merges.append(m)
    tj["model"]["merges"] = new_merges
    for at in tj.get("added_tokens", []):
        at["id"] = new_id[at["id"]]
    if tj.get("post_processor"):
        # special_tokens entries carry ids
        def remap(obj):
            if isinstance(obj, dict):
                if "ids" in obj and "tokens" in obj:
                    obj["ids"] = [new_id.get(i, i) for i in obj["ids"]]
                for v in obj.values():
                    remap(v)
            elif isinstance(obj, list):
                for v in obj:
                    remap(v)
        remap(tj["post_processor"])
    json.dump(tj, open(out / "tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"merges {len(merges)} -> {len(new_merges)}")

    # tokenizer_config.json: added_tokens_decoder keys are ids
    tc = json.load(open(src / "tokenizer_config.json", encoding="utf-8"))
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {str(new_id[int(k)]): v for k, v in tc["added_tokens_decoder"].items() if int(k) in new_id}
    json.dump(tc, open(out / "tokenizer_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for f in ("special_tokens_map.json", "generation_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)

    # config.json
    cfg = json.load(open(src / "config.json", encoding="utf-8"))
    tc_key = "text_config" if "text_config" in cfg else None
    target = cfg[tc_key] if tc_key else cfg
    target["vocab_size"] = len(keep_sorted)
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        for c in (cfg, target):
            if k in c and c[k] is not None:
                c[k] = [new_id[i] for i in c[k]] if isinstance(c[k], list) else new_id[c[k]]
    json.dump(cfg, open(out / "config.json", "w", encoding="utf-8"), indent=2)

    # weights
    idx = torch.tensor(keep_sorted)
    for shard in sorted(src.glob("*.safetensors")):
        sd = load_file(str(shard))
        for name, t in list(sd.items()):
            if t.ndim == 2 and t.shape[0] == V:
                sd[name] = t[idx].contiguous()
                print(f"  {name}: {tuple(t.shape)} -> {tuple(sd[name].shape)}")
        save_file(sd, str(out / shard.name), metadata={"format": "pt"})
    if (src / "model.safetensors.index.json").exists():
        shutil.copy(src / "model.safetensors.index.json", out / "model.safetensors.index.json")
    AutoConfig.from_pretrained(out)  # sanity
    t2 = AutoTokenizer.from_pretrained(out)
    s = "合言葉は「青い象」であるか"
    print("check:", tok.encode(s, add_special_tokens=False), "->", t2.encode(s, add_special_tokens=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
