"""Package a sentence embedder — multilingual-e5-small (BERT) or Ruri v3
(ModernBERT-Ja) — plus the (state, option) head of tools/e5_generic.py for
grande's wgpu engine (crates/grande-wgpu/src/e5.rs), natively and in the
browser.

Weights go to the engine's manifest layout (model::Manifest): f16, or Q8_0
codes with f16 block scales (`--dtype q8`). BERT: Q, K, V are fused into
one `[3d, d]` projection with 1/sqrt(head_dim) folded into Q (the
attention kernel has no scale; ModernBERT's RoPE kernel applies it, so its
fused Wqkv is copied as is), token type row 0 becomes the embedding bias.
ModernBERT: `mlp.Wi` is split into its value / gate halves. The head's
feature z-scoring (mu, sd) is folded into its first Linear. With
`--corpus`, the 250k XLM-R vocabulary is pruned to the pieces the corpus
uses, the `--keep-top` most probable pieces, every single-character piece
and the specials; Unigram has no merges, so the corpus tokenizes the same
and other text re-segments over the pieces that remain (JNLI valid lost 5
points with the corpus alone; with the 40k most probable pieces added it
matches the full vocabulary). The embedding is 82% of the model, so this
is what makes it a small download (118M f16: 235 MB -> ~50 MB q8).

Output directory:
    config.json            HF config + `grande_e5` (head, token ids, temperature, name)
    tokenizer.json         pruned (or copied) HF tokenizer
    tokenizer_config.json  special_tokens_map.json
    manifest.json          tensor -> file / byte ranges
    embed.bin  enc.bin  head.bin

    python tools/export_e5.py --head runs/e5/generic/head.pt \\
        --corpus .cache/jglue/*-train.jsonl .cache/kev/*.jsonl examples/*.json \\
        --dtype q8 --out web/models/multilingual-e5-small-wgpu
    python tools/export_e5.py --model cl-nagoya/ruri-v3-130m --prefix "" --temperature 1.75 \\
        --head runs/ruri/130m/generic/head.pt --corpus … --dtype q8 --out web/models/ruri-v3-130m-wgpu
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).parent))
from export_laya import SCAFFOLD, Writer, corpus_texts  # noqa: E402


def snapshot(name: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, allow_patterns=["*.json", "model.safetensors"]))


def prune(tok_json: dict, hf, texts: list[str], top: int) -> tuple[dict, np.ndarray]:
    """Kept old ids (sorted) and the rewritten Unigram tokenizer.json: the
    specials, every single-character piece, the `top` most probable pieces
    and whatever the corpus uses."""
    model = tok_json["model"]
    assert model["type"] == "Unigram", model["type"]
    vocab = model["vocab"]  # [[piece, score], ...]
    V = len(vocab)
    keep = set(a["id"] for a in tok_json.get("added_tokens", []))
    unk = model.get("unk_id")
    if unk is not None:
        keep.add(unk)
    for i, (piece, _) in enumerate(vocab):
        if len(piece) == 1 or (len(piece) == 2 and piece[0] == "▁"):
            keep.add(i)
    keep |= set(int(i) for i in np.argsort([-sc for _, sc in vocab], kind="stable")[:top])
    base = len(keep)
    used = set()
    for enc in hf.encode_batch(texts, add_special_tokens=False):
        used.update(enc.ids)
    keep |= used
    print(f"tokens: {base} structural, {len(used)} from the corpus")
    keep = sorted(i for i in keep if i < V)
    new_id = {old: new for new, old in enumerate(keep)}
    print(f"keeping {len(keep)} of {V} tokens ({100 * len(keep) / V:.1f}%)")
    out = json.loads(json.dumps(tok_json))
    out["model"]["vocab"] = [vocab[old] for old in keep]
    if unk is not None:
        out["model"]["unk_id"] = new_id[unk]
    out["added_tokens"] = [dict(a, id=new_id[a["id"]]) for a in tok_json.get("added_tokens", []) if a["id"] in new_id]
    return out, np.array(keep, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small", help="Hub id or local snapshot directory")
    ap.add_argument("--head", required=True, help="head.pt from tools/e5_generic.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f16", "q8"], default="q8", help="storage of the 2-D encoder weights")
    ap.add_argument("--corpus", nargs="*", default=[], help="files whose tokens the pruned vocabulary must cover; omit to keep all 250k")
    ap.add_argument("--extra", nargs="*", default=[], help="extra strings to keep tokens for")
    ap.add_argument("--keep-top", type=int, default=40000, help="with --corpus: also keep this many of the most probable pieces (Unigram scores), so text outside the corpus still segments as the model saw it")
    ap.add_argument("--prefix", default="query: ", help='text prefix the head was trained with ("query: " for e5, "" for Ruri v3)')
    ap.add_argument("--temperature", type=float, default=1.4, help="the head's calibration temperature (tools/e5_serve.py's default; the generic head's fitted T is in its summary.json)")
    ap.add_argument("--name", help="model name written to config.json (default: the output directory's)")
    a = ap.parse_args()
    src = Path(a.model) if Path(a.model).is_dir() else snapshot(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    tok_json = json.load(open(src / "tokenizer.json"))
    W = load_file(str(src / "model.safetensors"))
    d, L, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    hd = d // H
    modern = cfg["model_type"] == "modernbert"
    assert modern or cfg["model_type"] == "bert", cfg["model_type"]
    if modern and any(k.startswith("model.") for k in W):
        W = {k[len("model."):]: v for k, v in W.items() if k.startswith("model.")}

    from tokenizers import Tokenizer as Hf

    hf = Hf.from_file(str(src / "tokenizer.json"))
    embed = W["embeddings.tok_embeddings.weight" if modern else "embeddings.word_embeddings.weight"]
    if a.corpus:
        texts = SCAFFOLD + list(corpus_texts(a.corpus)) + a.extra
        tok_json, keep = prune(tok_json, hf, texts, a.keep_top)
        embed = embed[keep]
    else:
        embed = embed[: len(tok_json["model"]["vocab"])]
    vocab = len(tok_json["model"]["vocab"])
    json.dump(tok_json, open(out / "tokenizer.json", "w"), ensure_ascii=False)
    for f in ("tokenizer_config.json", "special_tokens_map.json"):
        if (src / f).is_file():
            shutil.copy(src / f, out / f)
    hf2 = Hf.from_file(str(out / "tokenizer.json"))
    ids = {t: hf2.token_to_id(t) for t in ("<s>", "</s>", "<pad>")}
    assert None not in ids.values(), ids

    head = torch.load(a.head, weights_only=False)
    mu, sd = head["mu"].numpy(), head["sd"].numpy()
    l1, l1_b = head["model"]["net.0.weight"].numpy(), head["model"]["net.0.bias"].numpy()
    l2, l2_b = head["model"]["net.3.weight"].numpy(), head["model"]["net.3.bias"].numpy()
    assert l1.shape[1] == 4 * d and l2.shape[0] == 1, (l1.shape, l2.shape)
    # z = W (x - mu) / sd + b = (W / sd) x + (b - W mu / sd)
    l1f = l1 / sd[None, :]
    l1f_b = l1_b - l1f @ mu

    g = {"name": a.name or out.name, "cls": ids["<s>"], "sep": ids["</s>"], "pad": ids["<pad>"], "prefix": a.prefix,
         "max_len": min(cfg["max_position_embeddings"], 512), "head_in": 4 * d, "head_hidden": int(l1.shape[0]), "temperature": a.temperature,
         "head_source": str(a.head)}
    json.dump(dict(cfg, vocab_size=vocab, grande_e5=g), open(out / "config.json", "w"), indent=2, ensure_ascii=False)

    files = []
    w = Writer(out / "embed.bin")
    w.tensor("embed", embed, a.dtype)
    if modern:
        norm_bias = cfg.get("norm_bias", False)
        w.f16("embed_norm_w", W["embeddings.norm.weight"])
        w.f16("final_norm_w", W["final_norm.weight"])
        if norm_bias:
            w.f16("embed_norm_b", W["embeddings.norm.bias"])
            w.f16("final_norm_b", W["final_norm.bias"])
        files.append(w.close())
        w = Writer(out / "enc.bin")
        ff = cfg["intermediate_size"]
        for l in range(L):
            p = f"layers.{l}."
            n = f"enc.{l}."
            if l > 0:
                w.f16(n + "attn_norm_w", W[p + "attn_norm.weight"])
                if norm_bias:
                    w.f16(n + "attn_norm_b", W[p + "attn_norm.bias"])
            w.tensor(n + "qkv", W[p + "attn.Wqkv.weight"], a.dtype)
            w.tensor(n + "o", W[p + "attn.Wo.weight"], a.dtype)
            w.f16(n + "mlp_norm_w", W[p + "mlp_norm.weight"])
            if norm_bias:
                w.f16(n + "mlp_norm_b", W[p + "mlp_norm.bias"])
            wi = W[p + "mlp.Wi.weight"]
            assert wi.shape == (2 * ff, d), wi.shape
            w.tensor(n + "wi_val", wi[:ff], a.dtype)
            w.tensor(n + "wi_gate", wi[ff:], a.dtype)
            w.tensor(n + "wo", W[p + "mlp.Wo.weight"], a.dtype)
        files.append(w.close())
    else:
        w.f16("pos_embed", W["embeddings.position_embeddings.weight"])
        w.f16("type_embed", W["embeddings.token_type_embeddings.weight"][0])
        w.f16("embed_norm_w", W["embeddings.LayerNorm.weight"])
        w.f16("embed_norm_b", W["embeddings.LayerNorm.bias"])
        files.append(w.close())
        w = Writer(out / "enc.bin")
    for l in range(0 if modern else L):
        p = f"encoder.layer.{l}."
        n = f"enc.{l}."
        q, k, v = (W[p + f"attention.self.{x}.weight"] for x in ("query", "key", "value"))
        qb, kb, vb = (W[p + f"attention.self.{x}.bias"] for x in ("query", "key", "value"))
        s = hd ** -0.5
        w.tensor(n + "qkv", np.concatenate([q * s, k, v]), a.dtype)
        w.f16(n + "qkv_b", np.concatenate([qb * s, kb, vb]))
        w.tensor(n + "o", W[p + "attention.output.dense.weight"], a.dtype)
        w.f16(n + "o_b", W[p + "attention.output.dense.bias"])
        w.f16(n + "attn_norm_w", W[p + "attention.output.LayerNorm.weight"])
        w.f16(n + "attn_norm_b", W[p + "attention.output.LayerNorm.bias"])
        w.tensor(n + "wi", W[p + "intermediate.dense.weight"], a.dtype)
        w.f16(n + "wi_b", W[p + "intermediate.dense.bias"])
        w.tensor(n + "wo", W[p + "output.dense.weight"], a.dtype)
        w.f16(n + "wo_b", W[p + "output.dense.bias"])
        w.f16(n + "mlp_norm_w", W[p + "output.LayerNorm.weight"])
        w.f16(n + "mlp_norm_b", W[p + "output.LayerNorm.bias"])
    if not modern:
        files.append(w.close())
    w = Writer(out / "head.bin")
    w.f16("head.l1", l1f)
    w.f16("head.l1_b", l1f_b)
    w.f16("head.l2", l2)
    w.f16("head.l2_b", l2_b)
    files.append(w.close())
    json.dump({"files": files}, open(out / "manifest.json", "w"), indent=1)
    total = sum((out / f["path"]).stat().st_size for f in files)
    print(f"wrote {out}: {total / 1e6:.1f} MB of weights, vocab {vocab}")


if __name__ == "__main__":
    main()
