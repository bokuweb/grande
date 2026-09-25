"""Package a cross-encoder trained by tools/cross_train.py (Ruri v3 +
scorer, docs/cross.md) for omg's wgpu engine. It is a Laya without the
question-type embedding, decision-head layers and act head, so it goes out
in the Laya export layout (tools/export_laya.py) and runs on `laya.rs`,
natively (`omg serve | probe --model <dir>`) and in the browser (kind
`laya` in web/engine.js).

Weights: Q8_0 for the 2-D encoder matrices (`--dtype q8`), f16 for the
norms and the scorer (host). `mlp.Wi` is split into value / gate halves;
the fused Wqkv is copied as is (the RoPE kernel applies 1/sqrt(head_dim)).
The vocabulary is kept whole (Ruri's 102k Japanese pieces: a corpus prune
keeps 84% of them, not worth a changed tokenisation).

    python tools/export_cross.py --model cl-nagoya/ruri-v3-310m --weights runs/cross/310m/model.pt \\
        --out web/models/ruri-v3-310m-cross-wgpu
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_laya import Writer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the Ruri v3 Hub id the weights were trained from (config + tokenizer)")
    ap.add_argument("--weights", required=True, help="model.pt from tools/cross_train.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f16", "q8"], default="q8")
    ap.add_argument("--name", help="model name written to config.json (default: the output directory's)")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(a.model, allow_patterns=["*.json"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    assert cfg["model_type"] == "modernbert", cfg["model_type"]
    W = {k: v.float().numpy() for k, v in torch.load(a.weights, map_location="cpu").items()}
    d, L, ff = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"]
    assert not cfg.get("norm_bias", False), "laya.rs reads bias-free encoder norms"

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(src / f, out / f)
    # The special tokens tools/cross_train.py laid out (Ruri's own cls / sep
    # are <cls> / <sep>; the trainer used <s> / </s>).
    tokens = {"cls_token": "<s>", "sep_token": "</s>", "pad_token": "<pad>", "mask_token": "<mask>"}
    agent = {"max_len": 512, "head_max_len": 192, "head_layers": 0, "type_emb": False, "act_head": False,
             "temperature": [1.0, 1.0, 1.0], "source": str(a.weights)}
    json.dump(dict(cfg, laya_agent=agent, laya_tokens=tokens, laya_name=a.name or out.name, omg_cross=1),
              open(out / "config.json", "w"), indent=2, ensure_ascii=False)

    files = []
    w = Writer(out / "embed.bin")
    w.tensor("embed", W["enc.embeddings.tok_embeddings.weight"], a.dtype)
    w.f16("embed_norm", W["enc.embeddings.norm.weight"])
    w.f16("final_norm", W["enc.final_norm.weight"])
    files.append(w.close())
    w = Writer(out / "enc.bin")
    for l in range(L):
        p, n = f"enc.layers.{l}.", f"enc.{l}."
        if l > 0:
            w.f16(n + "attn_norm", W[p + "attn_norm.weight"])
        w.tensor(n + "qkv", W[p + "attn.Wqkv.weight"], a.dtype)
        w.tensor(n + "o", W[p + "attn.Wo.weight"], a.dtype)
        w.f16(n + "mlp_norm", W[p + "mlp_norm.weight"])
        wi = W[p + "mlp.Wi.weight"]
        assert wi.shape == (2 * ff, d), wi.shape
        w.tensor(n + "wi_val", wi[:ff], a.dtype)
        w.tensor(n + "wi_gate", wi[ff:], a.dtype)
        w.tensor(n + "wo", W[p + "mlp.Wo.weight"], a.dtype)
    files.append(w.close())
    w = Writer(out / "head.bin")
    w.f16("scorer.norm_w", W["scorer.norm.weight"])
    w.f16("scorer.norm_b", W["scorer.norm.bias"])
    w.f16("scorer.l1", W["scorer.l1.weight"])
    w.f16("scorer.l1_b", W["scorer.l1.bias"])
    w.f16("scorer.l2", W["scorer.l2.weight"])
    w.f16("scorer.l2_b", W["scorer.l2.bias"])
    files.append(w.close())
    json.dump({"files": files}, open(out / "manifest.json", "w"), indent=1)
    total = sum((out / f["path"]).stat().st_size for f in files)
    print(f"wrote {out}: {total / 1e6:.1f} MB of weights")


if __name__ == "__main__":
    main()
