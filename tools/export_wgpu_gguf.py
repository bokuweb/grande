"""Package a Gemma 3 / Gemma 4 GGUF for the wgpu engine (crates/grande-wgpu).

The quantized weights are kept as they are (Q8_0 / Q4_0 codes) but repacked
into the engine's layout: per tensor, the f16 block scales in one contiguous
run and the codes in another, so both are 4-byte aligned storage buffers.
f32 / bf16 tensors become f16. q / k / v are fused into one `qkv` tensor per
layer, [q heads | k heads | v heads] (E4B has two K/V heads); layers that
share an earlier layer's K/V (Gemma 4) drop their unused k / v. The per-layer token table (Gemma 4) goes to its own file, gathered by
the host per request rather than uploaded to the GPU.

Output directory:
    config.json            HF-style text config derived from the GGUF metadata
    tokenizer.json         copied from --tokenizer (an HF checkpoint directory)
    tokenizer_config.json
    manifest.json          tensor -> file / byte ranges (model::Manifest)
    embed.bin              embedding, final norm, per-layer projection
    blk.N.bin              one file per layer
    per_layer_table.bin    Gemma 4 only

    python tools/export_wgpu_gguf.py models/gemma-4-E2B-it-Q4_0.gguf \\
        --tokenizer ~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B/snapshots/<rev> \\
        --out web/models/gemma-4-e2b-wgpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as T

BLOCK = 32
# GGUF block sizes in bytes: f16 scale + codes.
BLOCK_BYTES = {T.Q8_0: 34, T.Q4_0: 18}
DTYPE_NAME = {T.Q8_0: "q8", T.Q4_0: "q4"}


class Writer:
    """Appends tensors to one binary file and records manifest entries."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "wb")
        self.off = 0
        self.entries: list[dict] = []

    def _write(self, b: bytes | np.ndarray) -> tuple[int, int]:
        b = np.ascontiguousarray(b).tobytes() if isinstance(b, np.ndarray) else b
        off = self.off
        self.f.write(b)
        self.off += len(b)
        while self.off % 4:
            self.f.write(b"\0")
            self.off += 1
        return off, len(b)

    def f16(self, name: str, arr: np.ndarray, shape: list[int]):
        off, n = self._write(arr.astype(np.float16))
        self.entries.append({"name": name, "dtype": "f16", "shape": shape, "offset": off, "nbytes": n})

    def quant(self, name: str, kind: T, blocks: np.ndarray, shape: list[int]):
        """`blocks` is [nblocks, block_bytes] uint8 in GGUF layout."""
        bb = BLOCK_BYTES[kind]
        assert blocks.shape[1] == bb, (name, blocks.shape)
        soff, sn = self._write(blocks[:, :2])
        doff, dn = self._write(blocks[:, 2:])
        self.entries.append({
            "name": name, "dtype": DTYPE_NAME[kind], "shape": shape,
            "offset": doff, "nbytes": dn, "scales_offset": soff, "scales_nbytes": sn,
        })

    def tensor(self, name: str, t, shape: list[int] | None = None):
        """Any GGUF tensor: quantized kinds keep their codes, floats become f16."""
        shape = shape or [int(x) for x in reversed(t.shape)]
        if t.tensor_type in BLOCK_BYTES:
            self.quant(name, t.tensor_type, np.asarray(t.data).reshape(-1, BLOCK_BYTES[t.tensor_type]), shape)
        elif t.tensor_type == T.F32:
            self.f16(name, np.asarray(t.data, dtype=np.float32), shape)
        elif t.tensor_type == T.F16:
            self.f16(name, np.asarray(t.data, dtype=np.float16), shape)
        elif t.tensor_type == T.BF16:
            raw = np.asarray(t.data, dtype=np.uint8).reshape(-1, 2)
            u32 = (raw[:, 1].astype(np.uint32) << 24) | (raw[:, 0].astype(np.uint32) << 16)
            self.f16(name, u32.view(np.float32), shape)
        else:
            raise SystemExit(f"{t.name}: unsupported GGUF type {t.tensor_type.name}")

    def close(self) -> dict:
        self.f.close()
        return {"path": self.path.name, "tensors": self.entries}


def concat_quant(parts) -> tuple[T, np.ndarray]:
    kinds = {p.tensor_type for p in parts}
    if len(kinds) != 1:
        raise SystemExit(f"cannot fuse tensors of mixed types {[k.name for k in kinds]}")
    kind = kinds.pop()
    if kind in BLOCK_BYTES:
        return kind, np.concatenate([np.asarray(p.data).reshape(-1, BLOCK_BYTES[kind]) for p in parts])
    return kind, np.concatenate([np.asarray(p.data).reshape(-1) for p in parts])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("--tokenizer", help="directory with tokenizer.json (+ tokenizer_config.json); "
                    "default: the HF cache snapshot matching the GGUF's name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    r = GGUFReader(a.gguf)
    fields = {k: (f.contents() if hasattr(f, "contents") else None) for k, f in r.fields.items()}
    arch = fields["general.architecture"]
    if arch not in ("gemma3", "gemma4"):
        raise SystemExit(f"{arch}: only gemma3 / gemma4 GGUFs are supported")
    g = lambda k, default=None: fields.get(f"{arch}.{k}", default)  # noqa: E731
    tensors = {t.name: t for t in r.tensors}

    layers = int(g("block_count"))
    d = int(g("embedding_length"))
    heads = int(g("attention.head_count"))
    kv_heads = g("attention.head_count_kv")
    kv_heads = int(kv_heads[0] if isinstance(kv_heads, list) else kv_heads)
    vocab = int(tensors["token_embd.weight"].shape[1])
    swa = g("attention.sliding_window_pattern")
    if swa is None:
        k = int(g("attention.sliding_window_pattern", 6))
        swa = [(i + 1) % k != 0 for i in range(layers)]
    swa = [bool(x) for x in swa]
    hd_swa = int(g("attention.key_length_swa", g("attention.key_length")))
    hd_full = int(g("attention.key_length"))
    ff = g("feed_forward_length")
    ff = [int(x) for x in ff] if isinstance(ff, list) else [int(ff)] * layers
    shared = int(g("attention.shared_kv_layers", 0))
    pl_dim = int(g("embedding_length_per_layer_input", 0))
    first_shared = layers - shared

    # Partial RoPE on the global layers is expressed through rope_freqs
    # (1.0 for rotated pairs, 1e30 for the rest).
    rot_full = 1.0
    if "rope_freqs.weight" in tensors:
        freqs = np.asarray(tensors["rope_freqs.weight"].data, dtype=np.float32)
        rot_full = float((freqs < 1e10).sum() * 2 / hd_full)

    config = {
        "model_type": "gemma4_text" if arch == "gemma4" else "gemma3_text",
        "vocab_size": vocab,
        "hidden_size": d,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": hd_swa,
        "global_head_dim": hd_full,
        "intermediate_size": ff,
        "layer_types": ["sliding_attention" if s else "full_attention" for s in swa],
        "sliding_window": int(g("attention.sliding_window")),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon")),
        "rope_parameters": {
            "full_attention": {"rope_theta": float(g("rope.freq_base")), "partial_rotary_factor": rot_full},
            "sliding_attention": {"rope_theta": float(g("rope.freq_base_swa", 10000.0))},
        },
        "num_kv_shared_layers": shared,
        "hidden_size_per_layer_input": pl_dim,
        "final_logit_softcapping": float(g("final_logit_softcapping", 0.0)),
        "bos_token_id": int(fields.get("tokenizer.ggml.bos_token_id", 2)),
        # GGUF norm weights already include Gemma 3's +1.
        "grande_norm_offset": 0.0,
        "grande_source": os.path.basename(a.gguf),
    }
    if arch == "gemma3":
        config["query_pre_attn_scalar"] = float(g("attention.query_pre_attn_scalar", hd_swa))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    tok_dir = a.tokenizer
    if tok_dir is None:
        name = fields.get("general.name", "")
        hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--*{name.split('-it')[0]}*/snapshots/*/tokenizer.json"))
        if not hits:
            raise SystemExit("--tokenizer: no tokenizer.json found in the HF cache; pass the checkpoint directory")
        tok_dir = os.path.dirname(hits[0])
    for f in ("tokenizer.json", "tokenizer_config.json"):
        src = Path(tok_dir) / f
        if src.exists():
            shutil.copy(src, out / f)
        elif f == "tokenizer.json":
            raise SystemExit(f"{src} not found")

    files = []
    w = Writer(out / "embed.bin")
    w.tensor("embed", tensors["token_embd.weight"], [vocab, d])
    w.tensor("final_norm", tensors["output_norm.weight"], [d])
    if pl_dim:
        w.tensor("pl_model_proj", tensors["per_layer_model_proj.weight"], [pl_dim * layers, d])
        w.tensor("pl_proj_norm", tensors["per_layer_proj_norm.weight"], [pl_dim])
    files.append(w.close())

    total = 0
    for l in range(layers):
        t = lambda s: tensors[f"blk.{l}.{s}.weight"]  # noqa: E731
        hd = hd_swa if swa[l] else hd_full
        has_kv = l < first_shared
        w = Writer(out / f"blk.{l}.bin")
        w.tensor(f"blk.{l}.attn_norm", t("attn_norm"), [d])
        parts = [t("attn_q")] + ([t("attn_k"), t("attn_v")] if has_kv else [])
        kind, blocks = concat_quant(parts)
        width = heads * hd + (2 * kv_heads * hd if has_kv else 0)
        if kind in BLOCK_BYTES:
            w.quant(f"blk.{l}.qkv", kind, blocks, [width, d])
        else:
            w.f16(f"blk.{l}.qkv", blocks, [width, d])
        w.tensor(f"blk.{l}.q_norm", t("attn_q_norm"), [hd])
        if has_kv:
            w.tensor(f"blk.{l}.k_norm", t("attn_k_norm"), [hd])
        w.tensor(f"blk.{l}.o", t("attn_output"), [d, heads * hd])
        w.tensor(f"blk.{l}.post_attn_norm", t("post_attention_norm"), [d])
        w.tensor(f"blk.{l}.ffn_norm", t("ffn_norm"), [d])
        w.tensor(f"blk.{l}.gate", t("ffn_gate"), [ff[l], d])
        w.tensor(f"blk.{l}.up", t("ffn_up"), [ff[l], d])
        w.tensor(f"blk.{l}.down", t("ffn_down"), [d, ff[l]])
        w.tensor(f"blk.{l}.post_ffn_norm", t("post_ffw_norm"), [d])
        if pl_dim:
            w.tensor(f"blk.{l}.pl_gate", t("inp_gate"), [pl_dim, d])
            w.tensor(f"blk.{l}.pl_proj", t("proj"), [d, pl_dim])
            w.tensor(f"blk.{l}.pl_norm", t("post_norm"), [d])
            w.tensor(f"blk.{l}.out_scale", t("layer_output_scale"), [1])
        files.append(w.close())
        total += w.off
        print(f"blk.{l}: {w.off / 1e6:.1f} MB ({kind.name}, ff {ff[l]}, hd {hd}, {'own' if has_kv else 'shared'} kv)")

    manifest = {"files": files}
    if pl_dim:
        # The table is written in row chunks: scales first, then codes, so the
        # host can gather a row from two contiguous runs.
        t = tensors["per_layer_token_embd.weight"]
        kind = t.tensor_type
        bb = BLOCK_BYTES[kind]
        width = pl_dim * layers
        rows = int(t.shape[1])
        assert int(t.shape[0]) == width
        blocks_per_row = width // BLOCK
        data = np.asarray(t.data).reshape(rows, blocks_per_row, bb)
        path = out / "per_layer_table.bin"
        with open(path, "wb") as f:
            for start in range(0, rows, 8192):
                f.write(np.ascontiguousarray(data[start:start + 8192, :, :2]).tobytes())
            soff, sn = 0, rows * blocks_per_row * 2
            for start in range(0, rows, 8192):
                f.write(np.ascontiguousarray(data[start:start + 8192, :, 2:]).tobytes())
            dn = rows * blocks_per_row * (bb - 2)
        manifest["per_layer_table"] = {
            "name": "per_layer_table", "dtype": DTYPE_NAME[kind], "shape": [rows, width],
            "path": path.name, "offset": sn, "nbytes": dn, "scales_offset": soff, "scales_nbytes": sn,
        }
        total += sn + dn
        print(f"per_layer_table: {(sn + dn) / 1e6:.0f} MB ({kind.name})")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total += files[0]["tensors"] and os.path.getsize(out / "embed.bin")
    print(f"{out}: {len(files)} tensor files, {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
