"""Prune a Gemma 4 ONNX export (transformers.js layout) to the vocabulary of a
vocab-pruned GGUF, so the browser demo carries the same 25k tokens as native.

    python tools/prune_onnx_vocab.py --onnx-dir models/onnx/e2b \
        --keep-from models/gemma-4-E2B-it-Q4_0-pruned-v2.gguf --out models/onnx/bokuweb/gemma-4-E2B-it-ONNX-ja

`--onnx-dir` is a snapshot of onnx-community/gemma-4-E2B-it-ONNX (config.json,
tokenizer*.json, generation_config.json, onnx/embed_tokens_q4f16.onnx{,_data},
onnx/decoder_model_merged_q4f16.onnx{,_data}). The kept token set is the GGUF's
token strings looked up in tokenizer.json; ids are renumbered densely in the
same order as the GGUF, so the two files agree token for token.

What changes:
  embed_tokens:  the GatherBlockQuantized tables (token + per-layer) lose their
                 unused rows; the vocab-size / image / audio id constants are
                 rewritten.
  decoder:       lm_head MatMulNBits loses rows (N attribute, logits dim).
  tokenizer.json vocab / merges / added_tokens, tokenizer_config.json,
                 config.json and generation_config.json token ids.
The external weight files are streamed (never loaded through protobuf), so
this runs in a few hundred MB of memory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

V_OLD = None  # set from tokenizer.json


def keep_ids_from_gguf(path: str, vocab: dict[str, int]) -> list[int]:
    from gguf import GGUFReader

    r = GGUFReader(path)
    f = next(f for f in r.fields.values() if f.name == "tokenizer.ggml.tokens")
    toks = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
    missing = [t for t in toks if t not in vocab]
    if missing:
        raise SystemExit(f"{len(missing)} GGUF tokens not in tokenizer.json, e.g. {missing[:5]}")
    ids = [vocab[t] for t in toks]
    if ids != sorted(ids):
        raise SystemExit("GGUF token order is not ascending HF ids; refusing to guess the mapping")
    return ids


def remap_id(x, new_id: dict[int, int]):
    """Map an old token id (or list of them) to the new numbering; ids that were
    pruned map to -1 (never matches an input token)."""
    if isinstance(x, list):
        return [remap_id(v, new_id) for v in x]
    if isinstance(x, bool) or not isinstance(x, int):
        return x
    return new_id.get(x, -1)


def prune_tokenizer(src: Path, dst: Path, keep: list[int], new_id: dict[int, int]):
    t = json.load(open(src / "tokenizer.json", encoding="utf-8"))
    vocab = t["model"]["vocab"]
    kept = set(keep)
    t["model"]["vocab"] = {tok: new_id[i] for tok, i in vocab.items() if i in kept}
    merges = t["model"].get("merges") or []
    out = []
    for m in merges:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        if vocab.get(a) in kept and vocab.get(b) in kept and vocab.get(a + b) in kept:
            out.append(m)
    t["model"]["merges"] = out
    dropped = [a for a in t["added_tokens"] if a["id"] not in kept]
    t["added_tokens"] = [dict(a, id=new_id[a["id"]]) for a in t["added_tokens"] if a["id"] in kept]
    print(f"tokenizer.json: vocab {len(vocab)} -> {len(t['model']['vocab'])}, merges {len(merges)} -> {len(out)}, "
          f"added_tokens dropped {[a['content'] for a in dropped]}")
    json.dump(t, open(dst / "tokenizer.json", "w", encoding="utf-8"), ensure_ascii=False)

    tc = json.load(open(src / "tokenizer_config.json", encoding="utf-8"))
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {
            str(new_id[int(k)]): v for k, v in tc["added_tokens_decoder"].items() if int(k) in kept
        }
    json.dump(tc, open(dst / "tokenizer_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prune_config(src: Path, dst: Path, keep: list[int], new_id: dict[int, int]):
    def walk(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                walk(v)
            elif k.endswith("token_id") or k.endswith("token_ids"):
                d[k] = remap_id(v, new_id)
            elif k in ("vocab_size", "vocab_size_per_layer_input"):
                d[k] = len(keep)

    for name in ("config.json", "generation_config.json"):
        if not (src / name).exists():
            continue
        c = json.load(open(src / name, encoding="utf-8"))
        walk(c)
        json.dump(c, open(dst / name, "w", encoding="utf-8"), indent=2)


def ext_info(t):
    e = {x.key: x.value for x in t.external_data}
    return e.get("location"), int(e.get("offset", 0)), int(e.get("length", 0))


def prune_onnx(src: Path, dst: Path, name: str, keep: np.ndarray, patch):
    """Copy `name`.onnx + .onnx_data, slicing every external initializer whose
    first dim is the old vocab size along axis 0. `patch(graph)` fixes constants."""
    m = onnx.load(str(src / "onnx" / f"{name}.onnx"), load_external_data=False)
    g = m.graph
    data_in = np.memmap(src / "onnx" / f"{name}.onnx_data", dtype=np.uint8, mode="r")
    data_name = f"{name}.onnx_data"
    off = 0
    with open(dst / "onnx" / data_name, "wb") as out:
        for t in sorted((t for t in g.initializer if t.data_location == onnx.TensorProto.EXTERNAL), key=lambda t: ext_info(t)[1]):
            loc, o, ln = ext_info(t)
            chunk = data_in[o : o + ln]
            dims = list(t.dims)
            if dims and dims[0] == V_OLD:
                row = ln // V_OLD
                view = np.asarray(chunk).reshape(V_OLD, row)[keep]
                t.dims[0] = len(keep)
                buf = np.ascontiguousarray(view).tobytes()
                print(f"  {t.name}: {dims} -> {list(t.dims)}, {ln / 1e6:.1f} -> {len(buf) / 1e6:.1f} MB")
            else:
                buf = chunk.tobytes()
            out.write(buf)
            del t.external_data[:]
            for k, v in (("location", data_name), ("offset", str(off)), ("length", str(len(buf)))):
                t.external_data.add(key=k, value=v)
            off += len(buf)
    patch(g)
    onnx.save(m, str(dst / "onnx" / f"{name}.onnx"))
    print(f"  {data_name}: {off / 1e6:.1f} MB")


def main():
    global V_OLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--keep-from", required=True, help="vocab-pruned GGUF whose token list defines the kept set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="q4f16")
    a = ap.parse_args()
    src, dst = Path(a.onnx_dir), Path(a.out)
    (dst / "onnx").mkdir(parents=True, exist_ok=True)

    tok = json.load(open(src / "tokenizer.json", encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    V_OLD = len(vocab)
    keep = keep_ids_from_gguf(a.keep_from, vocab)
    new_id = {old: new for new, old in enumerate(keep)}
    V = len(keep)
    print(f"vocab {V_OLD} -> {V}")

    prune_tokenizer(src, dst, keep, new_id)
    prune_config(src, dst, keep, new_id)
    for name in ("chat_template.jinja", "preprocessor_config.json", "processor_config.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)

    cfg = json.load(open(src / "config.json"))
    image_id, audio_id = cfg.get("image_token_id"), cfg.get("audio_token_id")

    def patch_embed(g):
        # Scalar INT64 constants: vocab size (the "is a text token" bound) and the
        # image / audio placeholder ids the per-layer gather masks out.
        for t in g.initializer:
            if t.data_type == onnx.TensorProto.INT64 and not t.dims:
                v = int(numpy_helper.to_array(t))
                new = {V_OLD: V, image_id: new_id.get(image_id, -1), audio_id: new_id.get(audio_id, -1)}.get(v)
                if new is not None:
                    t.CopyFrom(numpy_helper.from_array(np.array(new, dtype=np.int64), t.name))
                    print(f"  const {t.name}: {v} -> {new}")

    def patch_decoder(g):
        for n in g.node:
            if n.op_type == "MatMulNBits":
                for at in n.attribute:
                    if at.name == "N" and at.i == V_OLD:
                        at.i = V
                        print(f"  {n.name}: N {V_OLD} -> {V}")

    def patch_shapes(g):
        # Declared shapes (outputs and intermediate value_info) still say 262144;
        # ORT's shape inference refuses a graph where they disagree with the weights.
        n = 0
        for vi in list(g.output) + list(g.value_info) + list(g.input):
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_value == V_OLD:
                    d.dim_value = V
                    n += 1
        print(f"  declared dims {V_OLD} -> {V}: {n}")

    keep_np = np.array(keep, dtype=np.int64)
    print("embed_tokens")
    prune_onnx(src, dst, f"embed_tokens_{a.dtype}", keep_np, lambda g: (patch_embed(g), patch_shapes(g)))
    print("decoder")
    prune_onnx(src, dst, f"decoder_model_merged_{a.dtype}", keep_np, lambda g: (patch_decoder(g), patch_shapes(g)))
    total = sum(os.path.getsize(p) for p in (dst / "onnx").iterdir()) + os.path.getsize(dst / "tokenizer.json")
    print(f"total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
