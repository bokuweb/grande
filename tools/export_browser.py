"""Export a merged omg checkpoint for the browser: ONNX fp16 of the text
backbone (last_hidden_state, no lm_head), embedding rows pruned to the
tokens a corpus uses, and an id map so the ORIGINAL tokenizer keeps running
unchanged in JS (ids outside the kept set fall back to <unk>).

    python tools/export_browser.py --run runs/grande-270m-12k --base unsloth/gemma-3-270m \
        --corpus ".cache/jglue/*-train.jsonl" "examples/*.json" ".cache/corpus/*.jsonl" \
        --out runs/grande-270m-12k/browser/grande-270m-ja
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from prune_vocab import corpus_texts  # noqa: E402

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True, help="tokenizer source (the base checkpoint)")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=["user", "model", "State:", "Question:"])
    ap.add_argument("--emb", choices=["fp16", "int8"], default="fp16", help="embedding table in the q8 export: fp16 (211 MB, exact) or int8 per row (162 MB, ~-3 pts JCQA)")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    full = AutoModelForCausalLM.from_pretrained(run / "merged", dtype=torch.float32)
    backbone = full.model
    for name in ("language_model",):
        if hasattr(backbone, name):
            backbone = getattr(backbone, name)
    V, d = backbone.embed_tokens.weight.shape
    print(f"vocab {V}, hidden {d}, layers {len(backbone.layers)}")

    # --- kept rows: specials / bytes / used tokens / labels -------------------------
    keep = {i for i in tok.all_special_ids if i < V}  # mirrors register an extra id past the table
    for t, i in tok.get_vocab().items():
        if i < V and t.startswith("<") and t.endswith(">") and (i < 300 or t.startswith("<0x")):
            keep.add(i)
    for text in list(corpus_texts(a.corpus)) + a.extra:
        keep.update(i for i in tok.encode(text, add_special_tokens=False) if i < V)
    for ch in LABELS:
        ids = tok.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            keep.add(ids[0])
    keep = sorted(keep)
    unk = tok.unk_token_id if tok.unk_token_id is not None else 0
    if unk not in keep:
        keep = sorted(set(keep) | {unk})
    new_id = {old: new for new, old in enumerate(keep)}
    id_map = np.full(V, new_id[unk], dtype=np.int32)
    for old, new in new_id.items():
        id_map[old] = new
    id_map.tofile(out / "id_map.bin")
    print(f"keeping {len(keep)} of {V} rows ({100 * len(keep) / V:.1f}%); id_map.bin {id_map.nbytes / 1e6:.1f} MB")

    # --- slice the embedding ----------------------------------------------------------
    with torch.no_grad():
        w = backbone.embed_tokens.weight[torch.tensor(keep)].clone()
        backbone.embed_tokens = torch.nn.Embedding(len(keep), d, _weight=w)
        backbone.config.vocab_size = len(keep)
        if hasattr(backbone.embed_tokens, "embed_scale"):
            pass
    # Gemma scales embeddings by sqrt(d) inside Gemma3TextScaledWordEmbedding; nn.Embedding
    # would drop that, so wrap the forward to apply it.
    class Wrapped(torch.nn.Module):
        def __init__(self, m, scale, window):
            super().__init__()
            self.m, self.scale, self.window = m, scale, window

        def forward(self, input_ids, attention_mask):
            emb = self.m.embed_tokens(input_ids) * self.scale
            # Build the masks here: transformers' vmap-based mask creation does
            # not trace. Gemma 3 accepts {"full_attention", "sliding_attention"}.
            B, L = input_ids.shape
            i = torch.arange(L, device=input_ids.device)
            causal = i[None, :] <= i[:, None]
            pad = attention_mask.bool()[:, None, None, :]
            full = causal[None, None] & pad
            window = self.window
            sliding = full & ((i[:, None] - i[None, :]) < window)[None, None]
            # Finite so the fp16 export does not overflow to -inf (WebGPU then returns zeros).
            neg = -1.0e4
            to_float = lambda m: torch.zeros(B, 1, L, L, dtype=emb.dtype, device=emb.device).masked_fill(~m, neg)
            masks = {"full_attention": to_float(full), "sliding_attention": to_float(sliding)}
            return self.m(inputs_embeds=emb, attention_mask=masks).last_hidden_state

    raw_scale = getattr(getattr(full.model, "embed_tokens", None), "embed_scale", None)
    scale = float(np.asarray(raw_scale.detach().cpu() if hasattr(raw_scale, "detach") else raw_scale).reshape(-1)[0]) if raw_scale is not None else float(d) ** 0.5
    window = int(getattr(backbone.config, "sliding_window", 512) or 512)
    wrapped = Wrapped(backbone, torch.tensor(scale, dtype=torch.float32), window).eval()

    # --- sanity: same hidden states as the unsliced model on a kept-token row ---------
    probe = tok("合言葉は「青い象」であるか。State: 契約", add_special_tokens=False, return_tensors="pt").input_ids
    ref_model = AutoModelForCausalLM.from_pretrained(run / "merged", dtype=torch.float32).model
    with torch.no_grad():
        ref = ref_model(input_ids=probe, attention_mask=torch.ones_like(probe)).last_hidden_state
        got = wrapped(torch.tensor(id_map[probe.numpy()]), torch.ones_like(probe))
    print("sliced vs original hidden max diff:", float((ref - got).abs().max()))

    # --- ONNX ---------------------------------------------------------------------------
    onnx_path = out / "onnx" / "model.onnx"
    torch.onnx.export(
        wrapped,
        (probe.new_tensor(id_map[probe.numpy()]), torch.ones_like(probe)),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "last_hidden_state": {0: "batch", 1: "seq"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o = s.run(None, {"input_ids": id_map[probe.numpy()].astype(np.int64), "attention_mask": np.ones(probe.shape, dtype=np.int64)})[0]
    print("onnx fp32 vs torch max diff:", float(np.abs(o - got.numpy()).max()))

    # --- q8: 8-bit MatMulNBits, embedding int8 per row (Gather -> Cast -> Mul scale) --
    from onnx import helper, numpy_helper
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    mq = onnx.load(str(onnx_path))
    inits = {t.name: t for t in mq.graph.initializer}
    for idx, node in enumerate(mq.graph.node):
        if node.op_type == "Gather" and node.input[0] in inits and inits[node.input[0]].dims[0] > 50000:
            t = inits[node.input[0]]
            w = numpy_helper.to_array(t).astype(np.float32)
            if a.emb == "fp16":
                t.CopyFrom(numpy_helper.from_array(w.astype(np.float16), t.name))
                out_name = node.output[0]
                node.output[0] = out_name + "_fp16"
                mq.graph.node.insert(idx + 1, helper.make_node("Cast", [out_name + "_fp16"], [out_name], to=onnx.TensorProto.FLOAT, name=out_name + "_cast"))
                print(f"embedding -> fp16: {t.name} {w.shape}")
                break
            row_scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
            row_scale[row_scale == 0] = 1.0
            q8 = np.clip(np.round(w / row_scale), -127, 127).astype(np.int8)
            t.CopyFrom(numpy_helper.from_array(q8, t.name))
            mq.graph.initializer.append(numpy_helper.from_array(row_scale.astype(np.float32).reshape(-1), t.name + "_scale"))
            out_name = node.output[0]
            node.output[0] = out_name + "_q8"
            new_nodes = [
                helper.make_node("Cast", [out_name + "_q8"], [out_name + "_f"], to=onnx.TensorProto.FLOAT, name=out_name + "_cast"),
                helper.make_node("Gather", [t.name + "_scale", node.input[1]], [out_name + "_s"], name=out_name + "_sgather"),
                helper.make_node("Unsqueeze", [out_name + "_s", out_name + "_axes"], [out_name + "_s1"], name=out_name + "_unsq"),
                helper.make_node("Mul", [out_name + "_f", out_name + "_s1"], [out_name], name=out_name + "_scale_mul"),
            ]
            mq.graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), out_name + "_axes"))
            for j, nn in enumerate(new_nodes):
                mq.graph.node.insert(idx + 1 + j, nn)
            print(f"embedding -> int8 per row: {t.name} {w.shape}")
            break
    quant = MatMulNBitsQuantizer(mq, algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8))
    quant.process()
    for f in (out / "onnx").glob("model_quantized.onnx*"):
        f.unlink()
    onnx.save_model(quant.model.model, str(out / "onnx" / "model_quantized.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="model_quantized.onnx_data", size_threshold=1024)
    sq = ort.InferenceSession(str(out / "onnx" / "model_quantized.onnx"), providers=["CPUExecutionProvider"])
    oq = sq.run(None, {"input_ids": id_map[probe.numpy()].astype(np.int64), "attention_mask": np.ones(probe.shape, dtype=np.int64)})[0]
    print("onnx q8 (int8 emb) vs torch: mean diff", float(np.abs(oq - got.numpy()).mean()), "max", float(np.abs(oq - got.numpy()).max()))

    from onnxruntime.transformers.float16 import convert_float_to_float16

    m16 = convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=True, disable_shape_infer=False, force_fp16_initializers=True)
    for f in (out / "onnx").glob("model_fp16.onnx*"):
        f.unlink()
    onnx.save_model(m16, str(out / "onnx" / "model_fp16.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="model_fp16.onnx_data", size_threshold=1024)
    # Keep the fp32 graph too (as external data) for backends where fp16 misbehaves.
    m32 = onnx.load(str(onnx_path))
    onnx_path.unlink()
    for f in (out / "onnx").glob("model.onnx*"):
        f.unlink()
    onnx.save_model(m32, str(out / "onnx" / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx_data", size_threshold=1024)
    s16 = ort.InferenceSession(str(out / "onnx" / "model_fp16.onnx"), providers=["CPUExecutionProvider"])
    o16 = s16.run(None, {"input_ids": id_map[probe.numpy()].astype(np.int64), "attention_mask": np.ones(probe.shape, dtype=np.int64)})[0]
    print("onnx fp16 vs torch max diff:", float(np.abs(o16 - got.numpy()).max()))

    # --- files --------------------------------------------------------------------------
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = Path(tok.name_or_path) if Path(tok.name_or_path).exists() else None
    tok.save_pretrained(out)
    cfg = json.load(open(run / "merged" / "config.json"))
    cfg["transformers.js_config"] = {"use_external_data_format": {"model_fp16.onnx": 1, "model.onnx": 1, "model_quantized.onnx": 1}}
    cfg["omg"] = {"readout": "pointer", "hidden_size": int(d), "layers": len(backbone.layers), "kept_vocab": len(keep), "original_vocab": int(V), "embed_scale": scale}
    def plain(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    json.dump(cfg, open(out / "config.json", "w"), indent=2, default=plain)
    shutil.copy(run / "head.safetensors", out / "head.safetensors")
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in list(out.iterdir()) + list((out / "onnx").iterdir()) if p.is_file()}
    print(sizes)


if __name__ == "__main__":
    main()
