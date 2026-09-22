"""Package a Laya checkpoint (ModernBERT / mmBERT encoder + decision head,
convaiinnovations/laya*) for omg's wgpu engine (crates/omg-wgpu/src/laya.rs),
natively and in the browser.

Weights go to the engine's manifest layout (model::Manifest): f16, or Q8_0
codes with the f16 block scales in a separate run (`--dtype q8`, half the
bytes, the same answers to ~1e-3). `mlp.Wi` is split into its value / gate
halves, the scorer and act head stay f16 (they run on the host). With
`--corpus`, the 256k Gemma vocabulary is pruned to the tokens the corpus
uses plus their BPE merge closure (see tools/prune_vocab.py: the corpus then
tokenizes byte-for-byte the same; other text may split into more pieces),
every added / byte / single-character token is kept, and the tokenizer.json
is rewritten with the new ids. The embedding is 61% of the checkpoint, so
this is what makes it a browser download (322M f16: 644 MB → ~160 MB q8).

Output directory:
    config.json            encoder config + `laya_agent` (rl_agent_config.json)
                           + `laya_tokens` (special token strings) + `laya_name`
    tokenizer.json         pruned (or copied) HF tokenizer
    tokenizer_config.json
    manifest.json          tensor -> file / byte ranges
    embed.bin  enc.bin  head.bin

    python tools/export_laya.py ~/.cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/<rev> \\
        --corpus .cache/jglue/*-train.jsonl .cache/kev/*.jsonl examples/*.json \\
        --dtype q8 --out web/models/laya-multilingual-wgpu
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

BLOCK = 32


class Bpe:
    """Rank-order BPE over a piece's characters, recording the merge path
    (as tools/prune_vocab.py: the merges inside a final piece fire in rank
    order whatever surrounds it, so the path is the closure to keep)."""

    def __init__(self, tokens, merges):
        self.rank = {m: i for i, m in enumerate(merges)}

    def bpe(self, word, trace=None):
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get(parts[i] + " " + parts[i + 1])
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi : bi + 2] = [parts[bi] + parts[bi + 1]]
            if trace is not None:
                trace.add(parts[bi])
        return parts


def walk(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def corpus_texts(paths):
    """Every string in the JSON / JSONL / text files (glob patterns)."""
    import glob

    for pat in paths:
        for p in glob.glob(pat):
            with open(p, encoding="utf-8") as fh:
                data = fh.read()
            try:
                yield from walk(json.loads(data))
                continue
            except json.JSONDecodeError:
                pass
            for line in data.splitlines():
                if line.strip():
                    try:
                        yield from walk(json.loads(line))
                    except json.JSONDecodeError:
                        yield line

# Laya's own prompt scaffolding, so its pieces survive any corpus.
SCAFFOLD = ["choice question: ", "score question: ", "noul question: ", " false: no, the statement does not hold",
            " true: yes, the statement holds", " level 0: ", " level 1: ", " level 2: ", " level 3: ", " level 4: ",
            '{"a": "b", "c": [1, 2.5, null, true, false]}', "What does the customer want in `message`?",
            "Does the response fully satisfy the request, using the supplied reference when present?"]


class Writer:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "wb")
        self.off = 0
        self.entries: list[dict] = []

    def _write(self, b: bytes) -> tuple[int, int]:
        off = self.off
        self.f.write(b)
        self.off += len(b)
        while self.off % 4:
            self.f.write(b"\0")
            self.off += 1
        return off, len(b)

    def f16(self, name: str, arr: np.ndarray):
        off, n = self._write(np.ascontiguousarray(arr.astype(np.float16)).tobytes())
        self.entries.append({"name": name, "dtype": "f16", "shape": list(arr.shape), "offset": off, "nbytes": n})

    def q8(self, name: str, arr: np.ndarray):
        """Q8_0 as QTensor::quantize_q8: per 32-element block, scale = max|v| / 127."""
        v = np.ascontiguousarray(arr.astype(np.float32)).reshape(-1, BLOCK)
        amax = np.abs(v).max(axis=1)
        d = (amax / 127.0).astype(np.float16)
        inv = np.where(d != 0, 1.0 / d.astype(np.float32), 0.0)
        q = np.clip(np.rint(v * inv[:, None]), -127, 127).astype(np.int8)
        soff, sn = self._write(d.tobytes())
        doff, dn = self._write(q.tobytes())
        self.entries.append({"name": name, "dtype": "q8", "shape": list(arr.shape), "offset": doff, "nbytes": dn,
                             "scales_offset": soff, "scales_nbytes": sn})

    def tensor(self, name: str, arr: np.ndarray, dtype: str):
        if dtype == "q8" and arr.ndim == 2:
            self.q8(name, arr)
        else:
            self.f16(name, arr)

    def close(self) -> dict:
        self.f.close()
        return {"path": self.path.name, "tensors": self.entries}


def prune(tok_json: dict, hf, texts: list[str]) -> tuple[dict, np.ndarray]:
    """Kept old ids (sorted) and the rewritten tokenizer.json."""
    model = tok_json["model"]
    vocab: dict[str, int] = model["vocab"]
    V = len(vocab)
    tokens = [""] * V
    for t, i in vocab.items():
        tokens[i] = t
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    keep = set(a["id"] for a in tok_json.get("added_tokens", []))
    for i, t in enumerate(tokens):
        # Byte fallback, single characters (with or without the word marker):
        # whatever falls outside the corpus still has a way in.
        if (t.startswith("<0x") and t.endswith(">")) or len(t) == 1 or (len(t) == 2 and t[0] == "▁"):
            keep.add(i)
    base = len(keep)
    used = set()
    for enc in hf.encode_batch(texts, add_special_tokens=False):
        used.update(enc.ids)
    keep |= used
    print(f"tokens: {base} structural, {len(used)} from the corpus")
    bpe = Bpe(tokens, merges)
    path: set[str] = set()
    for i in list(keep):
        t = tokens[i]
        if t.startswith("<") and t.endswith(">"):
            continue
        out = bpe.bpe(t, path)
        if out != [t]:
            path.update(out)
    tok2id = vocab
    for piece in path:
        j = tok2id.get(piece)
        if j is not None:
            keep.add(j)
    keep = sorted(keep)
    kept = set(keep)
    new_id = {old: new for new, old in enumerate(keep)}
    print(f"keeping {len(keep)} of {V} tokens ({100 * len(keep) / V:.1f}%)")
    new_vocab = {tokens[old]: new for old, new in new_id.items()}
    new_merges = []
    for m in model["merges"]:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        if tok2id.get(a) in kept and tok2id.get(b) in kept and tok2id.get(a + b) in kept:
            new_merges.append(m)
    print(f"merges {len(model['merges'])} -> {len(new_merges)}")
    out = json.loads(json.dumps(tok_json))
    out["model"]["vocab"] = new_vocab
    out["model"]["merges"] = new_merges
    out["added_tokens"] = [dict(a, id=new_id[a["id"]]) for a in tok_json.get("added_tokens", []) if a["id"] in kept]
    return out, np.array(keep, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="checkpoint directory (encoder/, tokenizer/, model.safetensors, rl_agent_config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f16", "q8"], default="q8", help="storage of the 2-D encoder / head weights")
    ap.add_argument("--corpus", nargs="*", default=[], help="files whose tokens the pruned vocabulary must cover; omit to keep all 256k")
    ap.add_argument("--extra", nargs="*", default=[], help="extra strings to keep tokens for")
    ap.add_argument("--name", help="model name written to config.json (default: the output directory's)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    enc_cfg = json.load(open(src / "encoder" / "config.json"))
    agent = json.load(open(src / "rl_agent_config.json"))
    tok_cfg = json.load(open(src / "tokenizer" / "tokenizer_config.json"))
    tok_json = json.load(open(src / "tokenizer" / "tokenizer.json"))
    W = load_file(str(src / "model.safetensors"))
    d, L, ff = enc_cfg["hidden_size"], enc_cfg["num_hidden_layers"], enc_cfg["intermediate_size"]
    head_layers = agent.get("head_layers", 2)

    embed = W["encoder.embeddings.tok_embeddings.weight"]
    if a.corpus:
        from tokenizers import Tokenizer as Hf
        hf = Hf.from_file(str(src / "tokenizer" / "tokenizer.json"))
        texts = SCAFFOLD + list(corpus_texts(a.corpus)) + a.extra
        tok_json, keep = prune(tok_json, hf, texts)
        embed = embed[keep]
        enc_cfg = dict(enc_cfg, vocab_size=int(len(keep)))
    json.dump(tok_json, open(out / "tokenizer.json", "w"), ensure_ascii=False)
    shutil.copy(src / "tokenizer" / "tokenizer_config.json", out / "tokenizer_config.json")

    special = {k: (tok_cfg[k]["content"] if isinstance(tok_cfg[k], dict) else tok_cfg[k])
               for k in ("cls_token", "sep_token", "pad_token", "mask_token")}
    cfg = dict(enc_cfg, laya_agent=agent, laya_tokens=special, laya_name=a.name or out.name, omg_laya=1)
    json.dump(cfg, open(out / "config.json", "w"), indent=2, ensure_ascii=False)

    files = []
    w = Writer(out / "embed.bin")
    w.tensor("embed", embed, a.dtype)
    w.f16("embed_norm", W["encoder.embeddings.norm.weight"])
    w.f16("final_norm", W["encoder.final_norm.weight"])
    w.f16("type_emb", W["type_emb.weight"])
    files.append(w.close())
    w = Writer(out / "enc.bin")
    for l in range(L):
        p = f"encoder.layers.{l}."
        n = f"enc.{l}."
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
    for h in range(head_layers):
        p = f"head.layers.{h}."
        n = f"head.{h}."
        g = lambda *names: next(W[p + x] for x in names if p + x in W)  # noqa: E731
        w.f16(n + "norm1_w", W[p + "norm1.weight"])
        w.f16(n + "norm1_b", W[p + "norm1.bias"])
        w.tensor(n + "in_proj", g("self_attn.in_proj.weight", "self_attn.in_proj_weight"), a.dtype)
        w.f16(n + "in_proj_b", g("self_attn.in_proj.bias", "self_attn.in_proj_bias"))
        w.tensor(n + "out_proj", W[p + "self_attn.out_proj.weight"], a.dtype)
        w.f16(n + "out_proj_b", W[p + "self_attn.out_proj.bias"])
        w.f16(n + "norm2_w", W[p + "norm2.weight"])
        w.f16(n + "norm2_b", W[p + "norm2.bias"])
        w.tensor(n + "l1", W[p + "linear1.weight"], a.dtype)
        w.f16(n + "l1_b", W[p + "linear1.bias"])
        w.tensor(n + "l2", W[p + "linear2.weight"], a.dtype)
        w.f16(n + "l2_b", W[p + "linear2.bias"])
    seq = lambda pfx, i, s: next(W[k] for k in (f"{pfx}.layers.{i}.{s}", f"{pfx}.{i}.{s}") if k in W)  # noqa: E731
    w.f16("scorer.norm_w", seq("scorer", 0, "weight"))
    w.f16("scorer.norm_b", seq("scorer", 0, "bias"))
    w.f16("scorer.l1", seq("scorer", 1, "weight"))
    w.f16("scorer.l1_b", seq("scorer", 1, "bias"))
    w.f16("scorer.l2", seq("scorer", 3, "weight"))
    w.f16("scorer.l2_b", seq("scorer", 3, "bias"))
    w.f16("act.l1", seq("act_head", 0, "weight"))
    w.f16("act.l1_b", seq("act_head", 0, "bias"))
    w.f16("act.l2", seq("act_head", 2, "weight"))
    w.f16("act.l2_b", seq("act_head", 2, "bias"))
    files.append(w.close())
    json.dump({"files": files}, open(out / "manifest.json", "w"), indent=1)
    total = sum((out / f["path"]).stat().st_size for f in files)
    print(f"wrote {out}: {total / 1e6:.1f} MB of weights, vocab {enc_cfg['vocab_size']}")


if __name__ == "__main__":
    main()
