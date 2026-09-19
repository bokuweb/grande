"""Prune a Gemma GGUF's vocabulary to the tokens a corpus actually uses.

Gemma's 262k vocabulary is 62% of the E2B Q4_0 file: the per-layer
embeddings (35 × 256 per token) and the input embedding table both scale
with it. A Japanese/English decision model never sees most of it.

    python tools/prune_vocab.py --model models/gemma-4-E2B-it-Q4_0.gguf \
        --corpus .cache/jglue/*-train.jsonl .cache/kev/*.jsonl examples/*.json \
        --out models/gemma-4-E2B-it-Q4_0-ja56k.gguf

Kept: every non-normal token (control, byte, user-defined), every token that
appears when the corpus is tokenized, and the BPE merge closure of those
(every intermediate piece on the merge path), so the corpus tokenizes
byte-for-byte the same. Text outside the corpus may tokenize into more,
shorter pieces; it never fails (byte fallback tokens are kept).

Token ids are renumbered densely, embeddings rows are sliced without
dequantizing, merges are filtered, special ids and vocab_size are rewritten.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crates"))

QUANT_BLOCK = {  # (elements per block, bytes per block)
    GGMLQuantizationType.F32: (1, 4),
    GGMLQuantizationType.F16: (1, 2),
    GGMLQuantizationType.BF16: (1, 2),
    GGMLQuantizationType.Q8_0: (32, 34),
    GGMLQuantizationType.Q4_0: (32, 18),
    GGMLQuantizationType.Q4_1: (32, 20),
    GGMLQuantizationType.Q5_0: (32, 22),
    GGMLQuantizationType.Q5_1: (32, 24),
    GGMLQuantizationType.Q4_K: (256, 144),
    GGMLQuantizationType.Q5_K: (256, 176),
    GGMLQuantizationType.Q6_K: (256, 210),
    GGMLQuantizationType.Q8_K: (256, 292),
    GGMLQuantizationType.Q2_K: (256, 84),
    GGMLQuantizationType.Q3_K: (256, 110),
}


def field_str_array(f):
    return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]


def field_scalar(f):
    return f.parts[f.data[0]][0]


# The label layout's own scaffolding (grande-core render.rs), so its pieces
# survive whatever the corpus happens to contain.
SCAFFOLD = ["State:\n", "\n\nQuestion: ", "\nA: ", "\nB: ", " — ", "\nAnswer with one letter.", "\nmodel\n", "user\n",
            "\nA: true\nB: false\n", "\nA: 0\nB: 1\nC: 2\nD: 3\nE: 4\n"]


def rendered_texts(doc):
    """What the runtime actually tokenizes for a request: an object state is
    rendered as `key: value` lines with nested values as compact JSON, so the
    JSON punctuation glued to the text (`":"振込`, `","`) must be seen too."""
    if not isinstance(doc, dict) or "state" not in doc:
        return
    st = doc["state"]
    if isinstance(st, dict):
        yield "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(',', ':'))}" for k, v in st.items())
    elif not isinstance(st, str):
        yield json.dumps(st, ensure_ascii=False, separators=(",", ":"))
    qs = doc.get("questions")
    for q in qs.values() if isinstance(qs, dict) else (qs or []):
        if not isinstance(q, dict):
            continue
        instr = q.get("instructions") or ""
        opts = q.get("criteria") or q.get("legend") or {}
        items = list(opts.items()) if isinstance(opts, dict) else [(str(o), None) for o in opts] if isinstance(opts, list) else []
        lines = [f"Question: {instr}"]
        for i, (k, d) in enumerate(items[:52]):
            lines.append(f"{chr(65 + i)}: {k} — {d}" if isinstance(d, str) and d else f"{chr(65 + i)}: {k}")
        yield "\n".join(lines) + "\nAnswer with one letter."


def corpus_texts(paths):
    yield from SCAFFOLD
    for pat in paths:
        for p in glob.glob(pat):
            with open(p, encoding="utf-8") as fh:
                data = fh.read()
            try:
                doc = json.loads(data)
                yield from walk(doc)
                yield from rendered_texts(doc)
                continue
            except json.JSONDecodeError:
                pass
            for line in data.splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                        yield from walk(row)
                        yield from rendered_texts(row)
                    except json.JSONDecodeError:
                        yield line


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


class Tokenizer:
    """Minimal byte-level BPE matching llama.cpp's gemma4 pretokenizer closely
    enough to collect used tokens: we only need which pieces occur, and we
    take the merge closure afterwards, so small pretokenization differences
    only make the kept set a little larger."""

    def __init__(self, tokens, merges):
        self.rank = {m: i for i, m in enumerate(merges)}
        self.tok2id = {t: i for i, t in enumerate(tokens)}

    def bpe(self, word, trace=None):
        """Rank-order BPE from characters; `trace` collects every piece that
        exists at some point on the way (the merge path)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="extra strings to keep tokens for")
    ap.add_argument("--hf-tokenizer", default="unsloth/gemma-4-E2B", help="tokenizer.json path or Hub id matching the GGUF's vocabulary")
    a = ap.parse_args()

    r = GGUFReader(a.model)
    F = {f.name: f for f in r.fields.values()}
    tokens = field_str_array(F["tokenizer.ggml.tokens"])
    scores = (
        np.array([F["tokenizer.ggml.scores"].parts[i][0] for i in F["tokenizer.ggml.scores"].data], dtype=np.float32)
        if "tokenizer.ggml.scores" in F
        else None
    )
    types = np.array([F["tokenizer.ggml.token_type"].parts[i][0] for i in F["tokenizer.ggml.token_type"].data], dtype=np.int32)
    # BPE (gemma4) carries merges; SentencePiece unigram (gemma3, "llama") does not.
    merges = field_str_array(F["tokenizer.ggml.merges"]) if "tokenizer.ggml.merges" in F else []
    V = len(tokens)
    print(f"vocab {V}, merges {len(merges) or 'none (unigram)'}")

    # --- which tokens does the corpus use ---------------------------------
    # Use the HF tokenizer when available (exact), else the fallback BPE.
    keep = set(i for i in range(V) if types[i] != 1)  # everything not NORMAL: control, byte, user-defined, unused
    used = set()
    try:
        # `tokenizers` directly (a tokenizer.json path or a Hub id): transformers'
        # AutoTokenizer chokes on the Gemma 4 tokenizer_config in some versions.
        from tokenizers import Tokenizer as HfTokenizer

        hf = HfTokenizer.from_file(a.hf_tokenizer) if a.hf_tokenizer.endswith(".json") else HfTokenizer.from_pretrained(a.hf_tokenizer)
        assert hf.get_vocab_size() >= V
        texts = list(corpus_texts(a.corpus)) + a.extra
        for enc in hf.encode_batch(texts, add_special_tokens=False):
            used.update(i for i in enc.ids if i < V)
        print(f"HF tokenizer: {len(texts)} texts, {len(used)} distinct tokens")
    except Exception as e:  # noqa: BLE001
        print(f"HF tokenizer unavailable ({e}); using fallback BPE")
        bpe = Tokenizer(tokens, merges)
        for text in list(corpus_texts(a.corpus)) + a.extra:
            for w in text.replace(" ", " ▁").split():  # crude
                for piece in bpe.bpe(w):
                    i = bpe.tok2id.get(piece)
                    if i is not None:
                        used.add(i)
        print(f"fallback: {len(used)} distinct tokens")
    keep |= used
    # Always keep the option labels and ASCII.
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
        i = tokens.index(ch) if ch in tokens else None
        if i is not None:
            keep.add(i)

    # --- merge closure ---------------------------------------------------------
    # BPE reaches a token through one specific path of lower-ranked merges (the
    # merges inside a final piece fire in rank order whatever surrounds it), so
    # replay that path for every kept token and keep each intermediate piece.
    # Keeping only *some* producing pair is not enough: the runtime would take
    # the real path, hit a dropped intermediate, and stop short ("An"+"sw"+"er").
    tok2id = {t: i for i, t in enumerate(tokens)}
    bpe = Tokenizer(tokens, merges)
    path = set()
    for i in list(keep):
        if types[i] == 1:
            path.update(tokens[i])  # the single-character start symbols
            out = bpe.bpe(tokens[i], path)
            if out != [tokens[i]]:
                path.update(out)  # no path (the token is not merge-reachable); keep what it splits into
    for piece in path:
        j = tok2id.get(piece)
        if j is not None:
            keep.add(j)
    keep = sorted(keep)
    new_id = {old: new for new, old in enumerate(keep)}
    kept_set = set(keep)
    print(f"keeping {len(keep)} tokens ({100 * len(keep) / V:.1f}%)")

    new_merges = []
    for m in merges:
        a_, b_ = m.split(" ", 1)
        if tok2id.get(a_) in kept_set and tok2id.get(b_) in kept_set and tok2id.get(a_ + b_) in kept_set:
            new_merges.append(m)
    print(f"merges {len(merges)} -> {len(new_merges)}")

    # --- write ---------------------------------------------------------------------
    arch = bytes(F["general.architecture"].parts[F["general.architecture"].data[0]]).decode()
    w = GGUFWriter(a.out, arch)
    skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    vocab_arrays = {"tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"}
    id_fields = {k for k in F if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
    for name, f in F.items():
        if name in skip or name in vocab_arrays:
            continue
        if name in id_fields:
            old = int(field_scalar(f))
            w.add_uint32(name, new_id[old])
            continue
        if name.endswith(".vocab_size"):
            w.add_uint32(name, len(keep))
            continue
        vt = f.types[0]
        if vt == GGUFValueType.ARRAY:
            inner = f.types[1]
            if inner == GGUFValueType.STRING:
                w.add_array(name, field_str_array(f))
            else:
                w.add_array(name, [f.parts[i][0].item() for i in f.data])
        elif vt == GGUFValueType.STRING:
            w.add_string(name, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"))
        else:
            val = f.parts[f.data[0]][0].item()
            {GGUFValueType.UINT8: w.add_uint8, GGUFValueType.INT8: w.add_int8, GGUFValueType.UINT16: w.add_uint16,
             GGUFValueType.INT16: w.add_int16, GGUFValueType.UINT32: w.add_uint32, GGUFValueType.INT32: w.add_int32,
             GGUFValueType.FLOAT32: w.add_float32, GGUFValueType.BOOL: w.add_bool, GGUFValueType.UINT64: w.add_uint64,
             GGUFValueType.INT64: w.add_int64, GGUFValueType.FLOAT64: w.add_float64}[vt](name, val)
    w.add_array("tokenizer.ggml.tokens", [tokens[i] for i in keep])
    if scores is not None:
        w.add_array("tokenizer.ggml.scores", [float(scores[i]) for i in keep])
    w.add_array("tokenizer.ggml.token_type", [int(types[i]) for i in keep])
    if merges:
        w.add_array("tokenizer.ggml.merges", new_merges)

    keep_np = np.array(keep)
    for t in r.tensors:
        shape = [int(x) for x in t.shape]  # ne0, ne1, ...
        raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
        blk, bpb = QUANT_BLOCK[t.tensor_type]
        row_bytes = shape[0] // blk * bpb
        rows = raw.size // row_bytes
        mat = raw.reshape(rows, row_bytes)
        if len(shape) == 2 and shape[1] == V:
            mat = np.ascontiguousarray(mat[keep_np])
            print(f"  {t.name}: {V} -> {len(keep)} rows, {raw.nbytes / 1e6:.0f} -> {mat.nbytes / 1e6:.0f} MB")
        elif len(shape) > 2:
            mat = mat.reshape(*reversed(shape[1:]), row_bytes)
        # uint8 [.., row_bytes] + raw_dtype: gguf-py derives the element shape.
        w.add_tensor(t.name, mat, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {a.out}: {Path(a.out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
