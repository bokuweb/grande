"""Packed branch layout, byte-for-byte the same as `grande-core::render`.

    <bos><unused0>{state}
    <unused1>{instructions}<unused2>{key — desc}<unused3>…<unused4>   (one per question)

Training data and live requests must go through the same layout; the Rust
side is the reference. `grande render --request r.json` dumps its token ids
so `check_parity()` below can compare.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

DELIMS = {"state": "<unused0>", "question": "<unused1>", "opt": "<unused2>", "opt_end": "<unused3>", "decide": "<unused4>"}
ZWNJ = "‌"


def content_text(v) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def state_text(v) -> str:
    if isinstance(v, dict):
        return "\n".join(f"{k}: {content_text(x)}" for k, x in v.items())
    return content_text(v)


def neutralize(text: str) -> str:
    """Mirror of grande-llama's `neutralize_specials`: break `<name` so caller
    text can never tokenize into a control / reserved token."""
    if "<" not in text:
        return text
    out = []
    chars = list(text)
    for i, c in enumerate(chars):
        out.append(c)
        if c == "<" and i + 1 < len(chars) and (chars[i + 1].isascii() and chars[i + 1].isalpha() or chars[i + 1] in "|/"):
            out.append(ZWNJ)
    return "".join(out)


def options_of(q: dict):
    """(kind, instructions, [(key, description|None)]) — same defaults as Rust."""
    kind = q["type"]
    instr = q.get("instructions")
    if kind == "choice":
        return kind, content_text(instr) if instr is not None else "Choose the best matching option.", [
            (k, content_text(d) if d is not None else None) for k, d in q["criteria"].items()
        ]
    if kind == "score":
        return kind, content_text(instr) if instr is not None else "Select the most appropriate level.", [
            (str(i), content_text(d)) for i, d in enumerate(q["criteria"])
        ]
    c = q.get("criteria") or {}
    get = lambda k: content_text(c[k]) if c.get(k) is not None else None
    return kind, content_text(instr) if instr is not None else "Is the statement true?", [("true", get("true")), ("false", get("false"))]


@dataclass
class Encoded:
    ids: list[int]
    seg: list[int]          # 0 = state, k = question k
    pos: list[int]          # branch positions restart at len(prefix)
    opt_idx: list[list[int]]  # per question, index of each </opt>
    decide_idx: list[int]
    keys: list[list[str]]
    orders: list[list[int]] = field(default_factory=list)


class Renderer:
    def __init__(self, tok):
        self.tok = tok
        self.bos = tok.bos_token_id
        self.delim = {k: self._single(v) for k, v in DELIMS.items()}

    def _single(self, s: str) -> int:
        ids = self.tok.convert_tokens_to_ids(s)
        if ids is None or ids == self.tok.unk_token_id:
            raise ValueError(f"{s!r} is not in the vocabulary")
        return ids

    def text(self, s: str) -> list[int]:
        if not s:
            return []
        return self.tok(neutralize(s), add_special_tokens=False).input_ids

    def encode(self, req: dict, orders: dict[str, list[int]] | None = None) -> Encoded:
        orders = orders or {}
        prefix = [self.bos, self.delim["state"]] + self.text(state_text(req["state"]))
        ids, seg, pos = list(prefix), [0] * len(prefix), list(range(len(prefix)))
        opt_idx, decide_idx, keys, used_orders = [], [], [], []
        for k, (qid, q) in enumerate(req["questions"].items(), start=1):
            _, instr, opts = options_of(q)
            order = orders.get(qid, list(range(len(opts))))
            ordered = [opts[i] for i in order]
            br = [self.delim["question"]] + self.text(instr)
            oi = []
            for key, desc in ordered:
                br += [self.delim["opt"]] + self.text(f"{key} — {desc}" if desc else key) + [self.delim["opt_end"]]
                oi.append(len(br) - 1)
            br.append(self.delim["decide"])
            base = len(ids)
            ids += br
            seg += [k] * len(br)
            pos += list(range(len(prefix), len(prefix) + len(br)))
            opt_idx.append([base + i for i in oi])
            decide_idx.append(base + len(br) - 1)
            keys.append([key for key, _ in ordered])
            used_orders.append(order)
        return Encoded(ids, seg, pos, opt_idx, decide_idx, keys, used_orders)


def check_parity(enc: Encoded, dump: dict) -> list[str]:
    """Compare against `grande render` output. Returns a list of mismatches."""
    problems = []
    n = len(dump["prefix"])
    if enc.ids[:n] != dump["prefix"]:
        problems.append(f"prefix differs: py {enc.ids[:n][:12]} vs rs {dump['prefix'][:12]}")
    off = n
    for k, b in enumerate(dump["branches"]):
        m = len(b["tokens"])
        if enc.ids[off:off + m] != b["tokens"]:
            problems.append(f"branch {b['id']} tokens differ")
        want = enc.opt_idx[k] + [enc.decide_idx[k]]
        if [w - off for w in want] != b["want"]:
            problems.append(f"branch {b['id']} want {[w - off for w in want]} vs {b['want']}")
        off += m
    return problems
