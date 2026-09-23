"""Answer the demo presets with the trained cross-encoder (tools/cross_train.py)
and with the Ruri (state, option) embedding head (docs/ruri.md), and compare
both with omg on Gemma 4 E2B (`omg probe` outputs). Every instruction in the
presets is one neither Ruri model was trained on.

    python tools/cross_compare.py --cross runs/cross/70m --e2b runs/cross/e2b \\
        --embed-head runs/ruri/70m/generic/head.pt --requests examples/ticket-ja.json runs/cross/preset-*_ja_.json

Per preset: agreement with E2B (choice: same argmax; noul: same side of 0.5;
score: same rounded level) and how many distinct values the noul answers
take (the embedding head gives every noul over one state the same value).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import cross_train as ct  # noqa: E402


def e2b_answers(path):
    out = {}
    for qid, a in json.load(open(path))["answers"].items():
        out[qid] = a.get("choice") if a["type"] == "choice" else (a["noul"] if a["type"] == "noul" else a["score"])
    return out


def agree(q, a, b):
    if q["type"] == "choice":
        return a == b
    if q["type"] == "noul":
        return (a >= 0.5) == (b >= 0.5)
    return round(a) == round(b)


def embed_answers(model, prefix, head_path, req, T):
    import e5_features
    import e5_serve
    e5_features.PREFIX = prefix
    emb, head = model
    out = e5_serve.answer(emb, head, emb.tok, req, T)["answers"]
    return {q: (a.get("choice") if a["type"] == "choice" else a.get("noul", a.get("score"))) for q, a in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cl-nagoya/ruri-v3-70m")
    ap.add_argument("--cross", default="runs/cross/70m")
    ap.add_argument("--e2b", default="runs/cross/e2b")
    ap.add_argument("--embed-head", default="runs/ruri/70m/generic/head.pt")
    ap.add_argument("--embed-temperature", type=float, default=1.5)
    ap.add_argument("--requests", nargs="+", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    b = ct.Builder(tok)
    cross = ct.Cross(a.model).to(a.device)
    cross.load_state_dict(torch.load(Path(a.cross) / "model.pt", map_location=a.device))
    emb = None
    if a.embed_head and Path(a.embed_head).exists():
        import e5_features
        import e5_serve
        e5_features.PREFIX = ""
        emb = (e5_features.Embedder(a.model, a.device, torch.float16, 512), e5_serve.Head(a.embed_head))
    rows, tot = [], {"cross": [0, 0], "embed": [0, 0]}
    distinct = {"cross": [], "embed": [], "e2b": []}
    for path in a.requests:
        req = json.load(open(path))
        name = Path(path).stem
        ref_path = Path(a.e2b) / f"{name}.json"
        if not ref_path.exists():
            continue
        ref = e2b_answers(ref_path)
        ca, ms = ct.answer(cross, b, req, a.device)
        ca, ms = ct.answer(cross, b, req, a.device)  # warm
        ea = embed_answers(emb, "", a.embed_head, req, a.embed_temperature) if emb else {}
        qs = req["questions"]
        row = {"request": name, "questions": len(qs), "cross_ms": round(ms)}
        for key, ans in (("cross", ca), ("embed", ea)):
            if not ans:
                continue
            ok = [agree(qs[q], ans[q], ref[q]) for q in qs]
            row[f"{key}_agree"] = f"{sum(ok)}/{len(ok)}"
            tot[key][0] += sum(ok)
            tot[key][1] += len(ok)
        nouls = [q for q in qs if qs[q]["type"] == "noul"]
        for key, ans in (("cross", ca), ("embed", ea), ("e2b", ref)):
            if ans and len(nouls) > 1:
                vals = {round(ans[q], 2) for q in nouls}
                row[f"{key}_noul_values"] = f"{len(vals)}/{len(nouls)}"
                distinct[key].append(len(vals) / len(nouls))
        row["answers"] = {q: {"e2b": ref[q], "cross": ca[q], "embed": ea.get(q)} for q in qs}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "answers"}, ensure_ascii=False), flush=True)
    summary = {k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in tot.items()}
    summary["noul_distinct_fraction"] = {k: float(np.mean(v)) if v else None for k, v in distinct.items()}
    print("agreement with E2B:", json.dumps(summary, ensure_ascii=False))
    json.dump({"summary": summary, "rows": rows}, open(Path(a.cross) / "presets.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
