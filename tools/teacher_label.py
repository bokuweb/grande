"""Label synthetic states with a teacher served at /v1/systemone (e.g.
`grande serve` with Gemma 4 E4B, zero-shot label readout). Writes the same
records with `probs` (teacher distribution over the question's options, in
criteria order) and `labels` (teacher argmax).

    python tools/teacher_label.py --url http://127.0.0.1:8089 --in .cache/distill/states.jsonl --out .cache/distill/labeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def call(url, body):
    req = urllib.request.Request(f"{url}/v1/systemone", json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    done = 0
    try:
        done = sum(1 for _ in open(a.out, encoding="utf-8"))
    except FileNotFoundError:
        pass
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f:
        for i, rec in enumerate(rows):
            if i < done:
                continue
            body = {"model": "grande-latest", "state": rec["state"], "questions": rec["questions"]}
            try:
                out = call(a.url, body)
            except Exception as e:  # noqa: BLE001
                print(f"row {i}: {e}", file=sys.stderr)
                continue
            probs, labels = {}, {}
            for qid, q in rec["questions"].items():
                ans = out["answers"][qid]
                if q["type"] == "noul":
                    p = [ans["noul"], 1 - ans["noul"]]
                elif q["type"] == "choice":
                    p = [ans["probabilities"][k] for k in q["criteria"]]
                else:
                    p = [ans["probabilities"][str(k)] for k in range(len(q["criteria"]))]
                probs[qid] = [round(x, 5) for x in p]
                labels[qid] = max(range(len(p)), key=lambda j: p[j])
            rec = {**rec, "probs": probs, "labels": labels, "teacher": out.get("model")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
