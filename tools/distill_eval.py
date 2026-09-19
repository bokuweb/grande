"""Agreement of a served student with the teacher on held-out labelled
synthetic records (tools/teacher_label.py output).

    python tools/distill_eval.py --url http://127.0.0.1:8090 --data .cache/distill/labeled.jsonl --skip 1800
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import urllib.request


def call(url, body):
    req = urllib.request.Request(f"{url}/v1/systemone", json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--data", required=True)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N records (the training portion)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")][a.skip :]
    if a.limit:
        rows = rows[: a.limit]
    by = collections.defaultdict(lambda: {"n": 0, "agree": 0, "l1": 0.0, "kl": 0.0})
    for rec in rows:
        out = call(a.url, {"model": "grande-latest", "state": rec["state"], "questions": rec["questions"]})
        for qid, q in rec["questions"].items():
            ans = out["answers"][qid]
            if q["type"] == "noul":
                p = [ans["noul"], 1 - ans["noul"]]
            elif q["type"] == "choice":
                p = [ans["probabilities"][k] for k in q["criteria"]]
            else:
                p = [ans["probabilities"][str(k)] for k in range(len(q["criteria"]))]
            t = rec["probs"][qid]
            for key in (q["type"], rec["family"], "ALL"):
                d = by[key]
                d["n"] += 1
                d["agree"] += int(max(range(len(p)), key=p.__getitem__) == max(range(len(t)), key=t.__getitem__))
                d["l1"] += sum(abs(x - y) for x, y in zip(p, t)) / 2
                d["kl"] += sum(y * math.log(max(y, 1e-9) / max(x, 1e-9)) for x, y in zip(p, t))
    print(f"{'group':10} {'n':>5} {'agree':>7} {'TV':>6} {'KL':>6}")
    for key in sorted(by):
        d = by[key]
        print(f"{key:10} {d['n']:>5} {d['agree'] / d['n']:>7.3f} {d['l1'] / d['n']:>6.3f} {d['kl'] / d['n']:>6.3f}")


if __name__ == "__main__":
    main()
