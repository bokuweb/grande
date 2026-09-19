"""Score any /v1/systemone server (grande, kev, jev_local, Jev) on JGLUE, or
time a request file. Same prompts as grande-eval / jev_local.

    python tools/http_eval.py --url http://127.0.0.1:8009 --task jnli --limit 300
    python tools/http_eval.py --url http://127.0.0.1:8009 --request examples/ticket-ja.json --repeat 5
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request

JNLI_LABELS = ["entailment", "contradiction", "neutral"]
JNLI_DESC = ["含意：前提から仮説が正しいと必ず言える", "矛盾：前提から仮説が誤りだと必ず言える", "中立：前提だけでは仮説が正しいとも誤りとも判断できない"]
JNLI_INSTR = "前提が正しいとき、仮説との論理的な関係を判定してください。前提から分からない情報を補わないでください。"
JCQA_INSTR = "質問に対して、常識に基づく最も適切な答えを選択肢から1つ選んでください。"


def payload(task, row, model):
    if task == "jnli":
        state = {"前提": row["sentence1"], "仮説": row["sentence2"]}
        criteria = dict(zip(JNLI_LABELS, JNLI_DESC))
        gold = row["label"]
        instr = JNLI_INSTR
    else:
        state = {"質問": row["question"]}
        criteria = {str(i): row[f"choice{i}"] for i in range(5)}
        gold = str(row["label"])
        instr = JCQA_INSTR
    return {"model": model, "state": state, "questions": {"answer": {"type": "choice", "instructions": instr, "criteria": criteria}}}, gold


def call(url, body, key):
    req = urllib.request.Request(f"{url}/v1/systemone", json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out, (time.perf_counter() - t) * 1000


def ece(confs, oks, bins=10):
    tot = [0.0] * bins
    acc = [0.0] * bins
    n = [0] * bins
    for c, o in zip(confs, oks):
        b = min(int(c * bins), bins - 1)
        tot[b] += c
        acc[b] += o
        n[b] += 1
    N = len(confs)
    return sum((n[b] / N) * abs(acc[b] / n[b] - tot[b] / n[b]) for b in range(bins) if n[b])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--key", default="local-dev")
    p.add_argument("--task", choices=["jnli", "jcqa"])
    p.add_argument("--data")
    p.add_argument("--limit", type=int)
    p.add_argument("--request")
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()

    if a.request:
        body = json.load(open(a.request))
        body["model"] = a.model
        times = []
        for _ in range(a.repeat):
            out, ms = call(a.url, body, a.key)
            times.append(ms)
        times.sort()
        print(json.dumps({"url": a.url, "questions": len(body["questions"]), "ms_min": round(times[0]), "ms_median": round(times[len(times) // 2]), "usage": out.get("usage")}, ensure_ascii=False))
        for qid, ans in out["answers"].items():
            print(" ", qid, ans.get("noul", ans.get("choice", ans.get("score"))))
        return

    data = a.data or f".cache/jglue/{'jnli' if a.task == 'jnli' else 'jcommonsenseqa'}-valid.jsonl"
    rows = [json.loads(l) for l in open(data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    oks, confs, nlls, times = [], [], [], []
    for i, row in enumerate(rows):
        body, gold = payload(a.task, row, a.model)
        out, ms = call(a.url, body, a.key)
        ans = out["answers"]["answer"]
        probs = ans["probabilities"]
        pred = ans["choice"]
        oks.append(int(pred == gold))
        confs.append(max(probs.values()))
        nlls.append(-math.log(max(probs.get(gold, 0.0), 1e-12)))
        times.append(ms)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} acc {sum(oks) / len(oks):.3f}", flush=True)
    n = len(oks)
    print(json.dumps({"url": a.url, "task": a.task, "n": n, "accuracy": sum(oks) / n, "ece": ece(confs, oks), "nll": sum(nlls) / n, "mean_confidence": sum(confs) / n, "mean_ms": sum(times) / n}, ensure_ascii=False))


if __name__ == "__main__":
    main()
