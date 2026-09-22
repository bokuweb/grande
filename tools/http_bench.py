"""Concurrency sweep against a /v1/systemone server: N clients each send a
request file with a cache-busting line prepended to the state (so every
request is a fresh document), and the sweep reports requests/s, decisions/s,
latency percentiles and how many requests the server put in one pass
(`X-Omg-Batch`). Same shape as the vLLM DiffusionGemma read benchmark
(concurrency sweep, three decisions per request, prefix cached).

    python tools/http_bench.py --url http://127.0.0.1:8080 --request examples/ticket-ja.json --concurrency 1,4,16,32 --requests 64
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


def call(url, body, key):
    req = urllib.request.Request(
        f"{url}/v1/systemone",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
        batch = int(r.headers.get("X-Omg-Batch", "1"))
        source = r.headers.get("X-Omg-State", "?")
    return out, (time.perf_counter() - t) * 1000, batch, source


def bust(body, i):
    b = json.loads(json.dumps(body))
    state = b["state"]
    if isinstance(state, dict):
        b["state"] = {"request_id": f"R-{i:05d}", **state}
    else:
        b["state"] = f"request_id: R-{i:05d}\n{state}"
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="local")
    ap.add_argument("--request", default="examples/ticket-ja.json")
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--requests", type=int, default=32, help="requests per concurrency level")
    args = ap.parse_args()
    body = json.load(open(args.request))
    body.setdefault("model", "grande-latest")
    decisions = len(body["questions"])
    # One warm-up so model load / first-pass costs are out of the numbers.
    first, _, _, _ = call(args.url, bust(body, 0), args.key)
    seq = 1
    for c in [int(x) for x in args.concurrency.split(",")]:
        n = max(args.requests, c)
        bodies = [bust(body, seq + i) for i in range(n)]
        seq += n
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            results = list(ex.map(lambda b: call(args.url, b, args.key), bodies))
        wall = time.perf_counter() - t0
        lat = sorted(r[1] for r in results)
        batches = [r[2] for r in results]
        sources = {}
        for r in results:
            sources[r[3]] = sources.get(r[3], 0) + 1
        # Every request must answer like the warm-up (same questions, only
        # the request id differs), or batching leaked something.
        disagree = 0
        for r in results:
            for q, a in r[0]["answers"].items():
                ref = first["answers"][q]
                for k in ("choice", "score", "noul"):
                    if k in a and k in ref and (a[k] != ref[k] if k == "choice" else abs(a[k] - ref[k]) > 0.05):
                        disagree += 1
        p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
        print(
            f"concurrency {c:3d}: {n} requests in {wall:6.2f} s = {n / wall:6.2f} req/s, {n * decisions / wall:6.1f} decisions/s; "
            f"latency p50 {statistics.median(lat):6.0f} ms  p90 {p(0.9):6.0f}  max {lat[-1]:6.0f}; "
            f"batch mean {statistics.mean(batches):.1f} max {max(batches)}; state {sources}"
            + (f"; {disagree} answers differ from the warm-up" if disagree else "")
        )


if __name__ == "__main__":
    main()
