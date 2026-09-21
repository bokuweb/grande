"""Serve a frozen embedding model + the generic (state, option) head from
tools/e5_generic.py behind grande's `/v1/systemone` shape, so
`tools/http_eval.py`, the JevBench harness and the web page can use it
unchanged.

    python tools/e5_serve.py --head runs/e5/generic/head.pt --port 8792
    python tools/http_eval.py --url http://127.0.0.1:8792 --task jnli

One forward per request: the rendered state and every option text not seen
before (option vectors are cached; a question's criteria are fixed text);
the head scores each (state, option) pair and a softmax over a question's
options is its answer.
Noul and score questions are served the same way (true/false or level
descriptions as the options) but the head never saw them in training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import e5_features  # noqa: E402
from e5_features import Embedder  # noqa: E402
from e5_generic import render_state  # noqa: E402
from e5_head import MLP, pair_feats  # noqa: E402


def option_text(key, desc):
    if desc is None:
        return str(key)
    return desc if isinstance(desc, str) else json.dumps(desc, ensure_ascii=False)


def state_text(state):
    if isinstance(state, dict):
        return render_state({k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in state.items()})
    return e5_features.PREFIX + (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))


class Head:
    def __init__(self, path):
        st = torch.load(path, weights_only=False)
        self.m = MLP(st["d"], st["out"], st["hidden"], 0.0)
        self.m.load_state_dict(st["model"])
        self.m.eval()
        self.mu, self.sd = st["mu"], st["sd"]

    @torch.no_grad()
    def __call__(self, s, opts):
        x = pair_feats(np.repeat(s[None], len(opts), 0), opts)
        x = (torch.as_tensor(x, dtype=torch.float32) - self.mu) / self.sd
        return self.m(x).squeeze(-1).numpy()


OPTION_CACHE: dict[str, np.ndarray] = {}


def embed_request(emb, state, options):
    """One forward for the state plus every option text not seen before. Option
    descriptions repeat across requests (a question's criteria are fixed text),
    so their vectors are kept; the state is never cached."""
    todo = [state] + list(dict.fromkeys(t for t in options if t not in OPTION_CACHE))
    vecs = emb(todo)
    for t, v in zip(todo[1:], vecs[1:]):
        OPTION_CACHE[t] = v
    if len(OPTION_CACHE) > 8192:
        for k in list(OPTION_CACHE)[:4096]:
            del OPTION_CACHE[k]
    return np.concatenate([vecs[:1], np.stack([OPTION_CACHE[t] for t in options])]) if options else vecs[:1]


def answer(emb, head, tok, req, T):
    questions = req["questions"]
    texts = [state_text(req["state"])]
    spans = {}
    for qid, q in questions.items():
        if q["type"] == "score":
            opts = [(str(i), option_text(i, c)) for i, c in enumerate(q["criteria"])]
        elif q["type"] == "noul":
            crit = q.get("criteria") or {"true": None, "false": None}
            opts = [(k, option_text(k, crit.get(k))) for k in ("true", "false")]
        else:
            opts = [(k, option_text(k, d)) for k, d in q["criteria"].items()]
        spans[qid] = (len(texts), opts)
        texts += [f"{e5_features.PREFIX}{t}" for _, t in opts]
    vecs = embed_request(emb, texts[0], texts[1:])
    n_tok = sum(len(ids) for ids in tok(texts)["input_ids"])
    answers = {}
    for qid, (start, opts) in spans.items():
        z = head(vecs[0], vecs[start : start + len(opts)]) / T
        p = np.exp(z - z.max())
        p /= p.sum()
        probs = {k: float(v) for (k, _), v in zip(opts, p)}
        q = questions[qid]
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": probs["true"]}
        elif q["type"] == "score":
            i = int(p.argmax())
            answers[qid] = {"type": "score", "score": float(sum(j * v for j, v in enumerate(p))), "legend": {str(j): c for j, c in enumerate(q["criteria"])}, "probabilities": probs, "confidence": float(p[i])}
        else:
            answers[qid] = {"type": "choice", "choice": opts[int(p.argmax())][0], "probabilities": probs, "confidence": float(p.max())}
    return {"answers": answers, "usage": {"input_tokens": n_tok, "output_tokens": 0}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--head", default="runs/e5/generic/head.pt")
    p.add_argument("--prefix", default="query: ", help='text prefix ("query: " for e5, "" for Ruri v3)')
    p.add_argument("--temperature", type=float, default=1.4)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8792)
    a = p.parse_args()
    e5_features.PREFIX = a.prefix

    emb = Embedder(a.model, a.device, torch.float16 if a.device != "cpu" else torch.float32, 512)
    head = Head(a.head)
    lock = Lock()
    name = f"{a.model.split('/')[-1]}+head"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"data": [{"id": name}]})
            self._json(404, {"detail": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._json(404, {"detail": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n))
                t = time.perf_counter()
                with lock:
                    out = answer(emb, head, emb.tok, req, a.temperature)
                ms = (time.perf_counter() - t) * 1000
            except Exception as e:  # noqa: BLE001
                return self._json(422, {"detail": [{"msg": str(e)}]})
            out["model"] = req.get("model") or name
            out["latency_ms"] = round(ms, 2)
            self._json(200, out)

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"e5: http://{a.host}:{a.port}/v1/systemone  model {name}  head {a.head}  T {a.temperature}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
