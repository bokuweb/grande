"""Serve a Laya checkpoint through laya-mlx behind omg's `/v1/systemone`
shape, so `tools/http_eval.py` and the JevBench harness can score it
unchanged. Apple Silicon only (MLX).

    uv venv -p 3.13 .laya && uv pip install -p .laya/bin/python laya-mlx
    python tools/laya_serve.py --model aac6fef/laya-multilingual-mlx --port 8790
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import laya_mlx as laya


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="aac6fef/laya-multilingual-mlx")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    a = p.parse_args()

    agent = laya.load(a.model, dtype=a.dtype)
    lock = Lock()
    name = a.model

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
                    out = agent.predict(req["state"], req["questions"])
                ms = (time.perf_counter() - t) * 1000
            except Exception as e:  # noqa: BLE001
                return self._json(422, {"detail": [{"msg": str(e)}]})
            out["model"] = req.get("model") or name
            out["latency_ms"] = round(ms, 2)
            self._json(200, out)

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"laya: http://{a.host}:{a.port}/v1/systemone  model {name}  dtype {a.dtype}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
