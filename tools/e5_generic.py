"""One head for every choice question: score(state, option text) from the
frozen embeddings of the rendered state and of each option's description,
softmax over the options. This is what `tools/e5_serve.py` serves behind
/v1/systemone, so it is trained on JNLI and JCQA train together and scored
per task with the protocol of tools/e5_head.py.

    python tools/e5_generic.py --feat runs/e5 --out runs/e5/generic

Texts (identical to what the server renders):
  state   "query: " + "key: value" pairs joined by spaces   (JNLI: 前提 / 仮説, JCQA: 質問)
  option  "query: " + the criterion's description            (JNLI: the three label descriptions, JCQA: the choice)
The JNLI state embedding is the `pair` vector tools/e5_features.py already
wrote; the JCQA state and the label descriptions are embedded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from e5_features import Embedder, rows  # noqa: E402
from e5_head import pair_feats, report, train  # noqa: E402
from http_eval import JNLI_DESC  # noqa: E402


def render_state(state):
    return "query: " + " ".join(f"{k}: {v}" for k, v in state.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--feat", default="runs/e5")
    ap.add_argument("--data", default=".cache/jglue")
    ap.add_argument("--out", default="runs/e5/generic")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--drop", type=float, default=0.2)
    a = ap.parse_args()
    feat, out = Path(a.feat), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    emb = Embedder(a.model, a.device, torch.float16, 512)
    desc = emb([f"query: {d}" for d in JNLI_DESC])  # [3, d]

    def jnli(split):
        d = np.load(feat / f"jnli-{split}.npz")
        s = d["pair"]
        o = np.broadcast_to(desc[None], (len(s), 3, desc.shape[-1]))
        return pair_feats(np.repeat(s[:, None], 3, 1), o), d["gold"]

    def jcqa(split):
        d = np.load(feat / f"jcqa-{split}.npz")
        recs = rows(f"{a.data}/jcommonsenseqa-{split}.jsonl")
        s = np.concatenate([emb([render_state({"質問": r["question"]}) for r in recs[i : i + 64]]) for i in range(0, len(recs), 64)])
        o = np.stack([d[f"c{i}"] for i in range(5)], 1)
        return pair_feats(np.repeat(s[:, None], 5, 1), o), d["gold"]

    xj, yj = jnli("train")
    xq, yq = jcqa("train")
    # both tasks in one array, padded to 5 options with -inf-scored rows after training
    k = 5
    x = np.concatenate([np.pad(xj, ((0, 0), (0, k - 3), (0, 0))), xq])
    y = np.concatenate([yj, yq])
    n_opts = np.concatenate([np.full(len(yj), 3), np.full(len(yq), 5)])
    perm = np.random.RandomState(0).permutation(len(y))
    x, y, n_opts = x[perm], y[perm], n_opts[perm]
    predict, info = train(x, y, k, a.hidden, a.epochs, a.lr, a.wd, a.drop, per_choice=True, n_opts=n_opts)
    print("selection:", info)

    res = {}
    xv, yv = jnli("valid")
    res["jnli"] = report("generic", predict(xv), yv)
    xv, yv = jcqa("valid")
    res["jcqa"] = report("generic", predict(xv), yv)
    torch.save(predict.state, out / "head.pt")
    json.dump({"args": vars(a), "selection": info, "results": res}, open(out / "summary.json", "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
