"""Markdown rows for docs/e5.md / docs/ruri.md from the run directories of
tools/e5_head.py (summary.json), tools/e5_generic.py (generic/summary.json)
and tools/e5_finetune.py (ft-*/summary.json).

    python tools/e5_table.py runs/e5 runs/ruri/30m runs/ruri/70m runs/ruri/130m runs/ruri/310m
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(r, lat=None):
    ece = f"{r['ece']:.3f} → {r['ece_T']:.3f}" if "ece" in r else "–"
    nll = f"{r['nll']:.3f} → {r['nll_T']:.3f}" if "nll" in r else "–"
    t = f"{r['T']:.2f}" if "T" in r else "–"
    f400 = f"{r['acc_400']:.3f} / {r['ece_400']:.3f}" if "ece_400" in r else f"{r['acc_400']:.3f} / –"
    ms = f"{lat:.0f}" if lat else (f"{r['ms']:.0f}" if "ms" in r else "–")
    return f"{r['acc']:.3f} | {ece} | {nll} | {t} | {f400} | {ms}"


def main():
    for d in sys.argv[1:]:
        d = Path(d)
        name = d.name
        s = json.load(open(d / "summary.json"))["results"]
        g = json.load(open(d / "generic" / "summary.json"))["results"] if (d / "generic" / "summary.json").exists() else {}
        lat = {}
        for task in ("jnli", "jcqa"):
            p = d / f"{task}-valid-latency.json"
            if p.exists():
                lat[task] = {k: v["mean_ms"] for k, v in json.load(open(p))["latency"].items()}
        for task in ("jnli", "jcqa"):
            print(f"\n{name} {task}")
            for r in s[task]:
                key = "pair" if r["head"].startswith("pair") else ("joint" if r["head"].startswith("joint") else "separate")
                print(f"| {name} `{r['head']}` | {fmt(r, lat.get(task, {}).get(key))} |")
            if task in g:
                print(f"| {name} `generic` | {fmt(g[task])} |")
            ft = d / f"ft-{task}" / "summary.json"
            if ft.exists():
                print(f"| {name} `ft` | {fmt(json.load(open(ft))['result'])} |")


if __name__ == "__main__":
    main()
