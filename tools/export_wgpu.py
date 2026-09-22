"""Package a merged (vocab-pruned) omg checkpoint for the wgpu engine:
config.json, tokenizer.json / tokenizer_config.json, head.safetensors and
model.safetensors cast to f16 — the same files serve `omg` natively (a
model directory instead of a GGUF) and the browser (fetched into web/models/).

    python tools/export_wgpu.py --run runs/grande-270m-12k --out web/models/grande-270m-ja-wgpu
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory (has merged-pruned/ and head.safetensors)")
    ap.add_argument("--merged", default="merged-pruned", help="checkpoint subdirectory to export")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    src = run / a.merged
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(src / name, out / name)
    shutil.copy(run / "head.safetensors", out / "head.safetensors")
    tensors = load_file(str(src / "model.safetensors"))
    tensors = {k: v.to("cpu").half().contiguous() for k, v in tensors.items() if not k.startswith("lm_head.")}
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt", "dtype": "f16"})
    total = sum(v.numel() * 2 for v in tensors.values())
    print(f"{out}: {len(tensors)} tensors, {total / 1e6:.0f} MB f16")


if __name__ == "__main__":
    main()
