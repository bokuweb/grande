"""Upload an exported wgpu model directory (tools/export_wgpu_gguf.py) to a
Hugging Face model repo, where the browser demo fetches it from (Hugging
Face serves CORS headers; GitHub Pages caps a site at 1 GB and GitHub
release assets are not fetchable cross-origin).

    hf auth login            # once (or HF_TOKEN in the environment)
    python tools/upload_wgpu_hf.py models/gemma-4-e2b-wgpu-q4-pruned bokuweb/gemma-4-E2B-it-grande-wgpu-ja
    python tools/upload_wgpu_hf.py models/gemma-4-e4b-wgpu-q4-pruned bokuweb/gemma-4-E4B-it-grande-wgpu-ja

The repo name is what web/engine.js lists as `hub` for the model. The model
card's size (E2B / E4B) and vocabulary note come from the export's
config.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi


README = """---
license: gemma
base_model: google/gemma-4-{size}-it
tags: [grande, webgpu, wgpu, gemma4]
---

# {repo}

Gemma 4 {size} ({quant} codes from the llama.cpp GGUF, repacked) for
[grande](https://github.com/bokuweb/grande)'s wgpu engine: state + every
question in one block-causal forward pass, in the browser on WebGPU or
natively on Metal / Vulkan. Exported with `tools/export_wgpu_gguf.py`;
`manifest.json` maps tensors to files. Not a standalone checkpoint format.
{vocab_note}
```bash
grande probe --model <this directory> --request examples/ticket-ja.json
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("repo")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    for f in ("manifest.json", "config.json", "tokenizer.json"):
        if not (d / f).exists():
            raise SystemExit(f"{d / f} missing; export first")
    cfg = json.loads((d / "config.json").read_text())
    source = cfg.get("grande_source", "")
    size = (re.search(r"E\dB", source) or re.search(r"E\dB", a.repo) or [None])[0]
    if not size:
        raise SystemExit(f"cannot tell the model size (E2B / E4B) from {source!r} or {a.repo!r}")
    quant = (re.search(r"Q\d_\w+", source) or ["Q4_0"])[0]
    vocab = int(cfg.get("vocab_size", 0))
    vocab_note = (
        f"\nVocabulary pruned to the {vocab:,} tokens a Japanese / English decision\n"
        "corpus uses (`tools/prune_vocab.py`); text inside that corpus tokenizes\n"
        "exactly as with the full vocabulary, other text into somewhat more pieces.\n"
        if vocab and vocab < 262144 else ""
    )
    (d / "README.md").write_text(README.format(repo=a.repo, size=size, quant=quant, vocab_note=vocab_note))
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)
    api.upload_folder(folder_path=str(d), repo_id=a.repo, repo_type="model",
                      commit_message="grande wgpu export", allow_patterns=["*.json", "*.bin", "README.md"])
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
