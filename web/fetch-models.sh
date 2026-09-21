#!/bin/bash
# Fetch the trained browser model from the GitHub release into web/models/.
# Used by the Pages workflow and for local development (needs `gh`).
set -euo pipefail
cd "$(dirname "$0")"
# The trained 270M models (releases models-v1 / wgpu-v1, 575 MB) are no
# longer in web/engine.js's model list, so they are not fetched: GitHub
# Pages caps a site at 1 GB and the Laya / e5 / Ruri exports below fill it.
TAG="${1:-models-v1}"
WTAG="${2:-wgpu-v1}"

# Laya (mmBERT encoder + decision head, tools/export_laya.py), 180 MB: small
# enough for the Pages site itself, so it is served same-origin from here
# rather than from the Hub.
LTAG="${3:-laya-v1}"
LDEST="models/laya-multilingual-wgpu"
mkdir -p "$LDEST"
for f in config.json tokenizer.json tokenizer_config.json manifest.json embed.bin enc.bin head.bin; do
  gh release download "$LTAG" --repo bokuweb/grande --pattern "$f" --output "$LDEST/$f" --clobber \
    || { echo "no $LTAG release; skipping the Laya model"; rm -rf "$LDEST"; break; }
done
ls -la "$LDEST" 2>/dev/null || true

# multilingual-e5-small + the (state, option) head (tools/export_e5.py), 58 MB.
ETAG="${4:-e5-v1}"
EDEST="models/multilingual-e5-small-wgpu"
mkdir -p "$EDEST"
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json manifest.json embed.bin enc.bin head.bin; do
  gh release download "$ETAG" --repo bokuweb/grande --pattern "$f" --output "$EDEST/$f" --clobber \
    || { echo "no $ETAG release; skipping the e5 model"; rm -rf "$EDEST"; break; }
done
ls -la "$EDEST" 2>/dev/null || true

# Ruri v3 130m / 310m + the (state, option) head (tools/export_e5.py), 146 / 314 MB.
RTAG="${5:-ruri-v1}"
for size in 130m 310m; do
  RDEST="models/ruri-v3-$size-wgpu"
  mkdir -p "$RDEST"
  for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json manifest.json embed.bin enc.bin head.bin; do
    gh release download "$RTAG" --repo bokuweb/grande --pattern "$size-$f" --output "$RDEST/$f" --clobber \
      || { echo "no $RTAG release; skipping ruri-v3-$size"; rm -rf "$RDEST"; break; }
  done
  ls -la "$RDEST" 2>/dev/null || true
done
