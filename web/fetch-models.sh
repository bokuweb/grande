#!/bin/bash
# Fetch the trained browser model from the GitHub release into web/models/.
# Used by the Pages workflow and for local development (needs `gh`).
set -euo pipefail
cd "$(dirname "$0")"
# The trained 270M models (releases models-v1 / wgpu-v1, 575 MB) are no
# longer in web/engine.js's model list, so they are not fetched (GitHub
# Pages caps a site at 1 GB).
TAG="${1:-models-v1}"
WTAG="${2:-wgpu-v1}"

# Laya (mmBERT encoder + decision head, tools/export_laya.py), 180 MB: small
# enough for the Pages site itself, so it is served same-origin from here
# rather than from the Hub.
LTAG="${3:-laya-v1}"
LDEST="models/laya-multilingual-wgpu"
mkdir -p "$LDEST"
for f in config.json tokenizer.json tokenizer_config.json manifest.json embed.bin enc.bin head.bin; do
  gh release download "$LTAG" --repo bokuweb/omg --pattern "$f" --output "$LDEST/$f" --clobber \
    || { echo "no $LTAG release; skipping the Laya model"; rm -rf "$LDEST"; break; }
done
ls -la "$LDEST" 2>/dev/null || true

# The e5 / Ruri v3 embedder exports (releases e5-v1, ruri-v1; tools/export_e5.py)
# are not fetched: they are not in the page's model list (their head does
# not read a question's instructions). `omg serve --model <export>`
# still runs them natively; download a release by hand to try one here.

# The Ruri v3 cross-encoders (tools/export_cross.py, docs/cross.md), 78 / 326 MB:
# release cross-v1, assets prefixed by size (70m-*, 310m-*).
CTAG="${4:-cross-v1}"
for size in 70m 310m; do
  CDEST="models/ruri-v3-$size-cross-wgpu"
  mkdir -p "$CDEST"
  for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json manifest.json embed.bin enc.bin head.bin; do
    gh release download "$CTAG" --repo bokuweb/omg --pattern "$size-$f" --output "$CDEST/$f" --clobber \
      || { echo "no $CTAG release; skipping ruri-v3-$size-cross"; rm -rf "$CDEST"; break; }
  done
  ls -la "$CDEST" 2>/dev/null || true
done
