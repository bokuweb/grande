#!/bin/bash
# Fetch the trained browser model from the GitHub release into web/models/.
# Used by the Pages workflow and for local development (needs `gh`).
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-models-v1}"
DEST="models/grande-270m-ja"
mkdir -p "$DEST/onnx"
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json head.safetensors id_map.bin; do
  gh release download "$TAG" --repo bokuweb/grande --pattern "$f" --output "$DEST/$f" --clobber
done
for f in model_quantized.onnx model_quantized.onnx_data; do
  gh release download "$TAG" --repo bokuweb/grande --pattern "$f" --output "$DEST/onnx/$f" --clobber
done
ls -la "$DEST" "$DEST/onnx"

# The same checkpoint for the wgpu engine (crates/grande-wgpu), if released.
WTAG="${2:-wgpu-v1}"
WDEST="models/grande-270m-ja-wgpu"
mkdir -p "$WDEST"
for f in config.json tokenizer.json tokenizer_config.json head.safetensors model.safetensors; do
  gh release download "$WTAG" --repo bokuweb/grande --pattern "$f" --output "$WDEST/$f" --clobber \
    || { echo "no $WTAG release; skipping the wgpu model"; rm -rf "$WDEST"; break; }
done
ls -la "$WDEST" 2>/dev/null || true

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
