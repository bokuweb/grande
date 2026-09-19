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
