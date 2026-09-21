#!/bin/zsh
# Ruri v3 (30m / 70m / 130m / 310m) through the e5 pipeline (docs/ruri.md):
# frozen features, heads and the generic head for every size, a prefix
# ablation on the 30m, then the whole-encoder fine-tunes smallest first.
#   tools/ruri.sh            # everything, ~7 h on an M4 (the fine-tunes are most of it)
#   tools/ruri.sh frozen     # the frozen part only, ~45 min
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python/.venv/bin/python}
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p runs/ruri
for size in 30m 70m 130m 310m; do
  M=cl-nagoya/ruri-v3-$size
  $PY tools/e5_features.py --model $M --prefix "" --out runs/ruri/$size --latency 200 > runs/ruri/$size.features.log 2>&1
  $PY tools/e5_head.py --feat runs/ruri/$size --out runs/ruri/$size/summary.json > runs/ruri/$size.head.log 2>&1
  $PY tools/e5_generic.py --model $M --prefix "" --feat runs/ruri/$size --out runs/ruri/$size/generic > runs/ruri/$size.generic.log 2>&1
  echo "$size done" >> runs/ruri/progress.log
done
$PY tools/e5_features.py --model cl-nagoya/ruri-v3-30m --prefix "トピック: " --out runs/ruri/30m-topic > runs/ruri/30m-topic.features.log 2>&1
$PY tools/e5_head.py --feat runs/ruri/30m-topic --out runs/ruri/30m-topic/summary.json > runs/ruri/30m-topic.head.log 2>&1
echo "all done" >> runs/ruri/progress.log
[ "${1:-}" = frozen ] && exit 0
export HF_HUB_OFFLINE=1
for size in 30m 70m 130m 310m; do
  M=cl-nagoya/ruri-v3-$size
  BS=32; BQ=8
  if [ $size = 310m ]; then BS=16; BQ=4; fi   # 16 GB: 315M params + Adam + activations
  $PY tools/e5_finetune.py --model $M --prefix "" --task jnli --bs $BS --out runs/ruri/$size/ft-jnli > runs/ruri/$size.ft-jnli.log 2>&1
  echo "$size ft-jnli done" >> runs/ruri/progress.log
  $PY tools/e5_finetune.py --model $M --prefix "" --task jcqa --bs $BQ --out runs/ruri/$size/ft-jcqa > runs/ruri/$size.ft-jcqa.log 2>&1
  echo "$size ft-jcqa done" >> runs/ruri/progress.log
done
echo "ft all done" >> runs/ruri/progress.log
