#!/usr/bin/env bash
# Run "exp:model" jobs one after another on one GPU:  bash run_queue.sh <gpu> exp:model [exp:model ...]
cd "$(dirname "$0")"
PY=${PY:-/home/server/miniforge3/envs/spanetv2/bin/python}
gpu=$1; shift
for job in "$@"; do
  exp=${job%%:*}; model=${job#*:}
  if [ -f "data/raw/${exp}_${model}.json" ]; then echo "skip $job (done)"; continue; fi
  echo "[$(date '+%F %T')] GPU$gpu start $job"
  CUDA_VISIBLE_DEVICES=$gpu $PY -W ignore exp_cls.py "$exp" --model "$model" > "logs/${exp}_${model}.log" 2>&1
  echo "[$(date '+%F %T')] GPU$gpu end   $job exit=$?"
done
