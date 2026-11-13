#!/bin/bash
# Run the 384 fine-tuning jobs one after another (each starts when the previous one exits).
#   bash scripts/spanetv2/run_finetune_384_queue.sh          # all jobs
#   bash scripts/spanetv2/run_finetune_384_queue.sh 1 4      # jobs 1-4 (S18/S36, run on the 4x RTX 4090 server)
#   bash scripts/spanetv2/run_finetune_384_queue.sh 5        # jobs 5-7 (M36/B36, meant for another server)
# Env vars (DATA_PATH, NUM_GPU, GRAD_ACCUM_STEPS, CUDA_VISIBLE_DEVICES) are passed to the job scripts, e.g.
#   DATA_PATH=/path/to/imagenet NUM_GPU=8 GRAD_ACCUM_STEPS=4 bash scripts/spanetv2/run_finetune_384_queue.sh 5
# Ctrl-C stops the whole queue. To skip only the current job, kill its torchrun process.
# A job that fails with CUDA OOM is retried once with half the per-GPU batch (2x grad accum).

CODE_PATH=${CODE_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)} # image_classification/
JOBS=(
    finetune_spanetv2_s18_pure_384
    finetune_spanetv2_s18_hybrid_384
    finetune_spanetv2_s36_pure_384
    finetune_spanetv2_s36_hybrid_384
    finetune_spanetv2_m36_pure_384
    finetune_spanetv2_b36_pure_384
    finetune_spanetv2_b36_hybrid_384
)
START=${1:-1}
END=${2:-${#JOBS[@]}}

cd $CODE_PATH || exit 1
mkdir -p output/logs
STATUS=output/logs/finetune_384_queue_status.txt
trap 'echo "[$(date "+%F %T")] queue interrupted" | tee -a $STATUS; exit 130' INT TERM

for ((i = START - 1; i < END; i++)); do
    JOB=${JOBS[$i]}
    SCRIPT=scripts/spanetv2/$JOB.sh
    ACCUM=${GRAD_ACCUM_STEPS:-$(grep -oP '^GRAD_ACCUM_STEPS=\$\{GRAD_ACCUM_STEPS:-\K[0-9]+' $SCRIPT)}
    for TRY in 1 2; do
        LOG=output/logs/${JOB}_$(date +%Y%m%d-%H%M%S).log
        echo "[$(date '+%F %T')] START $JOB  grad_accum=$ACCUM  log=$LOG" | tee -a $STATUS
        GRAD_ACCUM_STEPS=$ACCUM bash $SCRIPT 2>&1 | tee $LOG
        RC=${PIPESTATUS[0]}
        echo "[$(date '+%F %T')] END   $JOB  exit=$RC" | tee -a $STATUS
        if [ $RC -ne 0 ] && [ $TRY -eq 1 ] && grep -qE "OutOfMemoryError|CUDA out of memory" $LOG; then
            ACCUM=$((ACCUM * 2))
            echo "[$(date '+%F %T')] OOM -> retry $JOB with grad_accum=$ACCUM" | tee -a $STATUS
            sleep 30
            continue
        fi
        break
    done
    sleep 60  # let GPU memory and the rendezvous port free up before the next job
done
echo "[$(date '+%F %T')] QUEUE DONE" | tee -a $STATUS
