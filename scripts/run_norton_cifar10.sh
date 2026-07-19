#!/bin/bash
# NORTON native-benchmark validation: ResNet-56 / CIFAR-10.
#
# With an existing dense checkpoint:
#   DENSE_CHECKPOINT=/path/to/resnet56_dense.pth bash scripts/run_norton_cifar10.sh
#
# Without one, set DENSE_EPOCHS to train it first:
#   DENSE_EPOCHS=200 bash scripts/run_norton_cifar10.sh
set -e

set -a; [ -f .env ] && . ./.env; set +a

PY="${PY:-python}"
DATA="${DATA:-./cifar10-data}"
OUT="${OUT:-./output/norton_resnet56_cifar10}"
DEVICE="${DEVICE:-auto}"
WORKERS="${WORKERS:-2}"
BATCH="${BATCH:-256}"

DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-}"
DENSE_EPOCHS="${DENSE_EPOCHS:-0}"
DECOMPOSE_EPOCHS="${DECOMPOSE_EPOCHS:-400}"
PRUNE_EPOCHS="${PRUNE_EPOCHS:-400}"
NORTON_RANK="${NORTON_RANK:-7}"
NORTON_CPR="${NORTON_CPR:-[0.]+[0.18]*29}"
NORTON_CRITERION="${NORTON_CRITERION:-pabs}"
LR="${LR:-0.05}"
WANDB="${WANDB:-0}"
WANDB_GROUP="${WANDB_GROUP:-norton-resnet56-cifar10}"
ENV_FILE="${ENV_FILE:-}"

WANDB_ARGS=()
if [ "$WANDB" = "1" ]; then
    WANDB_ARGS=(--wandb --wandb-group "$WANDB_GROUP" \
        --wandb-run-name "resnet56-norton-cifar10")
    if [ -n "$ENV_FILE" ]; then
        WANDB_ARGS+=(--env-file "$ENV_FILE")
    fi
fi

DENSE_ARGS=()
if [ -n "$DENSE_CHECKPOINT" ]; then
    DENSE_ARGS=(--dense-checkpoint "$DENSE_CHECKPOINT")
else
    DENSE_ARGS=(--dense-epochs "$DENSE_EPOCHS")
fi

"$PY" norton_cifar10.py \
    --data-path "$DATA" \
    "${DENSE_ARGS[@]}" \
    --decompose-finetune-epochs "$DECOMPOSE_EPOCHS" \
    --prune-finetune-epochs "$PRUNE_EPOCHS" \
    --rank "$NORTON_RANK" \
    --compress-rate "$NORTON_CPR" \
    --criterion "$NORTON_CRITERION" \
    --batch-size "$BATCH" --workers "$WORKERS" \
    --lr "$LR" --device "$DEVICE" --output-dir "$OUT" \
    "${WANDB_ARGS[@]}"
