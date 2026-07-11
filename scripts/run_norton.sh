#!/bin/bash
# ===========================================================================
# NORTON comparison path inside the same YOLOv1 benchmark environment.
#   Step 1: train DENSE (same as OnestageDetectionPunner)
#   Step 2: CP-decompose eligible 3x3 convs with NORTON-style blocks + finetune
#   Step 3: benchmark DENSE vs NORTON
#
# Run from repo root:
#   bash scripts/run_norton.sh
#
# Override without editing:
#   BACKBONE=resnet18 NORTON_RANK=6 bash scripts/run_norton.sh
# ===========================================================================
set -e

set -a; [ -f .env ] && . ./.env; set +a

DATA="${DATA:-./NewDeepfish}"
BACKBONE="${BACKBONE:-resnet18}"
NUM_CLASSES="${NUM_CLASSES:-1}"
INPUT="${INPUT:-448}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
PY="${PY:-python}"

OUT="${OUT:-./output/yolo_${BACKBONE}_norton}"
DENSE_EPOCHS="${DENSE_EPOCHS:-70}"
NORTON_EPOCHS="${NORTON_EPOCHS:-80}"
NORTON_RANK="${NORTON_RANK:-6}"
NORTON_SCOPE="${NORTON_SCOPE:-all}"       # all | backbone | neck

WANDB="${WANDB:-0}"
WANDB_ARG=""; [ "$WANDB" = "1" ] && WANDB_ARG="--wandb"

PRETRAINED="${PRETRAINED:-1}"
PRET_ARG=""; [ "$PRETRAINED" = "1" ] && PRET_ARG="--pretrained-backbone"

echo "=============================================================="
echo " YOLOv1 ($BACKBONE) NORTON comparison"
echo "=============================================================="

echo "[Step 1/3] Train DENSE ..."
$PY train.py --data-path $DATA --backbone $BACKBONE --num-classes $NUM_CLASSES \
    --input-size $INPUT --epochs $DENSE_EPOCHS --batch-size $BATCH --workers $WORKERS \
    --device $DEVICE --augment $WANDB_ARG $PRET_ARG --output-dir $OUT/step1_dense

echo "[Step 2/3] Apply NORTON CP decomposition + finetune ..."
$PY norton.py --checkpoint $OUT/step1_dense/model_best.pth --data-path $DATA \
    --rank $NORTON_RANK --scope $NORTON_SCOPE --finetune-epochs $NORTON_EPOCHS \
    --batch-size $BATCH --workers $WORKERS --device $DEVICE \
    --output-dir $OUT/step2_norton

echo "[Step 3/3] Benchmark DENSE vs NORTON ..."
$PY benchmark.py --dense $OUT/step1_dense/model_best.pth \
    --pruned $OUT/step2_norton/model_best.pth \
    --data-path $DATA --num-classes $NUM_CLASSES \
    --batch-size $BATCH --workers $WORKERS --device $DEVICE

echo "=============================================================="
echo " DONE. Final: $OUT/step2_norton/model_best.pth"
echo "=============================================================="
