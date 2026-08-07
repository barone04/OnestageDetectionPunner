#!/bin/bash
# ===========================================================================
# Pipeline end-to-end — YOLOv1 + SVP (Singular Value Pruning)
#   Step 1: train DENSE  ->  Step 2: SVP-GAM prune + surgery  ->
#   Step 3: finetune LEAN  ->  Benchmark
# Chay tu repo root:  bash scripts/run_e2e_svp.sh
#   TARGET_RATE=0.6 BACKBONE=resnet50 bash scripts/run_e2e_svp.sh
# ===========================================================================
set -e

set -a; [ -f .env ] && . ./.env; set +a

# --------------------- CONFIG (deu env-overridable) ------------------------
DATA="${DATA:-./NewDeepfish}"
BACKBONE="${BACKBONE:-resnet18}"
NUM_CLASSES="${NUM_CLASSES:-1}"
INPUT="${INPUT:-448}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-auto}"

OUT="${OUT:-./output/yolo_svp_${BACKBONE}}"
PY="${PY:-python}"

# SVP pruning hyper-params
TARGET_RATE="${TARGET_RATE:-0.5}"         # SVP compress rate (ty le kenh bi cat)
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"   # finetune sau SVP prune (step 2)
SCOPE="${SCOPE:-all}"                     # all | backbone | neck

DENSE_EPOCHS="${DENSE_EPOCHS:-70}"
FINAL_EPOCHS="${FINAL_EPOCHS:-80}"

WANDB="${WANDB:-0}"
WANDB_ARG=""; [ "$WANDB" = "1" ] && WANDB_ARG="--wandb"

PRETRAINED="${PRETRAINED:-1}"
PRET_ARG=""; [ "$PRETRAINED" = "1" ] && PRET_ARG="--pretrained-backbone"
# ---------------------------------------------------------------------------

echo "=============================================================="
echo " YOLOv1 ($BACKBONE) SVP pruning (GAM)"
echo "=============================================================="

echo "[Step 1/4] Train DENSE ..."
$PY train.py --data-path $DATA --backbone $BACKBONE --num-classes $NUM_CLASSES \
    --input-size $INPUT --epochs $DENSE_EPOCHS --batch-size $BATCH --workers $WORKERS \
    --device $DEVICE --augment $WANDB_ARG $PRET_ARG --output-dir $OUT/step1_dense

echo "[Step 2/4] SVP prune (Huong B: GAM + GEM chon filter + COPY weight) ..."
$PY prune_svp.py --checkpoint $OUT/step1_dense/model_best.pth \
    --target-rate $TARGET_RATE --scope $SCOPE --device $DEVICE \
    --output-dir $OUT/step2_svp

echo "[Step 3/4] FINETUNE LEAN (Huong B: --weights = weight copy tu dense) ..."
$PY train.py --data-path $DATA --num-classes $NUM_CLASSES --input-size $INPUT \
    --init-config $OUT/step2_svp/model_lean.json \
    --weights $OUT/step2_svp/model_lean.pth \
    --epochs $FINAL_EPOCHS --batch-size $BATCH --workers $WORKERS \
    --device $DEVICE --augment $WANDB_ARG --output-dir $OUT/step3_final

echo "[Step 4/4] Benchmark DENSE vs PRUNED ..."
$PY benchmark.py --dense $OUT/step1_dense/model_best.pth \
    --pruned $OUT/step3_final/model_best.pth \
    --data-path $DATA --num-classes $NUM_CLASSES \
    --batch-size $BATCH --workers $WORKERS --device $DEVICE

echo "=============================================================="
echo " DONE. Final: $OUT/step3_final/model_best.pth"
echo "=============================================================="
