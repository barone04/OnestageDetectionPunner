#!/bin/bash
# ===========================================================================
# Pipeline 2 (end-to-end) — YOLOv1 bi-level pruning (standalone)
#   Step 1: train DENSE  ->  Step 2: bi-level prune + surgery  ->
#   Step 3: finetune LEAN  ->  Benchmark (Params/FLOPs/latency/FPS/mAP)
# Chay tu repo root:  bash scripts/run_e2e.sh
#   Doi flag khong can sua file -> them ENV truoc lenh:
#   BACKBONE=resnet50 TARGET_SPARSITY=0.7 bash scripts/run_e2e.sh
#   for bb in resnet18 resnet50 vgg16; do BACKBONE=$bb bash scripts/run_e2e.sh; done
# ===========================================================================
set -e

# --------------------- CONFIG (deu env-overridable) ------------------------
DATA="${DATA:-./NewDeepfish}"
BACKBONE="${BACKBONE:-resnet18}"       # resnet18/34/50/101, vgg16/19
NUM_CLASSES="${NUM_CLASSES:-1}"
INPUT="${INPUT:-448}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-auto}"

OUT="${OUT:-./output/yolo_${BACKBONE}}"
PY="${PY:-python}"

# Pruning hyper-params
TARGET_SPARSITY="${TARGET_SPARSITY:-0.5}"
PRUNE_ITERS="${PRUNE_ITERS:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-3}"
SCOPE="${SCOPE:-all}"                   # all | backbone | neck (component-aware)

DENSE_EPOCHS="${DENSE_EPOCHS:-5}"
FINAL_EPOCHS="${FINAL_EPOCHS:-5}"

WANDB="${WANDB:-0}"                     # WANDB=1 de bat log wandb (env key da set san tren FPT)
WANDB_ARG=""; [ "$WANDB" = "1" ] && WANDB_ARG="--wandb"
# ---------------------------------------------------------------------------

echo "=============================================================="
echo " YOLOv1 ($BACKBONE) bi-level pruning"
echo "=============================================================="

echo "[Step 1/4] Train DENSE ..."
$PY train.py --data-path $DATA --backbone $BACKBONE --num-classes $NUM_CLASSES \
    --input-size $INPUT --epochs $DENSE_EPOCHS --batch-size $BATCH --workers $WORKERS \
    --device $DEVICE --augment $WANDB_ARG --output-dir $OUT/step1_dense

echo "[Step 2/4] Bi-level prune (Song Han + Filter) + surgery ..."
$PY prune.py --data-path $DATA --checkpoint $OUT/step1_dense/model_best.pth \
    --target-sparsity $TARGET_SPARSITY --prune-iters $PRUNE_ITERS \
    --finetune-epochs $FINETUNE_EPOCHS --scope $SCOPE \
    --batch-size $BATCH --workers $WORKERS --device $DEVICE \
    --output-dir $OUT/step2_pruned

echo "[Step 3/4] Finetune LEAN ..."
$PY train.py --data-path $DATA --num-classes $NUM_CLASSES --input-size $INPUT \
    --init-config $OUT/step2_pruned/model_lean.json \
    --weights     $OUT/step2_pruned/model_lean.pth \
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
