#!/bin/bash
# ===========================================================================
# Pipeline 2 (end-to-end) — YOLOv1 bi-level pruning (standalone)
#   Step 1: train DENSE  ->  Step 2: bi-level prune + surgery  ->
#   Step 3: finetune LEAN  ->  Benchmark (Params/FLOPs/latency/FPS/mAP)
# Chay tu repo root:  bash scripts/run_e2e.sh
# ===========================================================================
set -e

# --------------------------- CONFIG ----------------------------------------
DATA="./fish"                 # root: images/{train,val} + labels/{train,val}
BACKBONE="resnet18"           # resnet18/34/50/101, vgg16/19
NUM_CLASSES=1
INPUT=448
BATCH=16
WORKERS=8
DEVICE="auto"

OUT="./output/yolo_${BACKBONE}"
PY="python"

# Pruning hyper-params
TARGET_SPARSITY=0.5
PRUNE_ITERS=8
FINETUNE_EPOCHS=5
SCOPE="all"                   # all | backbone | neck (component-aware)

DENSE_EPOCHS=120
FINAL_EPOCHS=60
# ---------------------------------------------------------------------------

echo "=============================================================="
echo " PIPELINE 2 — YOLOv1 ($BACKBONE) bi-level pruning"
echo "=============================================================="

echo "[Step 1/4] Train DENSE ..."
$PY train.py --data-path $DATA --backbone $BACKBONE --num-classes $NUM_CLASSES \
    --input-size $INPUT --epochs $DENSE_EPOCHS --batch-size $BATCH --workers $WORKERS \
    --device $DEVICE --augment --output-dir $OUT/step1_dense

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
    --device $DEVICE --augment --output-dir $OUT/step3_final

echo "[Step 4/4] Benchmark DENSE vs PRUNED ..."
$PY benchmark.py --dense $OUT/step1_dense/model_best.pth \
    --pruned $OUT/step3_final/model_best.pth \
    --data-path $DATA --num-classes $NUM_CLASSES \
    --batch-size $BATCH --workers $WORKERS --device $DEVICE

echo "=============================================================="
echo " DONE. Final: $OUT/step3_final/model_best.pth"
echo "=============================================================="
