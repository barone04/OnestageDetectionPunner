#!/bin/bash
# ===========================================================================
# Pipeline 2 (end-to-end) — YOLOv1 pruning (standalone)
#   Step 1: train DENSE  ->  Step 2: prune + surgery  ->
#   Step 3: finetune LEAN  ->  Benchmark (Params/FLOPs/latency/FPS/mAP)
# Chay tu repo root:  bash scripts/run_e2e.sh
#   Doi thuat toan prune:   METHOD=coring bash scripts/run_e2e.sh   (mac dinh bilevel = bai minh)
#   So 2 phuong phap:       for m in bilevel coring; do METHOD=$m bash scripts/run_e2e.sh; done
#   Doi flag khong can sua file -> them ENV truoc lenh:
#   BACKBONE=resnet50 TARGET_SPARSITY=0.7 bash scripts/run_e2e.sh
# ===========================================================================
set -e
# tu nap .env neu co (vd: WANDB_API_KEY=..., WANDB_PROJECT=..., WANDB_ENTITY=...)
set -a; [ -f .env ] && . ./.env; set +a

# --------------------- CONFIG (deu env-overridable) ------------------------
DATA="${DATA:-./NewDeepfish}"
BACKBONE="${BACKBONE:-resnet18}"       # resnet18/34/50/101, vgg16/19
METHOD="${METHOD:-bilevel}"            # bilevel (bai minh) | coring (baseline)
NUM_CLASSES="${NUM_CLASSES:-1}"
INPUT="${INPUT:-448}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-auto}"

OUT="${OUT:-./output}"
PY="${PY:-python}"

# Pruning hyper-params
TARGET_SPARSITY="${TARGET_SPARSITY:-0.5}"
PRUNE_ITERS="${PRUNE_ITERS:-8}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
SCOPE="${SCOPE:-all}"                   # all | backbone | neck (component-aware)

DENSE_EPOCHS="${DENSE_EPOCHS:-70}"
FINAL_EPOCHS="${FINAL_EPOCHS:-80}"

WANDB="${WANDB:-0}"                     # WANDB=1 de bat log wandb (CHI train.py; prune.py khong log)
WANDB_ARG=""; [ "$WANDB" = "1" ] && WANDB_ARG="--wandb"

PRETRAINED="${PRETRAINED:-1}"           # PRETRAINED=1 -> Step1 dung ImageNet backbone (mac dinh bat)
PRET_ARG=""; [ "$PRETRAINED" = "1" ] && PRET_ARG="--pretrained-backbone"

# ten thu muc = ten run wandb (basename), theo mau: step{N}_{stage}_yolov1_{backbone}[_method]
if [ "$METHOD" = "bilevel" ]; then METHOD_TAG="bi-level"; else METHOD_TAG="$METHOD"; fi
DENSE_DIR="$OUT/step1_dense_yolov1_${BACKBONE}"                 # dense dung chung moi method
PRUNE_DIR="$OUT/step2_prune_${METHOD}_yolov1_${BACKBONE}"
FINAL_DIR="$OUT/step3_final_yolov1_${BACKBONE}_${METHOD_TAG}"
# ---------------------------------------------------------------------------

echo "=============================================================="
echo " YOLOv1 ($BACKBONE) pruning | method=$METHOD"
echo "=============================================================="

if [ -f "$DENSE_DIR/model_best.pth" ]; then
    echo "[Step 1/4] DENSE da co -> bo qua ($DENSE_DIR/model_best.pth)"
else
    echo "[Step 1/4] Train DENSE ..."
    $PY train.py --data-path $DATA --backbone $BACKBONE --num-classes $NUM_CLASSES \
        --input-size $INPUT --epochs $DENSE_EPOCHS --batch-size $BATCH --workers $WORKERS \
        --device $DEVICE --augment $WANDB_ARG $PRET_ARG --output-dir $DENSE_DIR
fi

echo "[Step 2/4] Prune ($METHOD) + surgery ..."   # prune.py KHONG log wandb
$PY prune.py --data-path $DATA --checkpoint $DENSE_DIR/model_best.pth \
    --method $METHOD --target-sparsity $TARGET_SPARSITY --prune-iters $PRUNE_ITERS \
    --finetune-epochs $FINETUNE_EPOCHS --scope $SCOPE \
    --batch-size $BATCH --workers $WORKERS --device $DEVICE \
    --output-dir $PRUNE_DIR

echo "[Step 3/4] Finetune LEAN ..."
$PY train.py --data-path $DATA --num-classes $NUM_CLASSES --input-size $INPUT \
    --init-config $PRUNE_DIR/model_lean.json \
    --weights     $PRUNE_DIR/model_lean.pth \
    --epochs $FINAL_EPOCHS --batch-size $BATCH --workers $WORKERS \
    --device $DEVICE --augment $WANDB_ARG --output-dir $FINAL_DIR

echo "[Step 4/4] Benchmark DENSE vs PRUNED ..."
$PY benchmark.py --dense $DENSE_DIR/model_best.pth \
    --pruned $FINAL_DIR/model_best.pth \
    --data-path $DATA --num-classes $NUM_CLASSES \
    --batch-size $BATCH --workers $WORKERS --device $DEVICE

echo "=============================================================="
echo " DONE. method=$METHOD | Final: $FINAL_DIR/model_best.pth"
echo "=============================================================="
