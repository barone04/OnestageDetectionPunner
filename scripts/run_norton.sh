#!/bin/bash
# ===========================================================================
# NORTON comparison path inside the same YOLOv1 benchmark environment.
#   Step 1: train DENSE (same as OnestageDetectionPunner)
#   Step 2: CP-decompose eligible 3x3 convs + finetune + save checkpoint
#   Step 3: one-shot NORTON prune from that checkpoint + finetune
#   Step 4: benchmark DENSE vs NORTON
#
# Run from repo root:
#   bash scripts/run_norton.sh
#
# Override without editing:
#   BACKBONE=resnet18 NORTON_RANK=7 bash scripts/run_norton.sh
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
NORTON_DECOMPOSE_EPOCHS="${NORTON_DECOMPOSE_EPOCHS:-80}"
NORTON_PRUNE_EPOCHS="${NORTON_PRUNE_EPOCHS:-80}"
NORTON_RANK="${NORTON_RANK:-6}"
NORTON_SCOPE="${NORTON_SCOPE:-all}"       # all | backbone | neck
NORTON_PRUNE_RATIO="${NORTON_PRUNE_RATIO:-0.5}"
# Optional per-group expression, e.g. '[0.1]*8+[0.2]*5' for ResNet-18 + neck.
# When set, this takes the place of NORTON_PRUNE_RATIO.
NORTON_COMPRESS_RATE="${NORTON_COMPRESS_RATE:-}"
NORTON_CRITERION="${NORTON_CRITERION:-pabs}"  # pabs | csa | vbd
NORTON_COPY_BN="${NORTON_COPY_BN:-0}"          # 0 matches released ResNet-56 code
NORTON_WARMUP_EPOCHS="${NORTON_WARMUP_EPOCHS:-5}"
NORTON_WARMUP_DECAY="${NORTON_WARMUP_DECAY:-0.01}"
SEED="${SEED:-0}"
LR="${LR:-0.001}"

# W&B auto-bat neu .env co WANDB_API_KEY (norton.py). Dat WANDB=0 de tat.
WANDB="${WANDB:-1}"
WANDB_GROUP="${WANDB_GROUP:-norton-yolo-${BACKBONE}-deepfish}"
ENV_FILE="${ENV_FILE:-}"
# train.py van dung flag --wandb opt-in (khong doi); norton.py auto tu .env.
TRAIN_WANDB=()
[ "$WANDB" = "1" ] && TRAIN_WANDB=(--wandb --wandb-group "$WANDB_GROUP")
NORTON_WANDB=(--wandb-group "$WANDB_GROUP")
[ "$WANDB" = "0" ] && NORTON_WANDB+=(--no-wandb)
[ -n "$ENV_FILE" ] && NORTON_WANDB+=(--env-file "$ENV_FILE")

PRETRAINED="${PRETRAINED:-1}"
PRET_ARG=""; [ "$PRETRAINED" = "1" ] && PRET_ARG="--pretrained-backbone"

PRUNE_ARGS=(--prune-ratio "$NORTON_PRUNE_RATIO")
if [ -n "$NORTON_COMPRESS_RATE" ]; then
    PRUNE_ARGS=(--compress-rate "$NORTON_COMPRESS_RATE")
fi

BN_ARGS=()
if [ "$NORTON_COPY_BN" = "1" ]; then
    BN_ARGS=(--copy-bn)
fi

echo "=============================================================="
echo " YOLOv1 ($BACKBONE) NORTON comparison"
echo "=============================================================="

echo "[Step 1/4] Train DENSE ..."
"$PY" train.py --data-path "$DATA" --backbone "$BACKBONE" --num-classes "$NUM_CLASSES" \
    --input-size "$INPUT" --epochs "$DENSE_EPOCHS" --batch-size "$BATCH" --workers "$WORKERS" \
    --device "$DEVICE" --augment "${TRAIN_WANDB[@]}" $PRET_ARG \
    --wandb-run-name "${BACKBONE}-dense-deepfish" \
    --lr "$LR" \
    --lr-warmup-epochs "$NORTON_WARMUP_EPOCHS" \
    --lr-warmup-decay "$NORTON_WARMUP_DECAY" --seed "$SEED" \
    --output-dir "$OUT/step1_dense"

echo "[Steps 2-3/4] Decompose + finetune, then one-shot prune + finetune ..."
"$PY" norton.py --checkpoint "$OUT/step1_dense/model_best.pth" --data-path "$DATA" \
    --rank "$NORTON_RANK" --scope "$NORTON_SCOPE" \
    "${PRUNE_ARGS[@]}" "${BN_ARGS[@]}" --criterion "$NORTON_CRITERION" \
    --decompose-finetune-epochs "$NORTON_DECOMPOSE_EPOCHS" \
    --prune-finetune-epochs "$NORTON_PRUNE_EPOCHS" \
    --lr "$LR" \
    --lr-warmup-epochs "$NORTON_WARMUP_EPOCHS" \
    --lr-warmup-decay "$NORTON_WARMUP_DECAY" --seed "$SEED" \
    --batch-size "$BATCH" --workers "$WORKERS" --device "$DEVICE" \
    --output-dir "$OUT/step2_norton" \
    "${NORTON_WANDB[@]}" --wandb-run-name "${BACKBONE}-norton-deepfish"

echo "[Step 4/4] Benchmark DENSE vs NORTON ..."
"$PY" benchmark.py --dense "$OUT/step1_dense/model_best.pth" \
    --pruned "$OUT/step2_norton/model_best.pth" \
    --data-path "$DATA" --num-classes "$NUM_CLASSES" \
    --batch-size "$BATCH" --workers "$WORKERS" --device "$DEVICE"

echo "=============================================================="
echo " DONE. Final: $OUT/step2_norton/model_best.pth"
echo "=============================================================="
