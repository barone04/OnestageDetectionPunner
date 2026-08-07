# Config apply pruning — YOLOv1 / DeepFish (từng method)

## Chung (mọi method)

- **Dense (step1): train 1 lần, dùng chung** `model_best.pth` cho cả 4 method.
- **Finetune tổng = 200 epoch** mỗi method.
- **3 mức iso-%MACs**: Nhẹ ~40% · Vừa ~53% · Nặng ~65% (dense YOLO resnet18/448 = 9.36G MACs / 21.93M).
- **Prune scope: MID-only** cả 4 method (stem + output block + downsample GIỮ; residual-safe).
- **Hyperparam ISO** (mọi finetune, cả prune-time lẫn cuối): `--lr 1e-3 --weight-decay 5e-4 --momentum 0.9 --batch-size 16`, SGD + CosineAnnealingLR.
- **Giữ theo gốc** (thuật toán): `--prune-iters`, `--rank`, `--criterion`, scheduler-lúc-prune. `--workers` tùy máy (không ảnh hưởng acc).
- **Branch mỗi method** (checkout trước khi run): bi-level & CORING → `CORING` (`prune.py --method`); SVP → `Sliming` (`prune_svp.py`); NORTON → `NORTON` (`norton.py`). Chạy sai branch = thiếu driver/flag.

```bash
# Step1 dense (chung) — chay 1 lan
python train.py --data-path ./NewDeepfish --backbone resnet18 --num-classes 1 \
    --input-size 448 --epochs 70 --batch-size 16 --lr 1e-3 --weight-decay 5e-4 \
    --momentum 0.9 --augment --pretrained-backbone --output-dir ./output/dense
# => dung ./output/dense/model_best.pth cho ca 4 method
```

## Knob theo 3 mức %MACs (đã đo verify — shape-only, = số trên GPU)

Dense YOLO resnet18/448 = **9.361G MACs / 21.93M params**. bi-level ≡ CORING (cùng `round(n·r)` uniform, cùng tập layer → hình dạng y hệt).

| Case | %MACs | bi-level `--target-sparsity` | CORING `--target-sparsity` | SVP `--target-rate` | NORTON `--prune-ratio` |
|---|---|---|---|---|---|
| Nhẹ | ~40% | 0.37 | 0.37 | 0.44 | 0.30 |
| Vừa | ~53% | 0.50 | 0.50 | 0.64 | 0.45 |
| Nặng | ~65% | 0.64 | 0.64 | 0.76 | 0.60 |

**%MACs / %params thật tại iso-MACs** (ở cùng %MACs, %params khác nhau — không ép được cả hai):

**KẾT QUẢ:**

| Case | bi-level/CORING | SVP | NORTON |
|---|---|---|---|
| Nhẹ | 40.0% MACs / 47.8% params | 40.1% / 57.8% | 40.5% / 57.8% |
| Vừa | 52.6% MACs / 61.5% params | 53.4% / 70.3% | 53.0% / 69.4% |
| Nặng | 65.4% MACs / 74.5% params | 65.8% / 79.2% | 64.9% / 79.4% |


## Phân bổ 200 epoch finetune

| Method | Phân bổ |
|---|---|
| bi-level | prune-time 5×8 = 40 + final 160 |
| CORING | calibrate 5×8 = 40 + final 160 |
| SVP | 0 + final 200 |
| NORTON | decompose-ft 100 + prune-ft 100 |

---

## bi-level (`prune.py --method bilevel`)

```bash
DENSE=./output/dense/model_best.pth
S=0.50                       # {Nhẹ 0.37 | Vừa 0.50 | Nặng 0.64}

# step2: prune + prune-time finetune (5x8 = 40 ep) | lr iso 1e-3 (override default 5e-4)
python prune.py --method bilevel --checkpoint $DENSE --data-path ./NewDeepfish \
    --target-sparsity $S --prune-iters 5 --finetune-epochs 8 --scope all \
    --lr 1e-3 --weight-decay 5e-4 --momentum 0.9 --batch-size 16 \
    --output-dir ./output/bilevel/step2

# step3: finetune cuoi 160 ep (recipe CHUNG iso)
python train.py --data-path ./NewDeepfish --num-classes 1 --input-size 448 \
    --init-config ./output/bilevel/step2/model_lean.json \
    --weights     ./output/bilevel/step2/model_lean.pth \
    --epochs 160 --lr 1e-3 --weight-decay 5e-4 --momentum 0.9 --batch-size 16 \
    --augment --output-dir ./output/bilevel/step3
```

## CORING (`prune.py --method coring`)

```bash
DENSE=./output/dense/model_best.pth
S=0.50                       # {Nhẹ 0.37 | Vừa 0.50 | Nặng 0.64}

# step2: k-shot surgery-first + calibrate (5x8 = 40 ep) | lr iso 1e-3 (override default 5e-4)
python prune.py --method coring --checkpoint $DENSE --data-path ./NewDeepfish \
    --target-sparsity $S --prune-iters 5 --finetune-epochs 8 --scope all \
    --lr 1e-3 --weight-decay 5e-4 --momentum 0.9 --batch-size 16 \
    --output-dir ./output/coring/step2

# step3: finetune cuoi 160 ep (recipe CHUNG iso)
python train.py --data-path ./NewDeepfish --num-classes 1 --input-size 448 \
    --init-config ./output/coring/step2/model_lean.json \
    --weights     ./output/coring/step2/model_lean.pth \
    --epochs 160 --lr 1e-3 --weight-decay 5e-4 --momentum 0.9 --batch-size 16 \
    --augment --output-dir ./output/coring/step3
```

## SVP / SLIMING (`prune_svp.py`)

```bash
DENSE=./output/dense/model_best.pth
R=0.64                       # {Nhẹ 0.44 | Vừa 0.64 | Nặng 0.76}

# step2: GAM + GEM + copy weight (khong finetune -> khong co hyperparam train)
python prune_svp.py --checkpoint $DENSE --target-rate $R --scope all \
    --output-dir ./output/svp/step2

# step3: finetune cuoi 200 ep (recipe CHUNG iso)
python train.py --data-path ./NewDeepfish --num-classes 1 --input-size 448 \
    --init-config ./output/svp/step2/model_lean.json \
    --weights     ./output/svp/step2/model_lean.pth \
    --epochs 200 --lr 1e-3 --weight-decay 5e-4 --momentum 0.9 --batch-size 16 \
    --augment --output-dir ./output/svp/step3
```

## NORTON (`norton.py`)

```bash
DENSE=./output/dense/model_best.pth
PR=0.45                      # {Nhẹ 0.30 | Vừa 0.45 | Nặng 0.60}

# decompose-ft (100) + prune-ft (100) — TRONG norton.py, khong co step3 rieng
python norton.py --checkpoint $DENSE --data-path ./NewDeepfish \
    --rank 6 --prune-ratio $PR --criterion pabs --scope all \
    --decompose-finetune-epochs 100 --prune-finetune-epochs 100 \
    --lr 1e-3 --weight-decay 5e-4 --momentum 0.9 --batch-size 16 \
    --output-dir ./output/norton
```

## Benchmark (mọi method)

```bash
python benchmark.py --dense ./output/dense/model_best.pth \
    --pruned <PATH_model_best_cua_method> \
    --data-path ./NewDeepfish --num-classes 1
```
