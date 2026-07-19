## Extend from [PruneFishNet](https://github.com/barone04/PrunedFishNet.git). Prune one-stage detection models

### NORTON comparison path

`norton.py` applies the original two-stage NORTON flow inside the same YOLOv1
training/evaluation environment: CP decomposition, decomposition finetuning,
one-shot CPD-factor pruning from the best decomposed checkpoint, then a second
finetuning phase. It keeps the detector, dataset, loss, and benchmark stack
shared with the channel-pruning pipeline, but reports NORTON as a separate
decomposition-based variant because it changes topology.

Example:

```bash
bash scripts/run_norton.sh
```

Manual example:

```bash
python norton.py --checkpoint ./output/yolo_resnet18/step1_dense/model_best.pth \
  --data-path ./NewDeepfish --rank 7 --scope all --prune-ratio 0.5 \
  --criterion pabs --decompose-finetune-epochs 80 \
  --prune-finetune-epochs 80 --output-dir ./output/yolo_resnet18_norton
```

For original-style per-layer sparsity allocation, pass a compression vector
instead of the uniform `--prune-ratio`. For ResNet-18 with `--scope all`, the
first 8 entries control the safe mid-channel of its BasicBlocks and the last 5
entries control the neck convolutions:

```bash
python norton.py --checkpoint ./output/yolo_resnet18/step1_dense/model_best.pth \
  --data-path ./NewDeepfish --rank 7 --scope all \
  --compress-rate "[0.1]*8+[0.2]*5" --criterion pabs \
  --decompose-finetune-epochs 80 --prune-finetune-epochs 80 \
  --output-dir ./output/yolo_resnet18_norton
```

The ordered allocation is written to `compress_rate_layout.json`. Required
vector lengths for `--scope all` are: ResNet-18 13, ResNet-34 21, ResNet-50
37, ResNet-101 71, VGG-16 18, and VGG-19 21. With `--scope backbone`, omit the
5 neck entries; with `--scope neck`, provide exactly 5 entries. The detector
head and ResNet residual outputs are intentionally fixed.

Post-pruning BatchNorm is freshly initialized by default, matching the released
NORTON ResNet-56 pruning code. Pass `--copy-bn` only as an explicit ablation.
Dense training and both recovery phases use the native rebuild's linear-warmup plus cosine schedule
(`--lr-warmup-epochs 5 --lr-warmup-decay 0.01`) and record the seed, full run
arguments, stage, allocation, criterion, and BN policy in checkpoints.
All three phases also share the same detector learning rate (`1e-3` by default).

The runner exposes the same mode through `NORTON_COMPRESS_RATE`:

```bash
NORTON_COMPRESS_RATE='[0.1]*8+[0.2]*5' bash scripts/run_norton.sh
```

The intermediate `decomposed_best.pth` is the exact checkpoint used as input
to one-shot pruning. `model_best.pth` is the final post-pruning checkpoint used
by `benchmark.py`.

### NORTON native validation: ResNet-56 / CIFAR-10

`norton_cifar10.py` is a separate classification validation path modeled
directly on the original NORTON ResNet-56 code. It keeps the CIFAR ResNet
Option-A shortcuts, the original 30-value per-layer compression mapping, CPD
of every 3x3 convolution, the two long finetuning phases, and one-shot
PABS/CSA/VBD factor pruning. By default BN is freshly initialized after
pruning, matching the released NORTON pruning implementation; `--copy-bn` is
an explicit ablation rather than the default.

Run with an existing dense ResNet-56 checkpoint:

```bash
DENSE_CHECKPOINT=/path/to/resnet56_dense.pth \
  bash scripts/run_norton_cifar10.sh
```

Or train the dense model first in the same harness:

```bash
DENSE_EPOCHS=200 bash scripts/run_norton_cifar10.sh
```

The default NORTON configuration is rank 7, compression expression
`[0.]+[0.18]*29`, 400 decomposition-finetuning epochs, and 400 post-pruning
epochs. Outputs include `decomposed_best.pth`, final `model_best.pth`, and
`results.json` with Top-1, parameters, MACs, latency, and FPS. This validates
algorithm fidelity on the native benchmark; it is not mixed into the YOLO
comparison table.

### Weights & Biases logging

Put `WANDB_API_KEY` and `WANDB_PROJECT` in a `.env` file in this directory or
any parent directory. `WANDB_ENTITY` is optional. The scripts find that file
automatically and do not override variables already exported by Kaggle.
W&B SDK 0.22.3 or newer is required for the current long API-key format.

Enable metrics logging for the complete YOLO/DeepFish pipeline:

```bash
WANDB=1 bash scripts/run_norton.sh
```

Dense YOLO training and NORTON are separate W&B runs in one group. The NORTON
run records `decomposed/*` and `pruned/*` metrics. ResNet-56/CIFAR-10 uses one
run with `dense/*`, `decomposed/*`, `pruned/*`, and `profile/*` metrics:

```bash
WANDB=1 DENSE_EPOCHS=200 bash scripts/run_norton_cifar10.sh
```

Checkpoint files remain local by default. Direct `train.py` runs can opt into
uploading the best model with `--wandb-save-model`.
