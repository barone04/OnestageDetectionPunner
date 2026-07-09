# Image classification reference training scripts

This folder contains reference training scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs with 
the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--batch_size`           | `32`   |
| `--epochs`               | `90`   |
| `--lr`                   | `0.1`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--lr-step-size`         | `30`   |
| `--lr-gamma`             | `0.1`  |

### ResNet
```
torchrun --nproc_per_node=8 train.py --model resnet50 -cpr  [0.255]*20 --batch-size 128 --lr 0.5 --output-dir output/resnet50/[0.255]*20/ --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4
```


### MobileNetV2
```
torchrun --nproc_per_node=8 train.py\
     --model mobilenet_v2 --epochs 300 --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98
```
