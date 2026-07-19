"""
train.py — Train YOLOv1 (standalone).
- Step 1: train DENSE (mac dinh).
- Step 3: finetune LEAN model -> dung --init-config model_lean.json --weights model_lean.pth.

Chon best theo mAP@0.5:0.95 (fallback: val loss neu --no-map).

Vi du Step 1:
  python train.py --data-path ./fish --backbone resnet18 --num-classes 1 \
      --epochs 120 --batch-size 16 --output-dir ./output/yolo/step1_dense
Vi du Step 3:
  python train.py --data-path ./fish --num-classes 1 \
      --init-config ./output/yolo/step2_pruned/model_lean.json \
      --weights     ./output/yolo/step2_pruned/model_lean.pth \
      --epochs 60 --output-dir ./output/yolo/step3_final
"""
import os
import json
import argparse
import random

import numpy as np
import torch

from models.yolo import YoloModel, build_model
from loss import YoloLoss
from data import YoloDataset
from engine import train_one_epoch, evaluate
from utils import evaluate_map
from utils.tracking import init_wandb


def resolve_device(req):
    req = (req or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available -> CPU")
        return torch.device("cpu")
    return torch.device(req)


def get_args():
    p = argparse.ArgumentParser(description="Train/finetune YOLOv1")
    p.add_argument("--data-path", required=True)
    p.add_argument("--backbone", default="resnet18", help="resnet18/34/50/101, vgg16/19")
    p.add_argument("--input-size", default=448, type=int)
    p.add_argument("--num-classes", default=1, type=int)
    p.add_argument("--neck-out", default=512, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight-decay", default=5e-4, type=float)
    p.add_argument("--lr-warmup-epochs", default=5, type=int)
    p.add_argument("--lr-warmup-decay", default=0.01, type=float)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--pretrained-backbone", action="store_true",
                   help="nap ImageNet pretrained backbone (chi cho Step1 dense)")
    p.add_argument("--output-dir", default="./output/yolo_dense")
    p.add_argument("--resume", default="", help="resume full training state")
    # finetune LEAN (step 3)
    p.add_argument("--init-config", default="", help="model_lean.json (build lean model)")
    p.add_argument("--weights", default="", help="nap weights (lean .pth hoac dense)")
    # eval
    p.add_argument("--no-map", action="store_true", help="chon best theo val-loss thay vi mAP")
    p.add_argument("--map-every", default=1, type=int, help="tinh mAP moi N epoch")
    p.add_argument("--conf-thresh", default=0.001, type=float)
    p.add_argument("--nms-thresh", default=0.5, type=float)
    # wandb
    p.add_argument("--wandb", action="store_true", help="log len wandb")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-save-model", action="store_true",
                   help="upload best checkpoint; off logs metrics only")
    p.add_argument("--env-file", default="", help="optional path to .env")
    p.add_argument("--map-train", action="store_true",
                   help="tinh ca mAP tren TRAIN (cham, chay full train set moi epoch)")
    return p.parse_args()


def save_ckpt(path, model, optimizer, epoch, val_loss, metric, run_config=None):
    torch.save({
        "model": model.state_dict(),
        "config": model.config(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "metric": metric,
        "run_config": run_config,
    }, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, epochs, warmup_epochs, warmup_decay):
    warmup_epochs = min(max(warmup_epochs, 0), max(epochs - 1, 0))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_decay, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print("Device:", device)

    run_name = args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/"))
    wandb_run = init_wandb(
        args.wandb, vars(args), args.wandb_project or "prune-one-stage",
        run_name, env_file=args.env_file, group=args.wandb_group,
        job_type="dense-yolo",
    )

    train_ds = YoloDataset(args.data_path, "train", args.input_size, augment=args.augment)
    val_ds = YoloDataset(args.data_path, "val", args.input_size, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))
    if wandb_run:
        wandb_run.config.update({
            "train_images": len(train_ds),
            "val_images": len(val_ds),
            "resolved_device": str(device),
        }, allow_val_change=True)

    # --- build model: lean (step-3) hoac dense (step-1) ---
    if args.init_config:
        with open(args.init_config) as f:
            cfg = json.load(f)
        cfg["num_classes"] = args.num_classes
        model = build_model(cfg).to(device)
        print(f"[Step 3] Lean model from {args.init_config}")
    else:
        model = YoloModel(input_size=args.input_size, backbone=args.backbone,
                          num_classes=args.num_classes, neck_out=args.neck_out,
                          pretrained_backbone=args.pretrained_backbone).to(device)

    if args.weights and os.path.isfile(args.weights):
        wck = torch.load(args.weights, map_location="cpu", weights_only=False)
        sd = wck["model"] if isinstance(wck, dict) and "model" in wck else wck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded weights {args.weights} (missing={len(missing)}, unexpected={len(unexpected)})")

    criterion = YoloLoss(grid_size=model.grid_size, num_classes=args.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer, args.epochs, args.lr_warmup_epochs, args.lr_warmup_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args.backbone} | params={n_params/1e6:.2f}M | grid={model.grid_size}")

    print("Running initial validation...", flush=True)
    initial_val = evaluate(model, criterion, val_loader, device)
    if args.no_map:
        best = -initial_val["loss"]
        initial_tag = "-val_loss"
    else:
        print("Running initial mAP evaluation...", flush=True)
        initial_map = evaluate_map(
            model, val_loader, device, args.num_classes,
            args.conf_thresh, args.nms_thresh,
        )
        best = initial_map["mAP@0.5:0.95"]
        initial_tag = "mAP@0.5:0.95"
        print(f"  -> initial val mAP@0.5={initial_map['mAP@0.5']:.4f} "
              f"mAP@0.5:0.95={best:.4f}")
    best_path = os.path.join(args.output_dir, "model_best.pth")
    save_ckpt(
        best_path, model, optimizer, start_epoch - 1, initial_val["loss"], best,
        run_config=vars(args),
    )
    print(f"Initial best ({initial_tag}={best:.4f}) -> model_best.pth")
    if wandb_run:
        initial_log = {
            "epoch": start_epoch - 1,
            "val/loss": initial_val["loss"],
            "best_metric": best,
        }
        if not args.no_map:
            initial_log.update({f"val/{key}": value for key, value in initial_map.items()})
        wandb_run.log(initial_log)

    for epoch in range(start_epoch, args.epochs):
        tr = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch,
                             args.epochs, scaler=scaler)
        scheduler.step()
        val = evaluate(model, criterion, val_loader, device)

        log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        log.update({f"train/{k}": v for k, v in tr.items()})    # train loss + components
        log.update({f"val/{k}": v for k, v in val.items()})     # val loss + components

        use_map = (not args.no_map) and (epoch % max(args.map_every, 1) == 0
                                         or epoch == args.epochs - 1)
        if use_map:
            m = evaluate_map(model, val_loader, device, args.num_classes,
                             args.conf_thresh, args.nms_thresh)
            log.update({f"val/{k}": v for k, v in m.items()})   # val mAP@0.5/0.75/0.9/0.5:0.95
            print(f"  -> val mAP@0.5={m['mAP@0.5']:.4f} mAP@0.9={m['mAP@0.9']:.4f} "
                  f"mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}")
            metric = m["mAP@0.5:0.95"]
            if args.map_train:   # ponytail: cham (chay full train set) -> opt-in
                mt = evaluate_map(model, train_loader, device, args.num_classes,
                                  args.conf_thresh, args.nms_thresh)
                log.update({f"train/{k}": v for k, v in mt.items()})
        elif args.no_map:
            metric = -val["loss"]
        else:
            metric = None

        if wandb_run:
            wandb_run.log(log)

        save_ckpt(os.path.join(args.output_dir, "model_last.pth"),
                  model, optimizer, epoch, val["loss"], metric,
                  run_config=vars(args))
        if metric is not None and metric > best:
            best = metric
            save_ckpt(
                best_path, model, optimizer, epoch, val["loss"], metric,
                run_config=vars(args),
            )
            if wandb_run and args.wandb_save_model:
                wandb_run.save(best_path, policy="now")
            tag = "mAP@0.5:0.95" if use_map else "-val_loss"
            print(f"  ** new best ({tag}={best:.4f}) -> model_best.pth")

    print(f"Done. Best metric={best:.4f}")
    if wandb_run:
        wandb_run.summary["best_metric"] = best
        wandb_run.summary["best_checkpoint"] = best_path
        wandb_run.finish()


if __name__ == "__main__":
    main()
