"""
norton.py - Apply NORTON-style CP decomposition inside this YOLOv1 benchmark.

This is the decomposition-based comparison path:
  dense YOLO checkpoint -> replace eligible 3x3 convs with CPD convs
                         -> optional finetune using the same train/eval stack

It deliberately does not use channel-pruning surgery, because NORTON changes
topology rather than selecting a smaller set of channels.
"""
import os
import json
import argparse

import torch

from models.yolo import build_model
from models.norton import decompose_yolo_model
from loss import YoloLoss
from data import YoloDataset
from engine import train_one_epoch, evaluate
from utils import evaluate_map


def resolve_device(req):
    req = (req or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available -> CPU")
        return torch.device("cpu")
    return torch.device(req)


def get_args():
    p = argparse.ArgumentParser(description="NORTON CP decomposition for YOLOv1")
    p.add_argument("--checkpoint", required=True, help="dense model_best.pth with config")
    p.add_argument("--data-path", default="", help="optional; required for finetune/eval")
    p.add_argument("-r", "--rank", default=6, type=int, help="CP decomposition rank")
    p.add_argument("--scope", default="all", choices=["all", "backbone", "neck"])
    p.add_argument("--n-iter-max", default=300, type=int)
    p.add_argument("--n-iter-singular-error", default=3, type=int)
    p.add_argument("--finetune-epochs", default=0, type=int)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight-decay", default=5e-4, type=float)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="./output/yolo_norton")
    p.add_argument("--no-map", action="store_true", help="select best by val loss")
    p.add_argument("--map-every", default=1, type=int)
    p.add_argument("--conf-thresh", default=0.001, type=float)
    p.add_argument("--nms-thresh", default=0.5, type=float)
    return p.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_config_and_weights(model, output_dir, stem):
    cfg_path = os.path.join(output_dir, f"{stem}.json")
    weight_path = os.path.join(output_dir, f"{stem}.pth")
    with open(cfg_path, "w") as f:
        json.dump(model.config(), f, indent=2)
    torch.save(model.state_dict(), weight_path)
    return weight_path, cfg_path


def save_ckpt(path, model, optimizer, epoch, val_loss, metric):
    torch.save({
        "model": model.state_dict(),
        "config": model.config(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "metric": metric,
    }, path)


def make_loaders(args, input_size, device):
    train_ds = YoloDataset(args.data_path, "train", input_size, augment=True)
    val_ds = YoloDataset(args.data_path, "val", input_size, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))
    return train_loader, val_loader


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print("Device:", device)

    ckpt = load_checkpoint(args.checkpoint)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "model" not in ckpt:
        raise ValueError("--checkpoint must be a train.py checkpoint with 'config' and 'model'")

    cfg = dict(ckpt["config"])
    if cfg.get("variant") == "norton":
        raise ValueError("Input checkpoint is already a NORTON variant; use a dense checkpoint.")

    dense = build_model(cfg).to(device)
    dense.load_state_dict(ckpt["model"])
    print(f"Loaded dense {cfg['backbone']} | input={cfg['input_size']} | classes={cfg['num_classes']}")

    model, replaced = decompose_yolo_model(
        dense, args.rank, args.scope, args.n_iter_max, args.n_iter_singular_error
    )
    model.to(device)
    print(f"NORTON rank={args.rank} scope={args.scope} | decomposed 3x3 layers={replaced}")

    init_weight, init_cfg = save_config_and_weights(model, args.output_dir, "model_norton")
    print(f"Saved NORTON topology: {init_weight} + {init_cfg}")

    if args.finetune_epochs <= 0:
        print("No finetune requested. Done.")
        return
    if not args.data_path:
        raise ValueError("--data-path is required when --finetune-epochs > 0")

    train_loader, val_loader = make_loaders(args, cfg["input_size"], device)
    criterion = YoloLoss(grid_size=model.grid_size, num_classes=cfg["num_classes"])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.finetune_epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best = float("-inf")
    for epoch in range(args.finetune_epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch,
                        args.finetune_epochs, scaler=scaler)
        scheduler.step()
        val = evaluate(model, criterion, val_loader, device)

        use_map = (not args.no_map) and (epoch % max(args.map_every, 1) == 0
                                         or epoch == args.finetune_epochs - 1)
        if use_map:
            m = evaluate_map(model, val_loader, device, cfg["num_classes"],
                             args.conf_thresh, args.nms_thresh)
            print(f"  -> val mAP@0.5={m['mAP@0.5']:.4f} "
                  f"mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}")
            metric = m["mAP@0.5:0.95"]
        else:
            metric = -val["loss"]

        save_ckpt(os.path.join(args.output_dir, "model_last.pth"),
                  model, optimizer, epoch, val["loss"], metric)
        if metric > best:
            best = metric
            save_ckpt(os.path.join(args.output_dir, "model_best.pth"),
                      model, optimizer, epoch, val["loss"], metric)
            tag = "mAP@0.5:0.95" if use_map else "-val_loss"
            print(f"  ** new best ({tag}={best:.4f}) -> model_best.pth")

    print(f"Done. Best metric={best:.4f}")


if __name__ == "__main__":
    main()
