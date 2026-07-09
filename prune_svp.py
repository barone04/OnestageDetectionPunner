"""
prune_svp.py — SVP (Singular Value Pruning) + finetune + surgery.
Thay the bi-level (Song Han + L1-inf-inf) bang GAM tu SVP-main.

Quy trinh:
  1. GAM phan bo kenh structured theo target_rate (singular values)
  2. Finetune (giu zero qua apply_masks)
  3. Model surgery -> model_lean.pth + model_lean.json

Vi du:
  python prune_svp.py --data-path ./NewDeepfish \\
      --checkpoint ./output/yolo_resnet18/step1_dense/model_best.pth \\
      --target-rate 0.5 --finetune-epochs 5 --scope all \\
      --output-dir ./output/yolo_resnet18/step2_svp
"""
import os
import argparse

import torch

from models.yolo import build_model
from loss import YoloLoss
from data import YoloDataset
from engine import train_one_epoch, evaluate
from pruning import SVPPruner, convert_to_lean
from utils import evaluate_map


def resolve_device(req):
    req = (req or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available -> CPU")
        return torch.device("cpu")
    return torch.device(req)


class Scope:
    def __init__(self, model, which="all"):
        self.model = model
        self.which = which

    def get_prunable_layers(self, pruning_type="structured"):
        if self.which == "backbone":
            return self.model.get_backbone_prunable_layers(pruning_type)
        if self.which == "neck":
            return self.model.get_neck_prunable_layers(pruning_type)
        return self.model.get_prunable_layers(pruning_type)


def get_args():
    p = argparse.ArgumentParser(description="SVP prune YOLOv1")
    p.add_argument("--data-path", required=True)
    p.add_argument("--checkpoint", required=True, help="dense model_best.pth (co config)")
    p.add_argument("--target-rate", default=0.5, type=float,
                   help="SVP compress rate (ty le kenh bi cat), mac dinh 0.5")
    p.add_argument("--finetune-epochs", default=5, type=int)
    p.add_argument("--scope", default="all", choices=["all", "backbone", "neck"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight-decay", default=5e-4, type=float)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="./output/yolo_svp")
    p.add_argument("--no-map", action="store_true")
    p.add_argument("--conf-thresh", default=0.001, type=float)
    p.add_argument("--nms-thresh", default=0.5, type=float)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print("Device:", device)

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded dense {cfg['backbone']} | input={cfg['input_size']} | classes={cfg['num_classes']}")

    input_size = cfg["input_size"]
    train_ds = YoloDataset(args.data_path, "train", input_size, augment=True)
    val_ds = YoloDataset(args.data_path, "val", input_size, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))

    criterion = YoloLoss(grid_size=model.grid_size, num_classes=cfg["num_classes"])
    scope = Scope(model, args.scope)
    svp_pruner = SVPPruner(scope)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(args.finetune_epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    nc = cfg["num_classes"]

    def report_map(tag, mdl):
        if args.no_map:
            return
        m = evaluate_map(mdl, val_loader, device, nc, args.conf_thresh, args.nms_thresh)
        print(f"  [{tag}] mAP@0.5={m['mAP@0.5']:.4f}  mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}")

    p0 = sum(p.numel() for p in model.parameters())
    print(f"\nBaseline params={p0/1e6:.2f}M | scope={args.scope} | method=SVP-GAM")
    evaluate(model, criterion, val_loader, device)
    report_map("baseline", model)

    print(f"\n=== SVP-GAM prune | target_rate={args.target_rate:.3f} ===")
    svp_pruner.prune(target_rate=args.target_rate)

    if args.finetune_epochs > 0:
        print(f"\n=== Finetune {args.finetune_epochs} epoch(s) ===")
        for ep in range(args.finetune_epochs):
            train_one_epoch(model, criterion, train_loader, optimizer, device, ep,
                            args.finetune_epochs, pruner=svp_pruner, scaler=scaler)
            scheduler.step()
        evaluate(model, criterion, val_loader, device)
        report_map("after finetune", model)

    print("\n=== Model Surgery ===")
    save_path = os.path.join(args.output_dir, "model_lean.pth")
    lean, lean_cfg = convert_to_lean(model, save_path=save_path)
    lean.to(device)
    p1 = sum(p.numel() for p in lean.parameters())
    print(f"Lean saved: {save_path}")
    print(f"Params: {p0/1e6:.2f}M -> {p1/1e6:.2f}M  (-{(1-p1/p0)*100:.1f}%)")

    print("\nLean model eval:")
    lean_crit = YoloLoss(grid_size=lean.grid_size, num_classes=nc)
    evaluate(lean, lean_crit, val_loader, device)
    report_map("lean", lean)
    print(f"\nDone. Lean: {save_path}  +  {save_path.replace('.pth', '.json')}")


if __name__ == "__main__":
    main()
