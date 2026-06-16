"""
prune.py — Bi-level pruning (Song Han + Filter L1-inf-inf) + finetune + surgery.
Step 2 cua Pipeline 2 (standalone).

Quy trinh (lap prune_iters lan):
  sparsity tang dan -> Unstructured prune (sensitivity = mult * sparsity)
                    -> Structured prune  (prune_ratio = sparsity)
                    -> finetune (giu zero qua apply_masks)
Sau cung: model surgery -> luu model_lean.pth + model_lean.json (de step-3 finetune).

Component-aware: --scope {all, backbone, neck}.

Vi du:
  python prune.py --data-path ./fish \
      --checkpoint ./output/yolo/step1_dense/model_best.pth \
      --target-sparsity 0.5 --prune-iters 5 --finetune-epochs 3 \
      --scope all --output-dir ./output/yolo/step2_pruned
"""
import os
import argparse

import torch

from models.yolo import build_model
from loss import YoloLoss
from data import YoloDataset
from engine import train_one_epoch, evaluate
from pruning import UnstructuredPruner, StructuredPruner, convert_to_lean
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
    """Gioi han pham vi prune (component-aware)."""
    def __init__(self, model, which="all"):
        self.model = model
        self.which = which

    def get_prunable_layers(self, pruning_type="unstructured"):
        if self.which == "backbone":
            return self.model.get_backbone_prunable_layers(pruning_type)
        if self.which == "neck":
            return self.model.get_neck_prunable_layers(pruning_type)
        return self.model.get_prunable_layers(pruning_type)


def get_args():
    p = argparse.ArgumentParser(description="Bi-level prune YOLOv1")
    p.add_argument("--data-path", required=True)
    p.add_argument("--checkpoint", required=True, help="dense model_best.pth (co config)")
    p.add_argument("--target-sparsity", default=0.5, type=float)
    p.add_argument("--prune-iters", default=5, type=int)
    p.add_argument("--finetune-epochs", default=3, type=int)
    p.add_argument("--sensitivity-mult", default=2.0, type=float)
    p.add_argument("--scope", default="all", choices=["all", "backbone", "neck"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight-decay", default=5e-4, type=float)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="./output/yolo_pruned")
    p.add_argument("--no-map", action="store_true", help="bo qua mAP (chi val-loss)")
    p.add_argument("--conf-thresh", default=0.001, type=float)
    p.add_argument("--nms-thresh", default=0.5, type=float)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print("Device:", device)

    # --- load dense model tu checkpoint (kem config) ---
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded dense {cfg['backbone']} | input={cfg['input_size']} | classes={cfg['num_classes']}")

    # --- data ---
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
    u_pruner = UnstructuredPruner(scope)
    s_pruner = StructuredPruner(scope)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    nc = cfg["num_classes"]

    def report_map(tag, mdl):
        if args.no_map:
            return
        m = evaluate_map(mdl, val_loader, device, nc, args.conf_thresh, args.nms_thresh)
        print(f"  [{tag}] mAP@0.5={m['mAP@0.5']:.4f}  mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}")

    p0 = sum(p.numel() for p in model.parameters())
    print(f"\nBaseline params={p0/1e6:.2f}M | scope={args.scope}")
    evaluate(model, criterion, val_loader, device)
    report_map("baseline", model)

    # --- vong bi-level pruning ---
    for it in range(args.prune_iters):
        sparsity = args.target_sparsity * (it + 1) / args.prune_iters
        print(f"\n=== Prune iter {it+1}/{args.prune_iters} | sparsity={sparsity:.3f} ===")
        u_pruner.prune(sensitivity=args.sensitivity_mult * sparsity)
        s_pruner.prune(prune_ratio=sparsity)
        print(f"  Song Han global sparsity={u_pruner.global_sparsity():.3f}")

        for ep in range(args.finetune_epochs):
            train_one_epoch(model, criterion, train_loader, optimizer, device, ep,
                            args.finetune_epochs, pruner=u_pruner, scaler=scaler)
        evaluate(model, criterion, val_loader, device)

    # --- surgery -> lean model ---
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
