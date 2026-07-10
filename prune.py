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
import json
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
    # --- chon thuat toan pruning (bai minh vs baseline) ---
    p.add_argument("--method", default="bilevel", choices=["bilevel", "coring"],
                   help="bilevel = bai minh (Song Han + L1-inf-inf, mask); "
                        "coring = baseline (HOSVD+VBD+min_sum, k-shot surgery-first)")
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
    # wandb (log ca cac buoc prune: mAP + loss moi shot/iter)
    p.add_argument("--wandb", action="store_true", help="log len wandb")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print("Device:", device)

    wb = None
    if args.wandb:
        import wandb  # env WANDB_API_KEY/PROJECT/ENTITY tu lo (giong train.py)
        wb = wandb.init(name=os.path.basename(args.output_dir.rstrip("/")),
                        config=vars(args))

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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    nc = cfg["num_classes"]

    def report_map(tag, mdl):
        if args.no_map:
            return None
        m = evaluate_map(mdl, val_loader, device, nc, args.conf_thresh, args.nms_thresh)
        print(f"  [{tag}] mAP@0.5={m['mAP@0.5']:.4f}  mAP@0.5:0.95={m['mAP@0.5:0.95']:.4f}")
        return m

    def wb_log(step, sparsity, val, m, params_m):
        """Log 1 buoc prune len wandb (bo qua neu tat)."""
        if wb is None:
            return
        log = {"prune/step": step, "prune/sparsity": sparsity,
               "prune/params_M": params_m}
        if val:
            log.update({f"prune/val_{k}": v for k, v in val.items()})
        if m:
            log["prune/mAP@0.5"] = m["mAP@0.5"]
            log["prune/mAP@0.5:0.95"] = m["mAP@0.5:0.95"]
        wb.log(log)

    p0 = sum(p.numel() for p in model.parameters())
    print(f"\nMethod={args.method} | Baseline params={p0/1e6:.2f}M | scope={args.scope}")
    base_val = evaluate(model, criterion, val_loader, device)
    base_map = report_map("baseline", model)
    wb_log(0, 0.0, base_val, base_map, p0 / 1e6)

    if args.method == "coring":
        # ===== CORING baseline: k-shot, cat -> SURGERY NGAY -> finetune model nho =====
        from pruning.coring import CoringPruner
        cur, prev = model, 0.0
        for it in range(args.prune_iters):
            # ramp CORING theo code goc (get_cpr): cpr[shot] = r/(K-shot) -> phi tuyen, don ve cuoi
            target_abs = args.target_sparsity / (args.prune_iters - it)        # sparsity tuyet doi/goc
            ratio = (target_abs - prev) / (1.0 - prev) if prev < 1.0 else 0.0  # ty le tren model HIEN TAI
            print(f"\n=== CORING shot {it+1}/{args.prune_iters} | "
                  f"target={target_abs:.3f} | ratio_on_cur={ratio:.3f} ===")
            CoringPruner(Scope(cur, args.scope)).prune(prune_ratio=ratio)
            cur, _ = convert_to_lean(cur)          # surgery ngay -> model nho hon
            cur.to(device)
            # calibrate finetune tren model nho (cung budget finetune_epochs/shot nhu bi-level)
            opt = torch.optim.SGD(cur.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=max(args.finetune_epochs, 1))
            crit = YoloLoss(grid_size=cur.grid_size, num_classes=nc)
            for ep in range(args.finetune_epochs):
                train_one_epoch(cur, crit, train_loader, opt, device, ep,
                                args.finetune_epochs, scaler=scaler)
                sch.step()
            val = evaluate(cur, crit, val_loader, device)
            m = report_map(f"shot{it+1}", cur)
            wb_log(it + 1, target_abs, val, m, sum(p.numel() for p in cur.parameters()) / 1e6)
            prev = target_abs
        lean = cur
    else:
        # ===== Bi-level (bai minh): mask-finetune lap -> surgery 1 lan cuoi =====
        u_pruner = UnstructuredPruner(scope)
        s_pruner = StructuredPruner(scope)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        # warm-restart: lr reset dau moi vong prune (T_0 = so epoch finetune/vong)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(args.finetune_epochs, 1))
        for it in range(args.prune_iters):
            sparsity = args.target_sparsity * (it + 1) / args.prune_iters
            print(f"\n=== Prune iter {it+1}/{args.prune_iters} | sparsity={sparsity:.3f} ===")
            u_pruner.prune(sensitivity=args.sensitivity_mult * sparsity)
            s_pruner.prune(prune_ratio=sparsity)
            print(f"  Song Han global sparsity={u_pruner.global_sparsity():.3f}")

            for ep in range(args.finetune_epochs):
                train_one_epoch(model, criterion, train_loader, optimizer, device, ep,
                                args.finetune_epochs, pruner=u_pruner, scaler=scaler)
                scheduler.step()
            val = evaluate(model, criterion, val_loader, device)
            m = report_map(f"iter{it+1}", model)
            wb_log(it + 1, sparsity, val, m, sum(p.numel() for p in model.parameters()) / 1e6)

        print("\n=== Model Surgery ===")
        lean, _ = convert_to_lean(model)
        lean.to(device)

    # --- luu lean + eval (chung cho ca 2 method) ---
    save_path = os.path.join(args.output_dir, "model_lean.pth")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(lean.state_dict(), save_path)
    with open(save_path.replace(".pth", ".json"), "w") as f:
        json.dump(lean.config(), f, indent=2)
    p1 = sum(p.numel() for p in lean.parameters())
    print(f"\nLean saved: {save_path}")
    print(f"Params: {p0/1e6:.2f}M -> {p1/1e6:.2f}M  (-{(1-p1/p0)*100:.1f}%)")

    print("\nLean model eval:")
    lean_crit = YoloLoss(grid_size=lean.grid_size, num_classes=nc)
    lean_val = evaluate(lean, lean_crit, val_loader, device)
    lean_map = report_map("lean", lean)
    wb_log(args.prune_iters + 1, args.target_sparsity, lean_val, lean_map, p1 / 1e6)
    if wb is not None:
        wb.summary["final_params_M"] = p1 / 1e6
        wb.summary["params_reduction_pct"] = (1 - p1 / p0) * 100
        if lean_map:
            wb.summary["final_mAP@0.5"] = lean_map["mAP@0.5"]
            wb.summary["final_mAP@0.5:0.95"] = lean_map["mAP@0.5:0.95"]
        wb.finish()
    print(f"\nDone. Lean: {save_path}  +  {save_path.replace('.pth', '.json')}")


if __name__ == "__main__":
    main()
