"""
prune_svp.py — SVP (Singular Value Pruning) — HUONG A (dong bo voi resnet56):
  GAM config (KHONG GEM) -> build lean FRESH -> train FROM SCRATCH (o step 3).

SVP released trains pruned arch tu scratch; GEM chi dung khi copy+finetune.
De DONG BO voi ban resnet56 (train-from-scratch), YOLO cung lam vay:
  1. GAM (singular values) phan bo SO KENH giu moi layer (KHONG chon filter nao).
  2. Build lean config -> luu model_lean.json (+ .pth FRESH init, KHONG copy weight).
  3. Step 3 (train.py --init-config, KHONG --weights) train lean TU DAU.

Vi du:
  python prune_svp.py --checkpoint step1_dense/model_best.pth --target-rate 0.5 \\
      --scope all --output-dir step2_svp
"""
import os
import json
import argparse

import torch

from models.yolo import build_model
from pruning import SVPPruner, convert_to_lean


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
    p = argparse.ArgumentParser(
        description="SVP-GAM prune YOLOv1 — Huong A (config + train-from-scratch, khong GEM)")
    p.add_argument("--checkpoint", required=True, help="dense model_best.pth (co config) — CHI de GAM")
    p.add_argument("--target-rate", default=0.5, type=float,
                   help="SVP compress rate (ty le bi cat), mac dinh 0.5")
    p.add_argument("--budget", default="flops", choices=["flops", "channels"],
                   help="target-rate theo flops (mac dinh) hoac channels")
    p.add_argument("--scope", default="all", choices=["all", "backbone", "neck"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="./output/yolo_svp")
    # accept-and-ignore (tuong thich run_e2e_svp.sh; Huong A KHONG finetune o buoc nay):
    p.add_argument("--data-path", default="")
    p.add_argument("--finetune-epochs", default=0, type=int, help="(bo qua o Huong A)")
    p.add_argument("--batch-size", default=16, type=int, help="(bo qua)")
    p.add_argument("--workers", default=4, type=int, help="(bo qua)")
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
    print(f"Loaded dense {cfg['backbone']} | input={cfg['input_size']} | "
          f"classes={cfg['num_classes']} (CHI de tinh GAM singular values)")

    scope = Scope(model, args.scope)
    svp_pruner = SVPPruner(scope, input_size=cfg.get("input_size", 448))
    p0 = sum(p.numel() for p in model.parameters())
    print(f"\nBaseline params={p0/1e6:.2f}M | scope={args.scope} | method=SVP-GAM (Huong A)")

    # --- 1) GAM config (KHONG GEM): so kenh giu moi layer ---
    print(f"\n=== SVP-GAM allocation | target_rate={args.target_rate:.3f} | budget={args.budget} ===")
    layers, channels_to_keep, compress_rates = svp_pruner.compute_allocation(
        args.target_rate, use_flops=(args.budget == "flops"))

    # Train-from-scratch -> which-filter KHONG quan trong (khong copy weight) ->
    # giu FIRST-K filter (K tu GAM) de surgery ra dung SHAPE, KHONG dung GEM.
    for layer, k in zip(layers, channels_to_keep):
        w = layer.conv.weight
        n = w.size(0)
        mask = torch.zeros(n, device=w.device)
        mask[: max(1, int(k))] = 1.0
        layer.mask_handler.update(mask)
        layer.mask_handler.apply(layer.conv)

    # --- 2) Surgery lay lean CONFIG; build FRESH (KHONG copy weight) ---
    print("\n=== Build lean config (surgery) -> FRESH init (train from scratch) ===")
    _, lean_cfg = convert_to_lean(model, save_path=None)
    lean = build_model(lean_cfg).to(device)   # FRESH init — KHONG copy weight tu dense
    p1 = sum(p.numel() for p in lean.parameters())

    # --- 3) Luu config (+ fresh weight) cho step 3 train-from-scratch ---
    save_path = os.path.join(args.output_dir, "model_lean.pth")
    torch.save({"model": lean.state_dict(), "config": lean_cfg}, save_path)
    with open(save_path.replace(".pth", ".json"), "w") as f:
        json.dump(lean_cfg, f, indent=2)

    print(f"[GAM] target_rate={args.target_rate} -> giu {sum(channels_to_keep)} kenh "
          f"tren {len(layers)} layer structured")
    print(f"Lean config saved: {save_path} (+json)")
    print(f"Params: {p0/1e6:.2f}M -> {p1/1e6:.2f}M  (-{(1-p1/p0)*100:.1f}%)")
    print("\n[Huong A] Lean FRESH init -> TRAIN FROM SCRATCH o Step 3 "
          "(train.py --init-config, KHONG --weights).")
    print("Dong bo voi resnet56: GAM config, KHONG GEM, train-from-scratch.")


if __name__ == "__main__":
    main()
