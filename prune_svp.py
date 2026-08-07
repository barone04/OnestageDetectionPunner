"""
prune_svp.py — SVP (Singular Value Pruning) — HUONG B (khop resnet56 + fair):
  GAM (so kenh) -> GEM chon filter (nuclear-norm) -> COPY weight -> FINETUNE (step 3).

Bam Paper Algorithm 3 (copy + finetune), dong bo voi resnet56 Huong B va fair voi
CORING/NORTON (deu copy weight + finetune).
  1. GAM (singular values) phan bo SO KENH giu moi layer.
  2. GEM: chon FILTER nao giu (nuclear-norm top-K per-filter, SCALABLE cho conv lon
     YOLO; greedy-GEM O(N^2*SVD) bat kha thi tren 512-1024 kenh nen dung top-K).
  3. Surgery -> lean COPY weight (index-exact tu dense) -> luu model_lean.pth co weight thuc.
  4. Step 3 (train.py --init-config + --weights) FINETUNE tu weight copy.

Vi du:
  python prune_svp.py --checkpoint step1_dense/model_best.pth --target-rate 0.5 \\
      --scope all --output-dir step2_svp
"""
import os
import json
import time
import argparse

import torch

from models.yolo import build_model
from pruning import SVPPruner, convert_to_lean


def select_filters_topk_nuclear(weight, num_keep):
    """Huong B selection: giu num_keep filter co NUCLEAR NORM lon nhat (per-filter).
    O(N) small-SVD -> scalable cho conv lon YOLO (greedy-GEM O(N^2*SVD) bat kha thi).
    Cung tinh than 'giu filter salient theo singular values'. Tra bool mask theo out-channel."""
    n = weight.size(0)
    k = max(1, min(int(num_keep), n))
    norms = torch.empty(n, device=weight.device)
    for i in range(n):
        norms[i] = torch.linalg.svdvals(weight[i].reshape(weight[i].size(0), -1)).sum()
    mask = torch.zeros(n, dtype=torch.bool, device=weight.device)
    mask[torch.topk(norms, k).indices] = True
    return mask


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
        description="SVP prune YOLOv1 — Huong B (GAM + GEM + copy weight; finetune o step3)")
    p.add_argument("--checkpoint", required=True, help="dense model_best.pth (GAM + GEM + copy weight)")
    p.add_argument("--target-rate", default=0.5, type=float,
                   help="SVP compress rate (ty le bi cat), mac dinh 0.5")
    p.add_argument("--budget", default="flops", choices=["flops", "channels"],
                   help="target-rate theo flops (mac dinh) hoac channels")
    p.add_argument("--scope", default="all", choices=["all", "backbone", "neck"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="./output/yolo_svp")
    # accept-and-ignore (tuong thich run_e2e_svp.sh; finetune thuc hien o step3 train.py):
    p.add_argument("--data-path", default="")
    p.add_argument("--finetune-epochs", default=0, type=int, help="(finetune o step3 train.py)")
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
    print(f"\nBaseline params={p0/1e6:.2f}M | scope={args.scope} | method=SVP (Huong B: GAM+GEM+copy)")

    # --- 1) GAM: so kenh giu moi layer ---
    t_prune = time.perf_counter()   # do prune wall-clock (GAM + GEM + surgery)
    print(f"\n=== SVP-GAM allocation | target_rate={args.target_rate:.3f} | budget={args.budget} ===")
    layers, channels_to_keep, compress_rates = svp_pruner.compute_allocation(
        args.target_rate, use_flops=(args.budget == "flops"))

    # --- GEM: chon FILTER nao giu (nuclear-norm top-K per-filter) -> de copy weight ---
    print("=== GEM: chon filter (nuclear-norm top-K) tren tung layer ===")
    for layer, k in zip(layers, channels_to_keep):
        keep_mask = select_filters_topk_nuclear(layer.conv.weight.data, max(1, int(k)))
        layer.mask_handler.update(keep_mask.float().to(layer.conv.weight.device))
        layer.mask_handler.apply(layer.conv)

    # --- 2) Surgery -> lean COPY weight (index-exact tu dense; Huong B) ---
    print("\n=== Surgery -> lean (COPY weight tu dense) ===")
    lean, lean_cfg = convert_to_lean(model, save_path=None)
    lean.to(device)
    p1 = sum(p.numel() for p in lean.parameters())
    prune_secs = time.perf_counter() - t_prune
    print(f"[prune wall-clock] SVP (GAM+GEM+surgery) = {prune_secs:.2f}s")

    # --- 3) Luu lean (CO weight copy) cho step 3 FINETUNE ---
    save_path = os.path.join(args.output_dir, "model_lean.pth")
    torch.save({"model": lean.state_dict(), "config": lean_cfg}, save_path)
    with open(save_path.replace(".pth", ".json"), "w") as f:
        json.dump(lean_cfg, f, indent=2)

    print(f"[GAM] target_rate={args.target_rate} -> giu {sum(channels_to_keep)} kenh "
          f"tren {len(layers)} layer structured")
    print(f"Lean saved: {save_path} (+json)")
    print(f"Params: {p0/1e6:.2f}M -> {p1/1e6:.2f}M  (-{(1-p1/p0)*100:.1f}%)")
    print("\n[Huong B] GEM chon filter + COPY weight -> FINETUNE o Step 3 "
          "(train.py --init-config + --weights model_lean.pth).")
    print("Dong bo voi resnet56 + fair voi CORING/NORTON: GAM + GEM + copy weight + finetune.")


if __name__ == "__main__":
    main()
