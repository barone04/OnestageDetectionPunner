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

import torch

from models.yolo import YoloModel, build_model
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
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--device", default="auto")
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
    p.add_argument("--wandb-project", default="",
                   help="ten project wandb, vd yolov1_resnet18_bilevel_Deepfish")
    p.add_argument("--dense-ref", default="",
                   help="dense ckpt de log %MACs/%params reduction (summary)")
    p.add_argument("--map-train", action="store_true",
                   help="tinh ca mAP tren TRAIN (cham, chay full train set moi epoch)")
    return p.parse_args()


@torch.no_grad()
def count_macs(model, input_size):
    """forward-hook MACs (shape-only): Conv2d + Linear."""
    dev = next(model.parameters()).device
    total = [0]
    hooks = []
    def ch(m, i, o):
        _, _, H, W = o.shape
        kh, kw = m.kernel_size
        total[0] += (m.in_channels // m.groups) * m.out_channels * kh * kw * H * W
    def lh(m, i, o):
        total[0] += m.in_features * m.out_features
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv2d):
            hooks.append(mod.register_forward_hook(ch))
        elif isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lh))
    was_training = model.training
    model.eval()
    model(torch.zeros(1, 3, input_size, input_size, device=dev))
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return total[0]


def save_ckpt(path, model, optimizer, epoch, val_loss, metric):
    torch.save({
        "model": model.state_dict(),
        "config": model.config(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "metric": metric,
    }, path)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print("Device:", device)

    wandb = None
    if args.wandb:
        import wandb  # env WANDB_API_KEY/ENTITY tu lo (vd da set tren FPT)
        wandb.init(project=(args.wandb_project or None),
                   name=os.path.basename(args.output_dir.rstrip("/")), config=vars(args))

    train_ds = YoloDataset(args.data_path, "train", args.input_size, augment=args.augment)
    val_ds = YoloDataset(args.data_path, "val", args.input_size, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=YoloDataset.collate_fn, pin_memory=(device.type == "cuda"))

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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

    # --- log %MACs/%params reduction (summary) khi co dense ref ---
    if wandb and args.dense_ref and os.path.isfile(args.dense_ref):
        dref = torch.load(args.dense_ref, map_location="cpu", weights_only=False)
        dmodel = build_model(dref["config"])
        dmodel.load_state_dict(dref["model"])
        d_p, d_m = sum(p.numel() for p in dmodel.parameters()), count_macs(dmodel, args.input_size)
        l_m = count_macs(model, args.input_size)
        wandb.run.summary.update({
            "pruned_params_M": n_params / 1e6,
            "params_reduction_pct": (1 - n_params / d_p) * 100,
            "pruned_macs_M": l_m / 1e6,
            "macs_reduction_pct": (1 - l_m / d_m) * 100,
        })
        print(f"[reduction] MACs -{(1 - l_m / d_m) * 100:.1f}%  params -{(1 - n_params / d_p) * 100:.1f}%")

    best = float("-inf")
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
        else:
            metric = -val["loss"]

        if wandb:
            wandb.log(log)

        save_ckpt(os.path.join(args.output_dir, "model_last.pth"),
                  model, optimizer, epoch, val["loss"], metric)
        if metric > best:
            best = metric
            best_path = os.path.join(args.output_dir, "model_best.pth")
            save_ckpt(best_path, model, optimizer, epoch, val["loss"], metric)
            if wandb:
                wandb.save(best_path)
            tag = "mAP@0.5:0.95" if use_map else "-val_loss"
            print(f"  ** new best ({tag}={best:.4f}) -> model_best.pth")

    print(f"Done. Best metric={best:.4f}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
