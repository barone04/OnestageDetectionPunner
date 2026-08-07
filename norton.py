"""
norton.py - Apply NORTON-style CP decomposition inside this YOLOv1 benchmark.

This follows the original two-stage NORTON pipeline:
  dense YOLO checkpoint -> replace eligible 3x3 convs with CPD convs
                         -> finetune and save decomposed checkpoint
                         -> one-shot CPD factor/channel pruning + rebuild
                         -> finetune the pruned model

The pruning materialization is NORTON-specific: CPD layers are selected through
their factors, then a smaller NORTON topology is rebuilt and sliced.
"""
import os
import json
import time
import argparse
import random

import numpy as np
import torch

from models.yolo import build_model
from models.norton import (
    decompose_yolo_model,
    normalize_norton_compress_rate,
    prune_norton_model,
)
from loss import YoloLoss
from data import YoloDataset
from engine import train_one_epoch, evaluate
from utils import evaluate_map
from utils.tracking import init_wandb, load_dotenv


def resolve_device(req):
    req = (req or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available -> CPU")
        return torch.device("cpu")
    return torch.device(req)


@torch.no_grad()
def profile_model(model, input_size, device):
    """Params + MACs (thop, optional) tai batch=1 de tinh reduction."""
    import copy
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    macs = None
    try:
        from thop import profile
        sample = torch.randn(1, 3, input_size, input_size, device=device)
        macs, _ = profile(copy.deepcopy(model), inputs=(sample,), verbose=False)
        macs = float(macs)
    except Exception as exc:
        print(f"  [thop] MACs unavailable: {exc}")
    return params, macs


def get_args():
    p = argparse.ArgumentParser(description="NORTON CP decomposition for YOLOv1")
    p.add_argument("--checkpoint", required=True, help="dense model_best.pth with config")
    p.add_argument("--data-path", default="", help="optional; required for finetune/eval")
    p.add_argument("-r", "--rank", default=6, type=int, help="CP decomposition rank")
    p.add_argument("--scope", default="all", choices=["all", "backbone", "neck"])
    p.add_argument("--prune-ratio", default=0.0, type=float,
                   help="uniform structured ratio (legacy shortcut for all prune groups)")
    p.add_argument("--compress-rate", "--compress_rate", default=None,
                   help="per-group rates in original NORTON syntax, e.g. '[0.1]*8+[0.2]*5'")
    p.add_argument("--criterion", default="pabs", choices=["pabs", "csa", "vbd"],
                   help="NORTON factor-similarity criterion")
    p.add_argument("--copy-bn", action="store_true",
                   help="copy sliced BN; off matches released NORTON ResNet-56 pruning")
    p.add_argument("--n-iter-max", default=300, type=int)
    p.add_argument("--n-iter-singular-error", default=3, type=int)
    p.add_argument("--decompose-finetune-epochs", default=None, type=int,
                   help="finetune budget after CP decomposition")
    p.add_argument("--prune-finetune-epochs", default=None, type=int,
                   help="finetune budget after one-shot NORTON pruning")
    p.add_argument("--finetune-epochs", default=None, type=int,
                   help="legacy alias: use this value for both finetune phases")
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight-decay", default=5e-4, type=float)
    p.add_argument("--lr-warmup-epochs", default=5, type=int)
    p.add_argument("--lr-warmup-decay", default=0.01, type=float)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--output-dir", default="./output/yolo_norton")
    p.add_argument("--no-map", action="store_true", help="select best by val loss")
    p.add_argument("--map-every", default=1, type=int)
    p.add_argument("--conf-thresh", default=0.001, type=float)
    p.add_argument("--nms-thresh", default=0.5, type=float)
    p.add_argument("--no-wandb", action="store_true",
                   help="disable W&B (default: auto-enable if WANDB_API_KEY in .env)")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--env-file", default="", help="optional path to .env")
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


def save_ckpt(path, model, optimizer, epoch, val_loss, metric,
              run_config=None, stage=None):
    torch.save({
        "model": model.state_dict(),
        "config": model.config(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "metric": metric,
        "run_config": run_config,
        "stage": stage,
    }, path)


def resolve_phase_epochs(args):
    decompose_epochs = args.decompose_finetune_epochs
    prune_epochs = args.prune_finetune_epochs
    if args.finetune_epochs is not None:
        if decompose_epochs is None:
            decompose_epochs = args.finetune_epochs
        if prune_epochs is None:
            prune_epochs = args.finetune_epochs
    return int(decompose_epochs or 0), int(prune_epochs or 0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, epochs, warmup_epochs, warmup_decay):
    """Warmup + cosine schedule used by the native ResNet-56 rebuild."""
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


def make_grad_scaler(device):
    enabled = device.type == "cuda"
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def finetune_phase(model, criterion, train_loader, val_loader, device, args,
                   epochs, checkpoint_stem, phase_name, wandb_run=None,
                   metric_prefix=None):
    """Run one independent NORTON recovery phase and reload its best checkpoint."""
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer, epochs, args.lr_warmup_epochs, args.lr_warmup_decay
    )
    scaler = make_grad_scaler(device)
    log_prefix = metric_prefix or checkpoint_stem

    best_path = os.path.join(args.output_dir, f"{checkpoint_stem}_best.pth")
    last_path = os.path.join(args.output_dir, f"{checkpoint_stem}_last.pth")

    print(f"Running initial validation for {phase_name}...", flush=True)
    initial_val = evaluate(model, criterion, val_loader, device)
    if args.no_map:
        best = -initial_val["loss"]
        initial_tag = "-val_loss"
    else:
        print(f"Running initial mAP for {phase_name}...", flush=True)
        initial_map = evaluate_map(
            model, val_loader, device, model.num_classes,
            args.conf_thresh, args.nms_thresh,
        )
        best = initial_map["mAP@0.5:0.95"]
        initial_tag = "mAP@0.5:0.95"
        print(f"  -> initial mAP@0.5={initial_map['mAP@0.5']:.4f} "
              f"mAP@0.5:0.95={best:.4f}")
    save_ckpt(
        best_path, model, optimizer, -1, initial_val["loss"], best,
        run_config=vars(args), stage=phase_name,
    )
    print(f"  Initial {phase_name} best ({initial_tag}={best:.4f})")
    if wandb_run:
        initial_log = {
            "phase": log_prefix,
            "phase_epoch": -1,
            f"{log_prefix}/val/loss": initial_val["loss"],
            f"{log_prefix}/best_metric": best,
        }
        if not args.no_map:
            initial_log.update({
                f"{log_prefix}/val/{key}": value
                for key, value in initial_map.items()
            })
        wandb_run.log(initial_log)

    print(f"\n=== {phase_name}: {epochs} epochs ===")

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            epochs, scaler=scaler,
        )
        scheduler.step()
        val = evaluate(model, criterion, val_loader, device)

        metric = None
        use_map = (not args.no_map) and (
            epoch % max(args.map_every, 1) == 0 or epoch == epochs - 1
        )
        if use_map:
            result = evaluate_map(model, val_loader, device, model.num_classes,
                                  args.conf_thresh, args.nms_thresh)
            print(f"  -> val mAP@0.5={result['mAP@0.5']:.4f} "
                  f"mAP@0.5:0.95={result['mAP@0.5:0.95']:.4f}")
            metric = result["mAP@0.5:0.95"]
        elif args.no_map:
            metric = -val["loss"]

        if wandb_run:
            log = {
                "phase": log_prefix,
                "phase_epoch": epoch,
                f"{log_prefix}/lr": optimizer.param_groups[0]["lr"],
                f"{log_prefix}/best_metric": max(best, metric) if metric is not None else best,
            }
            log.update({
                f"{log_prefix}/train/{key}": value
                for key, value in train_metrics.items()
            })
            log.update({
                f"{log_prefix}/val/{key}": value
                for key, value in val.items()
            })
            if use_map:
                log.update({
                    f"{log_prefix}/val/{key}": value
                    for key, value in result.items()
                })
            wandb_run.log(log)

        save_ckpt(
            last_path, model, optimizer, epoch, val["loss"], metric,
            run_config=vars(args), stage=phase_name,
        )
        if metric is not None and metric > best:
            best = metric
            save_ckpt(
                best_path, model, optimizer, epoch, val["loss"], metric,
                run_config=vars(args), stage=phase_name,
            )
            tag = "mAP@0.5:0.95" if use_map else "-val_loss"
            print(f"  ** new {phase_name} best ({tag}={best:.4f})")

    if not os.path.isfile(best_path):
        raise RuntimeError(f"{phase_name} did not produce a best checkpoint")

    best_ckpt = load_checkpoint(best_path)
    model.load_state_dict(best_ckpt["model"], strict=True)
    print(f"Reloaded best {phase_name} checkpoint: {best_path}")
    if wandb_run:
        wandb_run.summary[f"{log_prefix}/best_metric"] = best
    return model, best_path, best


def main():
    args = get_args()
    decompose_epochs, prune_epochs = resolve_phase_epochs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print("Device:", device)
    load_dotenv(args.env_file or None)
    wandb_enabled = (not args.no_wandb) and bool(os.environ.get("WANDB_API_KEY"))
    run_name = args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/"))
    wandb_run = init_wandb(
        wandb_enabled, vars(args), args.wandb_project or "yolo-deepfish-norton",
        run_name, env_file=args.env_file, group=args.wandb_group,
        job_type="norton-yolo",
    )
    if wandb_run:
        wandb_run.define_metric("phase_epoch")
        for prefix in ("decomposed", "pruned"):
            wandb_run.define_metric(f"{prefix}/*", step_metric="phase_epoch")

    ckpt = load_checkpoint(args.checkpoint)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "model" not in ckpt:
        raise ValueError("--checkpoint must be a train.py checkpoint with 'config' and 'model'")

    cfg = dict(ckpt["config"])
    if cfg.get("variant") == "norton":
        raise ValueError("Input checkpoint is already a NORTON variant; use a dense checkpoint.")

    dense = build_model(cfg).to(device)
    dense.load_state_dict(ckpt["model"])
    print(f"Loaded dense {cfg['backbone']} | input={cfg['input_size']} | classes={cfg['num_classes']}")
    dense_params, dense_macs = profile_model(dense, cfg["input_size"], device)

    compress_rates, prune_layout = normalize_norton_compress_rate(
        dense, prune_ratio=args.prune_ratio,
        compress_rate=args.compress_rate, scope=args.scope,
    )
    pruning_requested = any(rate > 0.0 for rate in compress_rates)
    if pruning_requested:
        allocation_path = os.path.join(args.output_dir, "compress_rate_layout.json")
        with open(allocation_path, "w") as f:
            json.dump([
                {"index": index, "layer": layer, "compress_rate": rate}
                for index, (layer, rate) in enumerate(zip(prune_layout, compress_rates))
            ], f, indent=2)
        mode = "per-layer" if args.compress_rate is not None else "uniform"
        print(f"NORTON sparsity allocation: {mode}, groups={len(prune_layout)}")
        print(f"Saved ordered compress-rate layout: {allocation_path}")
        if wandb_run:
            wandb_run.config.update({
                "resolved_compress_rate": compress_rates,
                "prune_layout": prune_layout,
            }, allow_val_change=True)

    _t = time.perf_counter()
    model, replaced = decompose_yolo_model(
        dense, args.rank, args.scope, args.n_iter_max, args.n_iter_singular_error
    )
    prune_secs = time.perf_counter() - _t   # prune wall-clock: CP-decompose
    model.to(device)
    print(f"NORTON rank={args.rank} scope={args.scope} | decomposed 3x3 layers={replaced}")

    init_weight, init_cfg = save_config_and_weights(model, args.output_dir, "model_norton")
    print(f"Saved NORTON topology: {init_weight} + {init_cfg}")

    if decompose_epochs <= 0:
        if pruning_requested:
            raise ValueError(
                "Original NORTON pipeline requires --decompose-finetune-epochs > 0 "
                "before pruning"
            )
        print("No decomposition finetune requested. Saved initialized CPD model only.")
        if wandb_run:
            wandb_run.finish()
        return
    if not args.data_path:
        raise ValueError("--data-path is required for NORTON finetuning")

    train_loader, val_loader = make_loaders(args, cfg["input_size"], device)
    if wandb_run:
        wandb_run.config.update({
            "resolved_device": str(device),
            "train_images": len(train_loader.dataset),
            "val_images": len(val_loader.dataset),
            "decomposed_layers": replaced,
        }, allow_val_change=True)
    criterion = YoloLoss(grid_size=model.grid_size, num_classes=cfg["num_classes"])
    model, decomposed_best_path, decomposed_best = finetune_phase(
        model, criterion, train_loader, val_loader, device, args,
        decompose_epochs, "decomposed", "decomposition finetune", wandb_run,
    )

    if not pruning_requested:
        final_ckpt = load_checkpoint(decomposed_best_path)
        final_path = os.path.join(args.output_dir, "model_best.pth")
        torch.save(final_ckpt, final_path)
        print(f"Done (decomposition only). Best metric={decomposed_best:.4f}")
        print(f"Final checkpoint: {final_path}")
        if wandb_run:
            wandb_run.summary["final_checkpoint"] = final_path
            if os.path.exists(final_path):
                wandb_run.save(final_path)
            wandb_run.finish()
        return
    if prune_epochs <= 0:
        raise ValueError(
            "Original NORTON pipeline requires --prune-finetune-epochs > 0 "
            "after pruning"
        )

    # Preserve the original script boundary: pruning starts from the saved,
    # fine-tuned decomposed checkpoint rather than the current in-memory model.
    decomposed_ckpt = load_checkpoint(decomposed_best_path)
    model = build_model(decomposed_ckpt["config"]).to(device)
    model.load_state_dict(decomposed_ckpt["model"], strict=True)
    print(f"Pruning from trained decomposed checkpoint: {decomposed_best_path}")

    _t = time.perf_counter()
    model, _ = prune_norton_model(
        model, args.prune_ratio, criterion=args.criterion, scope=args.scope,
        compress_rate=(compress_rates if args.compress_rate is not None else None),
        copy_bn=args.copy_bn,
    )
    prune_secs += time.perf_counter() - _t   # + one-shot factor prune
    print(f"[prune wall-clock] norton (decompose + prune, excl. finetune) = {prune_secs:.2f}s")
    pruned_weight, pruned_cfg_path = save_config_and_weights(
        model, args.output_dir, "model_norton_pruned"
    )
    if args.compress_rate is None:
        allocation = f"uniform ratio={args.prune_ratio:.3f}"
    else:
        allocation = f"per-layer groups={len(compress_rates)}"
    bn_mode = "copy sliced BN" if args.copy_bn else "fresh BN (ResNet-56 protocol)"
    print(f"NORTON one-shot prune {allocation} criterion={args.criterion} | {bn_mode}")
    print(f"Saved pruned NORTON topology: {pruned_weight} + {pruned_cfg_path}")

    criterion = YoloLoss(grid_size=model.grid_size, num_classes=cfg["num_classes"])
    model, final_path, final_best = finetune_phase(
        model, criterion, train_loader, val_loader, device, args,
        prune_epochs, "model", "post-pruning finetune", wandb_run, "pruned",
    )
    final_params, final_macs = profile_model(model, cfg["input_size"], device)
    params_drop = (1.0 - final_params / dense_params) * 100.0
    line = f"\nPruned | params={final_params / 1e6:.3f}M (-{params_drop:.1f}%)"
    macs_drop = None
    if dense_macs and final_macs:
        macs_drop = (1.0 - final_macs / dense_macs) * 100.0
        line += f" | MACs={final_macs / 1e9:.3f}G (-{macs_drop:.1f}%)"
    print(line)
    print(f"Done. Decomposed best={decomposed_best:.4f} | final best={final_best:.4f}")
    print(f"Final checkpoint: {final_path}")
    if wandb_run:
        wandb_run.summary["decomposed_best_metric"] = decomposed_best
        wandb_run.summary["final_best_metric"] = final_best
        wandb_run.summary["final_checkpoint"] = final_path
        if os.path.exists(final_path):
            wandb_run.save(final_path)   # chi upload best cuoi cung (pruned finetune)
        wandb_run.summary["pruned_params_M"] = final_params / 1e6
        wandb_run.summary["params_reduction_pct"] = params_drop
        if macs_drop is not None:
            wandb_run.summary["pruned_macs_M"] = final_macs / 1e6
            wandb_run.summary["macs_reduction_pct"] = macs_drop
        wandb_run.finish()


if __name__ == "__main__":
    main()
