"""Validate the original NORTON pipeline on ResNet-56 / CIFAR-10.

Pipeline:
  dense ResNet-56 -> CP decomposition -> long finetune -> saved checkpoint
                  -> one-shot factor pruning -> long finetune -> evaluation
"""
import argparse
import ast
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data.cifar10 import make_cifar10_loaders
from models.norton_cifar import (
    build_cifar_resnet56,
    decompose_cifar_resnet56,
    normalize_compress_rate,
    prune_cifar_resnet56,
)
from utils.tracking import init_wandb, load_dotenv


def get_args():
    parser = argparse.ArgumentParser(
        description="Original NORTON pipeline on ResNet-56 / CIFAR-10"
    )
    parser.add_argument("--data-path", default="./cifar10-data")
    parser.add_argument("--dense-checkpoint", default="",
                        help="optional original/root or rebuilt dense checkpoint")
    parser.add_argument("--dense-epochs", default=0, type=int,
                        help="train dense model when no checkpoint is supplied")
    parser.add_argument("--decompose-finetune-epochs", default=400, type=int)
    parser.add_argument("--prune-finetune-epochs", default=400, type=int)
    parser.add_argument("-r", "--rank", default=6, type=int)
    parser.add_argument(
        "-cpr", "--compress-rate", default="[0.]+[0.18]*29",
        help="30-value NORTON ResNet-56 compression expression",
    )
    parser.add_argument("--criterion", default="pabs", choices=["pabs", "csa", "vbd"])
    parser.add_argument("--copy-bn", action="store_true",
                        help="copy sliced BN; off matches the released NORTON pruning code")
    parser.add_argument("--n-iter-max", default=300, type=int)
    parser.add_argument("--n-iter-singular-error", default=3, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--lr-warmup-epochs", default=5, type=int)
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--latency-warmup", default=10, type=int)
    parser.add_argument("--latency-iters", default=50, type=int)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--output-dir", default="./output/norton_resnet56_cifar10")
    parser.add_argument("--no-wandb", action="store_true",
                        help="disable W&B (default: auto-enable if WANDB_API_KEY in .env)")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--env-file", default="", help="optional path to .env")
    return parser.parse_args()


def _number(node):
    if (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _number(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("Compression expression contains a non-numeric value")


def _eval_rate_expression(node):
    if isinstance(node, ast.List):
        return [float(_number(item)) for item in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_rate_expression(node.left) + _eval_rate_expression(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if isinstance(node.left, ast.List):
            values = _eval_rate_expression(node.left)
            repeat_value = _number(node.right)
        elif isinstance(node.right, ast.List):
            values = _eval_rate_expression(node.right)
            repeat_value = _number(node.left)
        else:
            raise ValueError("Only list * integer is supported")
        if int(repeat_value) != repeat_value:
            raise ValueError("Compression list repeat must be an integer")
        repeat = int(repeat_value)
        if repeat < 0:
            raise ValueError("Compression list repeat must be non-negative")
        return values * repeat
    raise ValueError("Unsupported compression expression")


def parse_compress_rate(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        rates = _eval_rate_expression(tree.body)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid --compress-rate: {expression}") from exc
    return normalize_compress_rate(rates)


def resolve_device(request):
    request = (request or "auto").lower()
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, using CPU")
        return torch.device("cpu")
    return torch.device(request)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    normalized = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        normalized[key.replace(".feature.", ".")] = value
    return normalized


def save_checkpoint(path, model, optimizer, epoch, top1, stage):
    config = model.config()
    config["stage"] = stage
    torch.save({
        "model": model.state_dict(),
        "state_dict": model.state_dict(),
        "config": config,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "top1": float(top1),
    }, path)


def topk_accuracy(logits, targets, topk=(1, 5)):
    max_k = min(max(topk), logits.size(1))
    _, prediction = logits.topk(max_k, dim=1, largest=True, sorted=True)
    correct = prediction.t().eq(targets.view(1, -1))
    values = []
    for k in topk:
        k = min(k, logits.size(1))
        values.append(float(correct[:k].reshape(-1).float().sum().item()))
    return values


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs,
                    verbose=True):
    model.train()
    loss_sum = top1_sum = top5_sum = count = 0.0
    print_every = max(len(loader) // 10, 1)
    for index, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch = images.size(0)
        acc1, acc5 = topk_accuracy(logits, targets)
        loss_sum += float(loss.item()) * batch
        top1_sum += acc1
        top5_sum += acc5
        count += batch
        if verbose and index % print_every == 0:
            print(f"  [E{epoch + 1}/{epochs} {index:>4}/{len(loader)}] "
                  f"loss={loss.item():.4f} top1={100.0 * acc1 / batch:.2f}",
                  flush=True)
    return {
        "loss": loss_sum / max(count, 1),
        "top1": 100.0 * top1_sum / max(count, 1),
        "top5": 100.0 * top5_sum / max(count, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, verbose=True):
    model.eval()
    loss_sum = top1_sum = top5_sum = count = 0.0
    total = max(len(loader), 1)
    print_every = max(total // 10, 1)
    for index, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch = images.size(0)
        acc1, acc5 = topk_accuracy(logits, targets)
        loss_sum += float(loss.item()) * batch
        top1_sum += acc1
        top5_sum += acc5
        count += batch
        if verbose and (index % print_every == 0 or index + 1 == total):
            print(f"  [VAL {index + 1}/{total}]", flush=True)
    metrics = {
        "loss": loss_sum / max(count, 1),
        "top1": 100.0 * top1_sum / max(count, 1),
        "top5": 100.0 * top5_sum / max(count, 1),
    }
    if verbose:
        print(f"  -> val loss={metrics['loss']:.4f} top1={metrics['top1']:.3f} "
              f"top5={metrics['top5']:.3f}", flush=True)
    return metrics


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


def finetune_phase(model, train_loader, val_loader, criterion, device, args,
                   epochs, stem, stage, wandb_run=None, metric_prefix=None,
                   short_tag=None):
    if epochs <= 0:
        raise ValueError(f"{stage} requires a positive epoch budget")
    tag = short_tag or stem
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer, epochs, args.lr_warmup_epochs, args.lr_warmup_decay
    )
    best_path = os.path.join(args.output_dir, f"{stem}_best.pth")
    last_path = os.path.join(args.output_dir, f"{stem}_last.pth")
    initial = evaluate(model, val_loader, criterion, device, verbose=False)
    best_top1 = initial["top1"]
    save_checkpoint(best_path, model, optimizer, -1, best_top1, stage)

    print(f"\n=== {stage}: {epochs} epochs ===")
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
            verbose=False,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device, verbose=False)
        improved = val_metrics["top1"] > best_top1
        if improved:
            best_top1 = val_metrics["top1"]
        if wandb_run:
            wandb_run.log({
                "train_loss": train_metrics["loss"],
                "val_acc": val_metrics["top1"],
                "best_acc": best_top1,
                "val_loss": val_metrics["loss"],
                "lr": optimizer.param_groups[0]["lr"],
            })
        save_checkpoint(last_path, model, optimizer, epoch, val_metrics["top1"], stage)
        if improved:
            save_checkpoint(best_path, model, optimizer, epoch, best_top1, stage)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  {tag} [ep {epoch + 1}/{epochs}] loss={train_metrics['loss']:.3f} "
                  f"val_loss={val_metrics['loss']:.3f} acc={val_metrics['top1']:.2f}% "
                  f"best={best_top1:.2f}%", flush=True)

    best_checkpoint = load_checkpoint(best_path)
    model.load_state_dict(checkpoint_state(best_checkpoint), strict=True)
    print(f"Reloaded best {stage}: {best_path}")
    return model, best_path, best_top1


@torch.no_grad()
def profile_model(model, device, warmup, iterations):
    model = model.eval().to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    sample = torch.randn(1, 3, 32, 32, device=device)
    macs = None
    try:
        from thop import profile
        macs, _ = profile(copy.deepcopy(model), inputs=(sample,), verbose=False)
    except Exception as exc:
        print(f"  [thop] MACs unavailable: {exc}")

    for _ in range(max(warmup, 0)):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(iterations, 1)):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / max(iterations, 1)
    return {
        "params": int(params),
        "macs": None if macs is None else float(macs),
        "latency_ms": latency_ms,
        "fps": 1000.0 / latency_ms,
    }


def print_profile(name, metrics, profile):
    text = f"{name:<12} top1={metrics['top1']:.3f} params={profile['params'] / 1e6:.3f}M"
    if profile["macs"] is not None:
        text += f" MACs={profile['macs'] / 1e6:.3f}M"
    text += f" latency={profile['latency_ms']:.3f}ms FPS={profile['fps']:.2f}"
    print(text)


def main():
    args = get_args()
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    compress_rate = parse_compress_rate(args.compress_rate)
    if compress_rate[0] != 0.0:
        raise ValueError("Published NORTON ResNet-56 configs keep compress_rate[0] at 0")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print("Device:", device)
    print("Compression rate:", compress_rate)
    load_dotenv(args.env_file or None)
    wandb_enabled = (not args.no_wandb) and bool(os.environ.get("WANDB_API_KEY"))
    run_name = args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/"))
    wandb_run = init_wandb(
        wandb_enabled, vars(args), args.wandb_project or "resnet56-cifar10-norton",
        run_name, env_file=args.env_file, group=args.wandb_group,
        job_type="norton-resnet56-cifar10",
    )

    train_loader, val_loader = make_cifar10_loaders(
        args.data_path, args.batch_size, args.workers,
        download=not args.no_download, pin_memory=device.type == "cuda",
    )
    if wandb_run:
        wandb_run.config.update({
            "resolved_device": str(device),
            "train_images": len(train_loader.dataset),
            "val_images": len(val_loader.dataset),
            "resolved_compress_rate": compress_rate,
        }, allow_val_change=True)
    criterion = nn.CrossEntropyLoss().to(device)

    dense = build_cifar_resnet56().to(device)
    if args.dense_checkpoint:
        checkpoint = load_checkpoint(args.dense_checkpoint)
        dense.load_state_dict(checkpoint_state(checkpoint), strict=True)
        print(f"Loaded dense checkpoint: {args.dense_checkpoint}")
    elif args.dense_epochs > 0:
        dense, dense_path, _ = finetune_phase(
            dense, train_loader, val_loader, criterion, device, args,
            args.dense_epochs, "dense", "dense training", wandb_run,
            short_tag="dense",
        )
        print(f"Trained dense checkpoint: {dense_path}")
    else:
        raise ValueError("Provide --dense-checkpoint or set --dense-epochs > 0")
    dense_metrics = evaluate(dense, val_loader, criterion, device)
    dense_params = sum(p.numel() for p in dense.parameters())
    print(f"Baseline (dense) | params={dense_params / 1e6:.3f}M | top1={dense_metrics['top1']:.2f}%")

    decomposed, replaced = decompose_cifar_resnet56(
        dense, args.rank, args.n_iter_max, args.n_iter_singular_error
    )
    decomposed = decomposed.to(device)
    print(f"Decomposed 3x3 convolutions: {replaced}")
    decomposed_init_metrics = evaluate(decomposed, val_loader, criterion, device)
    save_checkpoint(
        os.path.join(args.output_dir, "decomposed_init.pth"),
        decomposed, None, -1, decomposed_init_metrics["top1"], "decomposed init",
    )
    decomposed, decomposed_path, decomposed_top1 = finetune_phase(
        decomposed, train_loader, val_loader, criterion, device, args,
        args.decompose_finetune_epochs, "decomposed", "decomposition finetune",
        wandb_run, short_tag="decomp",
    )

    # Match the original two-script boundary: prune only a reloaded checkpoint.
    decomposed_checkpoint = load_checkpoint(decomposed_path)
    decomposed = build_cifar_resnet56(decomposed_checkpoint["config"]).to(device)
    decomposed.load_state_dict(checkpoint_state(decomposed_checkpoint), strict=True)
    print(f"Pruning from trained checkpoint: {decomposed_path}")

    pruned = prune_cifar_resnet56(
        decomposed, compress_rate, criterion=args.criterion, copy_bn=args.copy_bn
    )
    pruned_init_metrics = evaluate(pruned, val_loader, criterion, device, verbose=False)
    pruned_params = sum(p.numel() for p in pruned.parameters())
    reduction = (1.0 - pruned_params / dense_params) * 100.0
    print(f"\nPruned | params={pruned_params / 1e6:.3f}M (-{reduction:.1f}%) | "
          f"acc truoc finetune={pruned_init_metrics['top1']:.2f}%")
    save_checkpoint(
        os.path.join(args.output_dir, "pruned_init.pth"),
        pruned, None, -1, pruned_init_metrics["top1"], "pruned init",
    )
    pruned, final_path, final_top1 = finetune_phase(
        pruned, train_loader, val_loader, criterion, device, args,
        args.prune_finetune_epochs, "model", "post-pruning finetune",
        wandb_run, "pruned", short_tag="ft",
    )
    final_metrics = evaluate(pruned, val_loader, criterion, device)

    results = {
        "config": vars(args),
        "compress_rate": compress_rate,
        "dense": dense_metrics,
        "decomposed_init": decomposed_init_metrics,
        "decomposed_best_top1": decomposed_top1,
        "pruned_init": pruned_init_metrics,
        "final": final_metrics,
        "final_best_top1": final_top1,
        "decomposed_checkpoint": decomposed_path,
        "final_checkpoint": final_path,
    }
    profiles = None
    if not args.skip_profile:
        print("\n=== Complexity and latency (batch=1) ===")
        profiles = {
            "dense": profile_model(dense, device, args.latency_warmup, args.latency_iters),
            "decomposed": profile_model(
                decomposed, device, args.latency_warmup, args.latency_iters
            ),
            "final": profile_model(pruned, device, args.latency_warmup, args.latency_iters),
        }
        results["profiles"] = profiles
        print_profile("Dense", dense_metrics, profiles["dense"])
        print_profile("Decomposed", {"top1": decomposed_top1}, profiles["decomposed"])
        print_profile("Final", final_metrics, profiles["final"])

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as file:
        json.dump(results, file, indent=2)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Results: {results_path}")
    if wandb_run:
        wandb_run.summary["baseline_acc"] = dense_metrics["top1"]
        wandb_run.summary["final_acc"] = final_metrics["top1"]
        wandb_run.summary["final_checkpoint"] = final_path
        wandb_run.summary["results_path"] = results_path
        if os.path.exists(final_path):
            wandb_run.save(final_path)   # chi upload best cuoi cung (pruned finetune)
        if not args.skip_profile:
            dprof, fprof = profiles["dense"], profiles["final"]
            wandb_run.summary["pruned_params_M"] = fprof["params"] / 1e6
            wandb_run.summary["params_reduction_pct"] = (
                1.0 - fprof["params"] / dprof["params"]) * 100.0
            if dprof["macs"] and fprof["macs"]:
                wandb_run.summary["pruned_macs_M"] = fprof["macs"] / 1e6
                wandb_run.summary["macs_reduction_pct"] = (
                    1.0 - fprof["macs"] / dprof["macs"]) * 100.0
        wandb_run.finish()


if __name__ == "__main__":
    main()
