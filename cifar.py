"""
cifar.py — Benchmark bi-level pruning tren CIFAR-10/100 (VGG16-BN, ResNet-56).

Muc dich: chung minh method tai lap duoc tren benchmark chuan, de so voi so PUBLISHED
cua L1 / HRank / GAL / SSS / SPSRC... ma khong phai chay lai tung baseline.

Protocol doi chieu tu repo CORING (github.com/vantienpham/CORING, dong HRank):
  - transform: RandomCrop(32, pad=4) + HFlip + Normalize(mean .4914/.4822/.4465,
    std .2023/.1994/.2010). KHONG cutout/autoaug/mixup — them vao la mat tinh so sanh.
  - dense (train tu scratch): SGD 0.9, wd 1e-4, batch 128,
    VGG16-BN 164 ep lr 0.1 x0.1@81,122 | ResNet-56 200 ep lr 0.1 x0.1@60,120,160
  - finetune sau prune: xem FINETUNE_PROTO — "coring" (300 ep, lr 0.01, @150,225,
    wd 5e-3) hoac "spsrc" (80 ep, lr 0.001, @20, wd 1e-4). CHON TRUOC KHI SO BANG:
    so cua bi-level voi 80 ep khong so duoc voi so cua CORING voi 300 ep.
  - GATE: mode=dense dung lai neu top-1 lech >0.5% so voi DENSE_TARGET. Dense sai
    moc thi moi so lieu prune ben tren deu vo nghia.

Vi du:
  python cifar.py --mode dense --model resnet56 --output-dir ./output/cifar/r56_dense
  python cifar.py --mode prune --model resnet56 --target-sparsity 0.5 \
      --checkpoint ./output/cifar/r56_dense/model_best.pth \
      --output-dir ./output/cifar/r56_p50
  python cifar.py --mode finetune --lean ./output/cifar/r56_p50/model_lean.pth \
      --epochs 80 --output-dir ./output/cifar/r56_p50_ft
"""
import os
import json
import time
import argparse

import torch
import torch.nn as nn

from models.cifar import CifarVGG, CifarResNet, build_cifar_model
from pruning import UnstructuredPruner, StructuredPruner
from pruning.surgery_cifar import convert_to_lean_cifar

# Normalize: DUNG DUNG gia tri cua HRank/CORING (main/data/cifar10.py cua repo CORING).
# Ban (0.2470,0.2435,0.2616) cung pho bien nhung KHAC -> doi la mat tinh so sanh.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# lich train dense tu scratch theo tung model (lr 0.1, wd 1e-4)
DENSE_SCHED = {
    "vgg16":    dict(epochs=164, milestones=[81, 122]),
    "resnet56": dict(epochs=200, milestones=[60, 120, 160]),
}

# recipe finetune SAU prune — moi dong paper mot kieu, phai chon truoc khi so bang
FINETUNE_PROTO = {
    # main/scripts/resnet56_cifar10/vbd.sh cua CORING
    "coring": dict(lr=0.01, epochs=300, milestones=[150, 225], weight_decay=5e-3),
    # SPSRC: "same optimization setting as baseline but lr 0.001, decaying at 20th epoch"
    "spsrc":  dict(lr=0.001, epochs=80, milestones=[20], weight_decay=1e-4),
}

# Muc tieu dense phai dat truoc khi duoc phep prune (top-1 %), theo tung dong paper
DENSE_TARGET = {
    ("resnet56", "cifar10"): 93.26,     # HRank/CORING baseline
    ("vgg16", "cifar10"): 93.96,        # HRank/CORING baseline (head="hrank")
}


def resolve_device(req):
    req = (req or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available -> CPU")
        return torch.device("cpu")
    return torch.device(req)


def get_loaders(data_path, dataset, batch_size, workers):
    import torchvision
    import torchvision.transforms as T

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    cls = torchvision.datasets.CIFAR100 if dataset == "cifar100" else torchvision.datasets.CIFAR10
    tr = cls(data_path, train=True, download=True, transform=train_tf)
    te = cls(data_path, train=False, download=True, transform=test_tf)
    return (torch.utils.data.DataLoader(tr, batch_size, shuffle=True, num_workers=workers,
                                        pin_memory=True, drop_last=False),
            torch.utils.data.DataLoader(te, batch_size, shuffle=False, num_workers=workers,
                                        pin_memory=True))


def build(model_name, num_classes, prune_set="all", head="hrank"):
    if model_name == "vgg16":
        return CifarVGG("vgg16", num_classes, prune_set=prune_set, head=head)
    return CifarResNet(int(model_name.replace("resnet", "")), num_classes)


def count_cost(model, device):
    """Params (M) + MACs (M). thop khong bat duoc mask nen goi TRUOC/SAU surgery."""
    params = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),), verbose=False)
        return params, macs / 1e6
    except Exception as e:                       # ponytail: thop optional, khong chan train
        print(f"[WARN] thop loi ({e}) -> bo qua MACs")
        return params, float("nan")


def train_one_epoch(model, criterion, loader, optimizer, device, epoch, total, pruner=None):
    model.train()
    loss_sum = correct = seen = 0
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if pruner is not None:                   # giu sparsity Song Han
            pruner.apply_masks()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    print(f"  [E{epoch+1}/{total}] train loss={loss_sum/seen:.4f} acc={100*correct/seen:.2f}% "
          f"({time.time()-t0:.1f}s)")
    return loss_sum / seen, 100 * correct / seen


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    loss_sum = correct = seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    acc = 100 * correct / seen
    print(f"  -> val loss={loss_sum/seen:.4f} top1={acc:.2f}%")
    return loss_sum / seen, acc


def run_training(model, loaders, args, device, epochs, lr, milestones, tag, pruner=None,
                 wandb=None, cfg=None):
    train_loader, val_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    best = 0.0
    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, criterion, train_loader, optimizer,
                                          device, ep, epochs, pruner)
        va_loss, va_acc = evaluate(model, criterion, val_loader, device)
        scheduler.step()
        if wandb:
            wandb.log({"epoch": ep, "lr": optimizer.param_groups[0]["lr"],
                       f"{tag}/train_loss": tr_loss, f"{tag}/train_acc": tr_acc,
                       f"{tag}/val_loss": va_loss, f"{tag}/val_top1": va_acc})
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "config": cfg or model.config(),
                        "top1": best, "epoch": ep},
                       os.path.join(args.output_dir, "model_best.pth"))
    print(f"[{tag}] best top1 = {best:.2f}%")
    return best


def get_args():
    p = argparse.ArgumentParser(description="Bi-level pruning benchmark tren CIFAR")
    p.add_argument("--mode", default="dense", choices=["dense", "prune", "finetune"])
    p.add_argument("--model", default="resnet56", choices=["vgg16", "resnet56", "resnet20",
                                                           "resnet32", "resnet44", "resnet110"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-path", default="./data")
    p.add_argument("--output-dir", default="./output/cifar")
    p.add_argument("--checkpoint", default=None, help="dense ckpt (mode=prune)")
    p.add_argument("--lean", default=None, help="lean ckpt (mode=finetune)")
    # protocol chuan — chi doi khi co ly do, doi la mat tinh so sanh voi so published
    p.add_argument("--batch-size", default=128, type=int)
    p.add_argument("--lr", default=None, type=float, help="None -> 0.1 (dense) / 0.001 (finetune)")
    p.add_argument("--epochs", default=None, type=int, help="None -> lich chuan theo model")
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--device", default="auto")
    # bi-level
    p.add_argument("--target-sparsity", default=0.5, type=float)
    p.add_argument("--prune-iters", default=5, type=int)
    p.add_argument("--prune-finetune-epochs", default=3, type=int)
    p.add_argument("--sensitivity-mult", default=2.0, type=float)
    p.add_argument("--vgg-prune-set", default="all", choices=["all", "l1a"],
                   help="l1a = cung tap layer voi L1 (VGG-16-pruned-A)")
    p.add_argument("--vgg-head", default="hrank", choices=["hrank", "single"],
                   help="hrank = bien the cua HRank/CORING (14.98M); single = cua SPSRC (14.72M)")
    p.add_argument("--protocol", default="coring", choices=["coring", "spsrc"],
                   help="recipe finetune sau prune, xem FINETUNE_PROTO")
    p.add_argument("--skip-gate", action="store_true",
                   help="bo qua kiem tra dense vs published (chi dung khi smoke test)")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    num_classes = 100 if args.dataset == "cifar100" else 10
    print(f"Device: {device} | {args.dataset} | {args.model} | seed={args.seed}")

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(name=os.path.basename(args.output_dir.rstrip("/")), config=vars(args))

    loaders = get_loaders(args.data_path, args.dataset, args.batch_size, args.workers)
    sched = DENSE_SCHED.get(args.model, DENSE_SCHED["resnet56"])

    # ---------------- dense ----------------
    if args.mode == "dense":
        model = build(args.model, num_classes, args.vgg_prune_set, args.vgg_head).to(device)
        pm, mm = count_cost(model, device)
        print(f"Dense: {pm:.2f}M params | {mm:.2f}M MACs")
        best = run_training(model, loaders, args, device,
                            args.epochs or sched["epochs"], args.lr or 0.1,
                            sched["milestones"], "dense", wandb=wandb)
        target = None if args.skip_gate else DENSE_TARGET.get((args.model, args.dataset))
        if target:
            gap = best - target
            ok = abs(gap) <= 0.5
            print(f"\n[GATE] dense={best:.2f}% vs published={target:.2f}% "
                  f"({gap:+.2f}) -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise SystemExit("Dense lech >0.5% so voi published -> KHONG duoc prune "
                                 "va KHONG duoc trich bang published. Sua training truoc.")

    # ---------------- bi-level prune ----------------
    elif args.mode == "prune":
        assert args.checkpoint, "--checkpoint la bat buoc voi mode=prune"
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = build_cifar_model(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        p0, m0 = count_cost(model, device)
        print(f"Loaded dense: {p0:.2f}M params | {m0:.2f}M MACs | top1={ckpt.get('top1')}")

        proto = FINETUNE_PROTO[args.protocol]
        criterion = nn.CrossEntropyLoss()
        u_pruner, s_pruner = UnstructuredPruner(model), StructuredPruner(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr or proto["lr"],
                                    momentum=args.momentum,
                                    weight_decay=proto["weight_decay"])

        for it in range(args.prune_iters):
            sparsity = args.target_sparsity * (it + 1) / args.prune_iters
            print(f"\n=== Prune iter {it+1}/{args.prune_iters} | sparsity={sparsity:.3f} ===")
            u_pruner.prune(sensitivity=args.sensitivity_mult * sparsity)
            s_pruner.prune(prune_ratio=sparsity)
            print(f"  Song Han global sparsity={u_pruner.global_sparsity():.3f}")
            for ep in range(args.prune_finetune_epochs):
                train_one_epoch(model, criterion, loaders[0], optimizer, device,
                                ep, args.prune_finetune_epochs, pruner=u_pruner)
            evaluate(model, criterion, loaders[1], device)

        print("\n=== Model Surgery ===")
        save_path = os.path.join(args.output_dir, "model_lean.pth")
        lean, lean_cfg = convert_to_lean_cifar(model, save_path=save_path)
        lean.to(device)
        p1, m1 = count_cost(lean, device)
        print(f"Params {p0:.2f}M -> {p1:.2f}M (-{(1-p1/p0)*100:.2f}%) | "
              f"MACs {m0:.2f}M -> {m1:.2f}M (-{(1-m1/m0)*100:.2f}%)")
        evaluate(lean, criterion, loaders[1], device)
        with open(os.path.join(args.output_dir, "cost.json"), "w") as f:
            json.dump({"params_M": p1, "macs_M": m1,
                       "params_red_pct": (1 - p1 / p0) * 100,
                       "macs_red_pct": (1 - m1 / m0) * 100}, f, indent=2)
        if wandb:
            wandb.log({"params_M": p1, "macs_M": m1,
                       "params_red_pct": (1 - p1 / p0) * 100,
                       "macs_red_pct": (1 - m1 / m0) * 100})

    # ---------------- finetune lean ----------------
    else:
        assert args.lean, "--lean la bat buoc voi mode=finetune"
        with open(args.lean.replace(".pth", ".json")) as f:
            cfg = json.load(f)
        model = build_cifar_model(cfg).to(device)
        model.load_state_dict(torch.load(args.lean, map_location="cpu"))
        pm, mm = count_cost(model, device)
        print(f"Lean: {pm:.2f}M params | {mm:.2f}M MACs")
        proto = FINETUNE_PROTO[args.protocol]
        args.weight_decay = proto["weight_decay"]
        print(f"Protocol '{args.protocol}': {proto}")
        run_training(model, loaders, args, device,
                     args.epochs or proto["epochs"], args.lr or proto["lr"],
                     proto["milestones"], "finetune", wandb=wandb, cfg=cfg)

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
