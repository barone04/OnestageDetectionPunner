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
  - NGAN SACH EPOCH: baseline (CORING) la one-shot -> prune 1 lan roi finetune 300 ep.
    Bi-level la iterative -> tieu prune_iters * prune_finetune_epochs epoch NGAY TRONG
    vong prune. So epoch do duoc ghi vao model_lean.json va mode=finetune TRU RA, nen
    tong epoch sau khi roi dense checkpoint bang dung baseline. --ignore-budget de tat.

Vi du:
  python cifar.py --mode dense --model resnet56 --output-dir ./output/cifar/r56_dense
  python cifar.py --mode prune --model resnet56 --target-sparsity 0.5 \
      --checkpoint ./output/cifar/r56_dense/model_best.pth \
      --output-dir ./output/cifar/r56_p50
  python cifar.py --mode finetune --lean ./output/cifar/r56_p50/model_lean.pth \
      --epochs 80 --output-dir ./output/cifar/r56_p50_ft
"""
import os
import re
import json
import time
import argparse

import torch
import torch.nn as nn

from models.cifar import (CifarVGG, CifarResNet, build_cifar_model,
                          load_hrank_state_dict)
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
    """Params (M) + MACs (M) cua Conv2d + Linear.

    KHONG dung thop: thop cong them BatchNorm nen ResNet-56 dense ra 127.62M, trong
    khi HRank/CORING/L1/HRank deu bao 125.49M (chi conv+linear). Lech 1.7% la du de
    con so cua ta khong dat canh bang published duoc. Ham nay cho dung 125.49M.
    """
    params = sum(p.numel() for p in model.parameters()) / 1e6
    total, hooks = [0], []

    def hk(m, i, o):
        total[0] += m.weight.numel() * (o.shape[-1] * o.shape[-2] if o.dim() == 4 else 1)

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(hk))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32).to(device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return params, total[0] / 1e6


def sparsity_for_target(cfg, target, metric="macs", lo=0.01, hi=0.95, iters=14):
    """Tim target_sparsity de lean model rot dung diem nen cua baseline (iso-FLOPs).

    Chay duoc ma KHONG can train vi so kenh con lai chi phu thuoc prune_ratio:
    StructuredPruner cat int(round(n * ratio)) filter moi lop, khong phu thuoc weight.
    => cost(ratio) don dieu giam -> chia doi ~14 lan la du.
    """
    cpu = torch.device("cpu")

    def cost_at(r):
        m = build_cifar_model(cfg)
        StructuredPruner(m).prune(prune_ratio=r, verbose=False)
        lean, _ = convert_to_lean_cifar(m)
        p, mac = count_cost(lean, cpu)
        return p if metric == "params" else mac

    c_lo, c_hi = cost_at(lo), cost_at(hi)
    if not (c_hi <= target <= c_lo):
        raise SystemExit(f"target {metric}={target} ngoai tam voi: "
                         f"[{c_hi:.2f} @r={hi}, {c_lo:.2f} @r={lo}]")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if cost_at(mid) > target:
            lo = mid
        else:
            hi = mid
    best = round((lo + hi) / 2, 3)
    got = cost_at(best)
    print(f"[match] target {metric}={target} -> target-sparsity={best} "
          f"(dat {got:.2f}, lech {100*(got-target)/target:+.2f}%)")
    if abs(got - target) / target > 0.02:
        print(f"[match] CANH BAO: lech >2% — grid kenh qua tho de khop diem nay")
    return best


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


def wandb_log_epoch(wandb, stage, ep, lr, tr_loss, tr_acc, va_loss, va_acc):
    """Mot cho duy nhat de log 1 epoch -> ten metric dong nhat giua cac stage.

    `epoch` la truc TONG (0..299): vong prune 0..14, finetune 15..299 -> ve chung
    duoc mot duong lien tuc tren ngan sach 300 epoch cua baseline.
    """
    if not wandb:
        return
    wandb.log({"epoch": ep, "lr": lr, "stage": stage,
               "train/loss": tr_loss, "train/acc": tr_acc,
               "val/loss": va_loss, "val/acc": va_acc})


def run_training(model, loaders, args, device, epochs, lr, milestones, tag, pruner=None,
                 wandb=None, cfg=None, epoch_offset=0):
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
        wandb_log_epoch(wandb, tag, epoch_offset + ep, optimizer.param_groups[0]["lr"],
                        tr_loss, tr_acc, va_loss, va_acc)
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "config": cfg or model.config(),
                        "top1": best, "epoch": ep},
                       os.path.join(args.output_dir, "model_best.pth"))
    # `best_top1` moi la con so dat canh bang published: HRank/CORING bao `best_prec1`,
    # tuc top-1 CAO NHAT tren tap test. Summary cua wandb chi giu val/acc epoch cuoi.
    if wandb:
        wandb.summary["best_top1"] = best
        wandb.summary[f"best_top1_{tag}"] = best
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
    p.add_argument("--checkpoint", default=None, help="dense ckpt cua ta (mode=prune)")
    p.add_argument("--pretrained", default=None,
                   help="checkpoint HRank/CORING (vd resnet_56.pt) — dung chung diem xuat "
                        "phat voi baseline. Thay cho --checkpoint.")
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
    p.add_argument("--no-unstructured", action="store_true",
                   help="tat muc unstructured -> bien the STRUCTURED-ONLY, cung loai voi "
                        "CORING/HRank. Bat buoc cho bang so sanh chinh; bi-level day du "
                        "de o bang ablation.")
    p.add_argument("--match-macs", default=None, type=float,
                   help="MACs (M) muc tieu -> tu tim target-sparsity, bo qua --target-sparsity")
    p.add_argument("--match-params", default=None, type=float,
                   help="Params (M) muc tieu -> tu tim target-sparsity")
    p.add_argument("--vgg-prune-set", default="all", choices=["all", "l1a"],
                   help="l1a = cung tap layer voi L1 (VGG-16-pruned-A)")
    p.add_argument("--vgg-head", default="hrank", choices=["hrank", "single"],
                   help="hrank = bien the cua HRank/CORING (14.98M); single = cua SPSRC (14.72M)")
    p.add_argument("--protocol", default="coring", choices=["coring", "spsrc"],
                   help="recipe finetune sau prune, xem FINETUNE_PROTO")
    p.add_argument("--budget", default=None, type=int,
                   help="Doi TONG ngan sach epoch (vong prune + finetune), milestone keo "
                        "theo ti le. Vd --budget 400 de bang CHIP. Mac dinh = cua protocol. "
                        "Doi la khong con so sanh ngan sach voi CORING duoc nua -> phai khai.")
    p.add_argument("--ignore-budget", action="store_true",
                   help="finetune du so epoch cua protocol, KHONG tru epoch vong prune "
                        "-> bi-level duoc nhieu epoch hon baseline (chi dung de ablation)")
    p.add_argument("--skip-gate", action="store_true",
                   help="bo qua kiem tra dense vs published (chi dung khi smoke test)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-run", default=None,
                   help="ten run wandb. Truyen CUNG mot ten cho buoc prune va finetune "
                        "-> gop thanh 1 run (resume), duong cong + params/MACs chung cho.")
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
        # --wandb-run: prune va finetune la 2 process nhung dung chung 1 run (resume)
        # -> duong cong training + params/MACs nam CHUNG mot cho.
        name = args.wandb_run or os.path.basename(args.output_dir.rstrip("/"))
        rid = re.sub(r"[^A-Za-z0-9_-]", "_", name)   # wandb id: chi chu/so/-/_
        wandb.init(name=name, id=rid, resume="allow", config=vars(args))

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
        assert args.checkpoint or args.pretrained, "can --checkpoint hoac --pretrained"
        if args.pretrained:
            # cung diem xuat phat voi baseline (CORING nap checkpoint HRank, khong tu train)
            dense_cfg = build(args.model, num_classes, args.vgg_prune_set,
                              args.vgg_head).config()
            model = build_cifar_model(dense_cfg)
            load_hrank_state_dict(model, args.pretrained)
            model = model.to(device)
        else:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            dense_cfg = ckpt["config"]
            model = build_cifar_model(dense_cfg).to(device)
            model.load_state_dict(ckpt["model"])
            print(f"top1 dense: {ckpt.get('top1')}")
        p0, m0 = count_cost(model, device)
        print(f"Dense: {p0:.2f}M params | {m0:.2f}M MACs")

        # match iso-FLOPs voi diem nen cua baseline (khong ton GPU, xem sparsity_for_target)
        if args.match_macs or args.match_params:
            args.target_sparsity = sparsity_for_target(
                dense_cfg, args.match_macs or args.match_params,
                "macs" if args.match_macs else "params")

        proto = FINETUNE_PROTO[args.protocol]
        # Ngan sach epoch SAU khi roi dense checkpoint phai bang cua baseline.
        # Bi-level la iterative nen tieu mot phan ngay trong vong prune -> ghi lai
        # de mode=finetune tru ra, neu khong bi-level duoc nhieu epoch hon baseline.
        prune_epochs = args.prune_iters * args.prune_finetune_epochs
        print(f"Ngan sach '{args.protocol}' = {proto['epochs']} ep | "
              f"vong prune tieu {prune_epochs} ep | con lai cho finetune "
              f"{proto['epochs'] - prune_epochs} ep")
        assert prune_epochs < proto["epochs"], (
            f"Vong prune ({prune_epochs} ep) da vuot ngan sach {proto['epochs']} ep")

        criterion = nn.CrossEntropyLoss()
        u_pruner, s_pruner = UnstructuredPruner(model), StructuredPruner(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr or proto["lr"],
                                    momentum=args.momentum,
                                    weight_decay=proto["weight_decay"])

        for it in range(args.prune_iters):
            sparsity = args.target_sparsity * (it + 1) / args.prune_iters
            print(f"\n=== Prune iter {it+1}/{args.prune_iters} | sparsity={sparsity:.3f} ===")
            if not args.no_unstructured:
                u_pruner.prune(sensitivity=args.sensitivity_mult * sparsity)
                print(f"  Song Han global sparsity={u_pruner.global_sparsity():.3f}")
            s_pruner.prune(prune_ratio=sparsity)
            for ep in range(args.prune_finetune_epochs):
                tr_loss, tr_acc = train_one_epoch(
                    model, criterion, loaders[0], optimizer, device,
                    ep, args.prune_finetune_epochs,
                    pruner=None if args.no_unstructured else u_pruner)
                va_loss, va_acc = evaluate(model, criterion, loaders[1], device)
                wandb_log_epoch(wandb, "prune", it * args.prune_finetune_epochs + ep,
                                optimizer.param_groups[0]["lr"],
                                tr_loss, tr_acc, va_loss, va_acc)

        print("\n=== Model Surgery ===")
        save_path = os.path.join(args.output_dir, "model_lean.pth")
        lean, lean_cfg = convert_to_lean_cifar(model, save_path=save_path)
        lean.to(device)
        p1, m1 = count_cost(lean, device)
        print(f"Params {p0:.2f}M -> {p1:.2f}M (-{(1-p1/p0)*100:.2f}%) | "
              f"MACs {m0:.2f}M -> {m1:.2f}M (-{(1-m1/m0)*100:.2f}%)")
        evaluate(lean, criterion, loaders[1], device)
        lean_cfg["prune_epochs"] = prune_epochs      # mode=finetune doc de tru ngan sach
        with open(save_path.replace(".pth", ".json"), "w") as f:
            json.dump(lean_cfg, f, indent=2)
        with open(os.path.join(args.output_dir, "cost.json"), "w") as f:
            json.dump({"params_M": p1, "macs_M": m1,
                       "params_red_pct": (1 - p1 / p0) * 100,
                       "macs_red_pct": (1 - m1 / m0) * 100,
                       "prune_epochs": prune_epochs,
                       "target_sparsity": args.target_sparsity,
                       "variant": "structured-only" if args.no_unstructured else "bi-level",
                       }, f, indent=2)
        if wandb:
            wandb.summary.update({
                "params_M": p1, "macs_M": m1,
                "params_red_pct": (1 - p1 / p0) * 100,
                "macs_red_pct": (1 - m1 / m0) * 100,
                "dense_params_M": p0, "dense_macs_M": m0,
                "target_sparsity": args.target_sparsity,
                "variant": "structured-only" if args.no_unstructured else "bi-level",
                "unstructured_sparsity": 0.0 if args.no_unstructured
                                         else u_pruner.global_sparsity(),
            })

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

        # Tru so epoch vong prune da tieu -> tong ngan sach sau dense BANG baseline.
        # Milestone dich theo cung so epoch de lich lr trung nhau tren truc tong.
        cost_path = os.path.join(os.path.dirname(os.path.abspath(args.lean)), "cost.json")
        if wandb and os.path.isfile(cost_path):
            with open(cost_path) as f:
                wandb.summary.update(json.load(f))     # params/MACs/reduction/variant
            print(f"Da day cost.json len wandb summary: {cost_path}")

        spent = 0 if args.ignore_budget else int(cfg.get("prune_epochs", 0))

        if args.budget:
            # Doi tong ngan sach (vd --budget 400 de bang CHIP): GIU NGUYEN HINH DANG
            # lich lr bang cach keo milestone theo ti le, roi moi tru phan vong prune.
            # Khong lam vay thi --epochs 400 van dung milestone 135/210 -> lich lr sai.
            k = args.budget / proto["epochs"]
            total = args.budget
            ms_global = [int(round(m * k)) for m in proto["milestones"]]
        else:
            total = proto["epochs"]
            ms_global = list(proto["milestones"])
        epochs = args.epochs or max(total - spent, 1)
        milestones = [max(m - spent, 1) for m in ms_global]

        print(f"Protocol '{args.protocol}': ngan sach {total} ep "
              f"- {spent} ep (vong prune) = {epochs} ep finetune | "
              f"lr={args.lr or proto['lr']} milestones={milestones} "
              f"(truc tong: {ms_global}) wd={proto['weight_decay']}")
        run_training(model, loaders, args, device,
                     epochs, args.lr or proto["lr"],
                     milestones, "finetune", wandb=wandb, cfg=cfg, epoch_offset=spent)

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
