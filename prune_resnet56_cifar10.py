"""
prune_resnet56_cifar10.py — Kiem chung setup (Quy chuan thuc nghiem §5).

Reproduce ket qua CORING tren ResNet-56/CIFAR-10 (paper Table 3: 94.76% @ 22.4% params),
NHUNG dung `coring_rank` CUA MINH (pruning/coring.py) de cham diem filter
-> neu ra accuracy ~ paper thi thuat toan coring cua minh ĐUNG.

Port tu code goc CORING (models/cifar10/resnet.py + main.py load_resnet_model),
chi thay get_rank -> coring_rank cua minh. CIFAR-10 tu tai (torchvision).

Baseline pretrained ResNet-56 (~93.26%): tai tu CORING releases
  https://github.com/pvti/CORING/releases/tag/v0.1.0  (file resnet_56.pt)

Vi du (one-shot, ~22.4% params):
  python prune_resnet56_cifar10.py --pretrain resnet_56.pt \
      --compress-rate "[0.]+[0.18]*29" --epochs 300 --lr-decay-step 150,225 \
      --weight-decay 0.005 --batch-size 256
K-shot (K=15, giong paper):
  ... --shot 15
"""
import os
import re
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from pruning.coring import coring_rank   # <-- THUAT TOAN CUA MINH


# ===========================================================================
# ResNet-56 cho CIFAR-10 (port tu CORING models/cifar10/resnet.py)
# ===========================================================================
def adapt_channel(compress_rate, num_layers=56):
    stage_repeat = [9, 9, 9]
    stage_out_channel = [16] + [16] * 9 + [32] * 9 + [64] * 9

    stage_oup_cprate = [compress_rate[0]]
    for i in range(len(stage_repeat) - 1):
        stage_oup_cprate += [compress_rate[i + 1]] * stage_repeat[i]
    stage_oup_cprate += [0.] * stage_repeat[-1]          # stage cuoi giu nguyen
    mid_cprate = compress_rate[len(stage_repeat):]

    overall_channel, mid_channel = [], []
    for i in range(len(stage_out_channel)):
        overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
        if i != 0:
            mid_channel += [int(stage_out_channel[i] * (1 - mid_cprate[i - 1]))]
    return overall_channel, mid_channel


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, midplanes, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            pad = (planes - inplanes) // 2
            if stride != 1:
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, pad, planes - inplanes - pad), "constant", 0))
            else:
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x, (0, 0, 0, 0, pad, planes - inplanes - pad), "constant", 0))

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu2(out)


class ResNet56(nn.Module):
    def __init__(self, compress_rate, num_classes=10):
        super().__init__()
        self.overall_channel, self.mid_channel = adapt_channel(compress_rate, 56)
        self.layer_num = 0
        self.conv1 = nn.Conv2d(3, self.overall_channel[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.overall_channel[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer_num = 1
        self.layer1 = self._make_layer(9, stride=1)
        self.layer2 = self._make_layer(9, stride=2)
        self.layer3 = self._make_layer(9, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, blocks, stride):
        layers = [BasicBlock(self.mid_channel[self.layer_num - 1],
                             self.overall_channel[self.layer_num - 1],
                             self.overall_channel[self.layer_num], stride)]
        self.layer_num += 1
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.mid_channel[self.layer_num - 1],
                                     self.overall_channel[self.layer_num - 1],
                                     self.overall_channel[self.layer_num]))
            self.layer_num += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)


# ===========================================================================
# Surgery: chon filter (coring_rank cua minh) + copy weight
# (port tu CORING main.py::load_resnet_model, thay get_rank -> coring_rank)
# ===========================================================================
def _copy_bn(state_dict, ori, bn_name, out_idx):
    """Copy BN sliced theo out_idx (kenh giu lai) neu prune, full neu None.
    Giong surgery YOLO cua minh (pruning/surgery.py). CORING goc KHONG lam buoc nay."""
    for suf in (".weight", ".bias", ".running_mean", ".running_var"):
        key = bn_name + suf
        if key in ori:
            state_dict[key] = ori[key][out_idx] if out_idx is not None else ori[key]
    nbt = bn_name + ".num_batches_tracked"
    if nbt in ori:
        state_dict[nbt] = ori[nbt]


@torch.no_grad()
def load_resnet56_pruned(model, oristate_dict, copy_bn=True):
    state_dict = model.state_dict()
    last_select_index = None
    all_conv_weight = []

    for layer in range(3):
        for k in range(9):
            for l in range(2):
                conv_name = f"layer{layer+1}.{k}.conv{l+1}"
                bn_name = f"layer{layer+1}.{k}.bn{l+1}"
                w = conv_name + ".weight"
                all_conv_weight.append(w)
                ori, cur = oristate_dict[w], state_dict[w]
                ori_n, cur_n = ori.size(0), cur.size(0)
                out_idx = None

                if ori_n != cur_n:                                    # output bi cat
                    rank = coring_rank(ori)                           # <-- CORING RANK CUA MINH
                    select = np.sort(np.argsort(rank)[ori_n - cur_n:])  # giu top-cur
                    if last_select_index is not None:
                        for ii, i in enumerate(select):
                            for jj, j in enumerate(last_select_index):
                                state_dict[w][ii][jj] = ori[i][j]
                    else:
                        for ii, i in enumerate(select):
                            state_dict[w][ii] = ori[i]
                    last_select_index = select
                    out_idx = select                                  # BN cat theo output nay
                elif last_select_index is not None:                  # chi in-channel cat
                    for ii in range(ori_n):
                        for jj, j in enumerate(last_select_index):
                            state_dict[w][ii][jj] = ori[ii][j]
                    last_select_index = None
                else:
                    state_dict[w] = ori
                    last_select_index = None

                if copy_bn:                                           # <-- COPY BN (cach minh)
                    _copy_bn(state_dict, oristate_dict, bn_name, out_idx)

    # stem conv1 (+bn1) + fc: khong cat -> copy full
    state_dict["conv1.weight"] = oristate_dict["conv1.weight"]
    if copy_bn:
        _copy_bn(state_dict, oristate_dict, "bn1", None)
    state_dict["fc.weight"] = oristate_dict["fc.weight"]
    state_dict["fc.bias"] = oristate_dict["fc.bias"]
    model.load_state_dict(state_dict)


# ===========================================================================
# CIFAR-10 + train/eval
# ===========================================================================
class _ParquetCIFAR(torch.utils.data.Dataset):
    """Doc CIFAR-10 tu parquet HuggingFace (uoft-cs/cifar10): img(bytes PNG) + label.
    Decode SAN toan bo -> numpy uint8 (N,32,32,3) trong __init__ (nhanh moi epoch)."""
    def __init__(self, path, transform):
        import io
        import pyarrow.parquet as pq
        from PIL import Image
        rows = pq.read_table(path).to_pylist()
        self.images = np.stack([np.asarray(Image.open(io.BytesIO(r["img"]["bytes"])).convert("RGB"))
                                for r in rows])              # (N,32,32,3) uint8
        self.labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.fromarray(self.images[i])                # tu array -> nhanh, khong decode PNG
        return self.transform(img), int(self.labels[i])


def cifar10_loaders(data_dir, batch_size, workers=4):
    """data_dir chua train.parquet + test.parquet (tai tu HF, xem README)."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)   # khop CORING data/cifar10.py
    tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(mean, std)])
    te_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    tr = _ParquetCIFAR(os.path.join(data_dir, "train.parquet"), tr_tf)
    te = _ParquetCIFAR(os.path.join(data_dir, "test.parquet"), te_tf)
    return (torch.utils.data.DataLoader(tr, batch_size, shuffle=True, num_workers=workers, pin_memory=True),
            torch.utils.data.DataLoader(te, batch_size, shuffle=False, num_workers=workers, pin_memory=True))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
    return loss_sum / max(n, 1)                          # train loss trung binh


@torch.no_grad()
def validate(model, loader, device, criterion=None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / total
    return (acc, loss_sum / total) if criterion is not None else acc


def parse_cpr(s):
    """'[0.]+[0.18]*29' -> list float."""
    out = []
    for tok in s.split("+"):
        num = re.findall(r"\*(\d+)", tok)
        rate = float(re.findall(r"\d+\.\d*", tok)[0])
        out += [rate] * (int(num[0]) if num else 1)
    return out


def get_cpr_kshot(compress_rate, K):
    """CORING get_cpr: cpr[shot] = r/(K-shot) (phi tuyen, absolute tu goc)."""
    return [[r / (K - shot) for r in compress_rate] for shot in range(K)]


def params_count(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser(description="Prune ResNet-56/CIFAR-10 bang coring_rank cua minh")
    ap.add_argument("--pretrain", required=True, help="resnet_56.pt (baseline ~93.26%, tu CORING releases)")
    ap.add_argument("--compress-rate", default="[0.]+[0.18]*29")
    ap.add_argument("--shot", default=1, type=int, help="1=one-shot; >1 = k-shot (r/(K-shot))")
    ap.add_argument("--epochs", default=300, type=int, help="finetune epoch cuoi cung")
    ap.add_argument("--lr", default=0.01, type=float)
    ap.add_argument("--lr-decay-step", default="150,225")
    ap.add_argument("--momentum", default=0.9, type=float)
    ap.add_argument("--weight-decay", default=0.005, type=float)
    ap.add_argument("--batch-size", default=256, type=int)
    ap.add_argument("--calib", default=100, type=int, help="tong calib epoch cho k-shot (chia ⌊ε/K⌋)")
    ap.add_argument("--no-bn-copy", action="store_true",
                    help="KHONG copy BN (giong CORING goc, can finetune lau). Mac dinh: copy BN (cach minh, nhanh hoi phuc)")
    ap.add_argument("--data-dir", default="./data/cifar10_hf")
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--output-dir", default="./output/resnet56_coring")
    ap.add_argument("--wandb", action="store_true", help="log len wandb")
    ap.add_argument("--wandb-name", default="resnet56-cifar10-coring")
    ap.add_argument("--wandb-project", default="resnet56-cifar10-coring")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    cpr = parse_cpr(args.compress_rate)
    train_loader, val_loader = cifar10_loaders(args.data_dir, args.batch_size, args.workers)
    criterion = nn.CrossEntropyLoss().to(device)

    # baseline pretrained (full)
    origin = ResNet56([0.] * 100).to(device)
    ck = torch.load(args.pretrain, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    origin.load_state_dict(sd)
    p0 = params_count(origin)
    base_acc = validate(origin, val_loader, device)
    print(f"Baseline ResNet-56 | params={p0/1e6:.3f}M | acc={base_acc:.2f}%")

    wb = None
    if args.wandb:
        import wandb  # env WANDB_API_KEY/PROJECT tu lo (giong train.py, vd da set tren FPT)
        wb = wandb.init(name=args.wandb_name, project=args.wandb_project, config=vars(args))
        wb.summary["baseline_acc"] = base_acc

    steps = list(map(int, args.lr_decay_step.split(",")))

    def finetune(model, epochs, tag="", save_path=None):
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=steps, gamma=0.1)
        best = 0.0
        for ep in range(epochs):
            tr_loss = train_epoch(model, train_loader, criterion, opt, device)
            sch.step()
            acc, val_loss = validate(model, val_loader, device, criterion)
            if acc > best:                                   # best moi -> luu weight epoch nay
                best = acc
                if save_path is not None:
                    torch.save({"state_dict": model.state_dict(), "acc": best}, save_path)
            if wb is not None:
                wb.log({"val_acc": acc, "best_acc": best, "train_loss": tr_loss,
                        "val_loss": val_loss, "lr": opt.param_groups[0]["lr"]})
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"  {tag}[ep {ep+1}/{epochs}] loss={tr_loss:.3f} "
                      f"val_loss={val_loss:.3f} acc={acc:.2f}% best={best:.2f}%")
        return best

    # ----- prune (one-shot hoac k-shot) -----
    if args.shot <= 1:
        print(f"\n=== ONE-SHOT prune | cpr={args.compress_rate} ===")
        model = ResNet56(cpr).to(device)
        load_resnet56_pruned(model, origin.state_dict(), copy_bn=not args.no_bn_copy)
    else:
        print(f"\n=== K-SHOT prune | K={args.shot} ===")
        cprs = get_cpr_kshot(cpr, args.shot)
        model = copy.deepcopy(origin)
        calib_ep = max(args.calib // args.shot, 1)                    # ⌊ε/K⌋
        for shot in range(args.shot):
            new = ResNet56(cprs[shot]).to(device)
            load_resnet56_pruned(new, model.state_dict(), copy_bn=not args.no_bn_copy)   # rank tinh lai moi shot
            print(f"  shot {shot+1}/{args.shot}: params={params_count(new)/1e6:.3f}M -> calibrate {calib_ep} ep")
            finetune(new, calib_ep, tag=f"shot{shot+1} ")
            model = new
        model = model.to(device)

    p1 = params_count(model)
    print(f"\nPruned | params={p1/1e6:.3f}M (-{(1-p1/p0)*100:.1f}%) | "
          f"acc truoc finetune={validate(model, val_loader, device):.2f}%")

    print(f"\n=== Finetune cuoi ({args.epochs} epoch) ===")
    save_path = os.path.join(args.output_dir, "model_best.pth")
    best = finetune(model, args.epochs, tag="ft ", save_path=save_path)  # tu luu best-epoch
    if wb is not None:
        wb.save(save_path, base_path=os.path.dirname(save_path), policy="now")  # upload ckpt best cuoi cung
        wb.summary["final_acc"] = best
        wb.summary["pruned_params_M"] = p1 / 1e6
        wb.summary["params_reduction_pct"] = (1 - p1 / p0) * 100
        wb.finish()
    print(f"\nDONE. Best acc={best:.2f}% | params -{(1-p1/p0)*100:.1f}% "
          f"(paper Table 3 ResNet-56 22.4% params -> 94.76%)")


if __name__ == "__main__":
    main()
