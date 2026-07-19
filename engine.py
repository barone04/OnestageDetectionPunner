"""
Vong train / eval dung chung cho train.py va prune.py (standalone).
Model.forward tra RAW prediction -> YoloLoss tinh loss (ca train lan eval).
"""
import time

import torch


def train_one_epoch(model, criterion, loader, optimizer, device, epoch,
                    total_epochs=0, pruner=None, log_every=50, scaler=None):
    model.train()
    agg = {}
    n = max(len(loader), 1)
    t0 = time.time()

    for i, (names, imgs, labels, shapes) in enumerate(loader):
        imgs = imgs.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            preds = model(imgs)
            ld = criterion(preds, labels)
            loss = ld["loss"]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # giu sparsity Song Han: ep weight bi prune ve 0 sau moi buoc
        if pruner is not None:
            pruner.apply_masks()

        for k, v in ld.items():
            agg[k] = agg.get(k, 0.0) + float(v.item())

        if i % log_every == 0:
            print(f"  [E{epoch+1}/{total_epochs} {i:>4}/{n}] "
                  + " ".join(f"{k}={v.item():.4f}" for k, v in ld.items()),
                  flush=True)

    dt = time.time() - t0
    avg = {k: v / n for k, v in agg.items()}
    print(f"  -> epoch {epoch+1} train: " + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
          + f"  ({dt:.1f}s)", flush=True)
    return avg


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    agg = {}
    n = max(len(loader), 1)
    total = max(len(loader), 1)
    print_every = max(total // 10, 1)
    for index, (names, imgs, labels, shapes) in enumerate(loader):
        imgs = imgs.to(device)
        preds = model(imgs)
        ld = criterion(preds, labels)
        for k, v in ld.items():
            agg[k] = agg.get(k, 0.0) + float(v.item())
        if index % print_every == 0 or index + 1 == total:
            print(f"  [VAL {index + 1}/{total}]", flush=True)
    avg = {k: v / n for k, v in agg.items()}
    print("  -> val: " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()),
          flush=True)
    return avg
