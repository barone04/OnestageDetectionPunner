"""
benchmark.py — So sanh hieu qua DENSE vs PRUNED: Params, FLOPs, latency, FPS, mAP.

Vi du:
  python benchmark.py \
    --dense  ./output/yolo/step1_dense/model_best.pth \
    --pruned ./output/yolo/step3_final/model_best.pth \
    --data-path ./fish --num-classes 1 --device cuda
"""
import os
import json
import time
import argparse

import torch

from models.yolo import build_model
from data import YoloDataset
from utils import evaluate_map


def resolve_device(req):
    req = (req or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(req)


def load_model(path, config_path=None):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg, sd = ckpt["config"], ckpt["model"]
    elif config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    else:
        raise ValueError(f"{path} khong co 'config' va khong co --*-config json")

    model = build_model(cfg)
    model.load_state_dict(sd, strict=True)
    return model, cfg


def measure_efficiency(model, input_size, device, iters=50, warmup=10):
    model.eval().to(device)
    params = sum(p.numel() for p in model.parameters())

    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    flops = None
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception as e:
        print(f"  [thop] skip FLOPs ({e})")

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

    latency = dt / iters * 1000.0    # ms
    fps = iters / dt
    return params, flops, latency, fps


def get_args():
    p = argparse.ArgumentParser(description="Benchmark dense vs pruned YOLOv1")
    p.add_argument("--dense", required=True)
    p.add_argument("--pruned", required=True)
    p.add_argument("--dense-config", default=None)
    p.add_argument("--pruned-config", default=None)
    p.add_argument("--data-path", default=None, help="neu co -> tinh mAP tren val")
    p.add_argument("--num-classes", default=1, type=int)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--input-size", default=None, type=int)
    p.add_argument("--conf-thresh", default=0.001, type=float)
    p.add_argument("--nms-thresh", default=0.5, type=float)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = get_args()
    device = resolve_device(args.device)
    print("Device:", device)

    dense, dcfg = load_model(args.dense, args.dense_config)
    pruned, pcfg = load_model(args.pruned, args.pruned_config)
    input_size = args.input_size or dcfg.get("input_size", 448)

    print("\nMeasuring DENSE...")
    p0, f0, l0, fps0 = measure_efficiency(dense, input_size, device)
    print("Measuring PRUNED...")
    p1, f1, l1, fps1 = measure_efficiency(pruned, input_size, device)

    m0 = m1 = None
    if args.data_path:
        val_ds = YoloDataset(args.data_path, "val", input_size, augment=False)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            collate_fn=YoloDataset.collate_fn)
        print("\nmAP DENSE...")
        m0 = evaluate_map(dense, val_loader, device, args.num_classes,
                          args.conf_thresh, args.nms_thresh)
        print("mAP PRUNED...")
        m1 = evaluate_map(pruned, val_loader, device, args.num_classes,
                          args.conf_thresh, args.nms_thresh)

    def pct(a, b):
        return (1 - b / a) * 100 if a else 0.0

    print("\n" + "=" * 72)
    print(f"{'Metric':<18} | {'Dense':>14} | {'Pruned':>14} | {'Delta':>14}")
    print("-" * 72)
    print(f"{'Params (M)':<18} | {p0/1e6:>14.2f} | {p1/1e6:>14.2f} | {'-'+format(pct(p0,p1),'.1f')+'%':>14}")
    if f0 and f1:
        print(f"{'FLOPs (G)':<18} | {f0/1e9:>14.2f} | {f1/1e9:>14.2f} | {'-'+format(pct(f0,f1),'.1f')+'%':>14}")
    print(f"{'Latency (ms)':<18} | {l0:>14.2f} | {l1:>14.2f} | {'-'+format(pct(l0,l1),'.1f')+'%':>14}")
    print(f"{'FPS':<18} | {fps0:>14.2f} | {fps1:>14.2f} | {'+'+format((fps1/fps0-1)*100,'.1f')+'%':>14}")
    if m0 and m1:
        print(f"{'mAP@0.5':<18} | {m0['mAP@0.5']:>14.4f} | {m1['mAP@0.5']:>14.4f} | {format(m1['mAP@0.5']-m0['mAP@0.5'],'+.4f'):>14}")
        print(f"{'mAP@0.5:0.95':<18} | {m0['mAP@0.5:0.95']:>14.4f} | {m1['mAP@0.5:0.95']:>14.4f} | {format(m1['mAP@0.5:0.95']-m0['mAP@0.5:0.95'],'+.4f'):>14}")
    print("=" * 72)


if __name__ == "__main__":
    main()
