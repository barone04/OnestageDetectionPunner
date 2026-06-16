"""
Dataset YOLO-format cho one-stage detector (standalone).

Layout (giong DeepFish/Etroplus):
    root/images/{train,val}/*.jpg
    root/labels/{train,val}/*.txt

Moi dong label: "cls xc yc w h".
- Tu dong nhan dang: neu max(toa do) <= 1 -> YOLO normalized cxcywh (chuan);
  nguoc lai -> coi la pixel x1 y1 x2 y2 -> doi sang normalized cxcywh.
Anh resize ve (input_size, input_size); label normalized nen KHONG can scale.
Tra ve: (filename, img_tensor[3,H,W] float[0,1], label[N,5]=[cls,xc,yc,w,h], (h0,w0)).
Anh khong co object -> label = [[-1,0,0,0,0]] (khop quy uoc YoloLoss).
"""
import os

import numpy as np
import torch
from PIL import Image

_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", input_size=448, augment=False):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.input_size = input_size
        self.augment = augment
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        self.imgs = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(_EXTS))
        print(f"[YoloDataset:{split}] {len(self.imgs)} images in {self.img_dir}")

    def __len__(self):
        return len(self.imgs)

    def _read_label(self, name, w, h):
        path = os.path.join(self.lbl_dir, os.path.splitext(name)[0] + ".txt")
        rows = []
        if os.path.isfile(path):
            with open(path, "r") as f:
                for line in f:
                    vals = line.split()
                    if len(vals) < 5:
                        continue
                    cls = float(vals[0])
                    a, b, c, d = (float(v) for v in vals[1:5])
                    if max(a, b, c, d) <= 1.0:
                        xc, yc, bw, bh = a, b, c, d
                    else:
                        # pixel x1y1x2y2 -> normalized cxcywh
                        xc = ((a + c) / 2) / max(w, 1)
                        yc = ((b + d) / 2) / max(h, 1)
                        bw = (c - a) / max(w, 1)
                        bh = (d - b) / max(h, 1)
                    if bw > 0 and bh > 0:
                        rows.append([cls, xc, yc, bw, bh])
        if not rows:
            return torch.tensor([[-1.0, 0, 0, 0, 0]], dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        w0, h0 = img.size
        img_r = img.resize((self.input_size, self.input_size))
        arr = np.asarray(img_r, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        label = self._read_label(name, w0, h0)

        if self.augment and torch.rand(1).item() < 0.5:
            t = torch.flip(t, dims=[2])  # hflip
            valid = label[:, 0] >= 0
            label[valid, 1] = 1.0 - label[valid, 1]

        return name, t, label, (h0, w0)

    @staticmethod
    def collate_fn(batch):
        names, imgs, labels, shapes = zip(*batch)
        return list(names), torch.stack(imgs, 0), list(labels), list(shapes)
