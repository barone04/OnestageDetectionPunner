"""
Dataset YOLO-format cho one-stage detector (standalone).

Layout (giong DeepFish/Etroplus):
    root/images/{train,val}/*.jpg
    root/labels/{train,val}/*.txt

Moi dong label: "cls xc yc w h".
- Tu dong nhan dang: max(toa do) <= 1 -> YOLO normalized cxcywh; nguoc lai -> pixel x1y1x2y2.
- **letterbox**: resize giu ti le + pad ve (S,S) -> box khong meo (giong YOLOv1 reference).
- **normalize**: /255 roi chuan hoa ImageNet (mean/std) -> hop pretrained backbone.
Tra ve: (filename, img_tensor[3,S,S], label[N,5]=[cls,xc,yc,w,h] (rel anh letterbox), (h0,w0)).
Anh khong co object -> label = [[-1,0,0,0,0]].
"""
import os

import numpy as np
import torch
from PIL import Image

_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet RGB
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", input_size=448, augment=False,
                 normalize=True, letterbox=True):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.input_size = input_size
        self.augment = augment
        self.normalize = normalize
        self.letterbox = letterbox
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        self.imgs = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(_EXTS))
        print(f"[YoloDataset:{split}] {len(self.imgs)} images in {self.img_dir} "
              f"(letterbox={letterbox}, normalize={normalize})")

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
                        xc = ((a + c) / 2) / max(w, 1)
                        yc = ((b + d) / 2) / max(h, 1)
                        bw = (c - a) / max(w, 1)
                        bh = (d - b) / max(h, 1)
                    if bw > 0 and bh > 0:
                        rows.append([cls, xc, yc, bw, bh])
        if not rows:
            return torch.tensor([[-1.0, 0, 0, 0, 0]], dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def _letterbox(self, img, S, color=(114, 114, 114)):
        """Resize giu ti le ve vua khung SxS roi pad. Tra (img_pad, nw, nh, pad_x, pad_y)."""
        w, h = img.size
        scale = S / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        img_r = img.resize((nw, nh))
        canvas = Image.new("RGB", (S, S), color)
        pad_x, pad_y = (S - nw) // 2, (S - nh) // 2
        canvas.paste(img_r, (pad_x, pad_y))
        return canvas, nw, nh, pad_x, pad_y

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        w0, h0 = img.size
        label = self._read_label(name, w0, h0)   # cxcywh normalized rel anh GOC
        S = self.input_size

        if self.letterbox:
            img_r, nw, nh, pad_x, pad_y = self._letterbox(img, S)
            valid = label[:, 0] >= 0
            if valid.any():
                # rel-goc [0,1] -> pixel trong vung resize -> +pad -> rel anh SxS
                label[valid, 1] = (label[valid, 1] * nw + pad_x) / S
                label[valid, 2] = (label[valid, 2] * nh + pad_y) / S
                label[valid, 3] = label[valid, 3] * nw / S
                label[valid, 4] = label[valid, 4] * nh / S
        else:
            img_r = img.resize((S, S))   # resize vuong (label normalized giu nguyen)

        arr = np.asarray(img_r, dtype=np.float32) / 255.0
        if self.normalize:
            arr = (arr - _MEAN) / _STD
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if self.augment and torch.rand(1).item() < 0.5:
            t = torch.flip(t, dims=[2])  # hflip
            valid = label[:, 0] >= 0
            label[valid, 1] = 1.0 - label[valid, 1]

        return name, t, label, (h0, w0)

    @staticmethod
    def collate_fn(batch):
        names, imgs, labels, shapes = zip(*batch)
        return list(names), torch.stack(imgs, 0), list(labels), list(shapes)
