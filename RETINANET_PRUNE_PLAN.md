# Master Plan — Prunable RetinaNet (multi-backbone, full prune)

> Mo rong framework pruning cua **PrunedFishNet** tu **Faster R-CNN (two-stage)** sang
> **RetinaNet (one-stage)**, prune duoc **backbone + FPN + head tower** cho **moi backbone**
> (resnet18/50/101, vgg16/19, alexnet, mobilenet_v3, efficientnet). Chi dung **Pipeline 2**.

---

## 0. Quyet dinh da chot

| Hang muc | Quyet dinh |
|---|---|
| Detector | **RetinaNet** — boc `torchvision.models.detection.RetinaNet`, giu focal loss + matcher + postprocess |
| num_classes | **2** (khong doi dataset; RetinaNet dung sigmoid focal, background = anchor khong match) |
| Pham vi prune | **backbone + FPN + head tower** |
| Backbone | Tat ca — ke ca non-ResNet → moi backbone phai viet lai bang `ConvBNReLU` de prune duoc |
| Pipeline | Chi **Pipeline 2** (train_det → prune_det → train_det) |

**Tai su dung 100% (khong dong vao):** `ConvBNReLU` + masks (`models/conv_bn_relu.py`),
`UnstructuredPruner` (Song Han), `StructuredPruner` (L1-inf-inf), vong iterative trong
`prune_det.py`, COCO eval trong `engines/trainer_det.evaluate`, toan bo `data/*`.

---

## 1. Y tuong cot loi: "Prune Protocol" (`export_lean`)

Surgery hien tai (`pruning/surgery.py:convert_to_lean_model`) la **1 ham khong lo, hardcode
topology ResNet/FPN + doc `roi_heads`** cua Faster R-CNN. Voi 6+ backbone (co SE, skip,
depthwise) cach do se vo tran. → **Refactor surgery thanh giao thuc theo module.**

Moi block prunable implement:

```python
def export_lean(self, in_kept_idx) -> (lean_module, out_kept_idx):
    """
    in_kept_idx : index kenh INPUT con song (tu block truoc tra ve)
    - tu build phien ban nho cua chinh no
    - copy weight: slice INPUT theo in_kept_idx, slice OUTPUT theo s_mask cua chinh no
    - xu ly noi bo: SE, depthwise (groups), skip/residual
    return: (module_nho, index kenh OUTPUT con song)  # feed cho block sau
    """
```

Surgery khi do = **fold tuyen tinh** qua mang:

```python
idx = backbone.export_lean(in_idx=[0,1,2])        # 3 kenh RGB
taps = backbone.collect_taps()                    # 4 tap-point cho FPN (kem out_idx moi tap)
fpn_lean, fpn_out_idx = fpn.export_lean(taps)     # slice lateral theo tap; prune inner block
head_lean = head.export_lean(in_idx=256)          # tower la chuoi linear
```

**Loi ich:** them backbone moi = chi viet `export_lean` cho block cua no, khong sua surgery
trung tam. Giu nguyen co che mask/pruner/Song-Han.

---

## 2. JSON config schema moi (tong quat)

Cu: `{"backbone":[...], "fpn":[...]}`. Moi:

```json
{
  "arch": "retinanet_resnet18",
  "backbone": { "...per-arch config (rates hoac widths)..." },
  "fpn":      [r0, r1, r2, r3],
  "head_cls": [w1, w2, w3, w4],
  "head_reg": [w1, w2, w3, w4]
}
```

- `head_cls`/`head_reg`: luu **so kenh con song tuyet doi** (khong luu rate) — tower la
  feed-forward, tranh lech rounding `int(256*(1-r))` giua step-2 va step-3.
- `arch`: de step-3 (`train_det.py`) goi dung builder, dung lai dung khung nho.

---

## 3. Kien truc one-stage vs two-stage

```
Two-stage (cu):  Backbone -> FPN -> RPN -> RoIAlign -> roi_heads(box_head MLP + predictor)
One-stage (moi): Backbone -> FPN -> Head (conv tower dung chung moi level) -> {cls_logits, bbox_reg}
```

- RetinaNet bo hoan toan **RPN + RoIAlign + RoI MLP head**.
- Head = 2 tower (classification + regression), moi tower 4 conv 3x3 **dung chung cho 5 level FPN**
  → chi 1 bo weight → prune 1 lan, de hon ResNet.
- Conv predictor cuoi (`cls_logits = A*K`, `bbox_reg = A*4`): **KHONG prune output** (rang buoc
  anchor/class) — chi **slice INPUT** theo tower conv cuoi.
- `PrunableFPN` luon expand ve 256 → head input on dinh 256 du backbone/FPN co prune.

---

## 4. Thay doi theo file

### File MOI
| File | Noi dung |
|---|---|
| `models/retinanet_custom.py` | builder `retinanet_<backbone>_fpn(...)` + `PrunableRetinaNetHead` (tower = `ConvBNReLU`) + registry backbone |
| `models/backbones/` (hybrid_vgg.py, hybrid_alexnet.py, hybrid_mobilenet.py, hybrid_efficientnet.py) | cac backbone viet lai bang `ConvBNReLU`, ho tro `compress_rate`, pretrained remap, 4 tap-point, `export_lean` |
| `scripts/pipeline2_retinanet.sh` | pipeline 2 cho RetinaNet (`--prune-fpn --prune-head`) |

### File SUA
| File | Sua gi |
|---|---|
| `engines/trainer_det.py` | **Fix crash**: block print hardcode loss keys Faster R-CNN → tong quat hoa theo key thuc co (RetinaNet tra `classification`, `bbox_regression`) |
| `pruning/surgery.py` | Refactor sang giao thuc `export_lean` (fold); nhanh one-stage; head surgery (linear chain); **FPN lateral input slice theo tap mask** (moi) |
| `prune_det.py` | them model `retinanet_*`; flag `--prune-head`; thu thap layer head/fpn/backbone |
| `train_det.py` | them model `retinanet_*`; doc `arch`/`head_cls`/`head_reg` tu JSON; nap lean |
| `models/__init__.py` | export builder RetinaNet |
| `benchmark.py`, `compare_acc.py`, `visualize_*.py` | chon builder theo `arch` |

---

## 5. Ke hoach tung backbone (do kho + diem surgery dac thu)

| Backbone | Connectivity | Do kho | `export_lean` phai xu ly |
|---|---|:---:|---|
| **VGG16/19** | Tuyen tinh (conv-relu-pool) | 🟢 De | Chuoi thang: in_idx = out_idx block truoc; tap sau cac maxpool |
| **AlexNet** | Tuyen tinh, **it scale** | 🟢 De (khong khuyen nghi lam detection BB) | Giong VGG; ~3 downsample → FPN ngheo scale, co the giam con 3-4 level |
| **ResNet18/50** | Residual add | 🟡 TB (da co) | Output block co dinh; chi prune mid. Port logic cu sang `export_lean` |
| **ResNet101** | Residual [3,4,23,3] | 🟡 TB | Them stage_repeat; reuse Bottleneck `export_lean` |
| **MobileNetV3** | Inverted residual + **SE** + skip | 🔴 Kho | Prune **hidden(expand) dim**; SE fc1/fc2 theo hidden; depthwise groups=hidden; project output co dinh khi co skip |
| **EfficientNet-B0** | MBConv + **SE** + skip + SiLU | 🔴 Kho nhat | Nhu MobileNet + nhieu block variant + stochastic depth (eval bo qua) |

**Chung moi backbone:** FPN expand ve 256 → head input on dinh; head tower linear → prune de nhat.

---

## 6. Lo trinh theo phase (moi phase co test gate)

### Phase 0 — Nen tang (refactor, chua them backbone)
- [ ] Fix `trainer_det` logging (loss keys tong quat) → het crash RetinaNet.
- [ ] Dinh nghia giao thuc `export_lean`; refactor surgery thanh fold.
- [ ] Regression: ResNet18/50 Faster R-CNN cu van surgery ra ket qua nhu truoc.
- [ ] Builder RetinaNet + registry + `PrunableRetinaNetHead` (tower ConvBNReLU) + head surgery.
- ✅ **Gate:** RetinaNet+ResNet18 forward + 1 step loss + surgery tren dummy 320x320 khong loi shape; `lean.load_state_dict` khong thieu/thua key.

### Phase 1 — Reference end-to-end: RetinaNet + ResNet18 (full prune)
- [ ] `pipeline2_retinanet.sh` smoke (1-2 epoch/step) tren DeepFish.
- ✅ **Gate:** step1 mAP>0 → step2 ra `model_lean.pth`+`.json` → step3 nap lai chay → `compare_acc`/`benchmark` ra so.

### Phase 2 — ResNet50 + ResNet101
- [ ] Port Bottleneck `export_lean`; them resnet101.

### Phase 3 — VGG16/19 + AlexNet (linear backbones)
- [ ] HybridVGG / HybridAlexNet bang `ConvBNReLU` + pretrained remap + tap-point + `export_lean` tuyen tinh.
- Validate giao thuc tren mang khong-residual.

### Phase 4 — MobileNetV3 (moc kho dau)
- [ ] HybridMobileNetV3: `ConvBNReLU` tren inner prunable + SE-aware `export_lean` + skip co dinh.

### Phase 5 — EfficientNet-B0 (moc kho nhat)
- [ ] HybridEfficientNet: MBConv + SE + SiLU.

### Phase 6 — Tich hop
- [ ] Tong quat `benchmark.py`/`compare_acc.py`/`visualize_*` chon builder theo `arch`.
- [ ] Script rieng tung backbone; cap nhat README.

---

## 7. Rui ro lon
- **SE module surgery** (MobileNet/EfficientNet): lech kenh giua fc-SE va hidden → viet test shape rieng cho block SE.
- **Depthwise groups** khi prune hidden → rebuild dw voi groups moi (tien le: `BottleneckFPNBlock` trong `custom_fpn.py`).
- **Pretrained weight remap** tung arch (giong `load_pretrained_weights` cua ResNet) — de mismatch key.
- **AlexNet** it scale → co the giam so level FPN cho rieng no.
- **Freeze vs prune backbone**: tier cu freeze MobileNet; gio prune backbone → backbone khong freeze → can train lai BB (anh huong LR/epoch).
- **Prune tap output → FPN lateral**: bat buoc xu ly trong `fpn.export_lean` (surgery cu chua co).

---

## 8. Cach chay Pipeline 2 (RetinaNet) — du kien

```bash
# Step 1: train dense RetinaNet
python train_det.py --data-path ./NewEtroplusMaculatus --model retinanet_resnet18 \
  --epochs 60 --batch-size 8 --min-size 320 --max-size 320 \
  --output-dir ./output/p2_retina/step1_dense

# Step 2: iterative prune (backbone + FPN + head)
python prune_det.py --data-path ./NewEtroplusMaculatus --model retinanet_resnet18 \
  --checkpoint ./output/p2_retina/step1_dense/model_best.pth \
  --target-sparsity 0.5 --prune-iters 8 --finetune-epochs 10 \
  --prune-fpn --prune-head --min-size 320 --max-size 320 \
  --output-dir ./output/p2_retina/step2_pruned

# Step 3: final finetune voi lean weights + JSON
python train_det.py --data-path ./NewEtroplusMaculatus --model retinanet_resnet18 \
  --weights-backbone ./output/p2_retina/step2_pruned/model_lean.pth \
  --compress-rate ./output/p2_retina/step2_pruned/model_lean.json \
  --epochs 50 --lr 0.02 --min-size 320 --max-size 320 \
  --output-dir ./output/p2_retina/step3_final
```

---

## 9. Trang thai

- [x] Doc & hieu Pipeline 2 (Faster R-CNN)
- [x] Chot quyet dinh (RetinaNet, num_classes=2, prune BB+FPN+head, moi backbone)
- [ ] **Phase 0** (dang cho duyet de bat dau)
- [ ] Phase 1..6
