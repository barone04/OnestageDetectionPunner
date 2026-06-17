# TIẾN ĐỘ CHI TIẾT — Component-aware Bi-level Pruning cho One-stage Detector

> Repo **standalone** mở rộng khung cắt tỉa 2-level (unstructured Song Han → structured
> filter → model surgery) từ Faster R-CNN sang **one-stage detector** cho bài journal.
> **YOLOv1** = baseline (ĐÃ XONG). **SSD** = phát triển sau.
>
> `PrunedFishNet/` (tham khảo phương pháp prune) + `YOLOv1/` (tham khảo kiến trúc) **chỉ để
> đọc tham khảo**; repo này **không import / không kế thừa** gì từ chúng.

---

## 0. Toàn cảnh pipeline (data flow)

```
        (1) PREPROC DATA           (2) BUILD MODEL            (3) PRETRAIN (dense)
  images/labels (YOLO fmt)  ->  YoloModel(backbone+neck+head) -> train.py -> model_best.pth (dense)
                                                                                   |
        (4) PRUNE (bi-level)                 (5) SURGERY                           v
  Song Han (unstruct) + Filter (struct)  ->  convert_to_lean  <----------  prune.py
  + finetune xen kẽ (mask giữ zero)          (cắt kênh thật)
                                                  |
                                                  v
                                       model_lean.pth + model_lean.json
                                                  |
        (6) FINETUNE (lean)                       v                  (7) BENCHMARK / mAP
  train.py --init-config lean.json   ->  model_best.pth (final)  ->  benchmark.py
           --weights lean.pth                                        (Params/FLOPs/FPS/mAP)
```

| Giai đoạn | File chính | Trạng thái |
|---|---|---|
| 1. Preproc data | `data/dataset.py` | ✅ |
| 2. Build model | `models/` | ✅ |
| 3. Pretrain (dense) | `train.py`, `engine.py`, `loss.py` | ✅ |
| 4. Prune (bi-level) | `prune.py`, `pruning/unstructured.py`, `pruning/structured.py` | ✅ |
| 5. Surgery | `pruning/surgery.py` | ✅ + test |
| 6. Finetune (lean) | `train.py --init-config` | ✅ |
| 7. Benchmark / mAP | `benchmark.py`, `utils/metrics.py` | ✅ |
| (SSD) | — | ⏳ chưa |
| (Chạy data cá thật, GPU) | — | ⏳ chưa |

---

## 1. PREPROC DATA — `data/dataset.py`

**Mục tiêu:** đọc ảnh + nhãn YOLO-format, chuẩn hoá về input cho model.

**Input:** thư mục theo layout
```
root/images/{train,val}/*.jpg
root/labels/{train,val}/*.txt     # mỗi dòng: "cls xc yc w h"
```
**Output mỗi sample:** `(filename, img_tensor[3,H,W] float[0,1], label[N,5]=[cls,xc,yc,w,h], (h0,w0))`

**Chi tiết:**
- Ảnh: PIL → RGB → resize `(input_size,input_size)` → `/255` → tensor CHW.
  (resize vuông; **nhãn normalized nên không cần scale theo resize**.)
- Nhãn **auto-detect**:
  - `max(toạ độ) <= 1` → YOLO **normalized cxcywh** (chuẩn).
  - ngược lại → coi là **pixel x1y1x2y2** → đổi sang normalized cxcywh.
  - bỏ box `w<=0 / h<=0`.
- Ảnh không có object → `label = [[-1,0,0,0,0]]` (quy ước cho YoloLoss bỏ qua).
- **Augment** (train): hflip xác suất 0.5 → lật ảnh + `xc → 1-xc` (chỉ hàng valid).
- `collate_fn`: stack ảnh thành batch, **labels giữ dạng list** (số box mỗi ảnh khác nhau).

**Trạng thái:** ✅ chạy OK. **Chưa có**: letterbox (giữ tỉ lệ), mosaic/color-jitter (mới chỉ hflip).

---

## 2. BUILD MODEL — `models/`

**Mục tiêu:** dựng YOLOv1 fully-conv, mọi conv **prune được**.

### 2.1. Đơn vị prune — `element.py: PrunableConv`
`Conv2d + BN + Act` (act = relu/leaky/identity) + 2 mask:
- `u_mask` (4D) cho Song Han; `s_mask` (1D theo out-channel) cho Filter.
- Trong `forward`: mask được `mul_` vào `conv.weight` **trước** khi conv → giữ zero bền vững.
- `get_prunable_layers(type)` trả `MaskProxy` (để pruner thao tác).

### 2.2. Backbone — `backbone.py`
- **ResNet18/34/50/101**: stem `PrunableConv(3→64,7,s2)` + maxpool + 4 stage.
  - BasicBlock: `conv1(in→mid,3)` → `conv2(mid→out,3,act=identity)` → +residual → relu.
  - BottleNeck: `conv1(1×1)` → `conv2(3×3)` → `conv3(1×1,identity)` → +residual → relu.
  - **An toàn residual:** structured prune **chỉ cắt mid-channel** (BasicBlock: conv1;
    BottleNeck: conv1+conv2). Stem & output block giữ nguyên.
- **VGG16/19**: chuỗi `PrunableConv` + MaxPool, prune tự do mọi conv.
- `feat_dims` = 512 (r18/34, vgg) hoặc 2048 (r50/101).
- **Width-based config** (`block_mids` / `widths`) → surgery rebuild khung nhỏ đúng 100%.

### 2.3. Neck — `neck.py: ConvBlock`
5 `PrunableConv`: `1×1→512, 3×3→1024, 1×1→512, 3×3→1024, 1×1→512` (leaky).
→ Đây là **điểm component-aware**: khi backbone nhẹ, neck thành bottleneck → prune trực tiếp.

### 2.4. Head — `head.py: YoloHead`
`Conv2d 1×1: neck_out → (1 obj + 4 box) + num_classes`. **Không prune output** (ràng buộc grid/box/class).

### 2.5. Lắp ráp — `yolo.py: YoloModel`
- `forward(x)` → RAW `(B, S*S, 5+C)` (obj,box đã sigmoid; cls raw) — dùng cho **loss**.
- `infer(x)` → decode `(B, S*S, 6)` `[score,xc,yc,w,h,label]` — dùng cho **detection/mAP**.
- grid `S = input/32` (=14 @448), **1 box/cell**.
- `config()` lưu `backbone_cfg/neck_widths` → reload lean chính xác.

**Trạng thái:** ✅. **Chưa có**: nạp **ImageNet-pretrained backbone** (hiện init ngẫu nhiên —
xem mục 3, đây là điểm cần thêm để tăng accuracy thật).

---

## 3. PRETRAIN (train DENSE) — `train.py` (Step 1)

**Mục tiêu:** huấn luyện model dense (chưa nén) → checkpoint gốc để đi prune.

**Input:** data. **Output:** `step1_dense/model_best.pth` (kèm `config`).

**Chi tiết:**
- Optimizer **SGD** (lr, momentum, weight_decay) + **CosineAnnealingLR** + **AMP**.
- Loss = **YoloLoss** (`loss.py`):
  `obj(MSE, target=IoU) + 0.5·noobj(MSE,0) + 5·box(MSE) + cls(CrossEntropy)`.
  (`λ_noobj=0.5` chống mất cân bằng fg/bg — YOLOv1 không dùng focal loss.)
- Mỗi epoch: `train_one_epoch` → `evaluate` (val loss) → `evaluate_map` →
  **chọn best theo mAP@0.5:0.95** (có `--no-map` → chọn theo val-loss).
- Lưu `model_best.pth` / `model_last.pth` (có `config()` để reload).

**Lệnh:**
```bash
python train.py --data-path ./fish --backbone resnet18 --num-classes 1 \
  --epochs 120 --batch-size 16 --augment --output-dir ./output/yolo/step1_dense
```

**Trạng thái:** ✅ chạy OK (smoke test). **Lưu ý:** backbone init **ngẫu nhiên** (chưa nạp
ImageNet) → trên data thật nên bổ sung pretrained backbone để hội tụ tốt hơn.

---

## 4. PRUNE (bi-level) — `prune.py` (Step 2)

**Mục tiêu:** áp **2-level pruning** + finetune phục hồi, tạo mask cắt tỉa.

**Input:** dense `model_best.pth`. **Output (sau khi qua mục 5):** `model_lean.pth/.json`.

**Vòng lặp iterative** (`prune_iters` lần), sparsity tăng dần:
```
sparsity = target_sparsity * (i+1) / prune_iters
  (L1) Song Han : sensitivity = mult * sparsity
  (L2) Filter   : prune_ratio = sparsity
  finetune finetune_epochs  (apply_masks() sau mỗi step -> giữ zero)
  evaluate (val loss)
```

- **L1 — Song Han (`pruning/unstructured.py`):** `threshold = sensitivity·std(W)` cho từng
  layer → zero `|w| <= threshold`. Ổn định saliency trước khi cắt kênh.
- **L2 — Filter mixed-norm L1-inf-inf (`pruning/structured.py`):**
  - chuẩn `l1inftyinfty(filter) = Σ_Cin max_{H,W}|w|`.
  - tính ma trận khoảng cách giữa các filter (**per-row vectorized** → nhanh hơn O(N²) gốc),
    ưu tiên cắt cặp **giống nhau nhất** (giữ filter norm lớn, diệt norm nhỏ); chưa đủ thì cắt
    tiếp theo norm nhỏ dần → tạo `s_mask`.
- **Component-aware** `--scope {all, backbone, neck}`: prune cả model / chỉ backbone / chỉ neck.

**Lệnh:**
```bash
python prune.py --data-path ./fish --checkpoint ./output/yolo/step1_dense/model_best.pth \
  --target-sparsity 0.5 --prune-iters 8 --finetune-epochs 5 --scope all \
  --output-dir ./output/yolo/step2_pruned
```
Có báo cáo **mAP** ở baseline + lean.

**Trạng thái:** ✅. **Lưu ý perf:** Filter pruner chậm trên CPU với layer lớn → nên chạy GPU.

---

## 5. SURGERY (materialize lean) — `pruning/surgery.py`

**Mục tiêu:** biến mask thành **model dense nhỏ thật** (giảm Params/FLOPs thật, không chỉ zero).

**Quy trình `convert_to_lean(model)`:**
1. Trích **width config** số kênh còn sống: ResNet `block_mids` (BasicBlock 1 mid / BottleNeck
   `[mid1,mid2]`), VGG `widths`, neck `neck_widths`.
2. Dựng lại `YoloModel` với config nhỏ.
3. Copy weight, slice theo index kênh còn sống theo **chuỗi phụ thuộc channel**:
   - ResNet: stem + output block **cố định** → chỉ mid co lại → backbone output **full**.
   - VGG/neck: chuỗi tuyến tính `conv_k.in = conv_{k-1}.out`.
   - Head 1×1: **output cố định**, input co lại theo neck.
4. Lưu `model_lean.pth` + `model_lean.json`; `build_lean_from_config()` rebuild + verify
   `load_state_dict(strict=True)`.

**Kết quả test (structural, Song Han 0.3 + Filter 0.4, scope=all):**

| Backbone | Params | FLOPs (@448) |
|---|---|---|
| ResNet-18 | 21.93M → **10.71M** (−51.2%) | 9.40G → **5.37G** (−42.9%) |
| VGG-16 | 25.48M → **9.17M** (−64.0%) | 63.71G → **23.05G** (−63.8%) |
| ResNet-50 | 35.05M → **16.93M** (−51.7%) | 18.79G → **9.82G** (−47.8%) |

Output shape giữ nguyên; rebuild từ JSON OK. **Trạng thái:** ✅ + test.

---

## 6. FINETUNE (lean) — `train.py` (Step 3)

**Mục tiêu:** huấn luyện lại model **đã nén** để hồi phục accuracy.

**Input:** `model_lean.json` (khung) + `model_lean.pth` (weight). **Output:** `step3_final/model_best.pth`.

**Chi tiết:**
- `--init-config model_lean.json` → `build_model` dựng **đúng khung nhỏ**;
  `--weights model_lean.pth` → nạp weight lean (verify `missing=0, unexpected=0`).
- Train như Step 1 (SGD + cosine + AMP), **chọn best theo mAP**.
- `model_best.pth` lưu kèm `config()` đầy đủ → benchmark reload chính xác.

**Lệnh:**
```bash
python train.py --data-path ./fish --num-classes 1 \
  --init-config ./output/yolo/step2_pruned/model_lean.json \
  --weights     ./output/yolo/step2_pruned/model_lean.pth \
  --epochs 60 --augment --output-dir ./output/yolo/step3_final
```
**Trạng thái:** ✅.

---

## 7. BENCHMARK + mAP — `benchmark.py`, `utils/metrics.py`

- **mAP eval (`evaluate_map`):** qua `infer()` + NMS theo lớp → **VOC all-point AP** →
  **mAP@0.5** và **mAP@0.5:0.95** (không cần pycocotools).
- **benchmark.py:** bảng so sánh **dense vs pruned**: Params, FLOPs (thop), latency (ms),
  FPS, mAP@0.5, mAP@0.5:0.95 (+ Δ%).

**Lệnh:**
```bash
python benchmark.py --dense ./output/yolo/step1_dense/model_best.pth \
  --pruned ./output/yolo/step3_final/model_best.pth \
  --data-path ./fish --num-classes 1
```
**Trạng thái:** ✅.

---

## 8. Chạy full Pipeline 2

```bash
bash scripts/run_e2e.sh     # sửa CONFIG block ở đầu file (DATA, BACKBONE, SCOPE, sparsity...)
```
Step1 train dense → Step2 prune+surgery → Step3 finetune lean → Step4 benchmark.

---

## 9. Cấu trúc repo (standalone)

```
models/   element.py backbone.py neck.py head.py yolo.py
pruning/  norms.py unstructured.py structured.py surgery.py
data/     dataset.py
utils/    metrics.py
loss.py  engine.py  train.py  prune.py  benchmark.py
scripts/  run_e2e.sh
PrunedFishNet/  YOLOv1/   (reference, read-only)
```

---

## 10. Đã xong vs Còn lại

**Đã xong (✅, smoke-tested CPU):** preproc data, build model (resnet18/34/50/101 + vgg16/19),
pretrain dense, bi-level prune (Song Han + Filter), surgery, finetune lean, mAP eval, benchmark, e2e script.

**Còn lại (⏳):**
- Nạp **ImageNet-pretrained backbone** cho Step 1 (hiện random init).
- Chạy trên **data cá thật** (DeepFish/Etroplus) trên **GPU** → số liệu mAP/FPS thật cho journal.
- Augment mạnh hơn (letterbox, mosaic) nếu cần.
- **SSD** (anchor-based): matcher IoU + focal/hard-neg + smooth-L1 + scale-aware pruning.
- (Tuỳ chọn) per-class mAP table, profiling latency CPU vs GPU.
