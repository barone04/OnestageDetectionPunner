# Nghiên cứu Prune RetinaNet (Phase 3)

---

## Kiến trúc (recap ngắn)

```
Backbone ResNet18/50/101, VGG16/19 ── C3,C4,C5 ──► FPN ──► P3 P4 P5 P6 P7 (đều 256ch)
                                         │
                         ┌───────────────┴───────────────┐
                   cls subnet (shared)             box subnet (shared)
                   4×conv3x3 256→256 + conv 256→(A·K)  4×conv3x3 256→256 + conv 256→(A·4)
```
- **Backbone**: ResNet50/101, trích C3/C4/C5.
- **FPN**: lateral 1×1 + top-down upsample + **ADD** + output 3×3; thêm P6/P7. Mọi level = 256ch.
- **Head**: 2 tower **dùng chung** mọi level; predictor cuối output cố định A·K / A·4 (A=9 anchor).
- **Loss**: Focal + smooth-L1 + anchor matcher (IoU).

---

## Thuật ngữ khó (giải thích ngắn)

- **Trục P** — chiều kênh 256 chung của FPN (P3..P7 đều 256). Đụng vào nó = phải sửa đồng loạt mọi nơi dùng.
- **ADD (top-down FPN)** — `P_l = lateral(C_l) + upsample(P_{l+1})`. Cộng theo kênh → 2 nhánh buộc **cùng số kênh + cùng thứ tự index**.
- **Share weight (head)** — cls/box subnet chỉ 1 bộ weight, chạy lại cho mọi level → sửa 1 lần áp mọi level.
- **Component-aware scope** — prune riêng theo vai trò từng khối (backbone / fpn / head). (YOLO đã có `--scope`.)
- **DepGraph (Torch-Pruning)** — trace forward → tự dựng đồ thị phụ thuộc kênh → cắt cả nhóm đồng bộ (kể cả residual/FPN add) bằng `index_select`, không hardcode. Paper CVPR'23 arXiv **2301.12900**; repo **github.com/VainF/Torch-Pruning**.

---

## Task tự nghiên cứu

1. **Input size** — nên đặt bao nhiêu cho từng dataset (COCO/VOC/cá)? Theo paper + các bài prune.
2. **Map module torchvision RetinaNet** — tên chính xác các Conv prunable ở backbone/FPN/head.
3. **Chuỗi phụ thuộc channel** — liệt kê mọi chỗ ADD / share → nhóm phải cắt đồng bộ.
4. **Prune in/out** — lập bảng: mỗi module cắt out? in? cả hai? chỗ nào output cố định (predictor).
5. **FPN trục P** — có prune được không? Cần grouped/DepGraph, bottleneck FPN (PrunedFishNet) hay có đề xuất phương pháp nào có thể p dụng khác ko? So (A) giữ P cố định vs (B) grouped-prune.
6. **Phương thức prune** — tái dùng bi-level (Song Han → Filter → surgery)? scope gồm gì?
7. **Loss** — tự thiết kế hay bọc torchvision (focal/matcher/nms)?
8. **Surgery topology rẽ nhánh** — **Tự xây pp Sugery** (orchestrator hay per-module đều phải làm lại cho mỗi arch) vs **DepGraph** (tự trace forward → group trục P → bỏ hardcode, khỏi viết lại). Chỉ DepGraph mới thực sự scale sang SSD/RetinaNet không viết lại — đọc Torch-Pruning để quyết.
9. **Pretrained** — cần pretrain gì (ImageNet/COCO)? Forward-verify khi swap PrunableConv.
10. **Scale-aware** — khảo sát phân bố size cá → sparsity khác nhau per-level P3..P7?
11. **Pipeline Step1–4** — thiết kế giống YOLO/SSD (train dense → prune+surgery → finetune lean → benchmark).

---

## Slide phân tích kiến trúc — yêu cầu

Bắt buộc thể hiện được:
- [ ] Số kênh (channel) từng tầng.
- [ ] Điểm ADD / concat / share weight (khoanh rõ).
- [ ] Layer output cố định (predictor = anchors×classes).
- [ ] Chuỗi phụ thuộc producer→consumer.
- [ ] Đánh dấu prunable vs giữ nguyên.
- [ ] Input size + stride mỗi level P3..P7.
