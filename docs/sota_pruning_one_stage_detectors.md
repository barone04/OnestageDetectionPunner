# Khảo sát method pruning one-stage detector (2023–2026)

Câu hỏi gốc: các bài pruning gần đây thực nghiệm trên one-stage model nào, dataset nào, kết quả ra
sao, pipeline của họ hoạt động thế nào.

Bảng các điểm vận hành xếp theo ô để vẽ đường cong: xem [pruning_cells_comparison.md](pruning_cells_comparison.md).

Cách lập: 5 luồng tìm kiếm song song. Mỗi bài đều **mở link gốc để xác nhận tiêu đề**; số nào không
đọc được trên trang đã mở thì ghi `[?]` hoặc "chưa kiểm chứng" chứ không đoán. Paywall thì xác minh
danh tính qua Crossref/PMC/OpenAlex.

---

## 1. Trả lời ngắn cho 3 câu hỏi

| Câu hỏi | Trả lời |
|---|---|
| Có ai prune **YOLOv1** 2023–2026 không? | **Không có bài nào.** Pruning YOLO gần đây nhắm v5/v7/v8/v10/v11, YOLOX, YOLOF. |
| Có ai prune trên **DeepFish** không? | **Không.** DeepFish có được dùng làm benchmark cho detector nhẹ (vd YOLOv10-nano, [arXiv 2509.17406](https://arxiv.org/abs/2509.17406)) nhưng chưa ai prune trên đó. |
| **SSD** còn được dùng làm benchmark pruning không? | **Còn, nhưng mỏng** — 5–6 bài 2023–2025, trong đó có TPAMI, AAAI, IJCV, WACV. Chi tiết §3. |

Hai khoảng trống lớn, cả hai đều xác minh được từ chính các bài:

1. **Không bài 2023–2026 nào áp dụng unstructured + structured cùng lúc lên một one-stage
   detector.** Gần nhất: IEE (WACV'25) dùng chung một tiêu chí cho cả hai mức nhưng chạy thành hai
   thí nghiệm tách rời; UniPTS/CAP kết hợp unstructured với N:M; YOLObile (AAAI'21, ngoài khung) chỉ
   *so sánh* hai loại trên YOLOv4.
2. **Gần như mọi bài chỉ prune backbone, giữ nguyên head.** HALP: "We only prune the backbone network
   in the detector". PAGCP loại output head khỏi tập layer. OCSPruner: "the SSD head was left
   unchanged… channel pruning in the head would be needed for larger parameter reductions". CAP giữ
   head dense. **Chưa ai prune neck/FPN + head có xử lý dependency tử tế.**

---

## 2. Bảng tổng hợp chính

mAP50 = IoU 0.5; mAP = COCO 50-95. "FLOPs↓" là mức giảm so với baseline của chính bài đó.

| # | Method | Venue, năm | Detector + backbone | Dataset (split) | Baseline → pruned | FLOPs↓ / Params↓ | Code |
|---|---|---|---|---|---|---|---|
| 1 | [PAGCP](https://arxiv.org/abs/2303.11923) | TPAMI 2023 | SSD300-VGG16 | VOC07+12 → VOC07 test | 77.5 → **75.6** mAP50 | 74.3% / 85.6% | [có](https://github.com/HankYe/PAGCP) |
| 1b | PAGCP | " | YOLOv5m | VOC07 test | 83.2 → **81.7** mAP50 | 64.8% / 78.0% | " |
| 1c | PAGCP | " | YOLOv5m | COCO val2017 | 43.6 → **41.5** mAP (AP50 62.7 → 60.7) | 54.2% / 64.0% | " |
| 1d | PAGCP | " | SSD512-VGG16 | COCO | 29.4 → **28.6** mAP (AP50 48.3 → 47.6) | 43.4% / 41.4% | " |
| 1e | PAGCP | " | CenterNet-R101 | VOC07 test | 76.6 → **77.2** mAP50 | 33.9% / – | " |
| 2 | [REPrune](https://arxiv.org/abs/2402.17862) | AAAI 2024 | SSD300-ResNet50 | COCO val2017 | 25.2 → **25.0** mAP (AP50 42.7 → 42.3) | 50% (chỉ backbone) | – |
| 3 | [OCSPruner](https://arxiv.org/abs/2501.13439) | arXiv 2025 (v2) | SSD300 + ResNet50 đã prune | COCO | 26.90 → 25.78 / 25.30 / **23.82** mAP | 25% / 34% / 56% | [có](https://github.com/ghimiredhikura/OCSPruner) |
| 4 | [IEE](https://arxiv.org/abs/2502.03658) | WACV 2025 | SSD512-ResNet50 | VOC07 test | `[chưa kiểm chứng]` — chỉ có Fig. 4 | – | – |
| 5 | [MDP](https://arxiv.org/abs/2504.02168) | CVPR 2025 | SSD512-ResNet50 | VOC (giao thức HALP) | 78.0 → **80.0** mAP50 | ×1.84 speedup | – |
| 6 | [AdaSAP](https://arxiv.org/abs/2306.14306) | ICLR 2024 | SSD512-ResNet50 | VOC + VOC-C | HALP 0.774 → **0.795** (size ratio 0.40) | – | – |
| 7 | [HALP](https://arxiv.org/abs/2210.06659) | NeurIPS 2022 | SSD512-ResNet50 | VOC07+12 → VOC07 test | 77.98 → **77.42** mAP50 | 65.6 → 15.4 GFLOPs, ×1.94 | – |
| 8 | [SeerNet](https://arxiv.org/abs/2304.06393) | IJCV 2023 | SSD-VGG16 | VOC07 test | 72.4 → **66.8** mAP50 | nén 36× (prune + quantize) | – |
| 9 | [Info-theoretic](https://arxiv.org/abs/2408.16772) | arXiv 2024 (DOI TNNLS chưa resolve) | RetinaNet-R50 | COCO val2017 | 36.5 → **37.6** mAP | ~50% MACs | – |
| 9b | " | " | FSAF / ATSS / PAA | COCO val2017 | 37.4→**38.0** / 38.1→**39.1** / 39.0→**40.0** | ~50% MACs | – |
| 10 | [IFSO](https://arxiv.org/abs/2308.06755) | arXiv 2023 | RetinaNet-R50 | COCO val2017 | 36.10 → **36.30** (50%) / 35.70 (60%) | 50–60% | [có](https://github.com/hrcheng1066/IFSO) |
| 11 | [R-TOSS](https://arxiv.org/abs/2303.02191) | DAC 2023 | RetinaNet, YOLOv5s | KITTI | 3EP 79.45 / 2EP 82.9 (baseline `[chưa kiểm chứng]`) | nén 2.89×, Jetson TX2 ×1.87 | – |
| 12 | [LDT](https://arxiv.org/abs/2509.17968) | WACV 2026 | ATSS / GFL / YOLOF | COCO | 39.4→39.5 / 40.2→39.8 / 37.5→**38.1** | 15–25% PR | – |
| 12b | LDT | " | YOLOX-S / ATSS / GFL | KITTI (8:2) | 90.7→89.6 / 89.7→**90.2** / 89.7→**90.4** | YOLOX-S 62.1% / 74.8% | – |
| 13 | [Saliency-guided](https://arxiv.org/abs/2303.02512) | IEEE IV 2023 | YOLOX-S | KITTI (nửa tập) | 87.3 → **89.1** (rate 0.3) / 86.6 / 81.1 | 29.2% / 40.2% | – |
| 13b | " | " | YOLOF-R50 | KITTI | 87.4 → **87.2** (rate 0.3) | 43.1% / 36.9% | – |
| 14 | [Neurocomputing 611](https://doi.org/10.1016/j.neucom.2024.128656) | Neurocomputing 2024 | YOLOX-S | KITTI | +1.8 mAP tại −40.2% params | – | – |
| 15 | [HESSO](https://arxiv.org/abs/2409.09085) | arXiv 2024 | YOLOv5l | COCO | 47.71 → **44.5** mAP (AP50 66.31 → 63.1) | còn 49% params | [có](https://github.com/microsoft/only_train_once) |
| 16 | [Iterative Filter Pruning](https://arxiv.org/abs/2405.03715) | IJCNN 2024 | YOLOv7, YOLOv7-tiny | COCO | `[chưa kiểm chứng]` — chỉ có Fig. 8–9 | tới 80% sparsity | [có](https://github.com/fzi-forschungszentrum-informatik/iterative-yolo-pruning) |
| 17 | [PruningBench](https://arxiv.org/abs/2406.12315) | arXiv 2024 | YOLOv8 (biến thể không ghi) | COCO | 49.993 → 44.78 (2×) / 40.85 (3×) / 38.71 (4×) | 2× / 3× / 4× | [có](https://github.com/HollyLee2000/PruningBench) |
| 18 | [RGP](https://doi.org/10.1016/j.isci.2024.111618) | iScience 2024 | YOLOv8n / YOLOv8x | GTSDB, Seaships, COCO | GTSDB 0.920→0.919 mAP50; COCO 0.519→0.450 mAP50 | 80–90% | [có](https://github.com/ydf131745/RT-model-compress) |
| 19 | [Sparse-IFT](https://arxiv.org/abs/2303.11525) | ICML 2024 | RetinaNet + Sparse-Wide R18 | COCO 2017 | 29.3 → **34.5** mAP (90% sparsity, iso-FLOP) | unstructured | [có](https://github.com/CerebrasResearch/Sparse-IFT) |
| 20 | [UniPTS](https://arxiv.org/abs/2405.18810) | CVPR 2024 | SSD-MobileNetV1 | VOC | POT 57.02 → **64.42** mAP50 (90% sparsity) | unstructured | – |
| 21 | [FPGM face detectors](https://arxiv.org/abs/2311.16613) | WACVW 2024 | EXTD, EResFD (SSD/FPN) | WIDER FACE | EResFD 0.866 → **0.873** (Easy, 10%) | 8–42% sparsity | [có](https://github.com/IDT-ITI/) |

---

## 3. Theo từng họ detector

### 3.1 SSD — còn dùng, nhưng là "mục phụ" trong các bài pruning tổng quát

Sáu bài 2023–2025 có số SSD: **PAGCP (TPAMI)**, **REPrune (AAAI)**, **SeerNet (IJCV)**,
**IEE (WACV'25)**, **OCSPruner (arXiv)**, **MIEP (ACM MM 2023, paywall)**.

Lưu ý quan trọng:
- **PAGCP là bài duy nhất prune toàn bộ SSD (kể cả extras) và công bố bảng VOC07 test đầy đủ theo
  từng vòng.** REPrune và OCSPruner chỉ prune backbone; SeerNet gộp prune với quantize.
- **Không bài peer-review nào 2023–2026 prune SSD-MobileNet.** Chỉ xuất hiện ở venue thấp.
- Baseline dense để đặt vào bảng của mình: SSD300-VGG16 = 77.5 mAP50 (VOC07 test, PAGCP);
  SSD300-ResNet50 = 25.2 mAP (COCO val2017, REPrune); SSD512-ResNet50 = 77.98 mAP50 (VOC, HALP).

### 3.2 RetinaNet và họ anchor-free — đây mới là "case study mặc định"

RetinaNet là detector mà các bài pruning tổng quát hay dùng nhất: info-theoretic (2024),
IFSO (2023), R-TOSS (DAC'23), Sparse-IFT (ICML'24). Giao thức chuẩn
(**RetinaNet/FSAF/ATSS/PAA trên COCO val2017, giảm 50–60% FLOPs, dùng MMDetection**) bắt nguồn từ
Group Fisher Pruning (ICML'21) và vẫn được dùng tới 2026.

**FCOS: gần như không ai prune.** Agent tìm qua arXiv API, OpenAlex full-text, Crossref và 4 lần
tìm web, không ra bài 2023–2026 nào prune FCOS. FCOS chỉ xuất hiện như một dòng baseline.
**EfficientDet: không có bài pruning nào** — BiFPN coupling quá nặng là lý do có thể.

### 3.3 YOLO — nhiều bài, nhưng không ở venue top

Agent xác nhận **DepGraph (CVPR'23), Isomorphic Pruning (ECCV'24), MDP (CVPR'25), ATO (CVPR'24),
OTOv2 (ICLR'23) đều không có kết quả YOLO**. Pruning YOLO nằm ở TPAMI/Neurocomputing/iScience,
IJCNN/WACV/DAC, và preprint.

Nguồn tốt nhất để so nhiều tiêu chí trên cùng một giao thức: **PruningBench** — 16 phương pháp trên
YOLOv8/COCO, finetune 100 epoch, SGD lr 0.01, batch 128. Nhận xét của chính bài:
*"in the YOLO experiments, the performance differences among various pruning methods are minimal"* —
tức trên YOLO thì **cách gom nhóm (grouping/strategy) quan trọng hơn tiêu chí importance**.
Nhược điểm: bảng YOLOv8 chỉ nằm trên repo, không ghi rõ biến thể model lẫn loại mAP.

**Không tìm thấy bài pruning nào cho YOLOv6, YOLO-NAS, PP-YOLOE, YOLOv10, YOLOv11 trên COCO.**

### 3.4 Dataset nhỏ chuyên ngành — publish được, nhưng chỉ ở tạp chí tầm trung

15 bài được xác minh ở nhóm này: 9 bài ở Sensors / Measurement / MST / PeerJ CS / PLOS ONE /
Buildings / Aquaculture International, 5 preprint, 1 workshop, **0 bài ở CVPR/ICCV/ECCV/NeurIPS**.

Vài mốc hữu ích:
- [PR-YOLO](https://doi.org/10.3390/s24237711) (Sensors 2024), YOLOv5m trên VisDrone: 38.1 → 36.2
  mAP50, params 20.9M → 1.2M, FLOPs 48.0G → 6.2G, Jetson AGX Xavier 29 → 49 FPS.
- [MCP-YOLO](https://doi.org/10.3390/s25227049) (Sensors 2025): dataset riêng **3.091 ảnh, 2 lớp**,
  vẫn đăng được; mAP50 0.909 → 0.921 sau prune.
- [YOLOX-S structural pruning](https://ceur-ws.org/Vol-4082/paper12.pdf) (workshop ATAIT 2025):
  dataset riêng **720 ảnh**; mAP50 97.76 → 97.18, params −43.4%; FPS trên Raspberry Pi là **ước
  lượng, không đo thật**.
- [RGP](https://doi.org/10.1016/j.isci.2024.111618) (iScience 2024) là khuôn mẫu đáng copy nhất:
  **một dataset nhỏ (GTSDB 900 ảnh) + một trung bình (Seaships 7k) + COCO**, báo cáo cả mAP50 lẫn
  mAP50-95, có code.

So với mức đó, DeepFish (4.505 ảnh, công khai, có trích dẫn) là **trên chuẩn** của nhóm venue này.

---

## 4. Tóm tắt pipeline các method chính

**PAGCP** (structured, iterative). Tiêu chí "performance-aware oracle": đo độ nhạy từng filter với
**từng task loss**, quy đổi qua hàm mũ âm của mức tụt hiệu năng; mỗi vòng bị ràng buộc bởi ngân sách
tụt hiệu năng toàn cục. Coupling: kênh skip-connection cùng index gom thành một nhóm, chấm điểm bằng
trung bình nhóm. Output head bị loại. Epoch finetune tỉ lệ với mức nén.

**REPrune** (structured, prune trong lúc train). Gom cụm agglomerative các kernel trong mỗi channel,
chọn filter phủ được các đại diện cụm (max-coverage).

**OCSPruner** (structured, one-cycle). Chỉ số "ổn định" (độ giống nhau của subnetwork giữa hai epoch)
quyết định **khi nào** cắt, thay vì lặp. Dùng **DepGraph** để gom nhóm.

**HALP → AdaSAP → IEE → MDP** (dòng NVIDIA). HALP: knapsack theo latency. AdaSAP: nhiễu trọng số thích
nghi đẩy vùng sắp cắt vào vùng loss phẳng. IEE: xen kẽ exploitation/exploration, **một tiêu chí dùng
chung cho cả weight và channel sparsity**. MDP: giải MIP đồng thời trên channel + layer + block dưới
ràng buộc latency.

**R-TOSS** (semi-structured, DAC'23). Pattern 3 hoặc 2 entry cho kernel 3×3; kernel 1×1 gom tạm thành
ma trận 3×3 rồi cắt theo pattern. Coupling: **DFS trên computational graph** dựng nhóm parent–child.

**Iterative Filter Pruning** (IJCNN'24, YOLOv7). 14 vòng. Xử lý concat/ELAN bằng đồ thị liên kết lớp
mà trọng số cạnh mã hoá **slice nào của concat** phải bị cắt. Finetune thích nghi theo mAP hiện tại.

**Saliency-guided / LDT** (Choi & Tian). Tiêu chí **đặc thù detection**: gradient của detection loss
theo feature map, **nhân trọng số theo vị trí GT box** cộng biên context. Importance tính trên 50–256
ảnh calibration.

**Sparse-IFT** (unstructured, ICML'24). Thay lớp dense bằng biến thể sparse **iso-FLOP**, sparsity động
kiểu RigL, backbone **giữ nguyên sparse** khi finetune.

---

## 5. Phụ lục:  Các bản YOLO mới nhất có được prune không? (13/10/2026)

**Không.** "YOLOv25" không tồn tại (YV25 = tên hội nghị YOLO Vision 2025, model là YOLO26).

| Bản | Ngày ra | Số bài pruning | Trễ so với release |
|---|---|---|---|
| [YOLOv9](https://arxiv.org/abs/2402.13616) | 02/2024 | **0** | 32 tháng, vẫn chưa có |
| [YOLOv10](https://arxiv.org/abs/2405.14458) | 05/2024 | 1 | ~22 tháng |
| [YOLO11](https://docs.ultralytics.com/models/yolo11/) | 09/2024 | ~16 | 7–11 tháng |
| [YOLOv12](https://arxiv.org/abs/2502.12524) | 02/2025 | 7 | ~9 tháng |
| [YOLOv13](https://arxiv.org/abs/2506.17733) | 06/2025 | 4 | ~2 tháng |
| [YOLO26](https://docs.ultralytics.com/models/yolo26/) | 01/2026 | 2 | ~4 tháng |
| [YOLO27](https://docs.ultralytics.com/models/yolo27/) | công bố 09/2026 | 0 | **chưa phát hành weights** |

~70–80% bài pruning YOLO của 2025–2026 vẫn dùng **v5/v7/v8**. Độ trễ tỉ lệ với mức tiện dụng trong hệ
sinh thái Ultralytics, không tỉ lệ với độ mới: YOLOv13 là fork Ultralytics nên bị prune sau 2 tháng,
còn YOLOv9/v10 nằm ngoài codebase nên gần như bị bỏ qua.

### Coupling của attention làm structured pruning gần như vô hiệu

Hai bài tạp chí 2026 độc lập prune YOLOv13 trung thực và đều thu được mức giảm gần bằng không:

- **RDD-YOLO** ([Sci. Rep. 2026](https://doi.org/10.1038/s41598-026-64451-6)): params **−0.9%**,
  GFLOPs −3.3%. Nguyên văn: *"the actual number of pruned layers is only 10, and a total of 184
  channels are removed"*, vì *"some layers cannot be pruned due to… the failure to satisfy the
  coupling conditions"*.
- **ISA-YOLO** ([Plant Methods 2026](https://doi.org/10.1186/s13007-026-01566-x)): params **−0.2%**.
  Nguyên văn: *"the coupled architecture leaves limited removable redundancy; pruning behaves more
  like mild regularization than the primary source of acceleration"*.
- **YOLOv13-Cone-Lite** ([Applied Sciences 15(17):9501](https://doi.org/10.3390/app15179501)) so hai
  tiêu chí trên YOLOv13s: **BN-γ sparsity 94.2 → 92.7** mAP50, **structural reparameterization
  94.2 → 93.8**. Tiêu chí theo BN hại nhiều hơn hẳn trên kiến trúc có attention.
- **Water 2026** ([18(16):2029](https://doi.org/10.3390/w18162029)) nói thẳng lý do không prune
  YOLO11/YOLO26: *"the C3k2/C2PSA components of these architectures do not adhere to the same pruning
  rule"*; và *"reductions in parameter count and GFLOPs do not serve as reliable predictors of
  round-trip latency on the target VPU"*.

Hệ quả: bài nào **tuyên bố giảm mạnh trên v12/v13 thì thực ra không prune phần attention** — hoặc chỉ
prune nhánh CNN do chính họ thêm vào (DELP-YOLOv12, F3-YOLO), hoặc prune cổng/đường dẫn (MambaACE),
hoặc chỉ đổi width multiplier trong YAML (YOLO-SAT, Aero-YOLO).

### Duy nhất một bài ≥ v9 có số COCO

[MambaACE-YOLO](https://doi.org/10.3390/app16157845) (Applied Sciences 2026), dòng YOLOv13, COCO 2017
val, mAP50-95: YOLOv13-N 41.6 → chưa prune 42.5 → **đã prune 42.4** (params 2.6M → 2.3M, −11.5%).
Nhưng đây là prune **đường dẫn/cổng**, không phải prune kênh; tác giả cũng thừa nhận không có control
YOLOv13-N cùng lịch 600 epoch.

### Rào cản công cụ

1. **Torch-Pruning không có ví dụ nào quá YOLOv8**; README của chính ví dụ yolov8 ghi *"This example
   is crashed due to ultralytics upgrade"*.
2. **Issue #454 "YOLO11 got stuck during pruning while generating pruner"** mở 02/2025, **vẫn open**.
3. **Ultralytics đóng PR #21977** xin thêm structured pruning (mở 07/09/2025, đóng không merge
   14/02/2026).
4. **Attention block không có BN γ để xếp hạng** — C2PSA của YOLO11, area-attention của YOLOv12.
5. **Head NMS-free/bỏ DFL** của YOLOv10/YOLO26 đổi ngữ nghĩa kênh ở head; **chưa có công trình nào xử
   lý coupling này**.
6. **AGPL-3.0 toàn bộ.**

### Hai rủi ro phải nói rõ nếu đi hướng model mới

- **mAP50 che giấu thiệt hại.** YOLO26-RipeLoc cắt 30%: mAP50 giảm 0.01 điểm nhưng **mAP50-95 giảm
  2.00 điểm**, và phần chữ của bài chỉ nói về mAP50.
- **params/GFLOPs không dự đoán được latency thật** trên VPU/edge. Phải đo trực tiếp.

---

## 6. Đã kiểm tra và loại

| Bài | Venue | Vì sao loại |
|---|---|---|
| [DepGraph](https://arxiv.org/abs/2301.12900) | CVPR 2023 | §4.1: *"This paper focuses on classification tasks"* — **không có detector**. YOLO chỉ có trong repo |
| [Isomorphic Pruning](https://arxiv.org/abs/2407.04616) | ECCV 2024 | Chỉ ImageNet |
| [ATO](https://arxiv.org/abs/2403.14729) | CVPR 2024 | Chỉ CIFAR + ImageNet |
| [OTOv2](https://arxiv.org/abs/2303.06862)/[v3](https://arxiv.org/abs/2312.09411) | ICLR 2023 | grep toàn PDF: 0 kết quả detection |
| [MaxQ](https://arxiv.org/abs/2312.07061) | CVPR 2024 | N:M sparsity nhưng chỉ two-stage |
| [CAP](https://arxiv.org/abs/2210.09223) | NeurIPS 2023 | Detection chỉ có DETR |
| [DCODP](https://doi.org/10.1007/s11063-024-11697-8) | NPL 2025 | Tên hứa hẹn về coupling nhưng chỉ YOLOv7/v8 + Faster R-CNN |
| [Pruning DETR](https://doi.org/10.1007/s11760-023-02719-4) | SIViP 2024 | Chỉ DETR |
| I-YOLOv11n | Sensors 25(15):4857 | Model "đã prune" **lớn hơn** baseline (3.87M vs 2.58M) — **không trích dẫn** |
| [Explainability-inspired](https://arxiv.org/abs/2602.14040) | arXiv 2026 | Baseline RetinaNet 0.151 mAP là phi lý — **không trích dẫn** |

**Nên lấy qua thư viện trường:** [MIEP](https://doi.org/10.1145/3581783.3612563) (ACM MM 2023, pruning
thiết kế riêng cho detector — đáng lấy nhất), [Neurocomputing 611](https://doi.org/10.1016/j.neucom.2024.128656),
[DASP](https://doi.org/10.1109/aibdf67964.2025.11440919) (claim YOLOv12-n −58.3% FLOPs / −1.2% mAP),
[MPCS](https://link.springer.com/chapter/10.1007/978-981-95-3052-6_14) (KSEM 2025, YOLOv12).

**Số cần kiểm chứng lại trước khi đưa vào luận văn:**
- PruningBench: baseline 49.993 không ghi biến thể YOLOv8 lẫn loại mAP.
- IEE (WACV'25): số SSD512 chỉ nằm trong hình.
- SMCP: mAP 78.3 là MDP đọc từ hình của SMCP, **không phải số SMCP công bố**.
- LDT: intro ghi YOLOX-S "+1.6% mAP" nhưng Table 4 là 90.7 → 89.6; Fig. 3 nói iterative còn §4.2.2 nói
  "even one iteration".
- Info-theoretic (arXiv 2408.16772): DOI TNNLS ghi trên arXiv **không resolve qua Crossref**.
- R-TOSS: tự mâu thuẫn giữa "mAP IoU 0.5" và "AP@[.5:.95]".
