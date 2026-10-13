# Pruning one-stage detector — các ô so sánh (2023–2026)

Mục đích: chọn **ô** (detector + backbone + dataset + split) để đặt kết quả bi-level của ta vào, sao
cho có sẵn nhiều điểm của người khác để vẽ thành đường cong accuracy–FLOPs.

Cách lập: 5 luồng tìm kiếm song song. Mỗi bài đều **mở link gốc để xác nhận tiêu đề**; số nào không
đọc được trên trang đã mở thì ghi `[?]` chứ không đoán. Paywall thì xác minh qua Crossref/PMC/OpenAlex.

---

## 0. Quy tắc đọc bảng

Một "ô" chỉ hợp lệ khi **cùng detector, cùng backbone, cùng dataset, cùng split, cùng metric**. Trong
một ô, các điểm của nhiều bài khác nhau ghép được thành đường cong. Khác ô thì **không được vẽ chung**.

Ba cái bẫy, cả ba đều có thật trong dữ liệu dưới đây:

| Bẫy | Biểu hiện | Xử lý |
|---|---|---|
| **Khác trục** | Nhóm NVIDIA báo *speedup đo trên Titan V* và *size ratio*, không báo %FLOPs. OCSPruner báo **FLOPs còn lại**, không phải FLOPs đã cắt | Quy về cùng một trục trước khi vẽ; ô D phải để riêng |
| **Khác phạm vi cắt** | Đa số chỉ prune **backbone**; PAGCP prune cả extras. REPrune báo %FLOPs **chỉ của backbone** | Ghi cột "Phạm vi" cho từng điểm; chỉ so điểm cùng phạm vi |
| **Khác metric** | Nhiều bài chỉ có mAP50; vài bài chỉ ghi "mAP" mà không nói IoU | Ghi rõ metric; không trộn mAP50 với mAP50-95 |

**Khác baseline dense thì so Δ, không so trị tuyệt đối.** Trong mỗi bảng dưới đây, Δ luôn tính theo
baseline của **chính bài đó**.

---

## Ô A — YOLOv8 / COCO ⭐ dày nhất

**48 điểm, một giao thức duy nhất, một nhóm chạy.** Đây là thứ mà bảng ghép từ nhiều bài không bao
giờ có được: nhiễu giữa các điểm là nhiễu của phương pháp, không phải nhiễu của recipe.

Nguồn: [PruningBench](https://arxiv.org/abs/2406.12315) + [leaderboard trên repo](https://github.com/HollyLee2000/PruningBench).
Nền tảng Torch-Pruning/DepGraph; finetune 100 epoch, SGD lr 0.01, batch 128, wd 5e-4.
Baseline = **49.993** (repo không ghi biến thể YOLOv8 lẫn loại mAP — khớp với mAP50-95 của YOLOv8m, nhưng đó là **suy đoán**).

### A.1 — 12 tiêu chí importance thuần

| Tiêu chí | 2× | 3× | 4× |
|---|---|---|---|
| ThiNet | **44.64** | 39.32 | 37.42 |
| LAMP | 44.46 | **40.11** | 37.95 |
| Magnitude-L2 | 44.38 | **40.64** | 38.36 |
| OBD-Hessian | 44.33 | 38.90 | 36.72 |
| BNScale | 44.16 | 39.35 | 38.13 |
| Magnitude-L1 | 44.00 | 39.26 | **38.58** |
| Taylor | 43.40 | 39.24 | 34.75 |
| FPGM | 43.17 | 37.52 | 34.00 |
| HRank | 42.98 | 38.96 | 34.27 |
| Random | 42.80 | 37.87 | 33.21 |
| CP | 42.41 | 38.51 | 35.41 |

### A.2 — kèm regularizer (tốt hơn mọi tiêu chí thuần)

| Tổ hợp | 2× | 3× | 4× |
|---|---|---|---|
| BNScale + BNScale | **44.78** | 40.10 | 38.38 |
| MagL2 + GroupLASSO | 44.75 | 40.53 | **38.71** |
| BNScale + GroupLASSO | 44.54 | 39.64 | 38.50 |
| MagL2 + GrowingReg | 44.44 | 40.74 | 38.51 |
| MagL2 + GroupNorm | 44.29 | **40.85** | 38.67 |

**Nhận xét của chính bài:** *"in the YOLO experiments, the performance differences among various
pruning methods are minimal"* — trên YOLO, **cách gom nhóm quan trọng hơn tiêu chí importance**.
Khoảng cách giữa tiêu chí tốt nhất và tệ nhất ở 2× chỉ 2.4 điểm, trong khi regularizer kéo thêm
được 0.3–0.5 điểm.

**Ý nghĩa với ta:** nếu bi-level chỉ hơn 0.3–0.5 điểm thì nằm trong vùng mà ngay cả việc đổi
regularizer cũng đạt được. Phải hơn rõ hơn thế, hoặc phải thắng ở trục khác (latency thật, mAP50-95).

---

## Ô B — SSD300-VGG16 / VOC07+12 → VOC07 test (mAP50) ⭐ rẻ nhất để chạy

Baseline dense ~77.4–77.5.

| Method | Venue, năm | FLOPs↓ | Params↓ | mAP50 | Δ | Phạm vi | Link |
|---|---|---|---|---|---|---|---|
| Localization-aware CP | arXiv 2019 | 50% | 50% | 77.2 | −0.2 | conv1-1→conv5-3 | [1911.02237](https://arxiv.org/pdf/1911.02237) |
| ThiNet (dòng so sánh) | " | 50% | 50% | 74.6 | −2.8 | " | " |
| AIP | arXiv 2021 | 54.1% | 57.7% | 75.20 | −0.90 | `[?]` | [2108.13591](https://arxiv.org/pdf/2108.13591) |
| PAGCP vòng 1 | TPAMI 2023 | 59.4% | 67.3% | 76.2 | −1.3 | **cả extras** | [2303.11923](https://arxiv.org/abs/2303.11923) |
| PAGCP vòng 2 | " | 66.3% | 78.3% | 75.9 | −1.6 | " | " |
| PAGCP vòng 3 | " | 71.1% | 82.1% | 75.7 | −1.8 | " | " |
| PAGCP vòng 4 | " | **74.3%** | **85.6%** | 75.6 | −1.9 | " | " |
| Localization-aware CP | arXiv 2019 | 75% | 75% | 75.2 | −2.2 | conv1-1→conv5-3 | [1911.02237](https://arxiv.org/pdf/1911.02237) |
| ThiNet (dòng so sánh) | " | 75% | 75% | 72.7 | −4.7 | " | " |

**Ưu:** PAGCP cho sẵn 4 mức → đã là một đường cong; có code công khai; là bài duy nhất prune cả phần
extras của SSD. VOC rẻ hơn COCO nhiều.
**Nhược:** đa số điểm là 2019–2021. Tốc độ 2.7–3.3× của PAGCP đo trên RTX 3090 / ARM / OpenCL.

---

## Ô C — RetinaNet-R50(-FPN) / COCO train2017 → val2017 (mAP50-95)

Baseline dense 36.1–36.5 tuỳ bài. Giao thức gốc từ Group Fisher Pruning (ICML'21), vẫn dùng tới 2026.

| Method | Venue, năm | FLOPs/MACs↓ | mAP | Δ | Phạm vi | Link |
|---|---|---|---|---|---|---|
| Info-theoretic "Ours-F" | arXiv 2024 | ~50% MACs | **37.6** | **+1.1** | backbone | [2408.16772](https://arxiv.org/abs/2408.16772) |
| Info-theoretic "Ours-P" | " | ~50% MACs | 37.3 | +0.8 | " | " |
| IFSO | arXiv 2023 | 50% | 36.30 | +0.20 | backbone + neck + head | [2308.06755](https://arxiv.org/abs/2308.06755) |
| Group Fisher | ICML 2021 | 50% | 36.20 | +0.10 | `[?]` | [2108.00708](https://arxiv.org/abs/2108.00708) |
| IFSO | arXiv 2023 | 60% | 35.70 | −0.40 | " | " |
| Info-theoretic "Ours-N" | arXiv 2024 | ~50% MACs | 35.6 | −0.9 | " | " |
| Group Fisher | ICML 2021 | 60% | 35.30 | −0.80 | `[?]` | " |

**Kèm miễn phí 3 detector anchor-free** từ cùng bài info-theoretic, cùng COCO val2017, ~50% MACs:
FSAF 37.4 → **38.0**; ATSS 38.1 → **39.1**; PAA 39.0 → **40.0**.

**Lưu ý quan trọng:** ở mức ~50%, **ba trong bảy điểm có Δ dương**. Tức là ở vùng nén này RetinaNet
còn dư thừa nhiều, prune còn giúp regularize. Muốn chứng minh phương pháp tốt thì phải đẩy lên 60–70%,
nơi các điểm bắt đầu âm.

---

## Ô D — SSD512-ResNet50 / VOC07+12 → VOC07 test ⚠️ trục là SPEEDUP, không phải FLOPs

Baseline dense 77.98 (HALP) / 78.0 (MDP). Năm bài cùng dòng NVIDIA, dùng chung baseline — ô dày
người nhất, nhưng **không ai báo %FLOPs**, nên không ghép thẳng vào trục của ô B/C được.

| Method | Venue, năm | Trục của họ | mAP50 | Δ | Link |
|---|---|---|---|---|---|
| HALP | NeurIPS 2022 | 65.56 → 15.38 GFLOPs, ×1.94 (BS=1) | 77.42 | −0.56 | [2210.06659](https://arxiv.org/abs/2210.06659) |
| SMCP | ECCV 2022 | ×2.12 | 78.3 `[?]` | `[?]` | [2211.02206](https://arxiv.org/abs/2211.02206) |
| AdaSAP | ICLR 2024 | size ratio 0.40 | 79.5 | vs HALP 77.4 | [2306.14306](https://arxiv.org/abs/2306.14306) |
| AdaSAP | " | size ratio 0.20 | 79.3 | vs HALP 77.0 | " |
| MDP | CVPR 2025 | ×1.84 | **80.0** | **+2.0** | [2504.02168](https://arxiv.org/abs/2504.02168) |
| IEE | WACV 2025 | — | `[?]` chỉ có hình scatter | `[?]` | [2502.03658](https://arxiv.org/abs/2502.03658) |

**Cả 5 bài đều chỉ prune backbone.** HALP ghi thẳng: *"We only prune the backbone network in the
detector."* Nếu ta prune cả neck/head thì **phải báo thêm một ablation chỉ-backbone** mới so được.

---

## Ô E — SSD300-ResNet50 / COCO val2017 (mAP50-95)

| Method | Venue, năm | FLOPs | mAP | AP50 | Phạm vi | Link |
|---|---|---|---|---|---|---|
| Baseline (REPrune) | AAAI 2024 | — | 25.2 | 42.7 | — | [2402.17862](https://arxiv.org/abs/2402.17862) |
| DMCP (dòng so sánh) | " | ↓50% backbone | 24.1 | 41.2 | backbone | " |
| REPrune | " | ↓50% backbone | **25.0** | 42.3 | backbone | " |
| Baseline (OCSPruner) | arXiv 2025 | — | 26.90 | — | — | [2501.13439](https://arxiv.org/abs/2501.13439) |
| OCSPruner | " | còn 74.58% | 25.78 | — | backbone | " |
| OCSPruner | " | còn 65.67% | 25.30 | — | backbone | " |
| OCSPruner | " | còn 44.06% | 23.82 | — | backbone | " |

⚠️ **Hai bài baseline khác nhau (25.2 vs 26.90)** do khác cách xử lý input/pretrain, và OCSPruner báo
**FLOPs còn lại** chứ không phải đã cắt. Ghép hai nhóm này vào một đường cong là sai.

---

## Ô F — YOLOX-S / KITTI (mAP50) ⭐ rẻ nhất cho 1 GPU

KITTI nửa tập hoặc 8:2, khoảng 3.7k ảnh train.

| Method | Venue, năm | Rate | Params↓ | FLOPs↓ | mAP50 | Δ | Link |
|---|---|---|---|---|---|---|---|
| Saliency-guided | IEEE IV 2023 | 0.3 | 40.2% | 29.2% | **89.1** | **+1.8** | [2303.02512](https://arxiv.org/abs/2303.02512) |
| Saliency-guided | " | 0.5 | — | — | 86.6 | −0.7 | " |
| Saliency-guided | " | 0.7 | — | — | 81.1 | −6.2 | " |
| LDT | WACV 2026 | — | 74.8% | 62.1% | 89.6 | −1.1 (baseline 90.7) | [2509.17968](https://arxiv.org/abs/2509.17968) |

Cùng ô, **YOLOF-R50 trên KITTI**: 87.4 → 87.2 ở rate 0.3 (−36.9% params, −43.1% FLOPs).
Cùng nhóm tác giả, **ATSS/GFL trên KITTI** (LDT): 89.7 → **90.2** và 89.7 → **90.4**.

Recipe của họ: SGD batch 8, lr 0.01, **300 epoch trên KITTI**, 100 epoch trên COCO-traffic.

---

## Ô G — YOLOv5 / COCO (mAP50-95)

| Method | Venue, năm | Mức nén | mAP | AP50 | Δ | Link |
|---|---|---|---|---|---|---|
| PAGCP (YOLOv5m) | TPAMI 2023 | −54.2% FLOPs, −64% params | 41.5 | 60.7 | −2.1 | [2303.11923](https://arxiv.org/abs/2303.11923) |
| PAGCP (YOLOv5m) | " | −61% FLOPs, −71% params | 41.0 | 60.3 | −2.6 | " |
| HESSO (YOLOv5l) | arXiv 2024 | còn 49% params | 44.5 | 63.1 | −3.2 | [2409.09085](https://arxiv.org/abs/2409.09085) |
| HFP (dòng so sánh) | 2021 | còn 50% params | 43.4 | 63.5 | −4.3 | " |
| TCFP (dòng so sánh) | 2022 | còn 50% params | 42.7 | 61.8 | −5.0 | " |

Baseline: YOLOv5m 43.6 mAP (PAGCP), YOLOv5l 47.71 mAP (HESSO). **Ba điểm của HESSO/HFP/TCFP cùng
ngân sách params ~50% → so sánh sạch nhất ở họ YOLO.**

---

## Ô H — YOLOv5m / VOC07 test (mAP50)

Chỉ có 1 điểm: PAGCP 83.2 → **81.7** (−64.8% FLOPs, −78.0% params). Thêm một điểm cũ ngoài khung:
group channel pruning trên YOLOv4/VOC (Applied Intelligence 2022, −64.7% params) nhưng **mAP không
đọc được**.

---

## Các điểm KHÔNG ghép được thành ô

**Nhóm YOLO11/v12/v13/YOLO26** (~30 bài): mỗi bài một dataset riêng tư khác nhau — mía, lúa, bông,
cúc vạn thọ, sứ cách điện, đường sắt, cọ dầu, chóp giao thông, cà chua nhà kính, ngô, cháy rừng. Đa số
chỉ báo mAP50. **Không có hai bài nào chung một ô.** Chi tiết ở §Phụ lục.

**Nhóm dataset nhỏ chuyên ngành** (~15 bài, Sensors/Measurement/PeerJ/PLOS ONE): cũng vậy. Mốc hữu ích
nhất là [RGP (iScience 2024)](https://doi.org/10.1016/j.isci.2024.111618) — khuôn mẫu "một dataset nhỏ
+ một trung bình + COCO", có code, báo cả hai metric.

---

## Khuyến nghị chọn ô

| Ô | Số điểm | Chi phí GPU | Dùng khi |
|---|---|---|---|
| **A. YOLOv8/COCO** | 48 | Cao (COCO) | Muốn con số mạnh nhất, cần thầy chạy hộ |
| **B. SSD300-VGG16/VOC** | ~9 | **Thấp** | Giữ SSD đang có, chạy được trên máy mình |
| **F. YOLOX-S/KITTI** | 4–6 | **Thấp nhất** | Muốn detector hiện đại mà vẫn 1 GPU |
| C. RetinaNet/COCO | 7 | Cao | Nếu cần thêm detector thứ hai cho reviewer |
| D. SSD512-R50/VOC | 5 | Trung bình | Chỉ khi chấp nhận trục speedup |

**Phương án đề xuất: ô B làm chính (giữ SSD, chạy rẻ) + ô F làm phụ (detector hiện đại)**, và nếu thầy
chạy hộ được thì thêm ô A để có con số đặt cạnh 48 điểm của PruningBench.

**Bắt buộc phải tự chạy, vì không ai làm sẵn:**
- Nhánh **structured-only** và **unstructured-only** của chính ta, ở cùng mức FLOPs — vì chưa có bài
  nào làm bi-level trên one-stage detector để mà so.
- Một ablation **chỉ-backbone** nếu ta prune cả neck/head, để so được với ô D.

---

## Hai khoảng trống còn thật sự trống

1. **Prune neck/FPN và head với dependency grouping đúng.** Mọi bài đều né head:
   HALP *"We only prune the backbone network in the detector"*; PAGCP loại output head khỏi tập layer;
   OCSPruner *"the SSD head was left unchanged… channel pruning in the head would be needed for larger
   parameter reductions"*; CAP giữ head dense.
2. **Bi-level unstructured + structured trên one-stage detector.** Chưa ai làm trong 2023–2026. Gần
   nhất: IEE (WACV'25) dùng chung một tiêu chí cho cả hai mức nhưng chạy thành hai thí nghiệm tách rời;
   R-TOSS (DAC'23) làm semi-structured theo pattern; YOLObile (AAAI'21, ngoài khung) chỉ *so sánh* hai
   loại.

Bài học từ CIFAR vẫn đúng: nếu surgery không biến sparsity thành FLOPs giảm thật thì tầng unstructured
chỉ tốn accuracy. [Sparse-IFT (ICML'24)](https://arxiv.org/abs/2303.11525) cho thấy cách hợp thức hoá:
giữ model sparse khi finetune và so ở chế độ **iso-FLOP** — RetinaNet-R18 29.3 → **34.5** mAP ở 90%
sparsity.

---

## Tóm tắt pipeline các method chính

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

## Phụ lục 1 — Các bản YOLO mới nhất có được prune không? (13/10/2026)

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

## Phụ lục 2 — Đã kiểm tra và loại

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
- SMCP: mAP ở ×2.12 chưa đọc được.
- LDT: intro ghi YOLOX-S "+1.6% mAP" nhưng Table 4 là 90.7 → 89.6; Fig. 3 nói iterative còn §4.2.2 nói
  "even one iteration".
- Info-theoretic (arXiv 2408.16772): DOI TNNLS ghi trên arXiv **không resolve qua Crossref**.
- R-TOSS: tự mâu thuẫn giữa "mAP IoU 0.5" và "AP@[.5:.95]".
