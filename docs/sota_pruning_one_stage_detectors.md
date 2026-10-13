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

### B.1 — Các điểm CÙNG TRỤC (baseline 77.4–77.68, đều báo %FLOPs thật)

| Method | Venue, năm | FLOPs↓ | Params↓ | mAP50 | Δ | Phạm vi | Link |
|---|---|---|---|---|---|---|---|
| OFP "Ours-0.05" | arXiv 2020 | 30.9% | 34.8% | **77.78** | **+0.10** (base 77.68) | filter trong SSD | [2010.06821](https://arxiv.org/abs/2010.06821) |
| OFP "Ours-0.1" | " | 41.8% | 46.1% | 77.14 | −0.54 | " | " |
| **LCP** | arXiv 2019 | 50% | 50% | **77.2** | **−0.2** (base 77.4) | backbone conv1-1→conv5-3 | [1911.02237](https://arxiv.org/abs/1911.02237) |
| ThiNet (LCP chạy lại) | " | 50% | 50% | 74.6 | −2.8 | backbone | " |
| MLP (PAGCP đo lại) | WACV 2019 | 57.1% | 85.2% | 75.0 | −2.5 (base 77.5) | filter ở conv liên tiếp | [2303.11923](https://arxiv.org/abs/2303.11923) |
| **PAGCP vòng 1** | TPAMI 2023 | 59.4% | 67.3% | **76.2** | −1.3 (base 77.5) | **mọi lớp trừ output head** | " |
| OFP "Ours-0.3" | arXiv 2020 | 61.1% | 64.7% | 76.45 | −1.23 | filter trong SSD | [2010.06821](https://arxiv.org/abs/2010.06821) |
| **PAGCP vòng 2** | TPAMI 2023 | 66.3% | 78.3% | **75.9** | −1.6 | mọi lớp trừ head | [2303.11923](https://arxiv.org/abs/2303.11923) |
| OFP "Ours-0.5" | arXiv 2020 | ~69.1% (tính lại) | ~73.9% (tính lại) | 75.83 | −1.85 | filter trong SSD | [2010.06821](https://arxiv.org/abs/2010.06821) |
| **PAGCP vòng 3** | TPAMI 2023 | 71.1% | 82.1% | **75.7** | −1.8 | mọi lớp trừ head | [2303.11923](https://arxiv.org/abs/2303.11923) |
| **PAGCP vòng 4** (dòng headline) | " | **74.3%** | **85.6%** | **75.6** | −1.9 | " | " |
| LCP η=0.75 | arXiv 2019 | 75% | 75% | 75.2 | −2.2 (base 77.4) | backbone | [1911.02237](https://arxiv.org/abs/1911.02237) |
| LCP (PAGCP đo lại) | — | 74.9% | 74.9% | 75.2 | −2.3 (base 77.5) | backbone | [2303.11923](https://arxiv.org/abs/2303.11923) |
| ThiNet η=0.75 | arXiv 2019 | 75% | 75% | 72.7 | −4.7 | backbone | [1911.02237](https://arxiv.org/abs/1911.02237) |
| ThiNet (PAGCP đo lại) | — | 74.9% | 74.9% | 72.7 | −4.8 | backbone | [2303.11923](https://arxiv.org/abs/2303.11923) |

**Lưu ý quan trọng về PAGCP:** 4 điểm đó là **4 vòng liên tiếp của MỘT lần chạy** (Table IV), không
phải 4 model được tinh chỉnh độc lập. Chỉ điểm cuối (75.6 @ 74.3%) là dòng PAGCP đặt vào bảng so sánh
chính của họ.

### B.2 — Các điểm KHÔNG ghép chung được

| Method | Vì sao để riêng |
|---|---|
| **AIP** ([2108.13591](https://arxiv.org/abs/2108.13591)) 76.10 → 75.60 @30.3% / 75.20 @54.1% | Baseline 76.10 ở **11.34 GFLOPs**, trong khi mọi dòng SSD300-VGG16 khác ở ~31.4 GFLOPs. Bài còn ghi backbone "VGG-16 trained on CIFAR-100" và không nói split |
| **Multi-layer Pruning** ([1811.08342](https://arxiv.org/abs/1811.08342)) 77.16 → 75.07 / 72.15 | **Không báo FLOPs**, chỉ báo params/kích thước. Và nó đổi kiến trúc (3 vs 6 lớp detection) |
| **NNCF filter pruning 40%** (SSD300-VGG-**BN**) 78.28 → 78.35 / 77.72 | Biến thể có BN nên baseline khác; "40%" là mức **đếm filter**, không phải FLOPs; hai phiên bản NNCF cho hai số khác nhau |

**Ưu của ô này:** rẻ nhất để chạy; PAGCP có code công khai và là bài duy nhất prune cả extras.
**Nhược:** đa số điểm là 2019–2021; baseline dense trải từ 77.16 đến 78.28 nên phải lọc.

---

## Ô C — RetinaNet-R50(-FPN) / COCO train2017 → val2017 (mAP50-95)

Baseline dense 36.1–36.5 tuỳ bài. Giao thức gốc từ Group Fisher Pruning (ICML'21), vẫn dùng tới 2026.

### C.1 — Nhóm baseline 36.5 (GFP và bài info-theoretic, đều cắt 50%)

| Method | Venue, năm | FLOPs/MACs↓ | Params↓ | AP | Δ | Phạm vi | Link |
|---|---|---|---|---|---|---|---|
| **Info-theoretic "Ours-F"** | arXiv 2024 | 50% MACs | 32.7% | **37.6** | **+1.1** | mọi kênh | [2408.16772](https://arxiv.org/abs/2408.16772) |
| **Info-theoretic "Ours-P"** | " | 50% MACs | 30.8% | **37.3** | **+0.8** | " | " |
| GFP "Retina-F" | ICML 2021 | 50% | 31.0% | 36.8 | +0.3 | cả FPN + head, gom nhóm kênh coupled | [2108.00708](https://arxiv.org/abs/2108.00708) |
| **GFP "Retina-M"** (bản chính) | " | 50% | 30.6% | **36.5** | **0.0** | " | " |
| GFP "Retina-I" | " | 50% | 20.2% | 36.3 | −0.2 | chỉ kênh độc lập | " |
| GFP "Retina-U" | " | 50% | 62.6% | 35.8 | −0.7 | cả FPN + head | " |
| Info-theoretic "Ours-N" | arXiv 2024 | 50% MACs | 64.5% | 35.6 | −0.9 | mọi kênh | [2408.16772](https://arxiv.org/abs/2408.16772) |

### C.2 — Nhóm baseline 36.10 (IFSO và bản IFSO chạy lại GFP)

| Method | FLOPs↓ | AP | AP50 | AP75 | Δ | Link |
|---|---|---|---|---|---|---|
| **IFSO-50** | 50% | **36.30** | 55.60 | 38.70 | **+0.20** | [2308.06755](https://arxiv.org/abs/2308.06755) |
| GFP-50 (IFSO chạy lại) | 50% | 36.20 | 55.50 | 38.50 | +0.10 | " |
| **IFSO-60** | 60% | **35.70** | 54.70 | 38.00 | **−0.40** | " |
| GFP-60 (IFSO chạy lại) | 60% | 35.30 | 54.60 | 37.40 | −0.80 | " |

⚠️ **Tôi đã ghi sai ở bản trước.** Hai số "GFP 36.20 @50% / 35.30 @60%" **không phải của bài GFP gốc**
mà là **bản IFSO chạy lại GFP theo baseline 36.10 của IFSO**. Bài GFP gốc (ICML'21) có baseline 36.5,
bốn biến thể ở 50%, và **không có điểm 60% nào**.

### C.3 — Không ghép chung được

| Method | Vì sao để riêng |
|---|---|
| **PSE-Net / BCNet** ([2408.16233](https://arxiv.org/abs/2408.16233)) 36.4 → 35.9 / 35.4 | "50%" là FLOPs **của riêng backbone** (ResNet50 4G→2G ở scale phân loại), neck và head không đụng tới → FLOPs toàn detector giảm ít hơn nhiều |
| **Anchor Pruning** ([2104.00432](https://arxiv.org/abs/2104.00432)) 36.9 → 35.1 | Prune **anchor ở head**, backbone nguyên vẹn; và đánh giá trên **test-dev2017**, không phải val2017 |

**Kèm miễn phí 3 detector anchor-free** từ bài info-theoretic, cùng COCO val2017, 50% MACs:
FSAF 37.4 → **38.0**; ATSS 38.1 → **39.1**; PAA 39.0 → **40.0**.

**Hai nhận xét quan trọng:**
1. Ở mức 50%, **5 trong 11 điểm có Δ ≥ 0**. RetinaNet ở vùng này còn dư thừa nhiều, prune còn giúp
   regularize. Muốn chứng minh phương pháp thì phải đẩy lên **60–70%**, nơi các điểm bắt đầu âm —
   nhưng hiện **chỉ IFSO có điểm 60%**, chưa ai có 70%. Đó là chỗ trống.
2. **Không có bài nào mới hơn 2024** trong ô này. Agent chạy 4 lượt tìm mở rộng cho 2025–2026 và không
   ra kết quả nào.

---

## Ô D — SSD512-ResNet50 / VOC07+12 → VOC07 test ⚠️ trục là SPEEDUP, không phải FLOPs

Baseline dense 77.98 (HALP) / 78.0 (MDP). Năm bài cùng dòng NVIDIA, dùng chung baseline — ô dày
người nhất, nhưng **không ai báo %FLOPs**, nên không ghép thẳng vào trục của ô B/C được.

| Method | Venue, năm | Trục của họ | FLOPs | mAP50 | Δ | Phạm vi | Link |
|---|---|---|---|---|---|---|---|
| **HALP** | NeurIPS 2022 | ×1.94 (BS=1) | **65.56 → 15.38 GFLOPs (−76.5%)**, 21.97 → 10.40M | 77.42 | −0.56 (dense 77.98) | **chỉ backbone** | [2210.06659](https://arxiv.org/abs/2210.06659) |
| SMCP | ECCV 2022 | ×2.12, giảm 75% latency (TITAN V, BS=1) | **không báo** | 78.3 `[?]` | `[?]` | backbone + feature layers | [2211.02206](https://arxiv.org/abs/2211.02206) |
| AdaSAP | ICLR 2024 | size ratio 0.40 | **không báo** | 79.5 | (HALP cùng mức: 77.4) | không nói | [2306.14306](https://arxiv.org/abs/2306.14306) |
| AdaSAP | " | size ratio 0.20 | **không báo** | 79.3 | (HALP cùng mức: 77.0) | không nói | " |
| MDP | CVPR 2025 | ×1.84 (TITAN V, BS=256) | **không báo** | **80.0** | **+2.0** (dense 78.0) | kênh + nguyên block | [2504.02168](https://arxiv.org/abs/2504.02168) |
| MDP | " | ×2.14 | **không báo** | 79.2 | +1.2 | " | " |
| IEE | WACV 2025 | — | **không báo** | `[?]` chỉ có hình scatter | `[?]` | không nói | [2502.03658](https://arxiv.org/abs/2502.03658) |

**Bốn cảnh báo cho ô này:**

1. **Chỉ HALP báo FLOPs.** Nếu trục x là FLOPs thì ô D co lại còn **đúng một điểm** cộng một đám điểm
   chỉ có speedup. Trục speedup cũng không sạch: HALP và SMCP đo ở **BS=1**, MDP đo ở **BS=256**, và
   HALP **không nói GPU nào** cho phần detection.
2. **Con số 78.3 của SMCP không phải do SMCP công bố** — đó là MDP **đọc từ hình** của SMCP. Bài SMCP
   gốc chỉ ghi "+2.63 mAP so với SSD300-RN50" và "×2.12, giảm 75% latency"; kết quả VOC của họ chỉ tồn
   tại dưới dạng Fig. 3.
3. **"Size ratio" của AdaSAP không phải tỉ lệ FLOPs** và không quy đổi được. Không đặt 0.40/0.20 lên
   trục FLOPs.
4. **Phạm vi cắt khác nhau và ảnh hưởng lớn:** HALP chỉ backbone, SMCP backbone + feature layers, MDP
   cắt cả nguyên block. Việc MDP hơn dense +2.0 mAP nhiều khả năng phản ánh **ngân sách retrain cỡ 800
   epoch** chứ không chỉ phản ánh pruning — nên Δ-so-với-dense ở ô này không phải hiệu ứng pruning sạch.

**Cả nhóm đều chỉ prune backbone hoặc không nói rõ.** HALP ghi thẳng: *"We only prune the backbone
network in the detector."* Nếu ta prune cả neck/head thì **phải báo thêm ablation chỉ-backbone**.

Mốc tham chiếu khác trong cùng bảng HALP: SSD300-RN50 75.69 @16.23G; SSD300-VGG16 76.72 @31.44G;
RetinaNet-RN50 77.27 @106.50G; SSD512-RN50-slim 75.83 @46.09G.

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
| **B. SSD300-VGG16/VOC** | **15** (9 cùng trục) | **Thấp** | Giữ SSD đang có, chạy được trên máy mình |
| **F. YOLOX-S/KITTI** | 4–6 | **Thấp nhất** | Muốn detector hiện đại mà vẫn 1 GPU |
| C. RetinaNet/COCO | 11 (2 nhóm baseline) | Cao | Nếu cần thêm detector thứ hai cho reviewer |
| D. SSD512-R50/VOC | 7 nhưng **chỉ 1 điểm có FLOPs** | Trung bình | Chỉ khi chấp nhận trục speedup |

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
- SMCP: mAP 78.3 là MDP đọc từ hình của SMCP, **không phải số SMCP công bố**.
- LDT: intro ghi YOLOX-S "+1.6% mAP" nhưng Table 4 là 90.7 → 89.6; Fig. 3 nói iterative còn §4.2.2 nói
  "even one iteration".
- Info-theoretic (arXiv 2408.16772): DOI TNNLS ghi trên arXiv **không resolve qua Crossref**.
- R-TOSS: tự mâu thuẫn giữa "mAP IoU 0.5" và "AP@[.5:.95]".
