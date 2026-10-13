# Kết quả SOTA pruning trên ResNet-56 / CIFAR-10 (2023–2025)

Tổng hợp ngày 2026-09-19. **Mọi con số đọc trực tiếp từ PDF gốc của từng bài**, không lấy
từ bảng so sánh của bài khác. Lý do: khi đối chiếu, bảng thứ cấp sai ở nhiều chỗ (xem §4).

Ký hiệu venue: **✓** = thấy trong chính bài (header, footer, DOI, URL proceedings);
**~** = chỉ thấy qua trích dẫn ở bài khác, bản arXiv không ghi.

FLOPs↓ / Params↓ là **% giảm so với dense của chính bài đó** — các bài đếm dense khác
nhau (xem §3), nên chỉ so được theo %, không so theo số tuyệt đối.

---

## 1. Bảng chính

### 1.1 Filter / channel pruning tĩnh (cùng loại với bi-level)

| Method | Venue | Baseline | Pruned | Δ | FLOPs↓ | Params↓ |
|---|---|---|---|---|---|---|
| [DepGraph](https://arxiv.org/abs/2301.12900) | CVPR 2023 ~ | 93.53 | 93.77 | +0.24 | 52.6% (2.11×) | — |
| | | 93.53 | 93.64 | +0.11 | 61.1% (2.57×) | — |
| [TPP](https://arxiv.org/abs/2207.12534) | ICLR 2023 ✓ | 93.78 | 93.81 ±0.11 | +0.03 | 31.0% (1.45×) | 31.1% |
| | | 93.78 | 93.46 ±0.06 | −0.32 | 49.7% (1.99×) | 49.8% |
| | | 93.78 | 92.35 ±0.12 | −1.43 | 72.1% (3.59×) | 70.6% |
| | | 93.78 | 89.63 ±0.10 | −4.15 | 91.2% (11.41×) | 90.4% |
| | | 93.78 | 85.86 ±0.08 | −7.92 | 94.8% (19.31×) | 95.2% |
| [KDFS](https://arxiv.org/abs/2307.00198) | arXiv 2023 | 93.26 | 93.78 | +0.52 | 40.9% | 41.7% |
| | | 93.26 | 93.58 | +0.32 | 51.2% | 51.3% |
| | | 93.26 | 93.19 | −0.07 | 59.2% | 60.9% |
| [FSCL](https://arxiv.org/abs/2304.13397) | arXiv 2023 | 93.26 | 93.65 | +0.39 | 52.2% | 50.3% |
| | | 93.26 | 93.52 | +0.26 | 54.4% | 49.7% |
| [ICI](https://arxiv.org/abs/2303.03645) | IEEE TNNLS ✓ | 93.26 | 93.53 | +0.27 | 58.6% | 57.6% |
| | | 93.26 | 92.49 | −0.77 | 73.3% | 74.1% |
| [LAASP](https://doi.org/10.1016/j.imavis.2023.104745) ([arXiv](https://arxiv.org/abs/2506.20152)) | Image and Vision Computing 2023 ✓ | 93.61 ±0.11 | 93.49 | −0.12 | 52.6% | — |
| | | 93.61 ±0.11 | 93.04 ±0.08 | −0.57 | 55.2% | — |
| [NuSPM](https://openaccess.thecvf.com/content/WACV2024/papers/Lee_Pruning_From_Scratch_via_Shared_Pruning_Module_and_Nuclear_Norm-Based_WACV_2024_paper.pdf) | WACV 2024 ✓ | — | 93.50 | — | 63.2M MACs (≈49.6%) | — |
| [SPSRC](https://openaccess.thecvf.com/content/WACV2024/papers/Sun_Towards_Better_Structured_Pruning_Saliency_by_Reorganizing_Convolution_WACV_2024_paper.pdf) | WACV 2024 ✓ | 93.59 | 93.52 ±0.01 | −0.07 | 31.0% | 14.1% |
| | | 93.59 | 93.19 ±0.01 | −0.40 | 43.0% | 44.7% |
| | | 93.59 | 91.65 ±0.01 | −1.94 | 63.9% | 64.7% |
| [ATO](https://arxiv.org/abs/2403.14729) | CVPR 2024 ~ | 93.50 | 93.74 | +0.24 | 55.0% | — |
| | | 93.50 | 93.48 | −0.02 | 65.3% | — |
| [RLAL](https://arxiv.org/abs/2403.19490) | CVPR 2024 ✓ | 93.41 | 93.86 | +0.45 | 50.0% | — |
| [Info-theoretic CP](https://arxiv.org/abs/2408.16772) | IEEE TNNLS ✓ | 93.39 | 93.60 | +0.21 | 45.5% | 40.3% |
| | | 93.39 | 93.11 | −0.28 | 58.1% | 43.9% |
| [OCSPruner](https://arxiv.org/abs/2501.13439) | arXiv 2025 | 93.97 | 93.80 | −0.17 | 53.1% | 49.7% |
| | | 93.97 | 93.65 | −0.32 | 61.1% | 58.6% |
| [SACP](https://arxiv.org/abs/2506.11469) | arXiv 2025 | 94.58 | 90.06 | −4.52 | 78.1% | 70.5% |

SPSRC có 3 biến thể (Spectral / Nuclear / Frobenius) ở mỗi mức; bảng ghi biến thể tốt nhất
(Nuclear). Đủ cả ba: 31.0% → 93.40 / 93.52 / 93.46; 43.0% → 92.82 / 93.19 / 92.89;
63.9% → 91.55 / 91.65 / 91.62.

### 1.2 Khác loại — không so trực tiếp với filter pruning tĩnh được

| Method | Loại | Venue | Baseline | Pruned | Δ | FLOPs↓ | Params↓ |
|---|---|---|---|---|---|---|---|
| [BilevelPruning / UDSP](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_BilevelPruning_Unified_Dynamic_and_Static_Channel_Pruning_for_Convolutional_Neural_CVPR_2024_paper.pdf) | **dynamic** + static | CVPR 2024 ✓ | 93.12 | 93.78 | +0.66 | 50.1% | 20.0% |
| [SR-GKP](https://proceedings.neurips.cc/paper_files/paper/2023/file/c3aba4234afd1c8116d879ba183f4835-Paper-Conference.pdf) | grouped **kernel** | NeurIPS 2023 ✓ | 93.24 | 94.08 | +0.84 | 43.2% | 43.5% |
| [Pons et al.](https://arxiv.org/abs/2405.17081) | **layer** pruning | arXiv 2024 | — | — | +0.95 | 56.3% | — |
| | | | — | — | +0.16 | 71.3% | — |
| | | | — | — | +0.08 | 75.1% | — |
| [Pruning Everything](https://arxiv.org/abs/2506.04513) | **filter + layer** | arXiv 2025 | — | — | +1.06 | 39.1% | — |
| | | | — | — | +0.95 | 50.4% | — |
| | | | — | — | +0.91 | 53.7% | — |
| | | | — | — | +0.96 | 62.8% | — |
| | | | — | — | +0.53 | 69.2% | — |
| | | | — | — | +0.19 | 72.7% | — |
| | | | — | — | −0.49 | 79.9% | — |
| | | | — | — | −1.36 | 86.4% | — |

- **BilevelPruning** trùng tên với method của bạn nhưng là **dynamic pruning** (chọn kênh
  theo từng ảnh lúc chạy) — nên params chỉ giảm 20%. Tên method trong bài là UDSP.
- **SR-GKP** là bài về độ bền đối kháng; con số là checkpoint "Best Benign" — bài tự ghi là
  chọn kiểu superscore (mỗi cột có thể từ một checkpoint khác).
- **Pons et al.** và **Pruning Everything** chỉ báo ΔAcc, không có accuracy tuyệt đối.

### 1.3 Biên — arXiv 2021–2022, xuất bản tạp chí 2022–2023

Method thuộc thế hệ trước dù năm xuất bản rơi vào khung. Cân nhắc trước khi coi là "SOTA 2–3
năm".

| Method | Venue | arXiv | Baseline | Pruned | Δ | FLOPs↓ | Params↓ |
|---|---|---|---|---|---|---|---|
| [WhiteBox](https://arxiv.org/abs/2104.11883) | IEEE TNNLS ✓ | 2021 | 93.26 | 93.54 | +0.28 | 55.6% | — |
| [CLR-RNF](https://arxiv.org/abs/2202.07190) | IEEE TNNLS ~ | 2022 | 93.26 | 93.27 | +0.01 | 57.3% | 55.5% |
| [DCFF](https://arxiv.org/abs/2107.06916) | chưa xác minh — bản arXiv ghi "TPAMI under review" | 2021 | 93.26 | 93.26 | 0.00 | 55.9% | 55.0% |

---

## 2. Đã kiểm tra và loại

| Method | Lý do |
|---|---|
| [OTOv2](https://arxiv.org/abs/2303.06862) (ICLR 2023) | PDF gốc 25 trang **không có ResNet-56** (chỉ ResNet50). Bảng của OCSPruner ghi OTOv2 đạt 93.30% trên ResNet-56 — không có trong bài gốc. |
| [BMRS](https://proceedings.neurips.cc/paper_files/paper/2024/file/7538e7a7a589a267f49527fe6fbc640e-Paper-Conference.pdf) (NeurIPS 2024) | Không có ResNet-56. |
| CORING | Loại theo yêu cầu. |

Chưa kiểm tra (được trích ở bài khác nhưng chưa mở bài gốc): DAIS, SOKS (đều TNNLS 2023);
DLRFC, DECORE (2022, ngoài khung).

---

## 3. Những điều phải biết trước khi so

**Dense đếm khác nhau.** Cùng một ResNet-56 mà mỗi bài ra một số:

| Dense MACs | Bài dùng | Cách đếm |
|---|---|---|
| 125.49M | CORING, KDFS, repo này (`count_cost`) | Conv + Linear |
| 126.56M | CLR-RNF, SR-GKP | — |
| 127.62M | DCFF | thop (cộng cả BatchNorm) |
| 129.32M | SPSRC | — |

→ Chỉ so theo **% giảm**, không so số MACs tuyệt đối.

**Baseline khác nhau**, từ 93.12 (UDSP) tới 94.58 (SACP). Bài có baseline cao thì
accuracy sau prune tự nhiên cao hơn. → So theo **Δ** (so với baseline của chính bài).

**"Baseline 93.26" không có nghĩa là cùng checkpoint.** Đã rà lại toàn văn từng bài
(2026-09-19): không bài nào ghi rõ là fine-tune từ checkpoint HRank. 93.26 là con số
tham chiếu lưu hành trong dòng HRank.

| Method | Baseline ghi | Xuất phát từ | Bằng chứng |
|---|---|---|---|
| bi-level (repo này) | 93.26 | checkpoint HRank | tự đo lại: 9326/10000 = 93.26% |
| CORING | 93.26 | checkpoint HRank | `vbd.sh`: `--pretrain_dir checkpoint/cifar/cifar10/resnet_56.pt` |
| KDFS | 93.26 | model pretrained | dùng pretrained làm teacher — không ghi nguồn |
| WhiteBox | 93.26 | model pretrained | ghi "pre-trained model → pruned model" |
| CLR-RNF | 93.26 | model pretrained | chọn filter từ model pretrained |
| DCFF | 93.26 | **train từ đầu** | "without the dependency on the pretrained model" |
| LAASP | 93.61 | **train từ đầu** | cột "Pretrain Used? No" |
| OCSPruner | 93.97 | **train từ đầu** | one-cycle từ khởi tạo ngẫu nhiên |
| ATO | 93.50 | **train từ đầu** | tên bài: "... Pruning from Scratch" |
| NuSPM | không có | **train từ đầu** | không có model dense nên không có baseline |
| Pons et al., Pruning Everything | không có | — | chỉ báo ΔAcc, bài không có accuracy tuyệt đối nào |
| FSCL, ICI | 93.26 | không nói | — |

Với method **train từ đầu**, "baseline" chỉ là một model dense train riêng để tham chiếu,
nên Δ của chúng không cùng nghĩa với Δ của method prune từ model pretrained.

**Thang độ công bằng** (từ sạch nhất tới kém nhất):
1. Cùng checkpoint + cùng recipe + cùng MACs → chỉ khác method. Chỉ đạt được khi tự chạy
   lại đối thủ trong pipeline của mình.
2. Cùng checkpoint, khác recipe → so accuracy tuyệt đối, nhưng recipe là biến nhiễu
   (xem §6: chính CORING dùng recipe khác script của nó).
3. Khác baseline → so Δ; nhiễu từ độ mạnh baseline và recipe.
4. Train từ đầu vs prune từ pretrained → Δ khác nghĩa, so rất thận trọng.

Bảng tổng hợp trong các paper (kể cả Table 3 của CORING) trộn cả bốn mức này.

**Ngân sách finetune không đồng nhất** (đã tra ở lần trước): HRank 30, TPP 120, CORING 300,
CHIP 400 epoch. ICI khai 300 epoch. Đa số bài trong bảng này không khai rõ trong phần chính.

---

## 4. Lỗi trong bảng thứ cấp — lý do phải đọc bài gốc

| Bảng của | Ghi về | Ghi | Bài gốc nói |
|---|---|---|---|
| OCSPruner | OTOv2 | 93.30% trên ResNet-56 | Bài OTOv2 không có ResNet-56 |
| OCSPruner | LAASP | 93.45% | 93.49% hoặc 93.04% |
| SACP | HRank | 92.17% | HRank công bố 93.17%; SACP chạy lại với baseline 94.58% riêng |
| DepGraph | HRank | 93.26 → 92.17, Δ −0.09 | Δ không khớp hai số — lỗi đánh máy |
| Pruning Everything | Pons et al. | +1.25 @ 37.52%, +0.86 @ 48.78% | Bản arXiv 2405.17081 không có hai điểm này |

Thêm: bảng OCSPruner để cột FLOPs là **phần còn lại** (38.91 = DepGraph ở 2.57×), trong khi
văn bản của chính bài viết "38.88% reduced FLOPs".

---

## 5. Kết quả bi-level đặt vào bảng

Kết quả hiện có (1 seed, ngân sách 400 epoch, checkpoint HRank 93.26):

| Cấu hình | Pruned | Δ | FLOPs↓ | Params↓ |
|---|---|---|---|---|
| bi-level, uniform + norm | 93.82 | +0.56 | 27.0% | 27.9% |
| bi-level, sensitivity + norm | 93.86 | +0.60 | 27.3% | 25.7% |

Ở vùng ~27–31% chỉ có hai bài báo cáo: TPP (+0.03 @ 31.0%) và SPSRC (−0.07 @ 31.0%).
Bi-level có Δ tốt hơn cả hai, nhưng ở mức nén **thấp hơn** 4 điểm.

**Vùng đông nhất là 50–58% FLOPs↓** — 12 điểm từ 10 method filter pruning tĩnh (chưa kể
UDSP, SR-GKP, Pruning Everything thuộc loại khác):

| Method | Δ | FLOPs↓ |
|---|---|---|
| RLAL | +0.45 | 50.0% |
| KDFS | +0.32 | 51.2% |
| FSCL | +0.39 | 52.2% |
| DepGraph | +0.24 | 52.6% |
| LAASP | −0.12 | 52.6% |
| OCSPruner | −0.17 | 53.1% |
| FSCL | +0.26 | 54.4% |
| ATO | +0.24 | 55.0% |
| LAASP | −0.57 | 55.2% |
| WhiteBox | +0.28 | 55.6% |
| DCFF | 0.00 | 55.9% |
| CLR-RNF | +0.01 | 57.3% |

Để có một dòng so được với nhiều method nhất, chạy bi-level ở **52.6% FLOPs↓** (trùng
DepGraph 2.11× và LAASP). Với scope conv1-only, mức này vẫn còn trong vùng an toàn — trần
cứng chỉ bắt đầu cắn quanh 60%.

---

## 6. Kiểm chứng CORING bằng checkpoint họ công bố

Cách đo: dùng **chính code model** của họ (`main/models/cifar10/resnet.py`), nạp **chính
checkpoint đã prune** ở release v0.1.0, đo trên đủ 10.000 ảnh test, normalize giống
`main/data/cifar10.py`. Trong checkpoint có 64 key `total_ops`/`total_params` do thop gắn
vào lúc đo FLOPs — bỏ riêng chúng, phần trọng số thật nạp `strict=True` không lỗi (script
test của họ dùng `strict=False` vì mấy key này).

| Dòng trong bài | Công bố | Đo lại | Params đo (công bố) | MACs đo (công bố) |
|---|---|---|---|---|
| CORING-E-5 | 94.76 | **94.76** | 0.660M (0.66M) | 89.46M (91.23M) |
| CORING @47% | 94.22 | **94.22** | 0.490M (0.48M) | 67.39M (65.94M) |
| CORING-E | 92.84 | **92.85** | 0.241M (0.24M) | 34.20M (34.79M) |

MACs đo bằng cùng cách đếm cho ra dense 125.49M — tức đúng quy ước của chính bài.

### Học được gì

**1. Accuracy là thật.** Cả ba khớp tới 0.01.

**2. Recipe thật khác script.** `vbd.sh` đang hoạt động ghi `--epochs 300` với
`MultiStepLR @150,225`. Nhưng checkpoint lưu ở **epoch 376 / 381 / 388**, và lr trong trạng
thái optimizer:

| epoch | lr lưu | cosine 0.01 → 0, T = 400 | MultiStep @150,225 |
|---|---|---|---|
| 376 | 9.08e-5 | 8.86e-5 | 1e-4 |
| 381 | 5.70e-5 | 5.55e-5 | 1e-4 |
| 388 | 2.28e-5 | 2.22e-5 | 1e-4 |

Fit ngược ra T ≈ 400.2 cho cả ba → **cosine ~400 epoch**, wd 0.005, momentum 0.9. Đây là
recipe của CHIP (`--epochs 400 --lr_type cos --weight_decay 0.005 --batch_size 256`) —
CORING mượn cả chuỗi `compress_rate` lẫn cách train của CHIP. Batch size không lưu trong
checkpoint; CHIP và "round 1" (đã comment) trong `vbd.sh` đều dùng 256.

**3. Họ cắt cả kênh đầu ra của stage** (`adapt_channel`: `compress_rate[1]`, `[2]` là độ
rộng đầu ra stage1/stage2; stage3 không cắt):

| | Mức nén | Đầu ra s1 / s2 / s3 | Mid s1 / s2 / s3 |
|---|---|---|---|
| CORING-E-5 | 27% | **13** / **26** / 64 | 13 / 26 / 52 |
| bi-level | 27% | 16 / 32 / 64 | 12 / 23 / 46 |
| CORING-E | 72% | **9** / **19** / 64 | 8 / 12 / 19 |
| bi-level | 72% | 16 / 32 / 64 | 4 / 9 / 18 |

Ở 27% họ cắt nhẹ 18% ở mọi nơi; bi-level cắt 28% chỉ ở mid. Ở 72% họ giữ mid stage1 ở 8
kênh nhờ cắt thêm đầu ra; bi-level bị ép xuống 4. Khớp chẩn đoán "scope là trần cứng".

**4. Hai trong ba dòng MACs không khớp model thật.** 65.94M và 34.79M trùng đúng số CHIP
công bố (cùng chuỗi `compress_rate` với CHIP). Dòng 94.22: model thật 67.39M, tức giảm
46.3% chứ không phải 47.4% như công bố. Hai dòng còn lại công bố MACs **cao hơn** thực tế
(thiệt cho họ, không phải lợi).

**5. Cả hai bên đều chọn "best top-1 trên tập test".** Checkpoint lưu `best_top1_acc` ở
epoch tốt nhất — cùng cách với `best_top1` của repo này, nên phần này so được.
