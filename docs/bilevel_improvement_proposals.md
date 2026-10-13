# Đề xuất cải tiến phương pháp bi-level pruning

Tài liệu này tổng hợp kết quả đo trên CIFAR-10 / ResNet-56, chẩn đoán nút thắt của
phương pháp bi-level hiện tại, và đề xuất ba hướng cải tiến kèm chi phí ước tính.

Mọi con số trong tài liệu đều **đo được**, không ước lượng. Script tái lập nằm ở cuối.

---

## 1. Hiện trạng

### 1.1 Thiết lập thí nghiệm

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Dense checkpoint | `resnet_56.pt`, top-1 93.26%, 0.853M params, 125.49M MACs | HRank, qua release CORING v0.1.0 |
| Transform | RandomCrop(32, pad=4) + HFlip, std (0.2023, 0.1994, 0.2010) | `main/data/cifar10.py` của CORING |
| Finetune | lr 0.01, ×0.1 @50%/75%, wd 5e-3, batch 128 | `main/scripts/resnet56_cifar10/vbd.sh` |
| Ngân sách epoch | 400 (15 trong vòng prune + 385 finetune) | tự đặt, xem §1.3 |
| MACs | đếm Conv2d + Linear, **không dùng thop** | thop cộng cả BatchNorm → 127.62M ≠ 125.49M |

Dense của ta đếm ra đúng **125.49M MACs / 0.853M params**, khớp từng chữ số với bảng
published của CORING → kiến trúc chính xác.

### 1.2 Kết quả đã có

Tất cả tại điểm nén ~27% MACs (đối chiếu CORING-E-5 = 91.23M):

| lịch | biến thể | top-1 | params | MACs |
|---|---|---|---|---|
| uniform | bi-level | **93.82%** | 0.615M | 91.64M |
| uniform | structured-only | 93.55% | 0.615M | 91.64M |
| steep | structured-only | 93.48% | 0.545M | 91.15M |
| steep | bi-level | 93.16% | 0.545M | 91.15M |

Tại điểm nén ~72% MACs (đối chiếu CORING-E = 34.79M), lịch `uniform`:

| biến thể | top-1 | params | MACs |
|---|---|---|---|
| structured-only | 90.00% | 0.241M | 34.28M |

### 1.3 Vị trí trong Table 3 của CORING

**Nhóm ~27% MACs** — cạnh tranh được:

| Method | Top-1 | Params | MACs |
|---|---|---|---|
| ResNet-56 dense | 93.26 | 0.85M | 125.49M |
| DECORE-450 (CVPR'22) | 93.34 | 0.64M | 92.48M |
| HRank-1 (CVPR'20) | 93.52 | 0.71M | 88.72M |
| TPP (ICLR'23) | 93.81 | N/A | 86.50M |
| **bi-level (uniform)** | **93.82** | **0.615M** | 91.64M |
| CORING | 94.76 | 0.66M | 91.23M |

Hoà TPP, trên DECORE-450 và HRank-1, **nén params tốt nhất nhóm**. Kém CORING 0.94.

**Nhóm ~72% MACs** — bét bảng:

| Method | Top-1 |
|---|---|
| **structured-only** | **90.00** |
| HRank-3 | 90.72 |
| CHIP | 92.05 |
| CORING | 92.84 |

### 1.4 Ghi chú về ngân sách epoch

Các method trong Table 3 **không** dùng chung ngân sách. Khai từ README/code của từng repo:

| Method | Epochs | lr schedule | batch | wd |
|---|---|---|---|---|
| HRank | 30 | 0.01, ×0.1 @5,10 | 128 | — |
| TPP | 120 | 0.01 → @60 → @90 | 128 | 5e-4 |
| CORING | 300 | 0.01, ×0.1 @150,225 | 128 | 5e-3 |
| CHIP | 400 | 0.01, cosine | 256 | 5e-3 |

Chênh tới 13 lần. Table 3 là bảng gom kết quả tốt nhất mỗi paper tự công bố, không phải
thí nghiệm có kiểm soát ngân sách. Ta chạy 400 để bằng CHIP (mức cao nhất trong bảng).

Ảnh hưởng của ngân sách với method của ta: 300 → 400 epoch cho +0.42 điểm. Có thật
nhưng không đủ lấp khoảng cách tới CORING.

---

## 2. Chẩn đoán

### 2.1 Nút thắt ở mức nén sâu: phạm vi lớp được cắt

Structured pruning hiện chỉ đụng `conv1` của mỗi BasicBlock
([models/backbone.py:40-46](../models/backbone.py#L40-L46)). Kênh đầu ra của block và
stem bị đóng băng vì chúng cộng vào nhánh residual.

Hệ quả, đo tại điểm 34.79M — số kênh mid còn lại:

| ratio | stage1 (gốc 16) | stage2 (32) | stage3 (64) |
|---|---|---|---|
| 0.281 (điểm 91.23M) | 12 | 23 | 46 |
| 0.719 (điểm 34.79M) | **4** | 9 | 18 |

Stage-1 bị bóp còn **4 kênh**. Để so sánh, `compress_rate` của CHIP/CORING là
`[0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9` — **30 mục**, gồm stem + 2 điểm chuyển stage + 27
block, tức là họ cắt cả kênh đầu ra.

Con số quyết định:

> Áp **đúng tỉ lệ của CHIP** (0.5/0.6/0.7) lên scope conv1-only chỉ xuống tới
> **50.71M MACs (−59.6%)**, không phải 34.79M (−72.3%).

Scope là **trần cứng**. Muốn tới 34.79M buộc phải cắt sâu hơn CHIP rất nhiều ở mọi lớp,
và đó là lý do accuracy sập còn 90.00%.

Nếu mở scope sang kênh đầu ra (đo analytic):

| cấu hình | stage1 mid | MACs |
|---|---|---|
| scope hiện tại, ép xuống 34.79M | 4 | 34.28M (−72.7%) |
| scope mở rộng, tỉ lệ CHIP cho cả hai | 8 | 24.50M (−80.5%) |
| scope mở rộng, cắt ~60% đều tay | 10 | 39.91M (−68.2%) |

Cùng mức nén, scope rộng giữ được 10 kênh thay vì 4.

### 2.2 Phân bổ tỉ lệ theo lớp: không phải nút thắt

Đã thử ba hình dạng phân bổ (`LAYER_SCHEDULES` trong [cifar.py](../cifar.py)):

| lịch | hình dạng s1/s2/s3 | tại 91.23M: kênh còn | params |
|---|---|---|---|
| uniform | đều | 12/23/46 | 0.615M |
| chip | 0.5/0.6/0.7 | 12/23/44 | 0.595M |
| steep | 0.3/0.6/0.8 | 14/22/39 | 0.545M |

`chip` gần như trùng `uniform`. `steep` thua 0.07–0.66 điểm. Trong phạm vi scope hiện
tại, điều chỉnh phân bổ **không cứu được gì** — vì cắt nhẹ chỗ này thì phải cắt nặng gấp
bội chỗ kia.

### 2.3 Thuật toán chọn filter: có trần n/2, nhưng là thiết kế đúng

Vòng lặp trong [pruning/structured.py](../pruning/structured.py) duyệt các cặp filter
theo khoảng cách tăng dần, giết cái norm nhỏ hơn, rồi đánh dấu **cả hai** là
`processed`. Hệ quả: tối đa n/2 filter cắt được bằng tiêu chí khoảng cách.

| ratio | cần cắt | theo khoảng cách | fallback L1-norm |
|---|---|---|---|
| 0.281 | 279 | 279 (100%) | 0 |
| 0.5 | 504 | 504 (100%) | 0 |
| 0.719 | 729 | 504 (69%) | **225 (31%)** |
| 0.836 | 846 | 504 (60%) | **342 (40%)** |

Vượt 50% nén, method thoái hoá dần thành L1-norm thuần. Điều này khớp với việc ta cạnh
tranh ở 27% và sập ở 72%.

**Nhưng đã thử sửa và đều thất bại.** Chỉ số dùng để so: tổng khoảng cách từ filter bị
cắt tới filter được giữ gần nhất (thấp hơn = phần bị cắt được biểu diễn tốt hơn), đo
trên checkpoint HRank đã train:

| Cách sửa | 0.281 | 0.5 | 0.719 | 0.836 |
|---|---|---|---|---|
| Bỏ khoá filter sống sót (phá trần n/2) | −1.3% | −2.3% | −1.7% | −1.1% |
| Gom cụm average-linkage | −0.8% | −2.9% | −2.1% | −2.0% |
| Gom cụm complete-linkage | −0.8% | −2.0% | −0.9% | −0.4% |

(số âm = tệ hơn hiện tại)

Khoá cả hai filter là **có chủ đích**: nó đảm bảo không bao giờ giết cả hai thành viên
của một cặp dư thừa. Trần n/2 là cái giá phải trả, và ở mức nén ≤50% ta không chạm tới
nó. **Không nên sửa chỗ này.**

### 2.4 Tầng unstructured: chưa có bằng chứng có tác dụng

| cấu hình | bi-level − structured-only |
|---|---|
| uniform, 400ep | **+0.27** |
| steep, 400ep | **−0.32** |

Cùng ngân sách, chỉ đổi lịch phân bổ mà dấu lật hẳn → ±0.3 nằm trong nhiễu seed. Sau 6
run vẫn chưa trả lời được câu hỏi trung tâm của phương pháp.

### 2.5 Vấn đề thật của tầng unstructured: không được tính công

Surgery chỉ cắt kênh, nên các trọng số bị zero-hoá vẫn nằm nguyên trong lean model và
được `params`/`MACs` đếm đủ. Bi-level **trả giá accuracy** cho tầng này mà **không nhận
được gì** trên trục đang dùng để so sánh.

Đo tại điểm 91.23M:

| | top-1 | params | **params ≠ 0** | MACs |
|---|---|---|---|---|
| CORING | 94.76% | 0.66M | 0.66M | 91.23M |
| bi-level | 93.82% | 0.615M | **0.321M** | 91.64M |

Tại điểm 34.79M còn chênh lệch hơn: bi-level cho **0.048M** params ≠ 0 so với
structured-only 0.241M — ít hơn 5 lần.

Thêm chẩn đoán: tầng unstructured đang làm việc trên cả những filter sắp bị xoá.

| ratio | zero-hoá vào filter sống | vào filter **bị xoá** |
|---|---|---|
| 0.281 | 69.4% | 30.6% (vứt đi) |
| 0.719 | 26.8% | **73.2% (vứt đi)** |

---

## 3. Ba đề xuất cải tiến

### 3.1 Đổi unstructured thành 2:4 semi-structured sparsity — **ưu tiên cao nhất**

**Vấn đề nó giải quyết.** Sparsity rải rác không cho tăng tốc trên bất kỳ phần cứng nào,
nên không ai tính công cho nó. 2:4 sparsity (mỗi nhóm 4 trọng số liên tiếp giữ 2 lớn
nhất) được tensor core NVIDIA Ampere trở lên thực thi ở khoảng **2× throughput**.

**Mức nén gần như không đổi:**

| | zero | params ≠ 0 | Tăng tốc thật? |
|---|---|---|---|
| ngưỡng `sensitivity × std(W)` hiện tại | 48.1% | 0.321M | không |
| 2:4 | 50.0% | 0.310M | **có, ~2×** |

Cùng mức nén, nhưng một bên đo được trên đồng hồ. Đây là thứ biến tầng unstructured từ
"vô dụng" thành đóng góp chính, và khớp thẳng với câu chuyện latency trên DeepFish.

**Chi phí:** ~20 dòng thay cho `UnstructuredPruner.prune`.

**Chi tiết cần xử lý:** một số lớp có `C_in × k × k` không chia hết 4 (đo được: 27, 207,
414) vì số kênh sau prune lẻ. Cách gỡ: ép số kênh còn lại về bội số của 4 hoặc 8 — việc
này **tự nó đã có lợi cho tốc độ thật**, kể cả không dùng 2:4.

### 3.2 Cho hai tầng tương tác — làm cho chữ "bi-level" có nghĩa

**Vấn đề nó giải quyết.** Hiện `UnstructuredPruner` và `StructuredPruner` chạy nối tiếp
nhưng hoàn toàn độc lập: một bên xét magnitude, một bên xét khoảng cách L1-∞-∞, không
bên nào biết bên kia làm gì. Thực chất là hai method xếp cạnh nhau.

**Cách ghép.** Dùng độ nhạy của từng lớp với tầng unstructured để quyết định tỉ lệ cắt
filter của tầng structured ở lớp đó. Lớp nào zero-hoá k% trọng số mà output gần như
không đổi → lớp đó dư thừa → cắt filter mạnh hơn.

Điều này giải quyết luôn bài toán phân bổ theo lớp (§2.2) nhưng bằng **số liệu đo được**
thay vì hình dạng đặt tay. Khi reviewer hỏi "tỉ lệ này từ đâu ra", có câu trả lời.

**Chi phí:** cần một calibration pass đo độ nhạy mỗi lớp. Vừa phải.

### 3.3 Đưa độ thưa của filter vào điểm chọn filter

**Vấn đề nó giải quyết.** Sau tầng unstructured, một filter bị zero 90% gần như đã chết —
xoá nó gần như miễn phí. Hàm `l1inftyinfty` bắt được phần nào (norm nhỏ) nhưng không
trực tiếp.

**Cách làm.** Thêm số trọng số khác 0 của filter vào công thức điểm, hoặc dùng làm
tie-breaker. Vài dòng, và tạo ra tương tác thật giữa hai tầng.

**Chi phí:** thấp.

---

## 4. Những thứ đã thử và thất bại

Ghi lại để khỏi làm lại.

| Ý tưởng | Kết quả đo |
|---|---|
| Tăng ngân sách 300 → 400 epoch | +0.42 điểm, không lấp được khoảng cách |
| Lịch phân bổ `chip` | gần như trùng `uniform` |
| Lịch phân bổ `steep` | thua 0.07–0.66 điểm |
| Đảo thứ tự structured → unstructured | sparsity cuối 47.2% vs 48.1%, không hơn |
| Bỏ khoá filter sống sót (phá trần n/2) | tệ hơn 1–2% |
| Gom cụm thay ghép cặp | tệ hơn 0.4–2.9% |

Lý do đảo thứ tự không ăn thua dù 73.2% công sức bị vứt: ngưỡng `sensitivity × std(W)`
tự thích nghi — structured zero-hoá filter xong thì `std(W)` tụt, ngưỡng tụt theo, triệt
tiêu phần tiết kiệm được.

---

## 5. Thứ tự đề xuất

1. **Thêm cột `params ≠ 0` vào `count_cost` và wandb summary.** Rẻ nhất, tác động lớn
   nhất — đây là trục mà CORING/CHIP/HRank (đều thuần structured) không có gì để đáp.
2. **Implement 2:4 sparsity (§3.1).** Biến trục đó thành tăng tốc đo được.
3. **Chạy 3 seed cho một cấu hình `uniform` 400ep, cả hai biến thể.** Trả lời dứt điểm
   câu hỏi §2.4. Nếu `params ≠ 0` thành trục chính thì phải chứng minh tầng unstructured
   không phá accuracy một cách hệ thống.
4. Chỉ khi cần dòng 34.79M trong bảng mới tính tới mở scope (§2.1). Việc lớn: mask dùng
   chung theo stage + viết lại surgery, ~150–250 dòng, và **không** liên quan gì tới việc
   làm tầng unstructured có tác dụng.

Việc **không** nên làm: sửa thuật toán ghép cặp (§2.3), thêm lịch phân bổ mới (§2.2),
tăng epoch tiếp (§1.4).

---

## 6. Tái lập các phép đo

Mọi con số trong tài liệu này đo bằng các script rời chạy trên checkpoint
`resnet_56.pt` của HRank. Các hàm liên quan trong repo:

| Phép đo | Hàm / file |
|---|---|
| params, MACs | `count_cost` trong [cifar.py](../cifar.py) |
| tỉ lệ cắt theo lớp | `layer_ratios`, `LAYER_SCHEDULES` trong [cifar.py](../cifar.py) |
| match iso-FLOPs | `sparsity_for_target` trong [cifar.py](../cifar.py) |
| tiêu chí chọn filter | [pruning/structured.py](../pruning/structured.py) |
| ngưỡng unstructured | [pruning/unstructured.py](../pruning/unstructured.py) |
| nạp checkpoint HRank | `load_hrank_state_dict` trong [models/cifar.py](../models/cifar.py) |

Self-check có sẵn:

```bash
python -m models.cifar            # params khớp bảng published
python -m pruning.surgery_cifar   # lean model khớp masked model, max|diff| < 1e-7
```
