# So sánh 3 chiều: Gốc · Rebuild ResNet-56 · Rebuild YOLO

Đọc code thật từng branch (`git show`). ✓ = khớp gốc. Mục tiêu: xác nhận rebuild YOLO **cùng thuật toán** với rebuild ResNet-56 và với bản gốc.

## CORING

| Khía cạnh | Gốc | Rebuild ResNet-56<br>`prune_resnet56_cifar10.py` | Rebuild YOLO<br>`prune.py --method coring` |
|---|---|---|---|
| Importance | HOSVD + VBD similarity + min_sum (Alg 1) | `coring_rank` (HOSVD+VBD+min_sum) ✓ | `coring_rank` (same) ✓ |
| Cấu trúc prune | k-shot, surgery-first | k-shot surgery-first + calibrate ✓ | k-shot surgery-first + calibrate ✓ |
| Weight | copy + finetune | copy (`load_resnet56_pruned`) ✓ | copy (`convert_to_lean`) ✓ |
| CPR / knob | compress_rate **vector** `[0.]+[r]*29` | vector (same) ✓ | scalar `--target-sparsity` (uniform) |

*YOLO khớp ResNet-56 (cùng `coring_rank`). Chỉ khác notation CPR: scalar-uniform (YOLO) vs vector-uniform (resnet56) — nội dung tương đương.*

## NORTON

| Khía cạnh | Gốc | Rebuild ResNet-56<br>`norton_cifar10.py` | Rebuild YOLO<br>`norton.py` |
|---|---|---|---|
| Decompose | CP-decomposition (CPDBlock) | CPD (`parafac`) ✓ | CPD (`parafac`) ✓ |
| Prune criterion | subspace-angle | `_factor_saliency` (pabs/csa/vbd) ✓ | `_factor_saliency` (same) ✓ |
| Cấu trúc | decompose → ft → prune → ft | same ✓ | same ✓ |
| Weight | copy (materialize factors) + finetune | copy ✓ | copy ✓ |
| CPR / knob | compress_rate **vector** + rank | vector + rank ✓ | scalar `--prune-ratio` (uniform) hoặc vector |

*YOLO khớp ResNet-56 hoàn toàn (cùng CPD + subspace-angle). Scalar `--prune-ratio` → uniform vector nội bộ.*

## SVP / SLIMING

| Khía cạnh | Gốc (paper / code) | Rebuild ResNet-56<br>`prune_resnet56_cifar10_svp.py` | Rebuild YOLO<br>`prune_svp.py` |
|---|---|---|---|
| Allocation (số kênh) | GAM — singular values (Alg 1) | GAM ✓ | GAM ✓ |
| Chọn filter | GEM greedy nuclear-norm (Alg 2) | **GEM greedy** ✓ | **GEM top-K** (xấp xỉ, scalable) ⚠️ |
| Weight | copy + finetune (Alg 3) — *code released: scratch* | copy + finetune ✓ (Alg 3) | copy + finetune ✓ (Alg 3) |
| Prune scope | mid + residual | mid + residual (GAM 54 conv) ✓ | structured layers |
| Model | resnet.py Option B (`embed`); ckpt release Option A | **Option A** (khớp ckpt release) | YOLO (n/a) |
| Recipe | code train.py: 600ep/lr0.5/wd2e-5/augment nặng · Table 2: 300ep/lr0.1/wd5e-3/crop+flip | theo **code** (reproduce) | recipe finetune chung |

*YOLO khớp ResNet-56 ở **GAM + copy weight**; KHÁC duy nhất ở **GEM**: resnet56 dùng greedy thật (conv nhỏ chạy được), YOLO dùng top-K vì greedy O(N²·SVD) bất khả thi trên conv 512–1024 kênh. Paper ≠ code (Table 2 vs train.py) → rebuild bám code để reproduce.*

---

**Tổng kết:** CORING & NORTON — rebuild YOLO **giống hệt** rebuild ResNet-56 về thuật toán (cùng hàm). SVP — giống ở GAM + copy, **chỉ khác GEM** (greedy → top-K, do scalability). CPR: gốc/resnet56 dùng vector-uniform, YOLO dùng scalar-uniform → tương đương.
