"""
CORING baseline (chuan) cho YOLO — self-contained, chi can torch + numpy.

CORING = effiCient tensOr decomposition-based filteR prunING (Neural Networks 2024).
Repo goc: https://github.com/vantienpham/CORING (Pham, Zniyed, Nguyen).

Thuat toan CHUAN cua CORING:
  HOSVD rank-1 (moi filter (C,h,w) -> 3 vector dai dien [a,b,c])
  -> VBD similarity giua cac filter
  -> saliency Algorithm 1 (min_sum) -> giu top-k filter khac biet nhat (cat cai trung lap).

Tinh vectorized (batch SVD + VBD dang dong var(a-b)=var(a)+var(b)-2cov) -> nhanh,
ket qua == cong thuc goc. Da chuan hoa dau vector ky di (dau SVD tuy y ma VBD nhay dau)
-> ket qua TAT DINH / tai hien duoc.

CoringPruner noi saliency -> s_mask (cung interface StructuredPruner); surgery + finetune
do prune.py lo (k-shot surgery-first).
"""
import numpy as np
import torch

from .norms import get_weight


# --- 1) HOSVD rank-1 (batch): weight (N,C,h,w) -> list factor (N,d) ---
@torch.no_grad()
def hosvd_rank1_factors(weight):
    """Left-singular troi nhat moi mode-unfold (batch ca N filter 1 lan).
    Bo fold d<2 (vd conv 1x1 o mode 1,2). Chuan hoa dau: phan tu |lon nhat| duong."""
    N, C, h, w = weight.shape
    wf = weight.float()
    unfolds = [
        wf.reshape(N, C, h * w),                       # mode 0 -> (N, C, h*w)
        wf.permute(0, 2, 1, 3).reshape(N, h, C * w),   # mode 1 -> (N, h, C*w)
        wf.permute(0, 3, 1, 2).reshape(N, w, C * h),   # mode 2 -> (N, w, C*h)
    ]
    factors = []
    for unf in unfolds:
        if unf.shape[1] < 2:
            continue
        u, _, _ = torch.linalg.svd(unf, full_matrices=False)  # (N, d, k)
        f = u[..., 0]                                         # (N, d)
        idx = f.abs().argmax(dim=1, keepdim=True)
        sign = torch.sign(torch.gather(f, 1, idx))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        factors.append(f * sign)
    return factors


# --- 2) VBD matrix (dang dong): var(a-b) = var(a) + var(b) - 2*cov(a,b) ---
@torch.no_grad()
def vbd_matrix(F, eps=1e-12):
    """F (N,d) -> ma tran VBD (N,N) khoang cach (nho = giong). 1 matmul."""
    d = F.shape[1]
    Fc = F - F.mean(dim=1, keepdim=True)
    cov = (Fc @ Fc.t()) / (d - 1)              # (N,N)
    v = torch.diagonal(cov)                    # var moi filter
    num = v[:, None] + v[None, :] - 2 * cov    # = var(a-b)
    return num / (v[:, None] + v[None, :] + eps)


# --- 3) saliency — Algorithm 1 (min_sum) ---
def _compare_sum(row, col, mat, inf):
    """VBD la distance -> bo cai co tong khoang-cach NHO hon (du thua hon)."""
    n = mat.shape[0]
    sum_row = sum_col = 0.0
    for i in range(n):
        if mat[row, i] != inf:
            sum_row += mat[row, i]
        if mat[i, col] != inf:
            sum_col += mat[i, col]
    return col if sum_row > sum_col else row


def get_saliency(mat):
    """mat (N,N) VBD distance -> saliency (N,): lon = quan trong (giu); nho = loai som.
    Moi vong: chon cap giong nhau nhat (argmin) -> bo cai du thua hon (compare_sum)."""
    n = mat.shape[0]
    saliency = np.full(n, n - 1, dtype=np.float32)
    inf = float("inf")
    for i in range(n):
        mat[i, i] = inf
    for i in range(n - 1):
        row, col = np.unravel_index(np.argmin(mat), mat.shape)
        mat[row, col] = inf
        mat[col, row] = inf
        idx = _compare_sum(row, col, mat, inf)
        mat[idx, :] = inf
        mat[:, idx] = inf
        saliency[idx] = i
    return saliency


@torch.no_grad()
def coring_rank(weight):
    """weight (N,C,h,w) -> saliency (N,) numpy (HOSVD + VBD + min_sum)."""
    factors = hosvd_rank1_factors(weight.detach())
    if not factors:
        return np.arange(weight.shape[0], dtype=np.float32)
    S = None
    for f in factors:                          # trung binh VBD qua cac fold
        m = vbd_matrix(f)
        S = m if S is None else S + m
    S = (S / len(factors)).cpu().numpy()
    return get_saliency(S)


# --- CoringPruner: saliency -> s_mask (cung interface StructuredPruner) ---
class CoringPruner:
    """Structured filter pruning bang saliency CORING (thay cho L1-inf-inf)."""

    def __init__(self, model):
        self.model = model     # Scope (co .get_prunable_layers)

    @torch.no_grad()
    def prune(self, prune_ratio=0.3, verbose=True):
        convs = self.model.get_prunable_layers(pruning_type="structured")
        total = 0
        for layer in convs:
            weight = get_weight(layer)             # (N, C, h, w)
            n = weight.shape[0]
            k = int(round(n * prune_ratio))
            if k <= 0:
                layer.mask_handler.update(torch.ones(n, device=weight.device))
                layer.mask_handler.apply(layer.conv)
                continue

            saliency = coring_rank(weight)                       # (N,)
            keep = np.argsort(saliency)[k:]                      # giu top-(n-k) cao nhat
            mask = torch.zeros(n, device=weight.device)
            mask[torch.as_tensor(keep, device=weight.device, dtype=torch.long)] = 1.0
            layer.mask_handler.update(mask)
            layer.mask_handler.apply(layer.conv)
            total += k

        if verbose:
            print(f"[CORING HOSVD+VBD+min_sum] ratio={prune_ratio:.3f} -> "
                  f"pruned {total} filters over {len(convs)} layers.")
