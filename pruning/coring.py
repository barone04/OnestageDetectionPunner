"""
CORING baseline — CHEP TRUNG THANH tu code goc (loop-based, KHONG vectorized).

CORING = effiCient tensOr decomposition-based filteR prunING (Neural Networks 2024).
Repo goc: https://github.com/pvti/CORING (Pham, Zniyed, Nguyen).

Chep dung logic 4 file goc, chi doi import cho self-contained:
  decompose.py        -> HOSVD/SVD/Tucker (get_u_svd, decompose)
  similarity.py       -> 6 metric (VBD mac dinh)
  ranking_strategy.py -> Algorithm 1 (get_saliency, min_sum)
  main/rank.py        -> get_correlation_mat, get_rank

CoringPruner (glue: saliency -> s_mask) cua bai YOLO. Yeu cau: tensorly, tqdm.
"""
import numpy as np
import torch
import tensorly as tl
from tensorly import unfold
from tensorly.decomposition import tucker
from tqdm.auto import tqdm

from .norms import get_weight

tl.set_backend("pytorch")
_cos = torch.nn.CosineSimilarity(dim=0)


# ===========================================================================
# decompose.py
# ===========================================================================
def get_num_unfold(decomposer):
    return 1 if decomposer == "svd" else 3


def get_u_svd(x, rank=1):
    u, _, _ = torch.linalg.svd(x, full_matrices=False)
    return u[:, :rank]


def _svd_factors(x, rank=1):
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    v = torch.transpose(vh, 0, 1)
    return [u[:, :rank], v[:, :rank]]


def decompose(x, decomposer="hosvd", rank=1, mode=0):
    if decomposer == "tucker":
        _, factors = tucker(x, rank=[rank, rank, rank])
        return factors
    elif decomposer == "hosvd":
        return [get_u_svd(unfold(x, i), rank=rank) for i in range(get_num_unfold(decomposer))]
    elif decomposer == "svd":
        return _svd_factors(unfold(x, mode), rank=rank)


# ===========================================================================
# similarity.py
# ===========================================================================
def cosine(a, b):
    return _cos(torch.flatten(a), torch.flatten(b))


def Euclide(a, b):
    return torch.dist(a, b)


def Manhattan(a, b):
    return torch.dist(a, b, 1)


def Pearson(a, b):
    from torchmetrics.functional import pearson_corrcoef  # lazy
    return pearson_corrcoef(torch.flatten(a), torch.flatten(b))


def SNR(signal, noise):
    return torch.var(signal - noise) / torch.var(signal)


def VBD(a, b):
    return torch.var(a - b) / (torch.var(a) + torch.var(b))


def similarity(a, b, criterion):
    if criterion == "cosine_sim":
        return abs(cosine(a, b))
    elif criterion == "Euclide_dis":
        return Euclide(a, b)
    elif criterion == "Manhattan_dis":
        return Manhattan(a, b)
    elif criterion == "Pearson_sim":
        return Pearson(a, b)
    elif criterion == "SNR_dis":
        return SNR(a, b)
    elif criterion == "VBD_dis":
        return VBD(a, b)


# ===========================================================================
# ranking_strategy.py — Algorithm 1
# ===========================================================================
def compare_sum(row, col, matrix, inf, dis=1):
    num_row = matrix.shape[0]
    sum_row = sum_col = 0
    for i in range(num_row):
        if matrix[row, i] != inf:
            sum_row += matrix[row, i]
        if matrix[i, col] != inf:
            sum_col += matrix[i, col]
    if dis == 1:
        return col if sum_row > sum_col else row
    return col if sum_row < sum_col else row


def get_saliency(mat, strategy="min_sum", dis=1):
    num_row = mat.shape[0]
    saliency = np.full(num_row, num_row - 1, dtype=np.float32)

    if strategy == "sum":
        for i in range(num_row):
            saliency[i] = dis * mat[i, :].sum()
        return saliency

    inf = dis * float("inf")
    for i in range(num_row):
        mat[i, i] = inf
    for i in range(num_row - 1):
        idx = mat.argmin() if dis == 1 else mat.argmax()
        row, col = np.unravel_index(idx, mat.shape)
        mat[row, col] = inf
        mat[col, row] = inf
        index = compare_sum(row, col, mat, inf, dis)
        mat[index, :] = inf
        mat[:, index] = inf
        saliency[index] = i
    return saliency


# ===========================================================================
# main/rank.py
# ===========================================================================
def get_correlation_mat(all_filters_u_dict, criterion="VBD_dis"):
    num_filters = len(all_filters_u_dict)
    correlation_mat = [[0.0] * num_filters for _ in range(num_filters)]
    for i in tqdm(range(num_filters), leave=False):
        for j in range(num_filters):
            ui = all_filters_u_dict[i]
            uj = all_filters_u_dict[j]
            fold = len(ui)
            s = 0.0
            for x in range(fold):
                s += similarity(torch.tensor(ui[x]), torch.tensor(uj[x]), criterion=criterion)
            correlation_mat[i][j] = (s / fold).item()
    return correlation_mat


def get_rank(weight, decomposer="hosvd", rank=1, mode=0,
             criterion="VBD_dis", strategy="min_sum"):
    """weight (N,C,h,w) -> saliency (N,) numpy; lon = quan trong (giu top-k)."""
    num_filters = weight.size(0)
    all_filters_u_dict = {}
    for i_filter in tqdm(range(num_filters), leave=False):
        f = weight.detach()[i_filter, :]
        u = decompose(f, decomposer=decomposer, rank=rank, mode=mode)
        all_filters_u_dict[i_filter] = [i.tolist() for i in u]

    correl_matrix = np.array(get_correlation_mat(all_filters_u_dict, criterion))
    dis = 1 if "dis" in criterion else -1
    return get_saliency(correl_matrix, strategy, dis)


@torch.no_grad()
def coring_rank(weight, criterion="VBD_dis", strategy="min_sum"):
    """weight (N,C,h,w) -> saliency (N,) numpy (goi get_rank goc)."""
    return get_rank(weight.detach().cpu(), criterion=criterion, strategy=strategy)


# ===========================================================================
# CoringPruner — glue: saliency -> s_mask (cung interface StructuredPruner)
# ===========================================================================
class CoringPruner:
    def __init__(self, model, criterion="VBD_dis", strategy="min_sum"):
        self.model = model            # Scope (co .get_prunable_layers)
        self.criterion = criterion
        self.strategy = strategy

    @torch.no_grad()
    def prune(self, prune_ratio=0.3, verbose=True):
        convs = self.model.get_prunable_layers(pruning_type="structured")
        total = 0
        for layer in convs:
            weight = get_weight(layer)
            n = weight.shape[0]
            k = int(round(n * prune_ratio))
            if k <= 0:
                layer.mask_handler.update(torch.ones(n, device=weight.device))
                layer.mask_handler.apply(layer.conv)
                continue

            saliency = coring_rank(weight, self.criterion, self.strategy)
            keep = np.argsort(saliency)[k:]
            mask = torch.zeros(n, device=weight.device)
            mask[torch.as_tensor(keep, device=weight.device, dtype=torch.long)] = 1.0
            layer.mask_handler.update(mask)
            layer.mask_handler.apply(layer.conv)
            total += k

        if verbose:
            print(f"[CORING {self.criterion}/{self.strategy}] ratio={prune_ratio:.3f} -> "
                  f"pruned {total} filters over {len(convs)} layers.")
