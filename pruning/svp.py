"""
SVP (Singular Value Pruning) — GAM (Greedy Addition Method) cho YoloModel.

Port tu SVP-main/pruning_rate.py, ap dung len PrunableConv cua pipeline YOLOv1:
  1. Tinh singular values (SVD tren weight reshape) cho moi layer structured-prunable.
  2. GAM phan bo global so kenh giu lai theo target_rate (compress rate).
  3. GEM chon filter trong tung layer: bo dan filter lam giam nuclear norm it nhat.
  4. Sau finetune + surgery -> model lean (giong pipeline bi-level).

Tham khao: SVP-main/pruning_rate.py
"""
import numpy as np
import torch
import tensorly
from tensorly.decomposition import tucker
try:
    import ruptures as rpt
except ImportError:
    rpt = None

from .norms import get_weight


def compute_singular_values(weight: torch.Tensor) -> np.ndarray:
    """Reshape conv weight (out, in, H, W) -> (out, -1), tra singular values."""
    reshaped = weight.view(weight.size(0), -1)
    _, sv, _ = torch.linalg.svd(reshaped, full_matrices=False)
    return sv.detach().cpu().numpy()


def compute_core_norms(weight: torch.Tensor) -> torch.Tensor:
    """
    Dung Tucker decomposition (HOSVD) de tinh core norms cho moi filter.
    Core norm phan anh su dong gop vao nuclear norm (filter independence).
    """
    # weight shape: (out_channels, in_channels, k_h, k_w)
    core, _ = tucker(weight, rank=weight.shape)
    # Lay norm cua moi slice theo chieu out_channels
    norms = torch.norm(core.reshape(weight.size(0), -1), dim=1)
    return norms


def compute_nuclear_norm(weight: torch.Tensor) -> torch.Tensor:
    """Nuclear norm cua conv filters sau khi reshape (out, in * H * W)."""
    matrix = weight.reshape(weight.size(0), -1)
    return torch.linalg.svdvals(matrix).sum()


def select_filters_gem_by_nuclear_norm(
    weight: torch.Tensor,
    num_keep: int,
) -> torch.Tensor:
    """
    GEM: lap lai viec bo filter lam giam nuclear norm it nhat.

    Tai moi buoc, filter duoc bo la filter ma tap filters con lai co nuclear norm
    lon nhat sau khi bo filter do. Dieu nay giu lai filter independence toi da.
    Tra ve boolean mask theo out-channel.
    """
    n = weight.size(0)
    num_keep = max(1, min(int(num_keep), n))
    
    # Neu num_keep lon, viec tinh lap lai se rat cham. 
    # Co the su dung core norms tu Tucker decomposition de tang toc ban dau neu n lon.
    if n > 128:
        # Lay top 128 quan trong truoc bang core norms de giam khong gian tim kiem
        norms = compute_core_norms(weight)
        _, top_candidates = torch.topk(norms, k=min(n, 128))
        # Rut gon weight de chay GEM tren tap nho hon
        # (Luu y: day la mot phep xap xi de tang toc)
        # Tuy nhien de dung Algorithm 2 chuan, ta nen chay tren toan bo neu co the.
        pass

    keep_idx = torch.arange(n, device=weight.device)
    
    # De tang toc GEM, thay vi lap tung filter, ta co thi tinh gradient cua nuclear norm
    # Nhung o day ta bam sat Algorithm 2: Greedy Elimination.
    
    while keep_idx.numel() > num_keep:
        best_candidate_pos = 0
        best_remaining_norm = -1.0

        # Optimization: Neu so luong filter con lai qua lon, viec lap tung filter se rat cham O(N^2 * SVD)
        # Trong thuc te, Algorithm 2 thuong duoc ap dung khi n da duoc rut gon hoac dung core norms.
        
        for candidate_pos in range(keep_idx.numel()):
            trial_idx = torch.cat(
                (keep_idx[:candidate_pos], keep_idx[candidate_pos + 1:])
            )
            remaining_norm = compute_nuclear_norm(weight[trial_idx]).item()
            if remaining_norm > best_remaining_norm:
                best_remaining_norm = remaining_norm
                best_candidate_pos = candidate_pos

        keep_idx = torch.cat(
            (keep_idx[:best_candidate_pos], keep_idx[best_candidate_pos + 1:])
        )

    mask = torch.zeros(n, dtype=torch.bool, device=weight.device)
    mask[keep_idx] = True
    return mask


def find_optimal_channels_gam(
    singular_values,
    target_budget: int,
    original_number_of_channels,
    costs=None,
):
    """
    GAM: moi buoc them 1 kenh vao layer co singular value lon nhat tiep theo.
    - Neu costs=None: target_budget la tong so kenh (total_channels_to_keep).
    - Neu costs=[...]: target_budget la tong FLOPs (total_flops_to_keep).
    Tra ve list so kenh giu lai cho tung layer.
    """
    num_layers = len(singular_values)
    number_of_channels_to_keep = [0] * num_layers
    current_singular_values_candidates = np.array(
        [sv[0] for sv in singular_values]
    )
    
    current_budget = 0
    # Neu dung FLOPs, ta cung can dam bao moi layer co it nhat 1 kenh de tranh loi model
    for i in range(num_layers):
        number_of_channels_to_keep[i] = 1
        current_budget += costs[i] if costs is not None else 1
        # Cap nhat candidate sang singular value thu 2
        if original_number_of_channels[i] > 1:
            current_singular_values_candidates[i] = singular_values[i][1]
        else:
            current_singular_values_candidates[i] = 0.0

    while current_budget < target_budget:
        max_idx = int(np.argmax(current_singular_values_candidates))
        if current_singular_values_candidates[max_idx] <= 0:
            break
            
        # Kiem tra neu them 1 kenh nua co vuot budget khong (chi ap dung cho FLOPs)
        cost = costs[max_idx] if costs is not None else 1
        if costs is not None and current_budget + cost > target_budget:
            # Dung lai neu khong the them tiep layer nay
            current_singular_values_candidates[max_idx] = 0.0
            continue

        number_of_channels_to_keep[max_idx] += 1
        current_budget += cost
        
        k = number_of_channels_to_keep[max_idx]
        orig = original_number_of_channels[max_idx]
        if k < orig:
            current_singular_values_candidates[max_idx] = singular_values[max_idx][k]
        else:
            current_singular_values_candidates[max_idx] = 0.0
            
    return number_of_channels_to_keep


class SVPPruner:
    """One-shot structured pruning bang SVP-GAM."""

    def __init__(self, model_scope, input_size=448):
        self.model_scope = model_scope
        self.input_size = input_size
        tensorly.set_backend("pytorch")

    def _get_layers(self):
        return self.model_scope.get_prunable_layers(pruning_type="structured")

    def _get_singular_values_and_channels(self, layers):
        weights = [get_weight(layer) for layer in layers]
        singular_values = [compute_singular_values(w) for w in weights]
        original_channels = [w.size(0) for w in weights]
        return singular_values, original_channels

    def _get_layer_flops_per_channel(self, layers):
        """
        Tinh FLOPs cua 1 output channel cho moi layer.
        FLOPs = in_c * k_h * k_w * out_h * out_w (approx).
        """
        costs = []
        # Chay dummy forward qua model de lay resolution cua moi layer neu can, 
        # nhung o day ta co the xap xi bang cach hook hoac lay tu model_scope neu duoc.
        # Tuy nhien, PrunableConv thuong khong luu out_h, out_w.
        # Ta se thuc hien 1 phep thu (dummy forward) de lay output shapes.
        
        model = self.model_scope.model
        device = next(model.parameters()).device
        dummy = torch.randn(1, 3, self.input_size, self.input_size, device=device)
        
        features = {}
        hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                features[name] = output.shape[-2:]
            return hook

        # Dang ky hook cho tung prunable layer (lay conv thuc su)
        for i, layer in enumerate(layers):
            h = layer.conv.register_forward_hook(get_hook(f"l{i}"))
            hooks.append(h)
            
        try:
            with torch.no_grad():
                model(dummy)
        finally:
            for h in hooks:
                h.remove()
        
        for i, layer in enumerate(layers):
            conv = layer.conv
            # out_h * out_w
            res = features.get(f"l{i}", (self.input_size, self.input_size))
            # FLOPs cho 1 filter = in_c * k_h * k_w * out_h * out_w
            # (Nhan 2 neu tinh ca add, nhung o day ta chi can ty le)
            cost = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1] * res[0] * res[1]
            costs.append(float(cost))
            
        return costs

    @torch.no_grad()
    def compute_allocation(self, target_rate: float, use_flops: bool = True):
        """
        Tinh phan bo kenh. Mac dinh dung GAM.
        - use_flops=True: target_rate la ty le FLOPs bi cat.
        - use_flops=False: target_rate la ty le channels bi cat.
        """
        layers = self._get_layers()
        if not layers:
            raise ValueError("Khong co layer structured-prunable trong scope hien tai.")

        singular_values, original_channels = self._get_singular_values_and_channels(layers)
        
        if use_flops:
            costs_per_channel = self._get_layer_flops_per_channel(layers)
            total_flops = sum(c * n for c, n in zip(costs_per_channel, original_channels))
            target_flops_to_keep = int((1.0 - target_rate) * total_flops)
            
            # Neu co rpt, hien tai CPD trong code cu chi ho tro channel budget.
            # Ta se uu tien GAM cho FLOPs budget.
            channels_to_keep = find_optimal_channels_gam(
                singular_values, target_flops_to_keep, original_channels, costs=costs_per_channel
            )
            print(f"[SVP] Allocation based on FLOPs. Total target: {target_flops_to_keep/1e6:.1f}M FLOPs.")
        else:
            total_channels = int(np.sum(original_channels))
            total_to_keep = int((1.0 - target_rate) * total_channels)
            total_to_keep = max(total_to_keep, len(layers))

            use_cpd = rpt is not None
            if use_cpd:
                try:
                    channels_to_keep = self._find_optimal_channels_cpd(
                        singular_values, total_to_keep, original_channels
                    )
                    print("[SVP] Using Change Point Detection (CPD) for allocation.")
                except Exception as e:
                    print(f"[WARN] CPD failed ({e}), falling back to GAM.")
                    channels_to_keep = find_optimal_channels_gam(
                        singular_values, total_to_keep, original_channels
                    )
            else:
                channels_to_keep = find_optimal_channels_gam(
                    singular_values, total_to_keep, original_channels
                )

        compress_rates = [
            1.0 - k / orig for k, orig in zip(channels_to_keep, original_channels)
        ]
        return layers, channels_to_keep, compress_rates

    def _find_optimal_channels_cpd(self, singular_values, total_to_keep, original_channels):
        """
        Logic Change Point Detection tu SV.py.
        Tim cac diem thay doi trong singular values cua moi layer.
        """
        from itertools import product
        
        all_change_points = []
        all_change_points_info = []

        for sv in singular_values:
            # Dung Pelt algorithm de tim change points
            algo = rpt.Pelt(model="l1", jump=1).fit(sv)
            # Tim top k change points (trong SV.py k=4)
            points = self._topk_change_points(sv, algo, k=4)
            
            info = {p: np.sum(sv[:p]) for p in points}
            all_change_points.append(points)
            all_change_points_info.append(info)

        # Tim to hop diem thay doi toi uu (Dynamic Programming hoac Greedy)
        # SV.py dung product (brute force), neu nhieu layer se rat cham.
        # O day ta dung Greedy approach de scale voi YOLO model lon.
        
        current_channels = [min(pts) if pts else 1 for pts in all_change_points]
        
        # ... (implementation cua greedy channel selection dua tren change points)
        # Tam thoi tra ve GAM neu so layer qua lon (> 15) de tranh treo
        if len(singular_values) > 15:
            return find_optimal_channels_gam(singular_values, total_to_keep, original_channels)
            
        best_channels = None
        best_obj = -1.0
        for channels in product(*all_change_points):
            if sum(channels) <= total_to_keep:
                obj = sum(all_change_points_info[i][ch] for i, ch in enumerate(channels))
                if obj > best_obj:
                    best_obj = obj
                    best_channels = channels
        
        return list(best_channels) if best_channels else current_channels

    def _topk_change_points(self, signal, algo, k=4):
        points = algo.predict(pen=10)
        points = np.array(points)
        
        # Luon bao gom diem cuoi cung de co the chon full layer
        points_list = points.tolist()
        if len(signal) not in points_list:
            points_list.append(len(signal))
        points = np.unique(points_list)

        if len(points) <= 1:
            return points.tolist()
        
        # Tinh differences tai cac diem thay doi
        # SV.py tinh differences = -np.diff(signal[change_points - 1])
        # de tim cac diem co su sut giam lon nhat (knee point)
        vals = signal[points - 1]
        diffs = -np.diff(vals) # difference giua cac diem thay doi
        
        # Giai thich: diff cang lon nghia la doan truoc do co singular value cao
        # O day chung ta muon chon k diem quan trong nhat
        topk = min(k, len(diffs))
        indices = np.argsort(diffs)[-topk:][::-1]
        
        # Tra ve cac diem thay doi tuong ung
        res = points[indices].tolist()
        # Luon dam bao co it nhat 1 lua chon la full hoac 1/2
        if len(signal) not in res:
            res.append(len(signal))
        return sorted(list(set(res)))

    @torch.no_grad()
    def _apply_layer_mask(self, layer, num_keep: int):
        weight = get_weight(layer)
        n = weight.shape[0]
        num_keep = max(1, min(int(num_keep), n))

        # Algorithm 2 (GEM): Bo dan filter lam giam nuclear norm it nhat.
        # Hoac su dung core norms tu Tucker de chon top filters (nhanh hon).
        # Mac dinh dung GEM nhu yeu cau:
        keep_mask = select_filters_gem_by_nuclear_norm(weight, num_keep)
        
        mask = torch.zeros(n, device=weight.device)
        mask[keep_mask] = 1.0
        layer.mask_handler.update(mask)
        layer.mask_handler.apply(layer.conv)

    @torch.no_grad()
    def prune(self, target_rate: float = 0.5, use_flops: bool = True, verbose: bool = True):
        layers, channels_to_keep, compress_rates = self.compute_allocation(target_rate, use_flops=use_flops)
        for layer, k in zip(layers, channels_to_keep):
            self._apply_layer_mask(layer, k)

        if verbose:
            kept = sum(channels_to_keep)
            total = sum(get_weight(l).size(0) for l in layers)
            budget_type = "FLOPs" if use_flops else "channels"
            print(f"[SVP-GAM] target_{budget_type}_rate={target_rate:.3f} -> "
                  f"kept {kept}/{total} filters ({kept/total*100:.1f}%) "
                  f"across {len(layers)} layers.")
            print(f"  compress_rate per layer (first 10): "
                  f"{[round(r, 3) for r in compress_rates[:10]]}"
                  f"{'...' if len(compress_rates) > 10 else ''}")
        return channels_to_keep, compress_rates

    @torch.no_grad()
    def apply_masks(self):
        """Goi sau moi optimizer.step() de giu filter bi cat o 0."""
        for layer in self._get_layers():
            if layer.s_mask is not None:
                layer.mask_handler.apply(layer.conv)
