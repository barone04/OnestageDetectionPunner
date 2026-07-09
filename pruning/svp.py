"""
SVP (Singular Value Pruning) — GAM (Greedy Addition Method) cho YoloModel.

Port tu SVP-main/pruning_rate.py, ap dung len PrunableConv cua pipeline YOLOv1:
  1. Tinh singular values (SVD tren weight reshape) cho moi layer structured-prunable.
  2. GAM phan bo global so kenh giu lai theo target_rate (compress rate).
  3. Chon filter giu lai theo norm L1-inf-inf (top-k) va cap nhat s_mask.
  4. Sau finetune + surgery -> model lean (giong pipeline bi-level).

Tham khao: SVP-main/pruning_rate.py
"""
import numpy as np
import torch

from .norms import get_weight, l1inftyinfty


def compute_singular_values(weight: torch.Tensor) -> np.ndarray:
    """Reshape conv weight (out, in, H, W) -> (out, -1), tra singular values."""
    reshaped = weight.view(weight.size(0), -1)
    _, sv, _ = torch.linalg.svd(reshaped, full_matrices=False)
    return sv.detach().cpu().numpy()


def find_optimal_channels_gam(
    singular_values,
    total_channels_to_keep: int,
    original_number_of_channels,
):
    """
    GAM: moi buoc them 1 kenh vao layer co singular value lon nhat tiep theo.
    Tra ve list so kenh giu lai cho tung layer.
    """
    num_layers = len(singular_values)
    number_of_channels_to_keep = [0] * num_layers
    current_singular_values_candidates = np.array(
        [sv[0] for sv in singular_values]
    )
    for _ in range(total_channels_to_keep):
        max_idx = int(np.argmax(current_singular_values_candidates))
        number_of_channels_to_keep[max_idx] += 1
        k = number_of_channels_to_keep[max_idx]
        orig = original_number_of_channels[max_idx]
        if k < orig:
            current_singular_values_candidates[max_idx] = singular_values[max_idx][k]
        else:
            current_singular_values_candidates[max_idx] = 0.0
    return number_of_channels_to_keep


class SVPPruner:
    """One-shot structured pruning bang SVP-GAM."""

    def __init__(self, model_scope):
        self.model_scope = model_scope

    def _get_layers(self):
        return self.model_scope.get_prunable_layers(pruning_type="structured")

    @torch.no_grad()
    def compute_allocation(self, target_rate: float):
        """
        Tinh phan bo kenh theo GAM.
        target_rate = ty le nen (compress rate), vd 0.5 -> cat ~50% kenh.
        Tra ve (layers, channels_to_keep, compress_rates).
        """
        layers = self._get_layers()
        if not layers:
            raise ValueError("Khong co layer structured-prunable trong scope hien tai.")

        weights = [get_weight(layer) for layer in layers]
        singular_values = [compute_singular_values(w) for w in weights]
        original_channels = [w.size(0) for w in weights]
        total_channels = int(np.sum(original_channels))
        total_to_keep = int((1.0 - target_rate) * total_channels)
        total_to_keep = max(total_to_keep, len(layers))  # toi thieu 1 kenh/layer

        channels_to_keep = find_optimal_channels_gam(
            singular_values, total_to_keep, original_channels
        )
        compress_rates = [
            1.0 - k / orig for k, orig in zip(channels_to_keep, original_channels)
        ]
        return layers, channels_to_keep, compress_rates

    @torch.no_grad()
    def _apply_layer_mask(self, layer, num_keep: int):
        weight = get_weight(layer)
        n = weight.shape[0]
        num_keep = max(1, min(int(num_keep), n))
        norms = l1inftyinfty(weight)
        _, top_idx = torch.topk(norms, num_keep)
        mask = torch.zeros(n, device=weight.device)
        mask[top_idx] = 1.0
        layer.mask_handler.update(mask)
        layer.mask_handler.apply(layer.conv)

    @torch.no_grad()
    def prune(self, target_rate: float = 0.5, verbose: bool = True):
        layers, channels_to_keep, compress_rates = self.compute_allocation(target_rate)
        for layer, k in zip(layers, channels_to_keep):
            self._apply_layer_mask(layer, k)

        if verbose:
            kept = sum(channels_to_keep)
            total = sum(get_weight(l).size(0) for l in layers)
            print(f"[SVP-GAM] target_rate={target_rate:.3f} -> "
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
