"""
Song Han unstructured pruning (level-1 cua bi-level pruning).

Threshold = sensitivity * std(weight) cho tung layer; zero cac trong so |w| <= threshold.
Mask duoc giu va ap lai sau moi optimizer.step() (qua apply_masks) de on dinh sparsity.
"""
import torch


class UnstructuredPruner:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def prune(self, sensitivity=1.0, verbose=True):
        convs = self.model.get_prunable_layers(pruning_type="unstructured")
        for layer in convs:
            weight = layer.conv.weight.data
            threshold = sensitivity * weight.std().item()
            new_mask = (weight.abs() > threshold).float()
            layer.mask_handler.update(new_mask)
            layer.mask_handler.apply(layer.conv)
        if verbose:
            print(f"[Song Han] sensitivity={sensitivity:.3f} -> applied to {len(convs)} layers.")

    @torch.no_grad()
    def apply_masks(self):
        """Goi sau moi optimizer.step() de ep cac trong so bi prune ve 0."""
        for layer in self.model.get_prunable_layers(pruning_type="unstructured"):
            if layer.u_mask is not None:
                layer.mask_handler.apply(layer.conv)

    @torch.no_grad()
    def global_sparsity(self):
        total = zeros = 0
        for layer in self.model.get_prunable_layers(pruning_type="unstructured"):
            w = layer.conv.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
        return zeros / max(total, 1)
