"""
Structured filter pruning (level-2 cua bi-level pruning) — mixed-norm L1-inf-inf.

Y tuong: cat cac filter TRUNG LAP/yeu. Tinh ma tran khoang cach L1-inf-inf giua cac
filter; uu tien cat cap giong nhau nhat (giu cai norm lon, diet cai norm nho); neu chua
du chi tieu thi cat tiep theo norm tang dan.

Toi uu: ma tran khoang cach tinh vectorized theo TUNG hang (per-row) -> tranh vong lap
Python O(N^2) cham nhu ban goc, van an toan bo nho.
"""
import torch

from .norms import l1inftyinfty, get_weight


class StructuredPruner:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def compute_distance_matrix(self, weight):
        """weight: (N, C_in, H, W) -> D (N, N) khoang cach L1-inf-inf."""
        n = weight.shape[0]
        D = torch.zeros(n, n, device=weight.device)
        for i in range(n):
            diff = (weight - weight[i:i + 1]).abs()          # (N, C_in, H, W)
            D[i] = diff.amax(dim=(-2, -1)).sum(dim=-1)        # (N,)
        return D

    @staticmethod
    def kill_key(weight, norms, kill_by):
        """Diem dung de chon filter nao bi giet (diem THAP bi giet truoc).

        Tra ve list tuple de so sanh lexicographic:
          "norm"    : (norm,)            — hanh vi goc.
          "density" : (mat do, norm)     — mat do = ti le trong so KHAC 0 sau tang
                      unstructured; hoa thi so norm. Do tren ResNet-56: tuong quan voi
                      tam quan trong that rho=0.325, trong khi norm chi rho=0.035 (norm
                      L1-inf-inf lay MAX trong kernel, ma tang unstructured xoa trong so
                      NHO -> norm gan nhu khong doi du filter mat ~48% trong so).
        Neu moi filter cung mat do (vd khong co tang unstructured, structured-only) thi
        mat do khong mang thong tin -> tu roi ve "norm".
        """
        nv = norms.tolist()
        if kill_by == "density":
            dens = (weight != 0).flatten(1).float().mean(1)
            if not torch.all(dens == dens[0]):
                return list(zip(dens.tolist(), nv))
        return [(v,) for v in nv]

    @torch.no_grad()
    def prune(self, prune_ratio=0.3, verbose=True, kill_by="norm"):
        """prune_ratio: mot so (ap deu moi lop) HOAC list/tuple mot ti le cho tung lop.

        List cho phep cat nhe o lop nong / nang o lop sau. Tim cap du thua van bang
        khoang cach L1-inf-inf; `kill_by` chi quyet dinh GIET CAI NAO trong cap du thua
        (va thu tu cua nhanh fallback) — xem kill_key.
        """
        assert kill_by in ("norm", "density"), kill_by
        convs = self.model.get_prunable_layers(pruning_type="structured")
        ratios = (list(prune_ratio) if isinstance(prune_ratio, (list, tuple))
                  else [prune_ratio] * len(convs))
        assert len(ratios) == len(convs), (
            f"prune_ratio co {len(ratios)} muc nhung model co {len(convs)} lop prunable")
        total_pruned = 0

        for layer, ratio in zip(convs, ratios):
            weight = get_weight(layer)
            n = weight.shape[0]
            num_to_remove = int(round(n * ratio))

            if num_to_remove <= 0:
                layer.mask_handler.update(torch.ones(n, device=weight.device))
                layer.mask_handler.apply(layer.conv)
                continue

            norms = l1inftyinfty(weight)                      # (N,)
            key = self.kill_key(weight, norms, kill_by)       # thap = bi giet truoc
            D = self.compute_distance_matrix(weight)          # tim cap du thua: KHONG doi

            # danh sach cap (dist, i, j) sap xep theo khoang cach tang dan
            iu, ju = torch.triu_indices(n, n, offset=1)
            dists = D[iu, ju]
            order = torch.argsort(dists)

            pruned, processed = set(), set()
            count = 0
            for idx in order.tolist():
                if count >= num_to_remove:
                    break
                i, j = int(iu[idx]), int(ju[idx])
                if i in processed or j in processed:
                    continue
                kill = j if key[i] >= key[j] else i
                pruned.add(kill)
                processed.add(i)
                processed.add(j)
                count += 1

            # neu chua du -> cat tiep theo diem nho nhat (norm, hoac mat do neu kill_by)
            if count < num_to_remove:
                remaining = [k for k in range(n) if k not in pruned]
                remaining.sort(key=lambda k: key[k])
                for k in remaining:
                    if count >= num_to_remove:
                        break
                    pruned.add(k)
                    count += 1

            mask = torch.ones(n, device=weight.device)
            if pruned:
                mask[list(pruned)] = 0.0
            layer.mask_handler.update(mask)
            layer.mask_handler.apply(layer.conv)
            total_pruned += count

        if verbose:
            desc = (f"{min(ratios):.3f}..{max(ratios):.3f}" if len(set(ratios)) > 1
                    else f"{ratios[0]:.3f}")
            print(f"[Filter L1-inf-inf] ratio={desc} -> pruned {total_pruned} filters "
                  f"over {len(convs)} layers.")
