"""L2-SP starting-point regularization (Li et al., 2018) and full-model displacement tracking.

    Omega(theta, theta_0) = 1/2 * sum_j (theta_j - theta_0,j)^2

Manual requirements: sum over all unique parameters (weights, biases, normalization parameters),
tied parameters counted once, theta_0 fixed, Omega NOT averaged over parameters. The gradient of
lambda*Omega is lambda*(theta - theta_0); it is added to the (already unscaled) SFT gradient before
clipping, which is exactly the gradient of the total objective L_SFT + lambda*Omega.
"""
import time
import logging
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StartingPointReference:
    """Frozen FP32 snapshot theta_0 of every unique parameter. Used as the L2-SP anchor for
    `l2sp` and as the displacement reference ||theta - theta_0||_2 for both Full FT variants."""

    def __init__(self, model: nn.Module, reg_lambda: float = 0.0, device: str = "cpu"):
        self.reg_lambda = float(reg_lambda)
        self.names: List[str] = []
        self.theta_0: Dict[str, torch.Tensor] = {}
        seen: Set[int] = set()
        n = 0
        # named_parameters() already de-duplicates shared tensors; the data_ptr check makes tied
        # embeddings/lm_head explicit even if a model returns both names.
        for name, p in model.named_parameters(remove_duplicate=True):
            if p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            ref = p.detach().float().clone()
            if device == "cpu":
                ref = ref.cpu()
                if torch.cuda.is_available():
                    ref = ref.pin_memory()
            else:
                ref = ref.to(device)
            self.theta_0[name] = ref
            self.names.append(name)
            n += p.numel()
        self.n_params = n
        self.device = device
        self.storage_mb = n * 4 / 2**20
        self.regularizer_time_sec = 0.0
        self.transfer_benchmark_sec = self._benchmark_transfer(model) if device == "cpu" else 0.0
        logger.info("theta_0 snapshot: %d unique tensors, %s params, %.1f MB on %s (one full transfer %.3fs)",
                    len(self.names), f"{n:,}", self.storage_mb, device, self.transfer_benchmark_sec)

    def _benchmark_transfer(self, model: nn.Module) -> float:
        dev = next(model.parameters()).device
        if dev.type != "cuda":
            return 0.0
        torch.cuda.synchronize()
        t0 = time.time()
        for name in self.names:
            _ = self.theta_0[name].to(dev, non_blocking=True)
        torch.cuda.synchronize()
        return time.time() - t0

    def _params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return dict(model.named_parameters(remove_duplicate=True))

    @torch.no_grad()
    def apply_l2sp_gradient(self, model: nn.Module) -> float:
        """grad += lambda * (theta - theta_0) for every unique parameter; returns Omega(theta_t)."""
        t0 = time.time()
        params = self._params(model)
        omega = None
        for name in self.names:
            p = params[name]
            diff = p.detach().float() - self.theta_0[name].to(p.device, non_blocking=True)
            sq = diff.pow(2).sum()
            omega = sq if omega is None else omega + sq
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.add_(diff.to(p.grad.dtype), alpha=self.reg_lambda)
        omega_val = 0.5 * float(omega.item())
        self.regularizer_time_sec += time.time() - t0
        return omega_val

    @torch.no_grad()
    def omega(self, model: nn.Module) -> float:
        params = self._params(model)
        total = None
        for name in self.names:
            p = params[name]
            sq = (p.detach().float() - self.theta_0[name].to(p.device, non_blocking=True)).pow(2).sum()
            total = sq if total is None else total + sq
        return 0.5 * float(total.item())

    def displacement(self, model: nn.Module) -> float:
        """||theta - theta_0||_2 = sqrt(2 * Omega)."""
        return (2.0 * max(self.omega(model), 0.0)) ** 0.5

    @torch.no_grad()
    def module_update_norms(self, model: nn.Module, target_suffixes: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
        """Per target linear module: ||W - W0||_F and ||W0||_F (FP32). Gives the layer-wise eta
        profile for Full FT variants, comparable with the adapters' merged corrections."""
        params = self._params(model)
        out: Dict[str, Dict[str, float]] = {}
        for name in self.names:
            if not name.endswith(".weight"):
                continue
            mod = name[: -len(".weight")]
            if not mod.split(".")[-1] in target_suffixes:
                continue
            p = params[name]
            w0 = self.theta_0[name].to(p.device, non_blocking=True)
            out[mod] = {"delta_fro": float((p.detach().float() - w0).norm().item()),
                        "w0_fro": float(w0.norm().item())}
        return out
