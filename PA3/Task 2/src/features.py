"""Frozen SigLIP feature cache.

google/siglip-base-patch16-224 has no CLS token: its 224 x 224 input gives exactly 196 patch tokens (14 x 14,
row-major, as produced by the patch-embedding conv + flatten). `SiglipVisionModel(...).last_hidden_state` is the
encoder output after SigLIP's final `post_layernorm`, i.e. the final layer normalization applied exactly once
(the attention-pooling head is not used). This is verified on the first batch against
`post_layernorm(hidden_states[-1])`, and LN is not re-applied anywhere downstream.

Preprocessing is the checkpoint's own deterministic SiglipImageProcessor (RGB, bicubic resize to 224 x 224,
rescale 1/255, normalize mean = std = 0.5, no crop, no augmentation). Features are stored in FP16; the vision
encoder is then deleted and GPU memory released before any Qwen training starts.
"""
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import torch

from src.data_prep import all_image_keys, locate_image
from src.utils import (atomic_torch_save, ensure_dir, free_cuda, peak_gb, reset_peak,
                       resolve_model_revision, sync, task_path)

logger = logging.getLogger(__name__)


def _load_rgb(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


@torch.no_grad()
def cache_features(cfg: Dict[str, Any], batch_size: int = 64) -> Dict[str, Any]:
    from transformers import SiglipImageProcessor, SiglipVisionModel
    out_dir = ensure_dir(task_path(cfg["paths"]["feature_dir"]))
    keys_by_split = all_image_keys(task_path(cfg["paths"]["build_dir"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = cfg["models"]["vision"]
    report: Dict[str, Any] = {"model": name, "model_revision_sha": resolve_model_revision(name), "splits": {}}

    reset_peak()
    t_load = time.time()
    proc = SiglipImageProcessor.from_pretrained(name)
    model = SiglipVisionModel.from_pretrained(name, dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    report["load_sec"] = time.time() - t_load
    logger.info("SigLIP on %s", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (no GPU visible)")
    report["preprocessing"] = {"size": proc.size, "resample": int(proc.resample), "do_rescale": proc.do_rescale,
                               "rescale_factor": proc.rescale_factor, "image_mean": proc.image_mean,
                               "image_std": proc.image_std, "do_center_crop": getattr(proc, "do_center_crop", None)}
    grid, dim = cfg["models"]["grid"], cfg["models"]["vision_dim"]
    checked = False
    pool = ThreadPoolExecutor(8)
    t0 = time.time()
    for split, keys in keys_by_split.items():
        path = os.path.join(out_dir, f"{split}.pt")
        if os.path.exists(path):
            logger.info("features for %s exist (%s)", split, path)
            continue
        feats = torch.empty((len(keys), grid * grid, dim), dtype=torch.float16)
        t_split = time.time()
        for s in range(0, len(keys), batch_size):
            batch = keys[s:s + batch_size]
            paths = [locate_image(cfg, k) for k in batch]
            if any(p is None for p in paths):
                raise FileNotFoundError(f"images missing for {[k for k, p in zip(batch, paths) if p is None][:3]}")
            imgs = list(pool.map(_load_rgb, paths))
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            if not checked:
                report["final_layernorm_check"] = check_layernorm_once(model, px)
            h = model(pixel_values=px).last_hidden_state
            if h.shape[1:] != (grid * grid, dim):
                raise RuntimeError(f"unexpected feature shape {tuple(h.shape)}")
            if not checked:
                report["final_layernorm_check"].update({
                    "pixel_shape": list(px.shape), "feature_shape": list(h.shape[1:]),
                    "token_norm_mean": float(h.float().norm(dim=-1).mean()),
                    "token_norm_std": float(h.float().norm(dim=-1).std())})
                checked = True
            feats[s:s + len(batch)] = h.to(torch.float16).cpu()
            if (s // batch_size) % 20 == 0:
                logger.info("  %s %d/%d", split, s + len(batch), len(keys))
        if not torch.isfinite(feats).all():
            raise RuntimeError(f"non-finite FP16 features in {split}")
        atomic_torch_save({"keys": keys, "feats": feats}, path)
        report["splits"][split] = {"n_images": len(keys), "sec": time.time() - t_split,
                                   "file_mb": os.path.getsize(path) / 2**20}
        del feats
    sync()
    report["total_sec"] = time.time() - t0
    report["peak_gpu_gb"] = peak_gb()
    report["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    del model
    free_cuda()
    report["gpu_allocated_after_release_gb"] = (torch.cuda.memory_allocated() / 2**30
                                                if torch.cuda.is_available() else 0.0)
    return report


@torch.no_grad()
def check_layernorm_once(model, px: torch.Tensor) -> Dict[str, Any]:
    """Verify that the cached features are SigLIP's final LayerNorm applied exactly once to the encoder output.

    A forward hook on `post_layernorm` counts its calls and records its input/output during one normal forward
    pass; independently, the encoder output is recomputed (embeddings -> encoder, without the vision model's
    wrapper) and normalized once. `output_hidden_states` is deliberately not used: in recent transformers its
    last entry is collected by hooks and can already be the post-LN output, which would make the check vacuous."""
    vm = model.vision_model
    seen = []
    handle = vm.post_layernorm.register_forward_hook(lambda mod, inp, out: seen.append((inp[0], out)))
    try:
        h = model(pixel_values=px).last_hidden_state
    finally:
        handle.remove()
    if len(seen) != 1:
        raise RuntimeError(f"post_layernorm ran {len(seen)} times in one forward pass (expected exactly 1)")
    ln_in, ln_out = seen[0]
    enc = vm.encoder(inputs_embeds=vm.embeddings(px)).last_hidden_state
    rep = {"post_layernorm_calls_per_forward": len(seen),
           "max_abs_diff_last_hidden_vs_hooked_ln_output": float((h - ln_out).abs().max()),
           "max_abs_diff_hooked_ln_input_vs_recomputed_encoder": float((ln_in - enc).abs().max()),
           "max_abs_diff_last_hidden_vs_ln(recomputed_encoder)": float((h - vm.post_layernorm(enc)).abs().max()),
           # informative only for a learned (non-identity) LN affine, as in the real checkpoint
           "max_abs_diff_if_applied_twice": float((vm.post_layernorm(h) - h).abs().max())}
    bad = [k for k in ("max_abs_diff_last_hidden_vs_hooked_ln_output",
                       "max_abs_diff_hooked_ln_input_vs_recomputed_encoder",
                       "max_abs_diff_last_hidden_vs_ln(recomputed_encoder)") if rep[k] > 1e-3]
    if bad:
        raise RuntimeError(f"last_hidden_state is not post_layernorm(encoder output) applied once: {rep}")
    return rep


class FeatureStore:
    """Read-only access to the cached features by image key ("train2014/123"), memory-mapped from disk.

    visual_tokens = 49 averages each spatial 2 x 2 block of the 14 x 14 grid (row-major 7 x 7 result)."""

    def __init__(self, cfg: Dict[str, Any], visual_tokens: int):
        d = task_path(cfg["paths"]["feature_dir"])
        self.grid = cfg["models"]["grid"]
        self.visual_tokens = visual_tokens
        if visual_tokens not in (self.grid ** 2, (self.grid // 2) ** 2):
            raise ValueError(f"unsupported number of visual tokens {visual_tokens}")
        self.tensors: Dict[str, torch.Tensor] = {}
        self.index: Dict[str, tuple] = {}
        for split in ("train2014", "val2014"):
            obj = torch.load(os.path.join(d, f"{split}.pt"), map_location="cpu", mmap=True)
            self.tensors[split] = obj["feats"]
            for i, k in enumerate(obj["keys"]):
                self.index[k] = (split, i)

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def get(self, keys: List[str]) -> torch.Tensor:
        """(B, visual_tokens, 768) float32 on CPU."""
        rows = []
        for k in keys:
            split, i = self.index[k]
            rows.append(self.tensors[split][i])
        x = torch.stack(rows).float()
        return x if self.visual_tokens == self.grid ** 2 else pool_2x2(x, self.grid)


def pool_2x2(x: torch.Tensor, grid: int) -> torch.Tensor:
    """(B, grid*grid, C) -> (B, (grid/2)^2, C): mean of each spatial 2 x 2 block, row-major."""
    b, _, c = x.shape
    x = x.view(b, grid // 2, 2, grid // 2, 2, c).mean(dim=(2, 4))
    return x.reshape(b, (grid // 2) ** 2, c)


def check_pooling() -> Dict[str, Any]:
    """The 2 x 2 average pooling keeps spatial order: token (r, c) of the 7 x 7 grid is the mean of patches
    (2r..2r+1, 2c..2c+1) of the 14 x 14 grid."""
    g, c = 14, 3
    x = torch.arange(g * g * c, dtype=torch.float32).view(1, g * g, c)
    pooled = pool_2x2(x, g)
    ok = True
    for r in range(7):
        for cc in range(7):
            idx = [(2 * r + a) * g + (2 * cc + b) for a in range(2) for b in range(2)]
            ok &= bool(torch.allclose(pooled[0, r * 7 + cc], x[0, idx].mean(0)))
    return {"pooling_order_ok": ok}
