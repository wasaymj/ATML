"""Calibration-set passes that run the model one block at a time on the 128 x 512 C4 calibration tokens.

* measure(): after a condition is built, runs the dense (pristine) and the compressed model side by side and
  records, for every eligible projection, the layer-output error on calibration activations, its diagonal
  prediction, the relative drift of every block's output, and the final-logit KL / NLL / top-1 agreement.
* dense_pass(): statistics of the unchanged model (input norms, AWQ |x| statistic, ShortGPT Block Influence,
  residual-stream magnitude), plus the dense stream at every block boundary for probes.
* probe(): isolated modifications (one block / one AWQ group / one projection type) evaluated by the KL of
  the final next-token distribution against the dense model.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .model_utils import (SHORT, BlockSwap, Stream, advance, get_layers, initial_stream, proj, run_block,
                          final_hidden_to_logprobs)
from .pruning import InputStats, act_norm_summary, score_summary, wanda_score
from .utils import log, sync


# ------------------------------------------------------------------------------------- logits comparison
@torch.no_grad()
def compare_final(model, h_ref: torch.Tensor, h_cmp: torch.Tensor | None, ids: torch.Tensor, cfg,
                  stride: int = 1) -> dict:
    """KL(p_dense || p_cmp), NLL of both and top-1 agreement at positions t = 0, stride, ... < L-1
    (position t predicts token t+1), pooled over all sequences. h_cmp=None -> dense only."""
    T = cfg["calib"]["kl_token_chunk"]
    dev = h_ref.device
    L = h_ref.shape[1]
    pos = torch.arange(0, L - 1, stride, device=dev)
    hr = h_ref[:, pos].reshape(-1, h_ref.shape[-1])
    hc = h_cmp[:, pos].reshape(-1, h_cmp.shape[-1]) if h_cmp is not None else None
    tg = ids.to(dev)[:, pos + 1].reshape(-1)
    acc = {"kl": 0.0, "nll_ref": 0.0, "nll_cmp": 0.0, "top1_agree": 0.0}
    for c in range(0, hr.shape[0], T):
        lr = final_hidden_to_logprobs(model, hr[c:c + T])
        acc["nll_ref"] -= float(lr.gather(1, tg[c:c + T, None]).sum())
        if hc is not None:
            lc = final_hidden_to_logprobs(model, hc[c:c + T])
            acc["kl"] += float((lr.exp() * (lr - lc)).sum())
            acc["nll_cmp"] -= float(lc.gather(1, tg[c:c + T, None]).sum())
            acc["top1_agree"] += float((lr.argmax(-1) == lc.argmax(-1)).sum())
            del lc
        del lr
    n = hr.shape[0]
    out = {k: v / n for k, v in acc.items()}
    out["n_positions"] = n
    return out


def _final(model, s: Stream) -> torch.Tensor:
    return s.h      # the last decoder block's output; the final norm is applied in compare_final


# ------------------------------------------------------------------------------------- measurement pass
class _ErrHook:
    """Layer-output error of one projection with the compressed model's own inputs, in original coordinates:
       err = || r * y_comp - (W x + b) ||^2 with x = c * x_comp (AWQ column scale c, row scale r; 1 otherwise)."""

    def __init__(self, mod, W, b, c, r):
        self.W, self.b, self.c, self.r = W, b, c, r
        self.err = 0.0
        self.ref = 0.0
        self.sumsq = None
        self.h = mod.register_forward_hook(self)

    def __call__(self, mod, inp, out):
        x = inp[0].reshape(-1, inp[0].shape[-1])
        y = out.reshape(-1, out.shape[-1]).float()
        if self.c is not None:
            x = (x.float() * self.c).to(x.dtype)
        if self.r is not None:
            y = y * self.r
        yr = F.linear(x, self.W, self.b).float()
        self.err += float(((y - yr) ** 2).sum())
        self.ref += float((yr ** 2).sum())
        xs = (x.float() ** 2).sum(0)
        self.sumsq = xs if self.sumsq is None else self.sumsq + xs


@torch.no_grad()
def measure(model, pristine, calib_ids: torch.Tensor, cfg, transforms: dict | None = None) -> dict:
    """Side-by-side dense / compressed pass on the calibration set. transforms[(l, short)] = (col, row) scales
    of AWQ conditions so errors are expressed in the original function's coordinates."""
    t0 = time.perf_counter()
    dev = next(model.parameters()).device
    s_c = initial_stream(model, calib_ids, cfg["calib"]["chunk_seqs"])
    s_d = Stream(s_c.h.clone(), s_c.kw, s_c.chunk)
    per_proj, drift = {}, []
    for l, block in enumerate(get_layers(model)):
        with BlockSwap(model, pristine, l):
            s_d_next = advance(block, s_d)
        hooks = {}
        for s in SHORT:
            W = pristine.weight(l, s, dev)
            b = pristine.bias(l, s, dev)
            c, r = (transforms or {}).get((l, s), (None, None))
            hooks[s] = _ErrHook(proj(block, s), W, b, c, r)
        s_c_next = advance(block, s_c)
        for s, hk in hooks.items():
            hk.h.remove()
            W_eff = proj(block, s).weight.data.float()
            c, r = (transforms or {}).get((l, s), (None, None))
            if r is not None:
                W_eff = W_eff * r[:, None]
            if c is not None:
                W_eff = W_eff / c[None, :]
            dW2 = (W_eff - hk.W.float()) ** 2
            pred = float((dW2 @ hk.sumsq).sum())
            per_proj[f"{l}.{s}"] = {"rel_err": hk.err / max(hk.ref, 1e-30), "err": hk.err, "ref": hk.ref,
                                    "diag_pred_rel_err": pred / max(hk.ref, 1e-30)}
            del dW2, W_eff
        d = (s_c_next.h.float() - s_d_next.h.float()).pow(2).sum() / s_d_next.h.float().pow(2).sum()
        cos = F.cosine_similarity(s_c_next.h.float(), s_d_next.h.float(), dim=-1).mean()
        drift.append({"block": l, "rel_mse": float(d), "cosine": float(cos)})
        s_c, s_d = s_c_next, s_d_next
        del hooks
    fin = compare_final(model, s_d.h, s_c.h, calib_ids, cfg, stride=1)
    sync()
    return {"per_proj": per_proj, "drift": drift, "final": fin, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------- dense pass
@torch.no_grad()
def dense_pass(model, calib_ids: torch.Tensor, cfg, keep_streams=False) -> dict:
    """Statistics of the unchanged model on the calibration stream (dense inputs at every block)."""
    s = initial_stream(model, calib_ids, cfg["calib"]["chunk_seqs"])
    out = {"sumsq": {}, "absmean": {}, "n_tokens": None, "bi": [], "resid_rms": [], "resid_max": [],
           "act_norm": {}, "score": {}}
    streams = [s.h.cpu()] if keep_streams else None
    for l, block in enumerate(get_layers(model)):
        st = InputStats(block, keep_mean_abs=True)
        s_next = advance(block, s)
        st.remove()
        for k in SHORT:
            out["sumsq"][(l, k)] = st.sumsq[k]
            out["absmean"][(l, k)] = st.absum[k] / st.n[k]
            W = proj(block, k).weight.data
            sc = wanda_score(W, st.sumsq[k])
            out["score"][f"{l}.{k}"] = score_summary(sc, cfg, cfg["seed"] + 7 * l + SHORT.index(k), W)
            out["act_norm"][f"{l}.{k}"] = act_norm_summary(st.sumsq[k])
            del sc
        out["n_tokens"] = st.n["q"]
        cos = F.cosine_similarity(s.h.float(), s_next.h.float(), dim=-1)
        out["bi"].append(float(1 - cos.mean()))                         # ShortGPT Block Influence
        out["resid_rms"].append(float(s.h.float().pow(2).mean().sqrt()))
        out["resid_max"].append(float(s.h.float().abs().max()))
        s = s_next
        if keep_streams:
            streams.append(s.h.cpu())
    out["final_h"] = s.h
    out["kw"] = s.kw
    out["streams"] = streams
    return out


def owl_lod(model, sumsq: dict, M: float) -> list[float]:
    """OWL layerwise outlier distribution per block: share of Wanda scores |W_ij| ||X_j|| above M times the
    block mean (all seven projections concatenated, as in the OWL code), on dense calibration inputs."""
    lod = []
    for l, block in enumerate(get_layers(model)):
        tot = 0.0
        n = 0
        scores = []
        for k in SHORT:
            sc = wanda_score(proj(block, k).weight.data, sumsq[(l, k)])
            tot += float(sc.double().sum())
            n += sc.numel()
            scores.append(sc)
        mean = tot / n
        cnt = sum(float((sc > M * mean).sum()) for sc in scores)
        lod.append(cnt / n)
        del scores
    return lod


# ----------------------------------------------------------------------------------------------- probes
class Prober:
    """Isolated sensitivity probes. Keeps the dense stream at the current block boundary and the dense final
    hidden states; each probe modifies some weights, runs the remaining blocks for all 128 calibration
    sequences (chunked), and returns KL(dense || probe) of the next-token distribution on every `stride`-th
    position. The caller restores the weights."""

    def __init__(self, model, calib_ids, cfg, dense_final: torch.Tensor):
        self.model, self.ids, self.cfg = model, calib_ids, cfg
        self.stride = cfg["probes"]["kl_stride"]
        self.s0 = initial_stream(model, calib_ids, cfg["calib"]["chunk_seqs"])
        self.dense_final = dense_final
        self.cur = self.s0
        self.cur_l = 0
        self.n_probes = 0
        self.seconds = 0.0

    def advance_to(self, l: int):
        if l < self.cur_l:
            self.cur, self.cur_l = self.s0, 0
        layers = get_layers(self.model)
        while self.cur_l < l:
            self.cur = advance(layers[self.cur_l], self.cur)
            self.cur_l += 1

    @torch.no_grad()
    def kl_from(self, l: int) -> dict:
        """Run blocks l..end from the dense boundary l with the CURRENT (modified) weights."""
        t0 = time.perf_counter()
        self.advance_to(l)
        layers = get_layers(self.model)
        out = torch.empty_like(self.cur.h)
        for i, h in self.cur.chunks():
            x = h
            for j in range(l, len(layers)):
                x = run_block(layers[j], x, self.cur.kw)
            out[i:i + x.shape[0]] = x
        res = compare_final(self.model, self.dense_final, out, self.ids, self.cfg, stride=self.stride)
        del out
        sync()
        self.n_probes += 1
        self.seconds += time.perf_counter() - t0
        return res
