"""PA4 Task 1 - mechanism diagnostic (supplementary analysis, run AFTER the main run; it changes no condition).

Why are some quantization units (block 0 q/k/v, block 1 MLP, the last two MLPs) far more sensitive than the rest?
Two explanations make different predictions:
  (a) local difficulty - the weights themselves quantize badly (outliers inside groups, large weight error);
  (b) downstream amplification - an ordinary local error lands where the network magnifies it, e.g. on the
      "massive activation" that early MLPs write into the residual stream of the first token, which later layers
      use as an attention sink.
This script measures, on the unchanged FP16 model plus the AWQ scales saved by the main run:
  1. dense map: residual-stream RMS / max per block boundary, where the massive entries sit (token, channel),
     which sublayer writes them, attention mass every layer puts on token 0, peak MLP neurons;
  2. weight difficulty of every Q3 unit at 2/3/4 bits (weight error, activation-weighted error, group outliers);
  3. propagation traces: one unit at 2 bits -> own sublayer error, drift at every later boundary (with and without
     the massive entries), change of the massive entries, change of attention-to-token-0, final KL;
  3b. the same trace for whole blocks pruned with Wanda (units "l.p75", the P3 block probe);
  4. causal tests: (i) activation patching at the boundary after the unit - restore the massive entries / token 0 /
     everything except token 0 / random entries to their dense values and re-measure KL; (ii) weight protection -
     keep k input channels (or k MLP neurons) of the 2-bit unit in FP16: the hypothesised ones vs random ones, and
     the top channels again with the rest re-quantized without them (do the outliers coarsen the grid?).
Reads results/sensitivity/{awq_scales.npz, awq_search.json, dense_input_norms.npz, q3_ and p3_sensitivity.json} and
results/data_selection/c4_calibration_locked.jsonl.gz; writes results/diagnostics/mechanism_diagnostic.json.

  python scripts/diagnose_task1.py [--n-seq 32] [--chunk 4] [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pruning as P  # noqa: E402
from src import quant as Q  # noqa: E402
from src.data import lock_revisions  # noqa: E402
from src.model_utils import SHORT, get_layers, load_model, proj  # noqa: E402
from src.utils import RESULTS, load_config, load_json, log, save_json, set_seed  # noqa: E402

TRACE_UNITS = ["0.qkv", "0.gateup", "1.qkv", "1.gateup", "1.down", "2.gateup", "2.down", "4.gateup", "13.qkv",
               "13.gateup", "26.gateup", "27.gateup", "27.down",
               "0.p75", "13.p75", "27.p75"]          # l.pN = whole block l pruned with Wanda at N% (P3 block probe)
PROTECT_UNITS = ["0.qkv", "1.gateup", "1.down", "2.down", "13.gateup", "26.gateup", "27.gateup", "27.down"]
INPUT_KEY = {"qkv": "q", "o": "o", "gateup": "gate", "down": "down"}     # dense_input_norms.npz key per group
PROJ_INPUT = {"q": "q", "k": "q", "v": "q", "o": "o", "gate": "gate", "up": "gate", "down": "down"}
MASSIVE_RATIO = 50.0          # an entry is "massive" if |h| > 50 x RMS of its boundary (dense model)

# Attention mass on token 0. HF's eager attention forms q.k^T in FP16 before scaling, which overflows on Qwen2.5
# (inf -> NaN). So the model keeps SDPA, exactly as in the main run, and this wrapper around the SDPA kernel also
# computes softmax(q k^T * scale) in FP32 and keeps only the mass on key 0 (queries 1..T-1, per head), when asked.
_REC = {"on": False, "mass": {}}


def _install_sdpa_recorder():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def sdpa_with_mass(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        out = sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                     is_causal=is_causal, **kw)
        if _REC["on"]:
            with torch.no_grad():
                B, H, T, d = query.shape
                rep = H // key.shape[1]
                k = key.float().repeat_interleave(rep, dim=1)
                sc = (query.float() @ k.transpose(-1, -2)) * (scaling if scaling is not None else d ** -0.5)
                causal = torch.ones(T, T, dtype=torch.bool, device=sc.device).triu(1)   # inputs are unpadded
                p = torch.softmax(sc.masked_fill(causal, float("-inf")), dim=-1)
                _REC["mass"][module.layer_idx] = p[:, :, 1:, 0].mean(dim=(0, 2))
                del sc, p, k
        return out

    ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa_with_mass)


class Diag:
    def __init__(self, a):
        self.a = a
        self.res = Path(a.results) if a.results else RESULTS
        cfg = load_config(a.config)
        cfg = copy.deepcopy(cfg)
        cfg["model"]["attn_implementation"] = "sdpa"            # as the main run; token-0 mass via the wrapper
        _install_sdpa_recorder()
        if a.model:
            cfg["model"]["name"] = a.model
        rev = a.revision or cfg["model"]["revision"]
        if rev is None and not Path(cfg["model"]["name"]).exists():
            rev = lock_revisions(cfg)["revisions"]["model_base"]
        self.cfg, self.rev = cfg, rev
        set_seed(cfg["seed"])
        log(f"loading {cfg['model']['name']} @ {rev} (SDPA + FP32 token-0 attention mass)")
        self.model, _ = load_model(cfg, a.device, revision=rev)
        self.dev = a.device
        self.layers = get_layers(self.model)
        self.L = len(self.layers)
        c = self.model.config
        self.nh, self.nkv = c.num_attention_heads, c.num_key_value_heads
        self.hd = getattr(c, "head_dim", None) or c.hidden_size // self.nh
        with gzip.open(self.res / "data_selection" / "c4_calibration_locked.jsonl.gz", "rt") as f:
            rows = [json.loads(x) for _, x in zip(range(a.n_seq), f)]
        self.ids = torch.tensor([r["token_ids"] for r in rows], dtype=torch.long)
        self.N, self.T = self.ids.shape
        sd = self.res / "sensitivity"
        self.awq = Q.table_from_npz(sd / "awq_scales.npz", load_json(sd / "awq_search.json")["table"])
        self.norms = dict(np.load(sd / "dense_input_norms.npz"))
        q3 = sd / "q3_sensitivity.json"
        self.saved_probe = load_json(q3)["probe_kl"] if q3.exists() else {}
        p3 = sd / "p3_sensitivity.json"
        self.saved_p3 = load_json(p3)["probe_kl"] if p3.exists() else {}
        self.hooks = []
        self.state = {}

    # ----------------------------------------------------------------------------------------- weights
    def W(self, l, k):
        return proj(self.layers[l], k).weight

    def fq_group(self, l, g, b):
        s = self.awq[l][g][b]["scales"].to(self.dev).float()
        if g == "o":
            s = Q._expand_o(s, self.nh, self.nkv, self.hd)
        return {k: Q.fq_scaled(self.W(l, k).data, s, b, self.cfg) for k in Q.GROUPS[g]}

    def apply(self, unit, b, keep_cols=None, keep_rows=None, requant=False):
        """Compress one unit in place and return the saved originals for restore().
        "l.g": AWQ group g of block l fake-quantized at b bits (the main run's Q3 probe); optionally keep some input
        columns / output rows in FP16. requant=True also removes the kept columns from the quantizer's group ranges
        (zeroed before quantization), so the other weights get the grid they would have without those outliers.
        "l.pN": all seven projections of block l pruned with Wanda at N% on dense input norms (the P3 block probe)."""
        l, g = int(unit.split(".")[0]), unit.split(".")[1]
        saved = {}
        with torch.no_grad():
            if g.startswith("p"):
                sp = int(g[1:]) / 100
                for k in SHORT:
                    w = self.W(l, k)
                    saved[(l, k)] = w.data.clone()
                    nrm = torch.from_numpy(self.norms[f"{l}.{PROJ_INPUT[k]}"]).to(self.dev).float()
                    w.data[P.wanda_mask_dense(w.data, nrm ** 2, sp)] = 0
                return saved
            if requant and keep_cols is not None:
                s = self.awq[l][g][b]["scales"].to(self.dev).float()
                if g == "o":
                    s = Q._expand_o(s, self.nh, self.nkv, self.hd)
                fq = {}
                for k in Q.GROUPS[g]:
                    W0 = self.W(l, k).data.clone()
                    W0[:, keep_cols[k]] = 0
                    fq[k] = Q.fq_scaled(W0, s, b, self.cfg)
            else:
                fq = self.fq_group(l, g, b)
            for k, Wq in fq.items():
                w = self.W(l, k)
                saved[(l, k)] = w.data.clone()
                if keep_cols is not None and k in keep_cols:
                    Wq[:, keep_cols[k]] = w.data[:, keep_cols[k]]
                if keep_rows is not None and k in keep_rows:
                    Wq[keep_rows[k], :] = w.data[keep_rows[k], :]
                w.data.copy_(Wq)
        return saved

    def restore(self, saved):
        with torch.no_grad():
            for (l, k), w in saved.items():
                self.W(l, k).data.copy_(w)

    # ------------------------------------------------------------------------------------------- hooks
    def clear(self):
        _REC["on"] = False
        _REC["mass"].clear()
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _install(self, record_h, record_attn, record_sub, patch):
        st = self.state
        for l, blk in enumerate(self.layers):
            if record_h:
                def pre(mod, args, kwargs, l=l):
                    h = args[0] if args else kwargs["hidden_states"]
                    st["h_in"][l] = h
                self.hooks.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
            if record_attn:
                _REC["on"] = True
                def ah(mod, inp, out, l=l):
                    st["attn0"][l] = _REC["mass"].pop(mod.layer_idx)     # (H,): mass on key 0, queries 1..T-1
                self.hooks.append(blk.self_attn.register_forward_hook(ah))
            if record_sub is not None and l in record_sub:
                def sa(mod, inp, out, l=l):
                    st["sub"][(l, "attn")] = out[0]
                def sm(mod, inp, out, l=l):
                    st["sub"][(l, "mlp")] = out
                def dn(mod, args, l=l):
                    st["down_in"][l] = args[0]
                self.hooks += [blk.self_attn.register_forward_hook(sa), blk.mlp.register_forward_hook(sm),
                               blk.mlp.down_proj.register_forward_pre_hook(dn)]
            if patch is not None and patch["boundary"] == l + 1:
                def ph(mod, inp, out):
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    i0 = st["chunk_start"]
                    ref = self.dense_h[patch["boundary"]][i0:i0 + h.shape[0]].to(h.device, non_blocking=True)
                    m = patch["mask"][i0:i0 + h.shape[0]].to(h.device)
                    h[m] = ref[m]
                    return out
                self.hooks.append(blk.register_forward_hook(ph))

    @torch.no_grad()
    def forward(self, record_h=False, record_attn=False, record_sub=None, patch=None, keep_final=False,
                on_chunk=None):
        """Run all N sequences in chunks; returns KL / top-1 vs the dense final (if stored) and per-chunk callbacks."""
        self._install(record_h, record_attn, record_sub, patch)
        st = self.state
        kl_sum = kl0 = top1 = 0.0
        finals = []
        try:
            for i0 in range(0, self.N, self.a.chunk):
                ids = self.ids[i0:i0 + self.a.chunk].to(self.dev)
                st.update(chunk_start=i0, h_in={}, attn0={}, sub={}, down_in={})
                hN = self.model.model(input_ids=ids, use_cache=False).last_hidden_state   # final norm applied
                if not torch.isfinite(hN).all():
                    raise FloatingPointError("non-finite hidden states")
                if keep_final:
                    finals.append(hN.cpu())
                if getattr(self, "dense_final", None) is not None and not keep_final:
                    ref = self.dense_final[i0:i0 + ids.shape[0]].to(self.dev)
                    for j in range(ids.shape[0]):
                        lr = torch.log_softmax(self.model.lm_head(ref[j]).float(), -1)
                        lc = torch.log_softmax(self.model.lm_head(hN[j]).float(), -1)
                        kl = (lr.exp() * (lr - lc)).sum(-1)[:-1]           # positions 0..T-2 predict a token
                        kl_sum += float(kl.sum())
                        kl0 += float(kl[0])
                        top1 += float((lr.argmax(-1) == lc.argmax(-1))[:-1].float().sum())
                        del lr, lc
                if on_chunk is not None:
                    on_chunk(i0, ids.shape[0], st)
        finally:
            self.clear()
        n = self.N * (self.T - 1)
        out = {"kl": kl_sum / n, "kl_pos0": kl0 / self.N, "kl_rest": (kl_sum - kl0) / (n - self.N), "top1_agree": top1 / n}
        if keep_final:
            out["finals"] = torch.cat(finals)
        return out

    # -------------------------------------------------------------------------------------- 1. dense map
    def dense(self):
        L, T = self.L, self.T
        self.dense_h = [None] * L
        attn0 = []
        peak = {}
        hidden_store = {l: [] for l in range(L)}
        sub_store = {}

        def cb(i0, n, st):
            for l in range(L):
                hidden_store[l].append(st["h_in"][l].cpu())
            for key, v in st["sub"].items():
                sub_store.setdefault(key, []).append(v.cpu())
            attn0.append(torch.stack([st["attn0"][l] for l in range(L)]).cpu())
            for l, x in st["down_in"].items():
                a = x.float().abs()                                            # (B, T, d_ff)
                pk = a.amax(dim=(0, 1)).cpu()
                p0 = a[:, 0].amax(0).cpu()
                peak[l] = (torch.maximum(peak[l][0], pk), torch.maximum(peak[l][1], p0)) if l in peak else (pk, p0)

        blocks = sorted({int(u.split(".")[0]) for u in TRACE_UNITS + PROTECT_UNITS})
        r = self.forward(record_h=True, record_attn=True, record_sub=blocks, keep_final=True, on_chunk=cb)
        self.dense_final = r["finals"]
        self.dense_h = [torch.cat(hidden_store[l]) for l in range(L)]
        self.dense_sub = {key: torch.cat(v) for key, v in sub_store.items()}
        try:
            self.dense_h = [h.pin_memory() for h in self.dense_h]
        except Exception:
            pass
        self.attn0_dense = torch.stack(attn0).mean(0)                          # (L, H)
        self.peak = peak
        # residual statistics and massive entries (boundary l = input of block l)
        bnd, self.massive = [], {}
        for l in range(L):
            h = self.dense_h[l].float()
            rms = float(h.pow(2).mean().sqrt())
            mx = float(h.abs().max())
            m = h.abs() > self.a.massive_ratio * rms
            self.massive[l] = m
            idx = m.nonzero()
            pos = idx[:, 1].tolist() if len(idx) else []
            ch = idx[:, 2].tolist() if len(idx) else []
            ch_count = {}
            for c in ch:
                ch_count[c] = ch_count.get(c, 0) + 1
            tok0 = h[:, 0].norm(dim=-1).mean() / h[:, 1:].norm(dim=-1).mean()
            bnd.append({"boundary": l, "rms": rms, "max": mx, "max_over_rms": mx / max(rms, 1e-30),
                        "n_massive": int(m.sum()), "massive_frac_at_pos0": float(np.mean([p == 0 for p in pos])) if pos else None,
                        "massive_channels": dict(sorted(ch_count.items(), key=lambda kv: -kv[1])[:8]),
                        "massive_share_of_sq_norm": float(h[m].pow(2).sum() / h.pow(2).sum()) if len(idx) else 0.0,
                        "tok0_norm_over_rest": float(tok0)})
        self.boundaries = bnd
        log("dense map: " + ", ".join(f"b{x['boundary']} max/rms {x['max_over_rms']:.0f} n_massive {x['n_massive']}"
                                     for x in bnd[:5]))
        # who writes the massive entries: MLP vs attention contribution of block l at the massive set of boundary l+1
        writers = []
        mlp_at = {l: [] for l in range(L - 1)}

        def cb2(i0, n, st):
            for l in range(L - 1):
                m = self.massive[l + 1][i0:i0 + n]
                if m.any():
                    mlp_at[l].append(st["sub"][(l, "mlp")].float().cpu()[m])
        sanity = self.forward(record_sub=list(range(L - 1)), on_chunk=cb2)     # dense vs dense: KL must be 0
        for l in range(L - 1):
            m = self.massive[l + 1]
            if not m.any():
                continue
            h_next = self.dense_h[l + 1].float()[m]
            h_prev = self.dense_h[l].float()[m]
            mlp = torch.cat(mlp_at[l])
            total = h_next - h_prev
            attn = total - mlp
            writers.append({"block": l, "n_entries": int(m.sum()),
                            "mean_abs_h_out": float(h_next.abs().mean()), "mean_abs_h_in": float(h_prev.abs().mean()),
                            "mean_abs_mlp_write": float(mlp.abs().mean()), "mean_abs_attn_write": float(attn.abs().mean()),
                            "mlp_share_of_increment": float((mlp * total).sum() / total.pow(2).sum().clamp(min=1e-30))})
        self.writers = writers
        return {"boundaries": bnd, "writers": writers, "dense_vs_dense_kl": sanity["kl"],
                "attn_mass_on_token0": {"per_layer_mean": self.attn0_dense.mean(1).tolist(),
                                        "per_layer_head": self.attn0_dense.tolist()},
                "mlp_peak_neurons": {str(l): {"top_by_peak": torch.topk(pk, 16).indices.tolist(),
                                              "peak_over_median": float(pk.max() / pk.median().clamp(min=1e-30)),
                                              "top_by_pos0_peak": torch.topk(p0, 16).indices.tolist(),
                                              "pos0_peak_over_median": float(p0.max() / p0.median().clamp(min=1e-30))}
                                     for l, (pk, p0) in self.peak.items()}}

    # ------------------------------------------------------------------------------- 2. weight difficulty
    @torch.no_grad()
    def weight_difficulty(self):
        out = {}
        G = self.cfg["quant"]["group_size"]
        for l in range(self.L):
            for g in Q.GROUP_NAMES:
                nrm = torch.from_numpy(self.norms[f"{l}.{INPUT_KEY[g]}"]).to(self.dev).float()
                s = self.awq[l][g][3]["scales"].to(self.dev).float()
                s = Q._expand_o(s, self.nh, self.nkv, self.hd) if g == "o" else s
                Ws = torch.cat([(self.W(l, k).data.float() * s[None]) for k in Q.GROUPS[g]])
                x = Ws.reshape(Ws.shape[0], -1, G)
                rng = (x.amax(-1) - x.amin(-1)) / x.std(-1).clamp(min=1e-12)          # group range in std units
                rec = {"group_range_over_std_mean": float(rng.mean()), "group_range_over_std_p99": float(rng.quantile(0.99)),
                       "input_norm_max_over_median": float(nrm.max() / nrm.median())}
                topc = torch.topk(nrm, 8).indices
                for b in (2, 3, 4):
                    num = den = wnum = wden = top = 0.0
                    for k, Wq in self.fq_group(l, g, b).items():
                        W = self.W(l, k).data.float()
                        d2 = (Wq.float() - W) ** 2
                        num += float(d2.sum())
                        den += float((W ** 2).sum())
                        e = d2.sum(0) * nrm ** 2                                       # per input channel
                        wnum += float(e.sum())
                        wden += float(((W ** 2).sum(0) * nrm ** 2).sum())
                        top += float(e[topc].sum())
                    rec[str(b)] = {"weight_rel_err": num / den, "act_weighted_rel_err": wnum / wden,
                                   "share_of_act_err_on_top8_channels": top / max(wnum, 1e-30)}
                out[f"{l}.{g}"] = rec
        return out

    # ------------------------------------------------------------------------------- 3. propagation trace
    def trace(self, unit, b=2):
        l = int(unit.split(".")[0])
        g = unit.split(".")[1]
        sub = None if g.startswith("p") else ("attn" if g in ("qkv", "o") else "mlp")
        L = self.L
        acc = {"drift_num": np.zeros(L), "drift_den": np.zeros(L), "drift_nm_num": np.zeros(L), "drift_nm_den": np.zeros(L),
               "mass_num": np.zeros(L), "mass_den": np.zeros(L), "attn0": [], "sub_err": 0.0, "sub_ref": 0.0,
               "sub_err0": 0.0, "sub_ref0": 0.0}

        def cb(i0, n, st):
            for j in range(L):
                hc = st["h_in"][j].float()
                hd = self.dense_h[j][i0:i0 + n].to(self.dev).float()
                d = (hc - hd) ** 2
                m = self.massive[j][i0:i0 + n].to(self.dev)
                acc["drift_num"][j] += float(d.sum())
                acc["drift_den"][j] += float((hd ** 2).sum())
                acc["drift_nm_num"][j] += float(d[~m].sum())
                acc["drift_nm_den"][j] += float((hd[~m] ** 2).sum())
                if m.any():
                    acc["mass_num"][j] += float((hc[m] - hd[m]).abs().sum())
                    acc["mass_den"][j] += float(hd[m].abs().sum())
            acc["attn0"].append(torch.stack([st["attn0"][j] for j in range(L)]).cpu())
            if sub is None:
                return
            oc = st["sub"][(l, sub)].float()
            od = self.dense_sub[(l, sub)][i0:i0 + n].to(self.dev).float()
            e = (oc - od) ** 2
            acc["sub_err"] += float(e.sum())
            acc["sub_ref"] += float((od ** 2).sum())
            acc["sub_err0"] += float(e[:, 0].sum())
            acc["sub_ref0"] += float((od[:, 0] ** 2).sum())
        saved = self.apply(unit, b)
        try:
            r = self.forward(record_h=True, record_attn=True, record_sub=[l] if sub else None, on_chunk=cb)
        finally:
            self.restore(saved)
        a0 = torch.stack(acc["attn0"]).mean(0)
        rest_err = acc["sub_err"] - acc["sub_err0"]
        own = sub is not None
        if g.startswith("p"):
            pp = self.saved_p3.get(str(int(g[1:]) / 100))
            saved_kl = pp[l]["kl"] if pp else None
        else:
            saved_kl = self.saved_probe.get(unit, {}).get(str(b), {}).get("kl")
        return {"unit": unit, "bits": None if g.startswith("p") else b,
                **{k: r[k] for k in ("kl", "kl_pos0", "kl_rest", "top1_agree")},
                "saved_probe_kl_128seq_sdpa": saved_kl,
                "own_sublayer": sub, "own_rel_err": acc["sub_err"] / max(acc["sub_ref"], 1e-30) if own else None,
                "own_err_share_at_pos0": acc["sub_err0"] / max(acc["sub_err"], 1e-30) if own else None,
                "own_rel_err_pos0": acc["sub_err0"] / max(acc["sub_ref0"], 1e-30) if own else None,
                "own_rel_err_rest": rest_err / max(acc["sub_ref"] - acc["sub_ref0"], 1e-30) if own else None,
                "drift": (acc["drift_num"] / np.maximum(acc["drift_den"], 1e-30)).tolist(),
                "drift_excl_massive": (acc["drift_nm_num"] / np.maximum(acc["drift_nm_den"], 1e-30)).tolist(),
                "massive_rel_change": [float(x / y) if y > 0 else None for x, y in zip(acc["mass_num"], acc["mass_den"])],
                "attn0_change": (a0 - self.attn0_dense).mean(1).tolist()}

    # ---------------------------------------------------------------------------------- 4. causal tests
    def patch_tests(self, unit, b=2):
        l = int(unit.split(".")[0])
        if l + 1 >= self.L:
            return None
        bd = l + 1
        g = torch.Generator().manual_seed(self.cfg["seed"])
        shape = self.dense_h[bd].shape
        masks = {}
        pos0 = torch.zeros(shape, dtype=torch.bool)
        pos0[:, 0] = True
        masks["token0"] = pos0
        masks["all_but_token0"] = ~pos0
        mb = next((j for j in range(bd, self.L) if self.massive[j].any()), None)
        out = {"boundary": bd}
        saved = self.apply(unit, b)
        try:
            for name, m in masks.items():
                out[name] = self.forward(patch={"boundary": bd, "mask": m})
            if mb is not None and mb < self.L:
                m = self.massive[mb]
                k = int(m.sum())
                rnd = torch.zeros(m.numel(), dtype=torch.bool)
                cand = (~m).flatten().nonzero().squeeze(1)
                rnd[cand[torch.randperm(cand.numel(), generator=g)[:k]]] = True
                out["massive_boundary"] = mb
                out["massive_entries"] = self.forward(patch={"boundary": mb, "mask": m})
                out["random_entries_same_count"] = self.forward(patch={"boundary": mb, "mask": rnd.reshape(m.shape)})
                out["n_patched_entries"] = k
        finally:
            self.restore(saved)
        return out

    def protect_tests(self, unit, b=2, k=8, n_rand=3):
        l, gname = int(unit.split(".")[0]), unit.split(".")[1]
        members = Q.GROUPS[gname]
        nrm = torch.from_numpy(self.norms[f"{l}.{INPUT_KEY[gname]}"])
        d_in = nrm.numel()
        g = torch.Generator().manual_seed(self.cfg["seed"] + l)
        res = {"k": k}

        def run(cols=None, rows=None):
            saved = self.apply(unit, b, keep_cols=cols, keep_rows=rows)
            try:
                return self.forward()
            finally:
                self.restore(saved)
        res["none"] = run()
        top = torch.topk(nrm, k).indices.to(self.dev)
        res["top_input_channels"] = run(cols={m: top for m in members})
        res["top_input_channels_ids"] = top.tolist()
        saved = self.apply(unit, b, keep_cols={m: top for m in members}, requant=True)
        try:
            res["top_input_channels_requant"] = self.forward()
        finally:
            self.restore(saved)
        res["random_input_channels"] = [run(cols={m: torch.randperm(d_in, generator=g)[:k].to(self.dev) for m in members})
                                        for _ in range(n_rand)]
        if gname in ("gateup", "down") and l in self.peak:
            pk = self.peak[l][1]                                               # peak |activation| at token 0
            neu = torch.topk(pk, k).indices.to(self.dev)
            res["top_token0_neurons_ids"] = neu.tolist()
            if gname == "gateup":
                res["top_token0_neurons"] = run(rows={"gate": neu, "up": neu})
                d_ff = pk.numel()
                res["random_neurons"] = []
                for _ in range(n_rand):
                    rn = torch.randperm(d_ff, generator=g)[:k].to(self.dev)
                    res["random_neurons"].append(run(rows={"gate": rn, "up": rn}))
            else:
                res["top_token0_neurons"] = run(cols={"down": neu})
        return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=None, help="override model path (smoke test)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--massive-ratio", type=float, default=MASSIVE_RATIO)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--trace-units", default=",".join(TRACE_UNITS))
    ap.add_argument("--protect-units", default=",".join(PROTECT_UNITS))
    a = ap.parse_args()
    d = Diag(a)
    out_dir = d.res / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mechanism_diagnostic.json"
    t, rec = {}, {"meta": {"n_seq": d.N, "seq_len": d.T, "revision": d.rev,
                           "attention": "sdpa (as the main run); token-0 attention mass recomputed in FP32", "k": a.k,
                           "n_random": a.n_random, "massive_ratio": a.massive_ratio, "bits": 2, "prune_units": "Wanda, dense input norms",
                           "note": "supplementary diagnostic, run after all conditions; selects nothing"}}
    t0 = time.perf_counter()
    rec["dense"] = d.dense()
    t["dense_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rec["weight_difficulty"] = d.weight_difficulty()
    t["weights_s"] = time.perf_counter() - t0
    save_json(path, rec)
    tu = [u for u in a.trace_units.split(",") if u and int(u.split(".")[0]) < d.L]
    pu = [u for u in a.protect_units.split(",") if u and int(u.split(".")[0]) < d.L]
    rec["trace"], rec["patch"], rec["protect"] = {}, {}, {}
    t0 = time.perf_counter()
    for u in tu:
        rec["trace"][u] = d.trace(u)
        rec["patch"][u] = d.patch_tests(u)
        tr = rec["trace"][u]
        log(f"{u}: KL {tr['kl']:.4g} (saved probe {tr['saved_probe_kl_128seq_sdpa']}), own rel err {tr['own_rel_err']}, "
            f"share at token0 {tr['own_err_share_at_pos0']}")
        save_json(path, rec)
    t["trace_patch_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for u in pu:
        rec["protect"][u] = d.protect_tests(u, k=a.k, n_rand=a.n_random)
        p = rec["protect"][u]
        log(f"{u}: protect none {p['none']['kl']:.4g} | top channels {p['top_input_channels']['kl']:.4g} | random "
            f"{np.mean([x['kl'] for x in p['random_input_channels']]):.4g} | top channels re-quantized "
            f"{p['top_input_channels_requant']['kl']:.4g}"
            + (f" | token0 neurons {p['top_token0_neurons']['kl']:.4g}" if "top_token0_neurons" in p else ""))
        save_json(path, rec)
    t["protect_s"] = time.perf_counter() - t0
    if torch.cuda.is_available():
        rec["meta"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    rec["meta"]["timings_s"] = t
    save_json(path, rec)
    log(f"wrote {path}  ({sum(t.values()) / 60:.1f} min)")


if __name__ == "__main__":
    main()
