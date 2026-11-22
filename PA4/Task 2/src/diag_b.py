"""Part B mechanism diagnostic (supplementary, run AFTER the main run; it changes no main-run condition).

The main run shows three things that need a cause, not a guess:
  F1  models that fail settle at ~1/K accuracy (linear 2 of 3 seeds at K=16, window trained on mixed, dilated);
  F2  a model trained with K_train pairs fails on pairs beyond K_train (dense / linear / window alike);
  F3  the delta rule learns K=16 with every seed, ELU+1 linear attention with one of three.
Measurements (all on the frozen test set; checkpoints from the main run):
  taxonomy    what a wrong answer is: another value of the same sequence, or a value not in the sequence;
              accuracy by the slot (pair index) of the queried pair                       [saved predictions only]
  dynamics    loss plateau vs ln K_train and the update at which a run leaves it          [saved training curves]
  relocation  causal test of F2: keep K = 4 pairs (constant load) and move the whole pair block to slots
              s..s+3; if accuracy follows the absolute slot, the limit is positional, not interference
  circuit     attention scores per head: value -> own key (layer 1, "binding") and query -> matching value
              (layer 2, "retrieval"), mass on the other values and on filler tokens (softmax and ELU+1 maps)
  ablation    causal test of the circuit: zero one head (or one layer's heads) at test time
  gates       delta rule: write strength beta by token type, and test-time interventions that remove its
              selectivity (fillers write as strongly as pairs / uniform beta / beta = 1); ELU+1 linear control:
              stop filler tokens from writing into the state at test time
  retrain     causal tests of F1 by retraining (same pools, same seeds, same 4,500-update recipe):
              window/dilated on mixed data with the unreachable (distance > 63) queries unsupervised, window on
              near data with the same thinned supervision count, dilated on near data, five extra linear seeds
"""
from __future__ import annotations

import gzip
import json
import math
import types
from pathlib import Path

import numpy as np
import torch

from . import mqar_data as MD
from .mqar_model import DeltaAttention, LinearAttention, MQARModel, SoftmaxAttention, elu1
from .part_b import cell_acc, evaluate, run_name, to_device, train_one
from .utils import data_repo, free_cuda, log

KMAX = 16
REACH = 63                       # one-hop reach of a W = 64 window: the value sits at distance d - 1 <= 63
SHIFTS = [0, 2, 4, 6, 8, 10, 12]  # pair-block offsets (slots) for the relocation test
EXTRA_LINEAR_SEEDS = [6307, 6308, 6309, 6310, 6311]


def run_list(cfg):
    pb = cfg["part_b"]
    return [run_name(m, K, reg, s) for m, K, reg in pb["runs"] for s in pb["seeds"]]


def parse(name):
    mech, k, reg, s = name.split("_")
    return mech, int(k[1:]), reg, int(s[1:])


def load_ckpt(cfg, name, ckpt_dir: Path, device):
    mech = parse(name)[0]
    m = MQARModel(cfg["part_b"], mech).to(device)
    m.load_state_dict(torch.load(ckpt_dir / f"{name}.pt", map_location=device))
    return m.eval()


def subsample(d, n):
    if not n:
        return d
    keep = np.zeros(len(d["ids"]), dtype=bool)
    for k in (4, 8, 16):
        for reg in ("near", "far", "mixed"):
            keep[np.nonzero((d["K"] == k) & (d["regime"] == reg))[0][:n]] = True
    return {key: v[keep] for key, v in d.items()}


def select(d, mask):
    return {k: v[mask] for k, v in d.items()}


# ----------------------------------------------------------------------------- 1. taxonomy (no model)
def taxonomy(cfg, res: Path, test) -> dict:
    """Wrong answers split into 'another value of this sequence' vs 'value not in this sequence'; accuracy by the
    slot of the queried pair (K = 16 test sequences); spread of the chosen slot when wrong (normalised entropy)."""
    qt = MD.query_table(test)
    tok = test["tokens"].astype(int)
    vals = tok[qt["example"]][:, 1:2 * KMAX:2]                       # values at slots 0..15
    kmask = np.arange(KMAX)[None] < qt["K"][:, None]
    slot = qt["source"] // 2
    out = {}
    for name in run_list(cfg):
        f = res / "part_b" / "runs" / name / "test_predictions.npz"
        if not f.exists():
            continue
        z = np.load(f)
        pred = z["pred"].astype(int) + 128
        corr = z["correct"].astype(bool)
        hit = (vals == pred[:, None]) & kmask
        inctx = hit.any(1)
        chosen = np.where(inctx, hit.argmax(1), -1)
        r = {"acc": float(corr.mean()), "wrong_in_context": float((inctx & ~corr).mean()),
             "not_in_context": float((~inctx).mean())}
        for k in (4, 8, 16):
            m = qt["K"] == k
            r[f"K{k}"] = {"acc": float(corr[m].mean()), "wrong_in_context": float((inctx & ~corr)[m].mean()),
                          "not_in_context": float((~inctx)[m].mean()), "chance_in_context": 1.0 / k}
        m16 = qt["K"] == 16
        r["slot_acc_K16"] = [float(corr[m16 & (slot == j)].mean()) for j in range(KMAX)]
        r["slot_acc_K8"] = [float(corr[(qt["K"] == 8) & (slot == j)].mean()) for j in range(8)]
        w = m16 & inctx & ~corr
        if w.sum():
            h = np.bincount(chosen[w], minlength=KMAX)[:KMAX] / w.sum()
            r["wrong_slot_entropy_norm_K16"] = float(-(h[h > 0] * np.log(h[h > 0])).sum() / math.log(KMAX))
            r["wrong_slot_hist_K16"] = h.tolist()
        mech = parse(name)[0]
        if mech in ("window", "dilated"):
            near = qt["distance"] <= REACH
            r["acc_reachable_d_le_63"] = float(corr[near].mean())
            r["wrong_in_context_reachable"] = float((inctx & ~corr)[near].mean())
        out[name] = r
    return out


# ----------------------------------------------------------------------------- 2. dynamics (no model)
def dynamics(cfg, res: Path) -> dict:
    out = {}
    for name in run_list(cfg):
        f = res / "part_b" / "runs" / name / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        tc = s["train_curve"]
        esc = next((c["update"] for c in tc if c["acc"] >= 0.5), None)
        by = {c["update"]: c for c in tc}
        out[name] = {"ln_K_train": math.log(s["train_K"]), "ln_64": math.log(64),
                     "loss_at": {u: by[u]["loss"] for u in (250, 500, 1000, 2000, 3000, 4500) if u in by},
                     "escape_update_acc50": esc, "final_train_acc": tc[-1]["acc"], "final_train_loss": tc[-1]["loss"],
                     "clipped_frac_mean": float(np.mean([c["clipped_frac"] for c in tc])),
                     "grad_norm_median": float(np.median([c["grad_norm"] for c in tc]))}
    return out


# ----------------------------------------------------------------------------- helpers: evaluation
def arrays_from(tokens, qpos, answers):
    return {"tokens": tokens.astype(np.uint8), "qpos": qpos.astype(np.int16), "answers": answers.astype(np.uint8)}


@torch.no_grad()
def acc_on(model, arr, device, bs=256):
    tok, lab = to_device(arr, device)
    c, p, l = evaluate(model, tok, lab, bs)
    return c, p, l


def shifted_k4(test, s: int, seed: int = 6304):
    """K = 4 test sequences with the 4-pair block moved from slots 0-3 to slots s..s+3 (positions 2s..2s+7).
    Old block positions become filler; queries that would now sit inside or before the new block are replaced by
    filler and dropped. Returns arrays + a per-query table (new slot, new distance, regime)."""
    sel = test["K"] == 4
    tok = test["tokens"][sel].astype(np.int64).copy()
    qpos = test["qpos"][sel].astype(np.int64).copy()
    ans = test["answers"][sel].astype(np.int64).copy()
    src = test["sources"][sel].astype(np.int64).copy()
    reg = test["regime"][sel]
    n = tok.shape[0]
    rng = np.random.default_rng(seed + s)
    block = tok[:, :8].copy()
    tok[:, :8] = rng.integers(2, 64, size=(n, 8))
    tok[:, 2 * s:2 * s + 8] = block
    valid = (qpos >= 0) & (qpos >= 2 * s + 8)
    drop = (qpos >= 0) & ~valid
    rows, cols = np.nonzero(drop)
    for r_, c_ in zip(rows, cols):
        p = qpos[r_, c_]
        if not (2 * s <= p < 2 * s + 8):               # outside the new block: plain filler
            tok[r_, p] = rng.integers(2, 64)
    q2 = np.where(valid, qpos, -1)
    keys = tok[:, 2 * s:2 * s + 8:2]
    for i in range(n):                                  # invariant: every key occurs once in the block + at its queries
        for kk in keys[i]:
            pos = np.nonzero(tok[i] == kk)[0]
            assert pos[0] >= 2 * s and set(pos[1:]) <= set(q2[i][q2[i] >= 0].tolist()), "relocation broke a key"
    m = q2 >= 0
    ex = np.nonzero(m)[0]
    newsrc = (src + 2 * s)[m]
    qtab = {"slot": newsrc // 2, "distance": q2[m] - newsrc, "regime": reg[ex], "example": ex}
    return arrays_from(tok, q2, ans), qtab


def relocation(cfg, test, ckpt_dir, device, names) -> dict:
    out = {}
    shifted = {s: shifted_k4(test, s) for s in SHIFTS}
    for name in names:
        mech, K, reg, seed = parse(name)
        if not (ckpt_dir / f"{name}.pt").exists():
            continue
        model = load_ckpt(cfg, name, ckpt_dir, device)
        r = {"by_shift": {}, "slot_acc": [None] * KMAX, "slot_n": [0] * KMAX}
        hits = np.zeros(KMAX)
        cnt = np.zeros(KMAX)
        for s, (arr, qtab) in shifted.items():
            c, _, _ = acc_on(model, arr, device)
            use = np.ones(len(c), bool)
            if mech == "window":                         # access limit: score only one-hop-reachable queries
                use = qtab["distance"] <= REACH
            r["by_shift"][str(s)] = {"acc": float(c[use].mean()), "n": int(use.sum()), "slots": [s, s + 3]}
            for j in range(s, s + 4):
                mm = use & (qtab["slot"] == j)
                hits[j] += c[mm].sum()
                cnt[j] += mm.sum()
        r["slot_acc"] = [float(hits[j] / cnt[j]) if cnt[j] else None for j in range(KMAX)]
        r["slot_n"] = cnt.astype(int).tolist()
        out[name] = r
        del model
        free_cuda()
        log(f"relocation {name}: " + " ".join(f"s{s}={v['acc']:.2f}" for s, v in r["by_shift"].items()))
    return out


# ----------------------------------------------------------------------------- 3. circuit scores
def attn_weights(att, h):
    """(B,H,T,T) attention weights of one block given its LayerNorm-ed input h; None for the delta rule."""
    B, T, d = h.shape
    if isinstance(att, SoftmaxAttention):
        q = att.q(h).view(B, T, att.H, att.dh).transpose(1, 2)
        k = att.k(h).view(B, T, att.H, att.dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(att.dh)
        s = s.masked_fill(~att.mask[:, :T, :T], float("-inf"))
        return torch.softmax(s, -1)
    if isinstance(att, LinearAttention):
        q = elu1(att.q(h).view(B, T, att.H, att.dh).transpose(1, 2))
        k = elu1(att.k(h).view(B, T, att.H, att.dh).transpose(1, 2))
        a = (q @ k.transpose(-1, -2)).tril()
        return a / (a.sum(-1, keepdim=True) + att.eps)
    return None


@torch.no_grad()
def capture(model, tok):
    x = model.emb(tok) + model.pe[:tok.shape[1]]
    A = []
    for b in model.blocks:
        h = b.ln1(x)
        A.append(attn_weights(b.attn, h))
        x = x + b.attn(h)
        x = x + b.mlp(b.ln2(x))
    return A


def circuit(cfg, test, ckpt_dir, device, names, n_seq=256, bs=32) -> dict:
    """K = 16 test sequences (n_seq per regime). Per layer and head:
    bind   = mean_j A[2j+1 -> 2j]          (value position reads its own key)
    ret    = mean_q A[q -> 2j*+1]          (query reads the value of the matching pair)
    ret_key= mean_q A[q -> 2j*]            (query reads the matching key)
    other  = mean_q mean_{j != j*} A[q -> 2j+1]   (query reads other values)
    filler = mean_q sum_{filler p} A[q -> p]      (mass on filler tokens)
    Window/dilated: queries with distance <= 63 only (one-hop reachable)."""
    sel = np.zeros(len(test["ids"]), bool)
    for reg in ("near", "far", "mixed"):
        sel[np.nonzero((test["K"] == 16) & (test["regime"] == reg))[0][:n_seq]] = True
    d = select(test, sel)
    tok_all = torch.as_tensor(d["tokens"].astype(np.int64), device=device)
    out = {}
    for name in names:
        mech = parse(name)[0]
        if mech == "delta" or not (ckpt_dir / f"{name}.pt").exists():
            continue
        model = load_ckpt(cfg, name, ckpt_dir, device)
        H = cfg["part_b"]["n_heads"]
        acc = {k: np.zeros((2, H)) for k in ("bind", "ret", "ret_key", "other", "filler")}
        bind_slot = np.zeros((2, H, KMAX))
        nq = np.zeros(2)
        nb = 0
        for i in range(0, tok_all.shape[0], bs):
            tok = tok_all[i:i + bs]
            A = capture(model, tok)
            B = tok.shape[0]
            jj = torch.arange(KMAX, device=device)
            for l, a in enumerate(A):
                bs_ = a[:, :, 2 * jj + 1, 2 * jj]                                          # (B,H,16)
                acc["bind"][l] += bs_.mean(-1).sum(0).cpu().numpy()
                bind_slot[l] += bs_.sum(0).cpu().numpy()
            nb += B
            qp = torch.as_tensor(d["qpos"][i:i + bs].astype(np.int64), device=device)       # (B,16)
            sp = torch.as_tensor(d["sources"][i:i + bs].astype(np.int64), device=device)
            dist = torch.as_tensor(d["distances"][i:i + bs].astype(np.int64), device=device)
            m = qp >= 0
            if mech in ("window", "dilated"):
                m &= dist <= REACH
            bi, qi = torch.nonzero(m, as_tuple=True)
            if len(bi) == 0:
                continue
            q, s = qp[bi, qi], sp[bi, qi]
            filler = torch.ones(B, tok.shape[1], dtype=torch.bool, device=device)
            filler[:, :2 * KMAX] = False
            filler[torch.arange(B, device=device)[:, None].expand_as(qp)[m], qp[m]] = False
            vals = 2 * jj + 1
            for l, a in enumerate(A):
                rows = a[bi, :, q]                                                         # (Nq,H,T)
                acc["ret"][l] += rows[torch.arange(len(q)), :, s + 1].sum(0).cpu().numpy()
                acc["ret_key"][l] += rows[torch.arange(len(q)), :, s].sum(0).cpu().numpy()
                vv = rows[:, :, vals]                                                      # (Nq,H,16)
                own = (vals[None] == (s + 1)[:, None])
                acc["other"][l] += ((vv * (~own)[:, None]).sum(-1) / (KMAX - 1)).sum(0).cpu().numpy()
                acc["filler"][l] += (rows * filler[bi][:, None]).sum(-1).sum(0).cpu().numpy()
                nq[l] += len(q)
        r = {"bind": (acc["bind"] / nb).tolist(), "bind_by_slot": (bind_slot / nb).tolist()}
        for k in ("ret", "ret_key", "other", "filler"):
            r[k] = (acc[k] / np.maximum(nq[:, None], 1)).tolist()
        r["n_queries"] = int(nq[0])
        out[name] = r
        del model
        free_cuda()
    return out


# ----------------------------------------------------------------------------- 4. head ablation
def _zero_head_hooks(model, heads):
    """heads: list of (layer, head). Zeroes the head's slice of the attention output before the o-projection."""
    hs = []
    for l, b in enumerate(model.blocks):
        mine = [h for (ll, h) in heads if ll == l]
        if not mine:
            continue
        dh = b.attn.dh

        def pre(mod, args, mine=mine, dh=dh):
            x = args[0].clone()
            for h in mine:
                x[..., h * dh:(h + 1) * dh] = 0
            return (x,)
        hs.append(b.attn.o.register_forward_pre_hook(pre))
    return hs


def ablation(cfg, test, ckpt_dir, device, names) -> dict:
    H = cfg["part_b"]["n_heads"]
    out = {}
    for name in names:
        if not (ckpt_dir / f"{name}.pt").exists():
            continue
        mech = parse(name)[0]
        sel = test["K"] == 16
        if mech == "window":
            sel &= test["regime"] == "near"
        arr = select(test, sel)
        model = load_ckpt(cfg, name, ckpt_dir, device)
        arms = {"none": []}
        for l in range(2):
            for h in range(H):
                arms[f"L{l + 1}H{h}"] = [(l, h)]
            arms[f"L{l + 1}_all"] = [(l, h) for h in range(H)]
        r = {}
        for arm, heads in arms.items():
            hooks = _zero_head_hooks(model, heads)
            try:
                c, p, loss = acc_on(model, arr, device)
            finally:
                for hk in hooks:
                    hk.remove()
            vals = arr["tokens"].astype(int)
            qt = MD.query_table(arr)
            v = vals[qt["example"]][:, 1:2 * KMAX:2]
            inctx = ((v == (p.astype(int) + 128)[:, None]) & (np.arange(KMAX)[None] < qt["K"][:, None])).any(1)
            r[arm] = {"acc": float(c.mean()), "loss": float(loss), "wrong_in_context": float((inctx & ~c.astype(bool)).mean())}
        out[name] = r
        del model
        free_cuda()
        log(f"ablation {name}: " + " ".join(f"{k}={v['acc']:.2f}" for k, v in r.items()))
    return out


# ----------------------------------------------------------------------------- 5. gates (delta) + linear control
def token_types(tok_np, qpos_np):
    """0 filler, 1 key (slot j < K), 2 value, 3 query. Keys/values are identified by token range at positions < 2K."""
    B, T = tok_np.shape
    ty = np.zeros((B, T), np.int64)
    pos = np.arange(T)[None]
    is_key = (tok_np >= 64) & (tok_np < 128)
    is_val = (tok_np >= 128) & (tok_np < 192)
    ty[is_val] = 2
    ty[is_key] = 1
    for i in range(B):
        q = qpos_np[i][qpos_np[i] >= 0]
        ty[i, q] = 3
    return ty


def _delta_forward(self, x, beta_mode=None, types_=None, record=None):
    B, T, d = x.shape
    q = self._phi(self.q(x).view(B, T, self.H, self.dh).transpose(1, 2))
    k = self._phi(self.k(x).view(B, T, self.H, self.dh).transpose(1, 2))
    v = self.v(x).view(B, T, self.H, self.dh).transpose(1, 2)
    beta = torch.sigmoid(self.beta(x)).transpose(1, 2)                 # (B,H,T)
    if record is not None:
        record.append(beta.detach())
    if beta_mode is not None:
        pair = ((types_ == 1) | (types_ == 2))[:, None].float()         # (B,1,T)
        fill = (types_ == 0)[:, None]
        if beta_mode == "fillers_as_pairs":
            mb = (beta * pair).sum(-1, keepdim=True) / pair.sum(-1, keepdim=True).clamp_min(1)
            beta = torch.where(fill, mb.expand_as(beta), beta)
        elif beta_mode == "uniform":
            beta = beta.mean(-1, keepdim=True).expand_as(beta)
        elif beta_mode == "one":
            beta = torch.ones_like(beta)
        elif beta_mode == "pairs_as_fillers":
            mf = (beta * fill).sum(-1, keepdim=True) / fill.float().sum(-1, keepdim=True).clamp_min(1)
            beta = torch.where(pair.bool(), mf.expand_as(beta), beta)
        else:
            raise ValueError(beta_mode)
    S = x.new_zeros(B, self.H, self.feat, self.dh)
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + beta[:, :, t, None, None] * kt[..., :, None] * (vt - pred)[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
    out = torch.stack(outs, 2)
    return self.o(out.transpose(1, 2).reshape(B, T, d))


def _linear_forward_masked(self, x, write=None):
    """ELU+1 linear attention where positions with write == 0 contribute nothing to the state or normaliser."""
    B, T, d = x.shape
    q = elu1(self.q(x).view(B, T, self.H, self.dh).transpose(1, 2))
    k = elu1(self.k(x).view(B, T, self.H, self.dh).transpose(1, 2))
    v = self.v(x).view(B, T, self.H, self.dh).transpose(1, 2)
    a = (q @ k.transpose(-1, -2)).tril()
    if write is not None:
        a = a * write[:, None, None, :]
    out = (a @ v) / (a.sum(-1, keepdim=True) + self.eps)
    return self.o(out.transpose(1, 2).reshape(B, T, d))


@torch.no_grad()
def _eval_with(model, arr, device, setup, bs=128):
    """evaluate() with a per-batch hook `setup(model, tok_np, qpos_np)` that patches the attention modules."""
    tok, lab = to_device(arr, device)
    corr = []
    for i in range(0, tok.shape[0], bs):
        setup(model, arr["tokens"][i:i + bs].astype(np.int64), arr["qpos"][i:i + bs].astype(np.int64))
        lg = model(tok[i:i + bs])
        m = lab[i:i + bs] >= 0
        corr.append((lg[..., 128:192][m].argmax(-1) == lab[i:i + bs][m]).cpu())
    return torch.cat(corr).numpy()


def gates(cfg, test, ckpt_dir, device, delta_names, linear_names) -> dict:
    out = {"delta": {}, "linear_filler_mask": {}}
    arr = select(test, test["K"] == 16)
    for name in delta_names:
        if not (ckpt_dir / f"{name}.pt").exists():
            continue
        model = load_ckpt(cfg, name, ckpt_dir, device)
        rec = {"beta_by_type": None, "acc": {}}
        sums = np.zeros((2, cfg["part_b"]["n_heads"], 4))
        cnts = np.zeros(4)
        for mode in [None, "fillers_as_pairs", "uniform", "one", "pairs_as_fillers"]:
            def setup(m, tok_np, q_np, mode=mode):
                ty = torch.as_tensor(token_types(tok_np, q_np), device=device)
                for l, b in enumerate(m.blocks):
                    if mode is None:
                        store = []
                        b.attn._store = store
                        b.attn.forward = types.MethodType(
                            lambda self, x, ty=ty, st=store: _delta_forward(self, x, None, ty, st), b.attn)
                    else:
                        b.attn.forward = types.MethodType(
                            lambda self, x, ty=ty, mo=mode: _delta_forward(self, x, mo, ty), b.attn)
                m._last_types = ty

            corr = []
            tok, lab = to_device(arr, device)
            for i in range(0, tok.shape[0], 128):
                setup(model, arr["tokens"][i:i + 128].astype(np.int64), arr["qpos"][i:i + 128].astype(np.int64))
                lg = model(tok[i:i + 128])
                mm = lab[i:i + 128] >= 0
                corr.append((lg[..., 128:192][mm].argmax(-1) == lab[i:i + 128][mm]).cpu())
                if mode is None:
                    ty = model._last_types
                    for l, b in enumerate(model.blocks):
                        beta = b.attn._store[0]                                    # (B,H,T)
                        for t in range(4):
                            sel = (ty == t)[:, None].float()
                            sums[l, :, t] += (beta * sel).sum((0, 2)).cpu().numpy()
                    for t in range(4):
                        cnts[t] += float((ty == t).sum())
            rec["acc"][mode or "none"] = float(torch.cat(corr).float().mean())
        for b in model.blocks:                                                     # restore the class method
            if "forward" in b.attn.__dict__:
                del b.attn.forward
        rec["beta_by_type"] = {nm: (sums[:, :, t] / max(cnts[t], 1)).tolist()
                               for t, nm in enumerate(["filler", "key", "value", "query"])}
        out["delta"][name] = rec
        del model
        free_cuda()
        log(f"gates {name}: " + " ".join(f"{k}={v:.2f}" for k, v in rec["acc"].items()))
    for name in linear_names:
        if not (ckpt_dir / f"{name}.pt").exists():
            continue
        model = load_ckpt(cfg, name, ckpt_dir, device)
        res = {}
        for mode in ("none", "fillers_not_written"):
            def setup(m, tok_np, q_np, mode=mode):
                ty = token_types(tok_np, q_np)
                w = torch.as_tensor((ty != 0).astype(np.float32), device=device) if mode != "none" else None
                for b in m.blocks:
                    b.attn.forward = types.MethodType(lambda self, x, w=w: _linear_forward_masked(self, x, w), b.attn)
            res[mode] = float(_eval_with(model, arr, device, setup).mean())
        for b in model.blocks:
            if "forward" in b.attn.__dict__:
                del b.attn.forward
        out["linear_filler_mask"][name] = res
        del model
        free_cuda()
    return out


# ----------------------------------------------------------------------------- 6. retraining arms
RETRAIN_ARMS = {
    # name: (mech, K, regime, supervision rule, seeds)
    "window_mixed_reach63": ("window", 16, "mixed", "reach", None),
    "window_near_thinned": ("window", 16, "near", "thin", None),
    "dilated_near": ("dilated", 16, "near", "all", None),
    "dilated_mixed_reach63": ("dilated", 16, "mixed", "reach", None),
    "linear_mixed_extra_seeds": ("linear", 16, "mixed", "all", EXTRA_LINEAR_SEEDS),
}


def supervision(pool, rule, keep_frac=None, seed=6304):
    """Label mask on a training pool: 'all', 'reach' (distance <= 63 only) or 'thin' (each query kept with
    probability keep_frac, fixed RNG)."""
    arr = {k: pool[k] for k in ("tokens", "qpos", "answers")}
    q = pool["qpos"].astype(np.int64).copy()
    if rule == "reach":
        q[(q >= 0) & (pool["distances"] > REACH)] = -1
    elif rule == "thin":
        keep = np.random.default_rng(seed).random(q.shape) < keep_frac
        q[(q >= 0) & ~keep] = -1
    elif rule != "all":
        raise ValueError(rule)
    arr["qpos"] = q.astype(np.int16)
    return arr


def retrain(cfg, device, out_dir: Path, cache_dir: Path, ckpt_root: Path, arms=None) -> dict:
    pb = cfg["part_b"]
    test, dev = MD.load_frozen(cfg, "test"), MD.load_frozen(cfg, "dev")
    if pb.get("eval_subsample"):
        test, dev = subsample(test, pb["eval_subsample"]), subsample(dev, pb["eval_subsample"])
    qt_test, qt_dev = MD.query_table(test), MD.query_table(dev)
    test_t, dev_t = to_device(test, device), to_device(dev, device)
    need = pb["batch"] * pb["updates"]
    mixed16 = MD.train_pool(cfg, 16, "mixed", need, cache_dir)
    m = mixed16["qpos"] >= 0
    reach_frac = float((mixed16["distances"][m] <= REACH).mean())            # supervised share under 'reach'
    summary = {"reach_fraction_mixed_K16": reach_frac, "arms": {}}
    for arm, (mech, K, reg, rule, seeds) in RETRAIN_ARMS.items():
        if arms and arm not in arms:
            continue
        pool = MD.train_pool(cfg, K, reg, need, cache_dir)
        sup = supervision(pool, rule, keep_frac=reach_frac)
        n_sup = int((sup["qpos"] >= 0).sum())
        tl = to_device(sup, device)
        rec = {"mech": mech, "K": K, "regime": reg, "rule": rule, "supervised_queries_per_seq": n_sup / len(sup["qpos"]),
               "runs": {}}
        for seed in seeds or pb["seeds"]:
            s = train_one(cfg, mech, K, reg, seed, tl, dev_t, test_t, qt_dev, qt_test,
                          out_dir / "runs" / f"{arm}_s{seed}", ckpt_root / arm, device)
            tc = s["train_curve"]
            rec["runs"][str(seed)] = {"test_acc_all": s["test_acc_all"], "test_cells": s["test_cells"],
                                      "escape_update_acc50": next((c["update"] for c in tc if c["acc"] >= 0.5), None),
                                      "final_train_loss": tc[-1]["loss"] if tc else None,
                                      "dev_curve": [{"update": c["update"], "dev_acc": c["dev_acc"]} for c in s["dev_curve"]]}
            cz = np.load(out_dir / "runs" / f"{arm}_s{seed}" / "test_predictions.npz")["correct"].astype(bool)
            near = qt_test["distance"] <= REACH
            rec["runs"][str(seed)]["acc_reachable_d_le_63"] = float(cz[near].mean())
        del tl
        free_cuda()
        summary["arms"][arm] = rec
        log(f"retrain {arm}: " + " ".join(f"{k}={v['test_acc_all']:.3f}" for k, v in rec["runs"].items()))
    return summary


# ----------------------------------------------------------------------------- driver
def run(cfg, device, res: Path, ckpt_dir: Path, cache_dir: Path, steps, subsample_n=None) -> dict:
    out_f = res / "diagnostics" / "part_b_diagnostic.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    test = MD.load_frozen(cfg, "test")
    test_s = subsample(test, subsample_n)
    names = run_list(cfg)
    have = [n for n in names if (ckpt_dir / f"{n}.pt").exists()]
    log(f"Part B diagnostic: {len(have)}/{len(names)} checkpoints in {ckpt_dir}")

    def save():
        from .utils import save_json
        save_json(out_f, out)

    if "taxonomy" in steps:
        out["taxonomy"] = taxonomy(cfg, res, test)
        save()
    if "dynamics" in steps:
        out["dynamics"] = dynamics(cfg, res)
        save()
    if "relocation" in steps:
        rel = [n for n in have if parse(n)[0] in ("dense", "linear", "delta") or
               (parse(n)[0] == "window" and parse(n)[2] == "near")]
        out["relocation"] = relocation(cfg, test_s, ckpt_dir, device, rel)
        save()
    if "circuit" in steps:
        out["circuit"] = circuit(cfg, test_s, ckpt_dir, device, have, n_seq=256 if not subsample_n else subsample_n)
        save()
    if "ablation" in steps:
        learned = [n for n in have if n.startswith(("dense_K16_mixed", "dense_K16_near", "window_K16_near", "delta_K16"))
                   or n == "linear_K16_mixed_s6305"]
        out["ablation"] = ablation(cfg, test_s, ckpt_dir, device, learned)
        save()
    if "gates" in steps:
        out["gates"] = gates(cfg, test_s, ckpt_dir, device, [n for n in have if n.startswith("delta")],
                             [n for n in have if n.startswith("linear")])
        save()
    if "retrain" in steps:
        out["retrain"] = retrain(cfg, device, res / "diagnostics" / "retrain", cache_dir, ckpt_dir / "diag")
        save()
    return out
