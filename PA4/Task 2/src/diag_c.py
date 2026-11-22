"""Part C mechanism diagnostic (supplementary; changes no main-run condition).

Finding to explain: INT4 with token-wise key groups fails on every prompt (0 % exact match, gold NLL ~24), while
channel-wise key groups at the same bytes are indistinguishable from FP16. The two layouts quantize V identically
(token-wise), so the difference must come from K. Candidate mechanisms make different predictions:
  (a) outlier key channels: a few channels with large magnitude set the range of every 64-channel token group, so
      the step size is larger than the spread of the ordinary channels and those collapse to one or two codes
      -> keeping the top-m channels in FP16 (excluded from the group range) should restore accuracy;
  (b) a constant per-channel offset (k_proj bias rotated by RoPE): removing the per-channel context mean before
      token-wise quantization should restore accuracy;
  (c) the attention sink: only the first tokens' keys matter -> keeping them in FP16 should restore accuracy;
  (d) plain resolution: INT8 token-wise should be fine (it does not separate (a)/(b) but bounds the effect);
  (e) a few layers: token-wise keys only in layers 0-1 (channel-wise elsewhere) vs everywhere except 0-1.
Arms (context cache only; suffix and generated tokens FP16 for every arm, so arms differ only in the context K/V):
  fp16, k_channel (= main INT4 channel-K/token-V), k_token (= main INT4 token-wise), k_token_v_fp16 (V exact),
  v_token_k_fp16 (K exact, V token-wise INT4), k_token_int8, k_token_center, k_token_outlier{2,8},
  k_token_sink8, k_token_layers01, k_token_except01.
Measurements per prompt and arm: exact match, gold NLL, KL vs FP16 at the first answer position; with the FP16 suffix
queries held fixed ("isolated" attention), per layer the KL of the attention distribution vs FP16 and the attention
mass on the gold-evidence tokens and on token 0. Observational key statistics per layer (channel offsets, step size of
the token-wise grid relative to the ordinary channels' spread)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

from . import kv_quant as KQ
from . import part_c as PC
from .utils import data_repo, free_cuda, log, read_jsonl, save_json, save_jsonl_gz

ARMS = ["fp16", "k_channel", "k_token", "k_token_v_fp16", "v_token_k_fp16", "k_token_int8", "k_token_center",
        "k_token_outlier2", "k_token_outlier8", "k_token_sink8", "k_token_layers01", "k_token_except01"]
ATTN_ARMS = ["k_channel", "k_token", "k_token_int8", "k_token_center", "k_token_outlier2", "k_token_outlier8",
             "k_token_sink8"]
G = KQ.G


def _token_k(k, lq, b=4):
    return KQ.quant_tokenwise(k, b, lq)


def _center(k, lq):
    mu = k[:, :, :lq].float().mean(2, keepdim=True).half().float()            # per (head, channel), stored FP16
    kq = KQ.quant_tokenwise((k.float() - mu).to(k.dtype), 4, lq).float() + mu
    out = kq.to(k.dtype)
    out[:, :, lq:] = k[:, :, lq:]
    return out


def outlier_channels(k, lq, m):
    """Top-m channels per KV head by mean |K| over the quantized context tokens. k (1,H,L,d) -> (H, m)."""
    return k[0, :, :lq].float().abs().mean(1).topk(m, dim=-1).indices


def _outlier(k, lq, m):
    """Token-wise INT4 where the top-m channels per head are kept in FP16 and excluded from their group's range
    (they are replaced by the mean of the group's other channels before quantization, then restored)."""
    if lq == 0:
        return k.clone()
    idx = outlier_channels(k, lq, m)                                           # (H, m)
    H, d = k.shape[1], k.shape[-1]
    mask = torch.zeros(H, d, dtype=torch.bool, device=k.device)
    mask[torch.arange(H, device=k.device)[:, None], idx] = True
    x = k.float().clone()
    xg = x[:, :, :lq].reshape(1, H, lq, d // G, G)
    mg = mask.reshape(1, H, 1, d // G, G)
    fill = (xg * ~mg).sum(-1, keepdim=True) / (~mg).sum(-1, keepdim=True).clamp_min(1)
    xg = torch.where(mg, fill, xg)
    q = KQ.qdq(xg, 4).float()
    q = torch.where(mg, x[:, :, :lq].reshape(1, H, lq, d // G, G), q)
    out = k.clone()
    out[:, :, :lq] = q.reshape(1, H, lq, d).to(k.dtype)
    return out


def _sink(k, lq, n=8):
    out = KQ.quant_tokenwise(k, 4, lq)
    out[:, :, :n] = k[:, :, :n]
    return out


def arm_kv(arm, ctx_kv):
    """List of (K, V) context tensors for one arm and its extra FP16 bytes (beyond the INT4 layout)."""
    kv, extra = [], 0
    for l, (k, v) in enumerate(ctx_kv):
        L = k.shape[-2]
        lq = (L // G) * G
        vq = KQ.quant_tokenwise(v, 4, lq)
        if arm == "fp16":
            kv.append((k.clone(), v.clone()))
            continue
        if arm == "k_channel":
            kh = KQ.quant_channelwise(k, 4, lq)
        elif arm in ("k_token", "k_token_v_fp16"):
            kh = _token_k(k, lq)
        elif arm == "v_token_k_fp16":
            kh = k.clone()
        elif arm == "k_token_int8":
            kh = _token_k(k, lq, 8)
        elif arm == "k_token_center":
            kh = _center(k, lq)
            extra += k.shape[1] * k.shape[-1] * 2
        elif arm.startswith("k_token_outlier"):
            m = int(arm[len("k_token_outlier"):])
            kh = _outlier(k, lq, m)
            extra += k.shape[1] * lq * m * 2 + k.shape[1] * m * 1
        elif arm == "k_token_sink8":
            kh = _sink(k, lq)
            extra += k.shape[1] * 8 * k.shape[-1] * 2
        elif arm == "k_token_layers01":
            kh = _token_k(k, lq) if l < 2 else KQ.quant_channelwise(k, 4, lq)
        elif arm == "k_token_except01":
            kh = KQ.quant_channelwise(k, 4, lq) if l < 2 else _token_k(k, lq)
        else:
            raise ValueError(arm)
        kv.append((kh, v.clone() if arm == "k_token_v_fp16" else vq))
    return kv, extra


@torch.no_grad()
def key_stats(ctx_kv):
    """Per layer (mean over KV heads): max |channel mean| / median channel std; token-wise step (range/15 of each
    64-channel group, median over tokens and groups) / median channel std; share of non-top-8 channels whose std is
    below half a step (they collapse to <= 2 codes)."""
    rows = []
    for k, _ in ctx_kv:
        L = k.shape[-2]
        lq = (L // G) * G
        x = k[0, :, :lq].float()                                                # (H, lq, d)
        mu, sd = x.mean(1), x.std(1)                                            # (H, d)
        med_sd = sd.median(-1).values
        g = x.reshape(x.shape[0], lq, -1, G)
        step = ((g.amax(-1) - g.amin(-1)) / 15).flatten(1).median(-1).values   # (H,)
        top = x.abs().mean(1).topk(8, -1).indices
        ordinary = torch.ones_like(sd, dtype=torch.bool)
        ordinary[torch.arange(x.shape[0])[:, None], top] = False
        collapse = ((sd < 0.5 * step[:, None]) & ordinary).float().sum(-1) / ordinary.float().sum(-1)
        rows.append({"offset_ratio": float((mu.abs().amax(-1) / med_sd).mean()),
                     "step_over_sd": float((step / med_sd).mean()),
                     "collapsed_share": float(collapse.mean()),
                     "top8_channels": top.cpu().tolist(),
                     "top8_mean_abs_over_median": float((x.abs().mean(1).topk(8, -1).values.mean(-1) /
                                                         x.abs().mean(1).median(-1).values).mean())})
    return rows


@torch.no_grad()
def suffix_queries(model, ctx_cache, suffix, L):
    """FP16 pass of the suffix on a copy of the context cache; returns per layer the rotary query states
    (1, Hq, S, d) and the suffix keys (1, Hkv, S, d)."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    cache = PC.clone_cache(ctx_cache)
    caps, hooks = {}, []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, l=l):
            caps[l] = (kwargs.get("hidden_states", args[0] if args else None), kwargs["position_embeddings"])
        hooks.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        PC.run_suffix(model, cache, suffix, L, L)
    finally:
        for h in hooks:
            h.remove()
    c = model.config
    Hq = c.num_attention_heads
    dh = getattr(c, "head_dim", None) or c.hidden_size // Hq
    S = suffix.shape[1]
    out = []
    for l, layer in enumerate(model.model.layers):
        hs, (cos, sin) = caps[l]
        q = layer.self_attn.q_proj(hs).view(1, S, Hq, dh).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        out.append((q, cache.layers[l].keys[:, :, L:].clone()))
    del cache
    return out


@torch.no_grad()
def attn_metrics(qs, ctx_k, ref_k, ev, L, Hq):
    """Isolated attention: same FP16 queries, context keys of an arm vs FP16. Per layer: KL(FP16 || arm) of the
    attention distribution (mean over heads and suffix queries), mass on evidence tokens and on token 0."""
    rows = []
    for l, (q, ks) in enumerate(qs):
        S = q.shape[2]

        def att(kc):
            K = torch.cat([kc, ks], 2).repeat_interleave(Hq // kc.shape[1], dim=1).float()
            s = q.float() @ K.transpose(-1, -2) / q.shape[-1] ** 0.5
            causal = torch.ones(S, L + S, dtype=torch.bool, device=s.device).tril(L)
            return torch.log_softmax(s.masked_fill(~causal, float("-inf")), -1)
        lp_ref, lp = att(ref_k[l]), att(ctx_k[l])
        p_ref = lp_ref.exp()
        kl = torch.where(p_ref > 0, p_ref * (lp_ref - lp), torch.zeros_like(p_ref)).sum(-1)
        p = lp.exp()
        evm = p[..., ev].sum(-1) if ev else torch.zeros_like(kl)
        evm_ref = p_ref[..., ev].sum(-1) if ev else torch.zeros_like(kl)
        rows.append({"kl": float(kl.mean()), "evidence_mass": float(evm.mean()), "evidence_mass_fp16": float(evm_ref.mean()),
                     "token0_mass": float(p[..., 0].mean()), "token0_mass_fp16": float(p_ref[..., 0].mean())})
    return rows


@torch.no_grad()
def run_prompt(model, tok, p, eos_ids, max_new, with_attn=True):
    dev = next(model.parameters()).device
    ids = torch.tensor([p["ids"]], device=dev)
    L, S = p["L"], p["S"]
    ctx, suffix = ids[:, :L], ids[:, L:]
    ctx_cache = DynamicCache()
    model(input_ids=ctx, past_key_values=ctx_cache, use_cache=True, logits_to_keep=1)
    ctx_kv = [(k.clone(), v.clone()) for k, v in PC.kv_list(ctx_cache)]
    rec = {k: p[k] for k in ("id", "target_tokens", "position", "distractors", "code", "L", "S")}
    rec["key_stats"] = key_stats(ctx_kv)
    qs = suffix_queries(model, ctx_cache, suffix, L) if with_attn else None
    del ctx_cache
    Hq = model.config.num_attention_heads
    ref_k = [k for k, _ in ctx_kv]
    arms, lp_ref, lptf_ref = {}, None, None
    for arm in ARMS:
        kv, extra = arm_kv(arm, ctx_kv)
        cache = KQ.build_cache(kv)
        o = PC.run_suffix(model, cache, suffix, L, L)
        first = o.logits[0, -1].float()
        tf_cache = PC.clone_cache(cache)
        gen = PC.greedy(model, cache, first, L + S, L + S, max_new, eos_ids)
        lp_tf, nll = PC.teacher_forced(model, tf_cache, first, p["gold_ids"], L + S, L + S)
        lp0 = torch.log_softmax(first, -1)
        text = tok.decode(gen, skip_special_tokens=True)
        r = {"answer_text": text, "exact_match": int(PC.extract(text) == p["code"]), "gold_nll": nll,
             "extra_fp16_bytes": extra, "k_rel_mse": float(np.mean([((a.float() - b.float()).pow(2).mean() /
                                                                     b.float().pow(2).mean()).item()
                                                                    for (a, _), (b, _) in zip(kv, ctx_kv)]))}
        if arm == "fp16":
            lp_ref, lptf_ref = lp0, lp_tf
        else:
            r["post_prefill_kl"] = PC.kl(lp_ref[None], lp0[None])
            r["teacher_forced_kl"] = PC.kl(lptf_ref, lp_tf)
        if with_attn and arm in ATTN_ARMS:
            r["attn"] = attn_metrics(qs, [k for k, _ in kv], ref_k, p["evidence_positions"], L, Hq)
        arms[arm] = r
        del cache, tf_cache, kv
    rec["arms"] = arms
    del qs, ctx_kv
    free_cuda()
    return rec


def run(cfg, device, res: Path, limit=None, max_new=12, attn_every=1) -> dict:
    out_dir = res / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(data_repo(cfg) / cfg["data"]["kv_cases"])
    if limit:
        cases = cases[:limit]
    model, tok, rev = PC.load_model(cfg, device)
    eos_ids = {tok.convert_tokens_to_ids(t) for t in PC.EOS_TEXTS if t in tok.get_vocab()}
    prompts = PC.prepare(cfg, tok, cases)
    f = out_dir / "part_c_per_prompt.jsonl.gz"
    done = {r["id"]: r for r in (read_jsonl(f) if f.exists() else [])}
    rows = list(done.values())
    for i, p in enumerate(prompts):
        if p["id"] in done:
            continue
        rows.append(run_prompt(model, tok, p, eos_ids, max_new, with_attn=(i % attn_every == 0)))
        if (i + 1) % 10 == 0 or i == len(prompts) - 1:
            save_jsonl_gz(f, rows)
            em = {a: np.mean([r["arms"][a]["exact_match"] for r in rows]) for a in ("k_channel", "k_token", "k_token_outlier8")}
            log(f"Part C diagnostic {i + 1}/{len(prompts)} prompts; EM so far " + " ".join(f"{k}={v:.2f}" for k, v in em.items()))
    save_jsonl_gz(f, rows)
    summ = summarize(rows)
    summ["model_revision"] = rev
    summ["max_new_tokens"] = max_new
    save_json(out_dir / "part_c_diagnostic.json", summ)
    return summ


def summarize(rows) -> dict:
    out = {"n_prompts": len(rows), "arms": {}}
    for arm in ARMS:
        a = [r["arms"][arm] for r in rows]
        s = {"exact_match": float(np.mean([x["exact_match"] for x in a])),
             "gold_nll_mean": float(np.mean([x["gold_nll"] for x in a])),
             "gold_nll_median": float(np.median([x["gold_nll"] for x in a])),
             "k_rel_mse": float(np.mean([x["k_rel_mse"] for x in a])),
             "extra_fp16_bytes_mean": float(np.mean([x["extra_fp16_bytes"] for x in a]))}
        if arm != "fp16":
            s["post_prefill_kl"] = float(np.mean([x["post_prefill_kl"] for x in a]))
            s["teacher_forced_kl"] = float(np.mean([x["teacher_forced_kl"] for x in a]))
        att = [x["attn"] for x in a if "attn" in x]
        if att:
            nl = len(att[0])
            s["attn_by_layer"] = {k: [float(np.mean([t[l][k] for t in att])) for l in range(nl)]
                                  for k in ("kl", "evidence_mass", "evidence_mass_fp16", "token0_mass", "token0_mass_fp16")}
        out["arms"][arm] = s
    ks = [r["key_stats"] for r in rows]
    nl = len(ks[0])
    out["key_stats_by_layer"] = {k: [float(np.mean([x[l][k] for x in ks])) for l in range(nl)]
                                 for k in ("offset_ratio", "step_over_sd", "collapsed_share", "top8_mean_abs_over_median")}
    # are the outlier channels the same across prompts? Jaccard of each prompt's top-8 set with the first prompt's
    jac = []
    for l in range(nl):
        ref = [set(h) for h in ks[0][l]["top8_channels"]]
        jac.append(float(np.mean([np.mean([len(set(h) & ref[i]) / len(set(h) | ref[i])
                                           for i, h in enumerate(x[l]["top8_channels"])]) for x in ks])))
    out["key_stats_by_layer"]["top8_jaccard_vs_first_prompt"] = jac
    for fac in ("target_tokens", "position", "distractors"):
        out[f"exact_match_by_{fac}"] = {str(v): {arm: float(np.mean([r["arms"][arm]["exact_match"] for r in rows
                                                                     if r[fac] == v])) for arm in ARMS}
                                        for v in sorted({r[fac] for r in rows})}
    return out
