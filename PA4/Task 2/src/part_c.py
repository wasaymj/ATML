"""Part C: KV-cache compression of Qwen2.5-1.5B-Instruct (FP16, SDPA, greedy) on the 180 frozen retrieval prompts.

Protocol for every prompt (identical for all policies):
  1. chat template -> token IDs; split at "QUESTION_BEGIN" into the pre-question context (L tokens) and the question
     suffix (S tokens, incl. the assistant header). Evidence token positions come from the gold character span and
     are used for evaluation only.
  2. prefill the context once -> FP16 context cache (the reference starts from it too).
  3. compress the context cache (this is where the case-specific INT4 budget is measured).
  4. process the suffix with the compressed cache, keeping the ORIGINAL absolute rotary positions
     (position_ids = L..L+S-1) and the physical cache slots (cache_position = N..N+S-1); then greedy decoding.
  5. metrics: exact match, gold-answer NLL and teacher-forced KL, post-prefill KL at the first answer position,
     retained evidence, realized bytes, timings, peak memory.
Policies at the same serialized-byte budget = uniform INT4 channel-K / token-V bytes of the context:
  sink_recent    FP16 eviction: 8 sink + most recent positions
  question_aware FP16 eviction: 8 sink + 64 recent + top SnapKV-style votes from the question suffix
  int4_channelK_tokenV, int4_tokenwise   quantization layouts (all positions kept)
  hybrid         8 sink + 64 recent + top-voted positions in FP16, all others INT2 channel-K/token-V
"""
from __future__ import annotations

import copy
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from . import kv_quant as KQ
from .utils import data_repo, free_cuda, gpu_mem, log, read_jsonl, reset_peak, save_json, save_jsonl_gz, sync

EOS_TEXTS = ["<|im_end|>", "<|endoftext|>"]
POLICIES = ["sink_recent", "question_aware", "int4_channelK_tokenV", "int4_tokenwise", "hybrid"]


# --------------------------------------------------------------------------------------------- loading
def load_model(cfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    pc = cfg["part_c"]
    rev = pc["revision"]
    if rev is None and not Path(pc["model"]).exists():
        lock = data_repo(cfg) / cfg["data"]["lock_file"]
        rev = json.loads(lock.read_text())["revisions"]["model_target"] if lock.exists() else None
    kw = dict(revision=rev, attn_implementation="sdpa")
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(pc["model"], dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(pc["model"], torch_dtype=dtype, **kw)
    model.to(device).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(pc["model"], revision=rev)
    return model, tok, rev


def prepare(cfg, tok, cases) -> list[dict]:
    out = []
    for c in cases:
        text = tok.apply_chat_template([{"role": "user", "content": c["user_text"]}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        ref_ids = tok.apply_chat_template([{"role": "user", "content": c["user_text"]}], tokenize=True, add_generation_prompt=True)
        if list(ref_ids) != list(ids):
            raise AssertionError(f"{c['id']}: tokenization of the templated string differs from apply_chat_template")
        offs = enc["offset_mapping"]
        qb = text.index("QUESTION_BEGIN")
        L = next(i for i, (a, b) in enumerate(offs) if b > qb)          # first token touching QUESTION_BEGIN
        u0 = text.index(c["user_text"])
        ea, eb = u0 + c["evidence_char_start"], u0 + c["evidence_char_end"]
        ev = [i for i, (a, b) in enumerate(offs) if a < eb and b > ea and i < L]
        gold = tok(c["code"], add_special_tokens=False)["input_ids"]
        out.append({"id": c["id"], "target_tokens": c["target_tokens"], "position": c["position"],
                    "distractors": c["distractors"], "code": c["code"], "ids": ids, "L": L, "S": len(ids) - L,
                    "evidence_positions": ev, "gold_ids": gold, "boundary_exact": offs[L][0] == qb})
    return out


# ---------------------------------------------------------------------------------- generation helpers
def kv_list(cache):
    return [(lay.keys, lay.values) for lay in cache.layers]


def clone_cache(cache):
    return copy.deepcopy(cache)


@torch.no_grad()
def run_suffix(model, cache, suffix, pos0, phys0):
    dev = next(model.parameters()).device
    S = suffix.shape[1]
    return model(input_ids=suffix.to(dev), past_key_values=cache, use_cache=True,
                 position_ids=torch.arange(pos0, pos0 + S, device=dev)[None],
                 cache_position=torch.arange(phys0, phys0 + S, device=dev))


@torch.no_grad()
def greedy(model, cache, first_logits, pos0, phys0, max_new, eos_ids):
    dev = next(model.parameters()).device
    toks = [int(first_logits.argmax())]
    pos, phys = pos0, phys0
    while len(toks) < max_new and toks[-1] not in eos_ids:
        o = model(input_ids=torch.tensor([[toks[-1]]], device=dev), past_key_values=cache, use_cache=True,
                  position_ids=torch.tensor([[pos]], device=dev), cache_position=torch.tensor([phys], device=dev))
        toks.append(int(o.logits[0, -1].argmax()))
        pos += 1
        phys += 1
    return toks


@torch.no_grad()
def teacher_forced(model, cache, first_logits, gold, pos0, phys0):
    """Log-probs of the gold digits (first from the post-suffix logits) and the logits at each gold position."""
    dev = next(model.parameters()).device
    g = torch.tensor([gold], device=dev)
    o = model(input_ids=g[:, :-1], past_key_values=cache, use_cache=True,
              position_ids=torch.arange(pos0, pos0 + len(gold) - 1, device=dev)[None],
              cache_position=torch.arange(phys0, phys0 + len(gold) - 1, device=dev))
    logits = torch.cat([first_logits[None], o.logits[0].float()], 0)       # (len(gold), V)
    lp = torch.log_softmax(logits, -1)
    return lp, float(-lp[torch.arange(len(gold)), g[0]].sum())


def extract(text: str):
    m = re.search(r"(?<!\d)\d{5}(?!\d)", text)
    return m.group(0) if m else None


def kl(lp_ref, lp):
    return float((lp_ref.exp() * (lp_ref - lp)).sum(-1).mean())


# ------------------------------------------------------------------------------------- question votes
@torch.no_grad()
def question_votes(model, ctx_cache, suffix, L, pool_kernel):
    """SnapKV-inspired fixed policy: run the suffix once on a copy of the full FP16 context cache, capture each
    layer's rotary-positioned query projections, compute masked FP32 attention against the cached rotary keys,
    sum the attention mass on each context position over the suffix queries and over the query heads that share
    a KV head, then 1-D max-pool (kernel 7). Returns votes (n_layers, H_kv, L)."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    cache = clone_cache(ctx_cache)
    caps, hooks = {}, []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, args, kwargs, l=l):
            caps[l] = (kwargs.get("hidden_states", args[0] if args else None), kwargs["position_embeddings"])
        hooks.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        run_suffix(model, cache, suffix, L, L)
    finally:
        for h in hooks:
            h.remove()
    c = model.config
    Hq, Hkv = c.num_attention_heads, c.num_key_value_heads
    dh = getattr(c, "head_dim", None) or c.hidden_size // Hq
    S = suffix.shape[1]
    votes = []
    for l, layer in enumerate(model.model.layers):
        hs, (cos, sin) = caps[l]
        q = layer.self_attn.q_proj(hs).view(1, S, Hq, dh).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        K = cache.layers[l].keys                                            # (1, Hkv, L+S, dh), rotary applied
        K = K.repeat_interleave(Hq // Hkv, dim=1)
        s = torch.matmul(q.float(), K.float().transpose(-1, -2)) / dh ** 0.5   # (1,Hq,S,L+S)
        causal = torch.ones(S, L + S, dtype=torch.bool, device=s.device).tril(L)
        a = torch.softmax(s.masked_fill(~causal, float("-inf")), -1)[..., :L]  # mass on context positions
        v = a.sum(2).view(Hkv, Hq // Hkv, L).sum(1)                          # (Hkv, L)
        v = F.max_pool1d(v[:, None], pool_kernel, stride=1, padding=pool_kernel // 2)[:, 0]
        votes.append(v)
    del cache
    return torch.stack(votes)


# ------------------------------------------------------------------------------------------- policies
def evict_sizes(cfg, L, budget, n, Hkv, dh):
    """Largest legal retained count per (layer, KV head) for the two eviction rules under the byte budget."""
    pc = cfg["part_c"]
    per_pos = 2 * 2 * dh                             # K + V, FP16, per (layer, head)
    nh = n * Hkv
    n_sr = min(L, budget // (nh * per_pos))          # sink+recent: implicit positions, no indices
    safe = pc["sink"] + pc["recent"]
    n_qa = min(L, (budget // nh + safe * pc["index_bytes"]) // (per_pos + pc["index_bytes"]))
    return int(n_sr), int(n_qa)


def hybrid_size(cfg, L, budget, n, Hkv, dh):
    """Largest number M of extra FP16 (voted) positions such that FP16 (sink+recent+M+INT2 remainder) +
    INT2 complete 64-groups + metadata + int16 indices of the M voted positions fit the budget."""
    pc = cfg["part_c"]
    b = pc["bits_hybrid_low"]
    safe = pc["sink"] + pc["recent"]
    per_group = 2 * dh * (KQ.G * b // 8 + 4)        # K channel-wise (dh groups) + V token-wise (64 x dh/64 groups)
    best = None
    for M in range(0, max(0, L - safe) + 1):
        R = L - safe - M
        Gc, r = R // KQ.G, R % KQ.G
        per = (safe + M + r) * 2 * 2 * dh + Gc * per_group + M * pc["index_bytes"]
        if per * n * Hkv <= budget:
            best = (M, Gc, r, per * n * Hkv)
    return best


def select_positions(L, n_keep, votes_h, sink, recent):
    """Indices (sorted) kept for one head: sink + recent + top votes (votes_h None -> fill with most recent)."""
    base = set(range(min(sink, L))) | set(range(max(0, L - recent), L))
    extra = n_keep - len(base)
    if extra <= 0:
        return sorted(base)[:n_keep] if n_keep < len(base) else sorted(base)
    mid = torch.arange(min(sink, L), max(min(sink, L), L - recent))
    if votes_h is None:
        chosen = mid[-extra:].tolist()
    else:
        chosen = mid[torch.topk(votes_h[mid], min(extra, len(mid))).indices.cpu()].tolist()
    return sorted(base | set(chosen))


def apply_policy(name, cfg, ctx_kv, L, votes, budget, dims):
    """Returns (cache, info). ctx_kv: list of (K, V) (1, H_kv, L, d) FP16 context tensors (not modified)."""
    pc = cfg["part_c"]
    n, Hkv, dh = dims
    info = {"policy": name}
    if name in ("sink_recent", "question_aware"):
        n_sr, n_qa = evict_sizes(cfg, L, budget, n, Hkv, dh)
        N = n_sr if name == "sink_recent" else n_qa
        kv, keep = [], []
        for l, (k, v) in enumerate(ctx_kv):
            idx = [select_positions(L, N, None if name == "sink_recent" else votes[l, h], pc["sink"], pc["recent"])
                   for h in range(Hkv)]
            it = torch.tensor(idx, device=k.device)                          # (Hkv, N)
            g = it[None, :, :, None].expand(1, Hkv, it.shape[1], dh)
            kv.append((torch.gather(k, 2, g).contiguous(), torch.gather(v, 2, g).contiguous()))
            keep.append(it.cpu())
        info.update({"retained_per_head": N, "bytes": n * Hkv * N * 4 * dh +
                     (n * Hkv * max(0, N - pc["sink"] - pc["recent"]) * pc["index_bytes"] if name == "question_aware" else 0),
                     "keep": torch.stack(keep)})                             # (n, Hkv, N)
        return KQ.build_cache(kv), info
    if name.startswith("int4_"):
        layout = name[len("int4_"):]
        kv, qend, errs = [], [], []
        for k, v in ctx_kv:
            kh, vh, lq = KQ.quantize_kv(k, v, layout, 4)
            errs.append(((kh.float() - k.float()).pow(2).mean() / k.float().pow(2).mean().clamp_min(1e-30),
                         (vh.float() - v.float()).pow(2).mean() / v.float().pow(2).mean().clamp_min(1e-30)))
            kv.append((kh, vh))
            qend.append(lq)
        cache = KQ.build_cache(kv, KQ.TailQuantCache).setup(layout, 4, qend)
        packed = sum(KQ.encoded_bytes(KQ.encode_layer(k, v, layout, 4)) for k, v in ctx_kv)   # actual bit-packed bytes
        if packed != KQ.bytes_quant(L, 4, n, Hkv, dh):
            raise AssertionError(f"packed bytes {packed} != formula {KQ.bytes_quant(L, 4, n, Hkv, dh)}")
        info.update({"bytes": packed, "k_rel_mse": [float(e[0]) for e in errs],
                     "v_rel_mse": [float(e[1]) for e in errs]})
        return cache, info
    if name == "hybrid":
        M, Gc, r, nbytes = hybrid_size(cfg, L, budget, n, Hkv, dh)
        b = pc["bits_hybrid_low"]
        kv, fp16_sets, errs = [], [], []
        for l, (k, v) in enumerate(ctx_kv):
            kh, vh = k.clone(), v.clone()
            sets = []
            for h in range(Hkv):
                hp = select_positions(L, pc["sink"] + pc["recent"] + M, votes[l, h], pc["sink"], pc["recent"])
                low = sorted(set(range(L)) - set(hp))
                lowq, rem = low[:Gc * KQ.G], low[Gc * KQ.G:]                 # incomplete remainder stays FP16
                if lowq:
                    li = torch.tensor(lowq, device=k.device)
                    kk = k[0, h, li].reshape(Gc, KQ.G, dh)                    # channel-wise: 64 tokens per channel
                    kh[0, h, li] = KQ.qdq(kk.transpose(-1, -2), b).transpose(-1, -2).reshape(-1, dh)
                    vv = v[0, h, li].reshape(-1, dh // KQ.G, KQ.G)            # token-wise: 64 channels per token
                    vh[0, h, li] = KQ.qdq(vv, b).reshape(-1, dh)
                sets.append(sorted(hp + rem))
            errs.append(((kh.float() - k.float()).pow(2).mean() / k.float().pow(2).mean(),
                         (vh.float() - v.float()).pow(2).mean() / v.float().pow(2).mean()))
            kv.append((kh, vh))
            fp16_sets.append(sets)
        fp16 = torch.zeros(n, Hkv, L, dtype=torch.bool)
        for l, sets in enumerate(fp16_sets):
            for h, s in enumerate(sets):
                fp16[l, h, s] = True
        info.update({"bytes": nbytes, "voted_fp16_per_head": M, "int2_groups_per_head": Gc, "fp16_remainder_per_head": r,
                     "fp16_positions_per_head": int(fp16[0, 0].sum()), "fp16_mask": fp16,
                     "k_rel_mse": [float(e[0]) for e in errs], "v_rel_mse": [float(e[1]) for e in errs]})
        return KQ.build_cache(kv), info
    raise ValueError(name)


# ------------------------------------------------------------------------------------------ main loop
def evidence_retained(mask_or_keep, ev, L, is_keep):
    """Fraction of evidence tokens retained, averaged over (layer, KV head); also the share retained everywhere."""
    if not ev:
        return None, None
    if is_keep:
        n, H, N = mask_or_keep.shape
        m = torch.zeros(n, H, L, dtype=torch.bool)
        m.scatter_(2, mask_or_keep.long(), True)
    else:
        m = mask_or_keep
    e = m[:, :, ev].float()
    return float(e.mean()), float(e.all(0).all(0).float().mean())


@torch.no_grad()
def run_prompt(model, tok, cfg, p, eos_ids, keep_masks=False):
    pc = cfg["part_c"]
    dev = next(model.parameters()).device
    c = model.config
    n, Hkv = c.num_hidden_layers, c.num_key_value_heads
    dh = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    ids = torch.tensor([p["ids"]], device=dev)
    L, S = p["L"], p["S"]
    ctx, suffix = ids[:, :L], ids[:, L:]
    rec = {k: p[k] for k in ("id", "target_tokens", "position", "distractors", "code", "L", "S")}
    rec["n_evidence_tokens"] = len(p["evidence_positions"])
    # 1. prefill (shared)
    reset_peak()
    sync()
    t0 = time.perf_counter()
    ctx_cache = DynamicCache()
    model(input_ids=ctx, past_key_values=ctx_cache, use_cache=True, logits_to_keep=1)
    sync()
    rec["prefill_s"] = time.perf_counter() - t0
    rec["prefill_peak_allocated"] = gpu_mem().get("max_allocated_bytes")
    budget = KQ.bytes_quant(L, pc["bits_ref"], n, Hkv, dh)
    full = KQ.bytes_fp16(L, n, Hkv, dh)
    rec.update({"budget_bytes": budget, "fp16_context_bytes": full, "budget_frac_fp16": budget / full})
    ctx_kv = [(k.clone(), v.clone()) for k, v in kv_list(ctx_cache)]
    # 2. FP16 reference
    res = {}

    def finish(name, cache, phys0, extra):
        sync()
        t1 = time.perf_counter()
        o = run_suffix(model, cache, suffix, L, phys0)
        sync()
        t_suffix = time.perf_counter() - t1
        first = o.logits[0, -1].float()
        tf_cache = clone_cache(cache)
        t2 = time.perf_counter()
        gen = greedy(model, cache, first, L + S, phys0 + S, pc["max_new_tokens"], eos_ids)
        sync()
        t_dec = time.perf_counter() - t2
        lp_tf, nll = teacher_forced(model, tf_cache, first, p["gold_ids"], L + S, phys0 + S)
        text = tok.decode(gen, skip_special_tokens=True)
        ans = extract(text)
        ctx_bytes = extra.get("bytes", full)
        appended = cache.get_seq_length() - phys0          # suffix + fed generated tokens (the last one is never fed)
        if isinstance(cache, KQ.TailQuantCache):            # completed 64-token blocks of the tail were quantized too
            live = KQ.bytes_quant_split(cache.qend[0], cache.get_seq_length(), cache.bits, n, Hkv, dh)
        else:
            live = ctx_bytes + KQ.bytes_fp16(appended, n, Hkv, dh)
        r = {"answer_text": text, "answer": ans, "exact_match": int(ans == p["code"]), "gold_nll": nll,
             "suffix_s": t_suffix, "decode_s": t_dec, "decode_tokens": len(gen), "first_token": gen[0],
             "peak_allocated_policy": gpu_mem().get("max_allocated_bytes"),
             "shared_suffix_and_generated_fp16_bytes": KQ.bytes_fp16(appended, n, Hkv, dh),
             "appended_tokens": appended, "total_live_cache_bytes_end": live,
             "decode_time_cache_bytes_beyond_context": live - ctx_bytes}
        del tf_cache
        return r, torch.log_softmax(first, -1), lp_tf, gen

    reset_peak()
    r_ref, lp0_ref, lptf_ref, gen_ref = finish("fp16", KQ.build_cache([(k.clone(), v.clone()) for k, v in ctx_kv]), L, {})
    r_ref.update({"bytes": full, "frac_fp16_bytes": 1.0})
    res["fp16"] = r_ref
    # 3. question votes (used by question_aware and hybrid); time = scoring overhead
    sync()
    t0 = time.perf_counter()
    votes = question_votes(model, ctx_cache, suffix, L, pc["snapkv_pool_kernel"])
    sync()
    t_vote = time.perf_counter() - t0
    del ctx_cache
    masks = {}
    for name in POLICIES:
        reset_peak()                       # peak covers compression + suffix + decoding of this policy
        sync()
        t0 = time.perf_counter()
        cache, info = apply_policy(name, cfg, ctx_kv, L, votes, budget, (n, Hkv, dh))
        sync()
        t_comp = time.perf_counter() - t0
        phys0 = cache.get_seq_length()
        r, lp0, lptf, gen = finish(name, cache, phys0, info)
        r["compress_s"] = t_comp
        r["scoring_s"] = t_vote if name in ("question_aware", "hybrid") else 0.0
        r["bytes"] = info["bytes"]
        r["frac_fp16_bytes"] = info["bytes"] / full
        r["within_budget"] = info["bytes"] <= budget
        r["post_prefill_kl"] = kl(lp0_ref[None], lp0[None])
        r["teacher_forced_kl"] = kl(lptf_ref, lptf)
        r["same_first_token_as_fp16"] = int(gen[0] == gen_ref[0])
        for k in ("retained_per_head", "voted_fp16_per_head", "int2_groups_per_head", "fp16_remainder_per_head",
                  "fp16_positions_per_head", "k_rel_mse", "v_rel_mse"):
            if k in info:
                r[k] = info[k]
        if "keep" in info:
            r["evidence_retained_mean"], r["evidence_retained_all_heads"] = evidence_retained(info["keep"], p["evidence_positions"], L, True)
            masks[name] = info["keep"]
        if "fp16_mask" in info:
            r["evidence_fp16_mean"], r["evidence_fp16_all_heads"] = evidence_retained(info["fp16_mask"], p["evidence_positions"], L, False)
            masks[name] = info["fp16_mask"]
        res[name] = r
        del cache
    rec["policies"] = res
    rec["vote_evidence_rank"] = _evidence_rank(votes, p["evidence_positions"])
    free_cuda()
    return rec, (masks, votes.cpu()) if keep_masks else None


def _evidence_rank(votes, ev):
    """Median (over layers and heads) percentile rank of the best evidence token among context votes."""
    if not ev:
        return None
    r = (votes[:, :, None, :] > votes[:, :, ev][..., None]).sum(-1).min(-1).values.float() / votes.shape[-1]
    return float(r.median())


# ------------------------------------------------------------------------------------------ validation
@torch.no_grad()
def validate(model, tok, cfg, prompts, eos_ids) -> dict:
    """C1: split prefill == full forward; 100 % retention through the cache-editing path == ordinary generation;
    eviction with preserved rotary positions == full cache with the same positions masked; finite logits."""
    dev = next(model.parameters()).device
    out = {"split_vs_full": [], "retain_all_vs_generate": [], "evict_vs_mask": []}
    for i, p in enumerate(prompts):
        ids = torch.tensor([p["ids"]], device=dev)
        L, S = p["L"], p["S"]
        full = model(input_ids=ids, use_cache=False, logits_to_keep=S).logits[0].float()
        c = DynamicCache()
        model(input_ids=ids[:, :L], past_key_values=c, use_cache=True, logits_to_keep=1)
        o = run_suffix(model, c, ids[:, L:], L, L).logits[0].float()
        out["split_vs_full"].append({"id": p["id"], "max_abs_logit_diff": float((full - o).abs().max()),
                                     "kl_last": kl(torch.log_softmax(full[-1:], -1), torch.log_softmax(o[-1:], -1)),
                                     "finite": bool(torch.isfinite(o).all() and torch.isfinite(full).all())})
        # 100 % retention via gather + explicit positions vs HF generate on the full prompt
        c2 = DynamicCache()
        model(input_ids=ids[:, :L], past_key_values=c2, use_cache=True, logits_to_keep=1)
        Hkv = model.config.num_key_value_heads
        allidx = torch.arange(L, device=dev)[None, None, :, None]
        kv = [(torch.gather(k, 2, allidx.expand(1, Hkv, L, k.shape[-1])), torch.gather(v, 2, allidx.expand(1, Hkv, L, v.shape[-1])))
              for k, v in kv_list(c2)]
        c3 = KQ.build_cache(kv)
        o3 = run_suffix(model, c3, ids[:, L:], L, L)
        mine = greedy(model, c3, o3.logits[0, -1].float(), L + S, L + S, cfg["part_c"]["max_new_tokens"], eos_ids)
        # plain greedy: the checkpoint's generation_config may set a repetition penalty / sampling defaults,
        # which generate() would otherwise apply even with do_sample=False (codes often repeat digits)
        ref = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=cfg["part_c"]["max_new_tokens"],
                             do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
        out["retain_all_vs_generate"].append({"id": p["id"], "identical_tokens": mine == ref[:len(mine)],
                                              "mine": tok.decode(mine), "generate": tok.decode(ref)})
        # eviction (shared positions) vs full cache with those positions masked
        if i < 6:
            keep = sorted(set(range(8)) | set(range(L // 3, L // 3 + 200)) | set(range(L - 64, L)))
            ki = torch.tensor(keep, device=dev)[None, None, :, None]
            kv = [(torch.gather(k, 2, ki.expand(1, Hkv, len(keep), k.shape[-1])), torch.gather(v, 2, ki.expand(1, Hkv, len(keep), v.shape[-1])))
                  for k, v in kv_list(c2)]
            oe = run_suffix(model, KQ.build_cache(kv), ids[:, L:], L, len(keep)).logits[0].float()
            c4 = DynamicCache()
            model(input_ids=ids[:, :L], past_key_values=c4, use_cache=True, logits_to_keep=1)
            allowed = torch.zeros(L + S, dtype=torch.bool, device=dev)
            allowed[keep] = True
            mask = torch.full((1, 1, S, L + S), float("-inf"), device=dev, dtype=model.dtype)
            for j in range(S):
                a = allowed.clone()
                a[L:L + j + 1] = True
                mask[0, 0, j, a] = 0
            om = model(input_ids=ids[:, L:], past_key_values=c4, use_cache=True, attention_mask=mask,
                       position_ids=torch.arange(L, L + S, device=dev)[None], cache_position=torch.arange(L, L + S, device=dev)).logits[0].float()
            out["evict_vs_mask"].append({"id": p["id"], "max_abs_logit_diff": float((oe - om).abs().max())})
        free_cuda()
    out["summary"] = {"split_max_abs_logit_diff": max(x["max_abs_logit_diff"] for x in out["split_vs_full"]),
                      "all_finite": all(x["finite"] for x in out["split_vs_full"]),
                      "retain_all_identical_fraction": float(np.mean([x["identical_tokens"] for x in out["retain_all_vs_generate"]])),
                      "evict_vs_mask_max_abs_logit_diff": max([x["max_abs_logit_diff"] for x in out["evict_vs_mask"]] or [0.0])}
    return out


# ----------------------------------------------------------------------------------------- C5 bench
@torch.no_grad()
def bench_packed(model, cfg, prompts, device) -> dict:
    """Encode (quantize + bit-pack) and decode (unpack + dequantize) of a whole context cache, serialized bytes,
    and one decode step with an FP16 cache vs with unpack-then-SDPA, at batch 1 and 8 and L = 2,048 / 4,096."""
    bc = cfg["part_c"]["bench"]
    c = model.config
    n, Hkv = c.num_hidden_layers, c.num_key_value_heads
    dh = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    rows = []

    def timed(fn):
        for _ in range(bc["warmup"]):
            fn()
        sync()
        ms = []
        for _ in range(bc["reps"]):
            if device == "cuda":
                a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                ms.append(a.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                fn()
                ms.append(1000 * (time.perf_counter() - t0))
        return {"median_ms": float(np.median(ms)), "mean_ms": float(np.mean(ms)), "std_ms": float(np.std(ms, ddof=1)),
                "reps": len(ms), "all_ms": ms}

    longest = max(prompts, key=lambda p: p["L"])
    ids = torch.tensor([longest["ids"][:longest["L"]]], device=device)
    base = DynamicCache()
    model(input_ids=ids, past_key_values=base, use_cache=True, logits_to_keep=1)
    src = kv_list(base)
    for L in bc["lengths"]:
        for B in bc["batches"]:
            rec = {"L": L, "batch": B}
            try:
                reps_ = -(-L // src[0][0].shape[2])
                kv = [(k.repeat(1, 1, reps_, 1)[:, :, :L].expand(B, -1, -1, -1).contiguous(),
                       v.repeat(1, 1, reps_, 1)[:, :, :L].expand(B, -1, -1, -1).contiguous()) for k, v in src]
                tok1 = torch.full((B, 1), int(ids[0, -1]), device=device)
                posid = torch.full((B, 1), L, device=device)
                cpos = torch.tensor([L], device=device)

                def step_fp16():
                    cc = KQ.build_cache([(k, v) for k, v in kv])
                    model(input_ids=tok1, past_key_values=cc, use_cache=True, position_ids=posid, cache_position=cpos)
                rec["decode_step_fp16"] = timed(step_fp16)
                rec["fp16_bytes"] = KQ.bytes_fp16(L, n, Hkv, dh) * B
                for layout, b in [("channelK_tokenV", 4), ("tokenwise", 4), ("channelK_tokenV", 2)]:
                    key = f"{layout}_int{b}"
                    enc = [KQ.encode_layer(k, v, layout, b) for k, v in kv]
                    rec[f"{key}_serialized_bytes"] = sum(KQ.encoded_bytes(e) for e in enc)
                    rec[f"{key}_formula_bytes"] = KQ.bytes_quant(L, b, n, Hkv, dh) * B
                    rec[f"{key}_encode"] = timed(lambda: [KQ.encode_layer(k, v, layout, b) for k, v in kv])
                    rec[f"{key}_decode"] = timed(lambda: [KQ.decode_layer(e, b) for e in enc])

                    def step_packed():
                        dec = [KQ.decode_layer(e, b) for e in enc]
                        cc = KQ.build_cache(dec)
                        model(input_ids=tok1, past_key_values=cc, use_cache=True, position_ids=posid, cache_position=cpos)
                    rec[f"{key}_decode_step_with_unpack"] = timed(step_packed)
                    dk, dv = KQ.decode_layer(enc[0], b)
                    rec[f"{key}_roundtrip_layer0_k_rel_mse"] = float((dk.float() - kv[0][0].float()).pow(2).mean() / kv[0][0].float().pow(2).mean())
                    del enc
                    free_cuda()
            except RuntimeError as e:
                rec["error"] = "OOM" if "out of memory" in str(e).lower() else repr(e)[:200]
            rows.append(rec)
            free_cuda()
            log(f"C5 bench L={L} B={B} done")
    return {"rows": rows, "unit": "ms", "note": "content = real context cache of the longest prompt tiled to L; "
            "decode step includes rebuilding the cache object; packed = unpack all layers then ordinary SDPA"}


# ------------------------------------------------------------------------------------------- driver
def run_part_c(cfg, device, out_dir: Path, limit=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(data_repo(cfg) / cfg["data"]["kv_cases"])
    if limit:
        cases = cases[:limit]
    model, tok, rev = load_model(cfg, device)
    eos_ids = {tok.convert_tokens_to_ids(t) for t in EOS_TEXTS if t in tok.get_vocab()}
    prompts = prepare(cfg, tok, cases)
    save_json(out_dir / "prompts_meta.json", {"model_revision": rev, "eos_ids": sorted(eos_ids), "prompts": [
        {k: p[k] for k in ("id", "target_tokens", "position", "distractors", "L", "S", "evidence_positions", "gold_ids",
                           "boundary_exact")} | {"total_tokens": len(p["ids"])} for p in prompts]})
    if not (out_dir / "validation.json").exists():
        vp = [p for p in prompts if p["id"].endswith("-00")] or prompts[:2]      # first case of each of the 18 cells
        v = validate(model, tok, cfg, vp, eos_ids)
        save_json(out_dir / "validation.json", v)
        log(f"C1 validation: {v['summary']}")
    res_path = out_dir / "per_prompt.jsonl.gz"
    done = {r["id"]: r for r in (read_jsonl(res_path) if res_path.exists() else [])}
    qual = set(cfg["part_c"]["qualitative_ids"])
    (out_dir / "qualitative").mkdir(exist_ok=True)
    rows = list(done.values())
    for i, p in enumerate(prompts):
        if p["id"] in done:
            continue
        rec, extra = run_prompt(model, tok, cfg, p, eos_ids, keep_masks=p["id"] in qual)
        rows.append(rec)
        if extra is not None:
            masks, votes = extra
            np.savez_compressed(out_dir / "qualitative" / f"{p['id']}.npz", votes=votes.numpy().astype(np.float16),
                                **{f"mask_{k}": v.numpy() if v.dtype == torch.bool else v.numpy().astype(np.int16) for k, v in masks.items()},
                                evidence=np.array(p["evidence_positions"]), ids=np.array(p["ids"]))
        if (i + 1) % 10 == 0 or i == len(prompts) - 1:
            save_jsonl_gz(res_path, rows)
            log(f"Part C {i + 1}/{len(prompts)} prompts")
    save_jsonl_gz(res_path, rows)
    if (device == "cuda" or cfg["part_c"].get("bench_on_cpu")) and not (out_dir / "bench_packed.json").exists():
        save_json(out_dir / "bench_packed.json", bench_packed(model, cfg, prompts, device))
