"""Quality evaluation shared by every condition: perplexity windows, HellaSwag log-likelihood scoring,
greedy continuations. All conditions use exactly this code, the same batches and the same scored tokens."""
from __future__ import annotations

import numpy as np
import torch

from .utils import log


@torch.no_grad()
def score_targets(model, seqs: list[list[int]], starts: list[int], max_tokens: int, logit_chunk: int,
                  pad_id: int) -> list[np.ndarray]:
    """For each sequence s and target positions t in [starts[i], len(s)), return log p(s[t] | s[:t]).

    Sequences are right-padded (no attention mask is needed: causal attention never lets a real token see a
    later pad). Batches are formed deterministically by length, so every model sees identical batches.
    Only the hidden states that predict scored targets go through the LM head (FP32 log-softmax)."""
    dev = next(model.parameters()).device
    order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]), i))
    out: list[np.ndarray | None] = [None] * len(seqs)
    b = 0
    while b < len(order):
        L = len(seqs[order[b]])
        n = max(1, max_tokens // L)
        idx = order[b:b + n]
        b += n
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
        x = x.to(dev)
        h = model.model(input_ids=x, use_cache=False).last_hidden_state          # final norm applied
        rows, cols, tgt = [], [], []
        for r, i in enumerate(idx):
            T = len(seqs[i])
            pos = torch.arange(starts[i], T)
            rows.append(torch.full_like(pos, r))
            cols.append(pos - 1)
            tgt.append(torch.tensor(seqs[i][starts[i]:T]))
        rows, cols, tgt = torch.cat(rows).to(dev), torch.cat(cols).to(dev), torch.cat(tgt).to(dev)
        sel = h[rows, cols]
        lp = torch.empty(sel.shape[0], dtype=torch.float32, device=dev)
        for c in range(0, sel.shape[0], logit_chunk):
            logits = model.lm_head(sel[c:c + logit_chunk]).float()
            lp[c:c + logit_chunk] = torch.log_softmax(logits, -1).gather(1, tgt[c:c + logit_chunk, None])[:, 0]
        lp = lp.cpu().numpy()
        k = 0
        for r, i in enumerate(idx):
            m = len(seqs[i]) - starts[i]
            out[i] = lp[k:k + m]
            k += m
        del h, sel
    return out


def perplexity(model, windows: list[dict], cfg, pad_id: int) -> dict:
    seqs = [w["token_ids"] for w in windows]
    lps = score_targets(model, seqs, [1] * len(seqs), cfg["eval"]["max_tokens_per_batch"],
                        cfg["eval"]["logit_chunk_tokens"], pad_id)
    nll_tok = np.concatenate([-x for x in lps]).astype(np.float32)
    win_sum = np.array([float(-x.astype(np.float64).sum()) for x in lps])
    win_n = np.array([len(x) for x in lps], dtype=np.int64)
    if not np.all(np.isfinite(nll_tok)):
        raise FloatingPointError("non-finite token log-probabilities")
    for w, n in zip(windows, win_n):
        assert n == w["n_scored"]
    N = int(win_n.sum())
    mean = float(win_sum.sum() / N)
    return {"N": N, "mean_nll": mean, "ppl": float(np.exp(mean)), "n_windows": len(windows),
            "_token_nll": nll_tok, "_win_sum": win_sum, "_win_n": win_n}


def _encode_pair(tok, context: str, continuation: str):
    """lm-evaluation-harness HFLM._encode_pair (add_special_tokens=False)."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]


def hellaswag_requests(tok, docs: list[dict]):
    seqs, starts, meta = [], [], []
    for d in docs:
        for j, ch in enumerate(d["choices"]):
            ctx, cont = _encode_pair(tok, d["query"], " " + ch)
            if len(cont) == 0:
                raise ValueError(f"empty continuation for HellaSwag order {d['order']}")
            seqs.append(ctx + cont)
            starts.append(len(ctx))
            meta.append((d["order"], j, len(cont)))
    return seqs, starts, meta


def hellaswag(model, tok, docs: list[dict], cfg, requests=None) -> dict:
    seqs, starts, meta = requests or hellaswag_requests(tok, docs)
    lps = score_targets(model, seqs, starts, cfg["eval"]["max_tokens_per_batch"], cfg["eval"]["logit_chunk_tokens"],
                        tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    per = []
    k = 0
    for d in docs:
        ll = []
        ntok = []
        for j in range(len(d["choices"])):
            assert meta[k][0] == d["order"] and meta[k][1] == j
            ll.append(float(lps[k].astype(np.float64).sum()))
            ntok.append(int(meta[k][2]))
            k += 1
        ll = np.array(ll)
        clen = np.array([float(len(c)) for c in d["choices"]])
        pred = int(np.argmax(ll))
        pred_n = int(np.argmax(ll / clen))
        per.append({"order": d["order"], "gold": d["gold"], "ll": ll.tolist(), "n_tokens": ntok,
                    "char_len": clen.astype(int).tolist(), "pred": pred, "pred_norm": pred_n,
                    "correct": int(pred == d["gold"]), "correct_norm": int(pred_n == d["gold"])})
    if not all(np.all(np.isfinite(p["ll"])) for p in per):
        raise FloatingPointError("non-finite HellaSwag log-likelihoods")
    acc = float(np.mean([p["correct"] for p in per]))
    accn = float(np.mean([p["correct_norm"] for p in per]))
    d512 = per[:512]
    return {"n": len(per), "acc": acc, "acc_norm": accn,
            "acc_first512": float(np.mean([p["correct"] for p in d512])),
            "acc_norm_first512": float(np.mean([p["correct_norm"] for p in d512])),
            "_per_example": per}


@torch.no_grad()
def continuations(model, tok, prompts: torch.Tensor, new_tokens: int) -> list[dict]:
    dev = next(model.parameters()).device
    x = prompts.to(dev)
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=new_tokens,
                         min_new_tokens=new_tokens, pad_token_id=tok.eos_token_id, use_cache=True)
    res = []
    for i in range(x.shape[0]):
        gen = out[i, x.shape[1]:].tolist()
        toks = gen
        bigrams = list(zip(toks, toks[1:]))
        res.append({"prompt_index": i, "prompt": tok.decode(x[i].tolist()), "continuation": tok.decode(gen),
                    "token_ids": gen, "distinct2": len(set(bigrams)) / max(1, len(bigrams))})
    return res
