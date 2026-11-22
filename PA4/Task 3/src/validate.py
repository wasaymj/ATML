"""Step 1 (manual): validate the exact speculative sampler before any timing.

toy        100,000 fixed-seed length-3 sequences on a 3-symbol second-order Markov target/draft; Pearson chi-square
           on the 27 joint categories (correct sampler at gamma 1/2/4 must not be rejected at p < 0.01; the incorrect
           sampler that resamples from p instead of the residual must be rejected).
tokenizer  target/draft token IDs and chat-template strings agree on all 920 prompts.
split      real-model split-prefill equivalence: one full forward vs prefill + token-by-token cached decoding.
cache      cache-position consistency inside the speculative sampler: every target/draft distribution the sampler
           uses is recomputed from scratch on the uncached token history and compared.
rounds     fully accepted rounds and the bonus token (draft = target), one-token rounds (a flattened draft),
           variable proposal length / final-token truncation (ignore EOS, cap not a multiple of gamma + 1),
           EOS handling (short-form prompts: EOS ends the output and never appears earlier).
padding    left-padded batched forward vs unpadded single forward (our samplers are batch-1 and never pad).
consistency  real-model distributional check (extra): for 4 prompts, 400 target-only vs 400 speculative samples of the
           first 3 tokens; chi-square homogeneity test on the first-token and on the 3-token joint counts.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import chi2_contingency

from .data import encode, eos_ids, test_prompts, train_pool
from .models import load_model, load_tokenizer, tokenizer_agreement
from .sampler import HFLM, sample_plain, sample_speculative
from .toy import run_chi2
from .utils import log, save_json


class CheckedLM(HFLM):
    """HFLM that recomputes every returned distribution from the uncached token history and records the error."""

    def __init__(self, model, temperature=1.0):
        super().__init__(model, temperature)
        self.hist, self.max_err = [], 0.0

    def reset(self):
        super().reset()
        self.hist = []

    def feed(self, tokens):
        probs = super().feed(tokens)
        self.hist.extend(int(t) for t in tokens)
        with torch.no_grad():
            full = self.model(input_ids=torch.tensor([self.hist], device=self.dev)).logits[0, -len(tokens):]
            ref = torch.softmax(full.float() / self.T, -1)
        self.max_err = max(self.max_err, float((ref - probs).abs().max()))
        return probs

    def crop(self, n):
        super().crop(n)
        del self.hist[n:]


class FlatLM(HFLM):
    """A deliberately poor draft: the model's distribution at temperature 8 (frequent rejections)."""

    def __init__(self, model):
        super().__init__(model, 8.0)


def _chi_homogeneity(a: list, b: list, min_count=5) -> dict:
    keys = sorted(set(a) | set(b))
    ca = {k: a.count(k) for k in keys}
    cb = {k: b.count(k) for k in keys}
    big = [k for k in keys if ca[k] + cb[k] >= 2 * min_count]
    rest = [k for k in keys if k not in big]
    tab = [[ca[k] for k in big] + ([sum(ca[k] for k in rest)] if rest else []),
           [cb[k] for k in big] + ([sum(cb[k] for k in rest)] if rest else [])]
    tab = np.array(tab)
    tab = tab[:, tab.sum(0) > 0]
    if tab.shape[1] < 2:
        return {"categories": int(tab.shape[1]), "p_value": 1.0, "note": "single category"}
    st = chi2_contingency(tab)
    return {"categories": int(tab.shape[1]), "chi2": float(st.statistic), "dof": int(st.dof), "p_value": float(st.pvalue)}


def run(cfg, out_path, dev="cuda") -> dict:
    sc = cfg["sampler_tests"]
    res = {}
    t0 = time.perf_counter()
    res["toy_chi2"] = run_chi2(sc["n_sequences"], sc["alpha"], cfg["seed"])
    log(f"toy chi2: passed={res['toy_chi2']['passed']} " + ", ".join(
        f"{k} p={v['p_value']:.3g}" for k, v in res["toy_chi2"]["runs"].items()))
    save_json(out_path, res)
    pool, tests = train_pool(cfg), test_prompts(cfg)
    res["tokenizer"] = tokenizer_agreement(cfg, [p["prompt"] for c in ("math", "short") for p in pool[c]] +
                                           [p["prompt"] for p in tests])
    log(f"tokenizer agreement: {res['tokenizer']['passed']}")
    tok = load_tokenizer(cfg, "target")
    eos = eos_ids(tok, cfg)
    target, draft = load_model(cfg, "target", dev), load_model(cfg, "draft", dev)
    assert target.config.vocab_size == draft.config.vocab_size, "target/draft vocabularies differ"
    res["vocab"] = {"target": target.config.vocab_size, "draft": draft.config.vocab_size}
    # ---- split-prefill equivalence
    split = []
    for p in tests[:3] + tests[60:62]:
        ids = encode(tok, p["prompt"])
        g = torch.Generator(device=dev).manual_seed(1)
        cont = sample_plain(HFLM(target), ids, 24, set(), g).tokens
        full_ids = ids + cont
        for name, m in (("target", target), ("draft", draft)):
            with torch.no_grad():
                full = torch.softmax(m(input_ids=torch.tensor([full_ids], device=dev)).logits[0].float(), -1)
            lm = HFLM(m)
            lm.reset()
            inc = [lm.feed(ids)]
            for t in cont[:-1]:
                inc.append(lm.feed([t]))
            inc = torch.cat(inc)
            split.append({"prompt": p["id"], "model": name, "max_abs_prob_diff": float((inc - full[:len(full_ids) - 1]).abs().max()),
                          "argmax_agreement": float((inc.argmax(-1) == full[:len(full_ids) - 1].argmax(-1)).float().mean())})
    res["split_prefill"] = {"rows": split, "max_abs_prob_diff": max(x["max_abs_prob_diff"] for x in split)}
    log(f"split-prefill max |dp| = {res['split_prefill']['max_abs_prob_diff']:.2e}")
    # ---- cache consistency inside the speculative sampler
    ck = []
    for p in tests[:3] + tests[60:63]:
        T, D = CheckedLM(target), CheckedLM(draft)
        g = torch.Generator(device=dev).manual_seed(7)
        r = sample_speculative(T, D, encode(tok, p["prompt"]), 4, 64, eos, g)
        ck.append({"prompt": p["id"], "target_max_err": T.max_err, "draft_max_err": D.max_err, "tokens": len(r.tokens),
                   "rounds": r.rounds, "accepted": r.accepted, "examined": r.examined})
    res["cache_consistency"] = {"rows": ck, "target_max_err": max(x["target_max_err"] for x in ck),
                                "draft_max_err": max(x["draft_max_err"] for x in ck)}
    log(f"cache consistency max |dp|: target {res['cache_consistency']['target_max_err']:.2e}, "
        f"draft {res['cache_consistency']['draft_max_err']:.2e}")
    # ---- round types, truncation, EOS
    rounds = {}
    agg = {"accepted": 0, "examined": 0, "bonus": 0, "rounds": 0, "fallbacks": 0}
    for p in tests[:4]:
        r = sample_speculative(HFLM(target), HFLM(target), encode(tok, p["prompt"]), 4, 48, eos,
                               torch.Generator(device=dev).manual_seed(3))
        for k, v in (("accepted", r.accepted), ("examined", r.examined), ("bonus", r.bonus), ("rounds", r.rounds),
                     ("fallbacks", r.residual_fallbacks)):
            agg[k] += v
    rounds["draft_equals_target"] = {**agg, "acceptance": agg["accepted"] / max(1, agg["examined"])}
    one = {"accepted_per_round": [], "rounds": 0}
    for p in tests[:4]:
        r = sample_speculative(HFLM(target), FlatLM(draft), encode(tok, p["prompt"]), 4, 48, eos,
                               torch.Generator(device=dev).manual_seed(4))
        one["accepted_per_round"] += r.accepted_per_round
        one["rounds"] += r.rounds
    hist = np.bincount(one["accepted_per_round"], minlength=5).tolist()
    rounds["flat_draft"] = {"rounds": one["rounds"], "accepted_per_round_histogram_0_to_4": hist,
                            "zero_acceptance_rounds": hist[0]}
    trunc = []
    for cap in (1, 2, 5, 7, 11):
        r = sample_speculative(HFLM(target), HFLM(draft), encode(tok, tests[0]["prompt"]), 4, cap, eos,
                               torch.Generator(device=dev).manual_seed(cap), ignore_eos=True)
        trunc.append({"cap": cap, "tokens": len(r.tokens), "rounds": r.rounds, "plain_steps": r.plain_steps,
                      "ok": len(r.tokens) == cap})
    rounds["final_token_truncation"] = trunc
    eos_rows = []
    for p in [x for x in tests if x["domain"] == "short"][:10]:
        for gm in (2, 4):
            r = sample_speculative(HFLM(target), HFLM(draft), encode(tok, p["prompt"]), gm, 32, eos,
                                   torch.Generator(device=dev).manual_seed(gm))
            inner = [t for t in r.tokens[:-1] if t in eos]
            eos_rows.append({"prompt": p["id"], "gamma": gm, "termination": r.termination, "n": len(r.tokens),
                             "last_is_eos": bool(r.tokens and r.tokens[-1] in eos), "eos_inside": len(inner)})
    rounds["eos"] = {"rows": eos_rows, "all_ok": all((x["termination"] == "eos") == x["last_is_eos"] and x["eos_inside"] == 0
                                                     for x in eos_rows)}
    res["rounds"] = rounds
    log(f"rounds: draft=target acceptance {rounds['draft_equals_target']['acceptance']:.4f}, flat-draft zero-acc "
        f"rounds {hist[0]}, truncation ok {all(x['ok'] for x in trunc)}, EOS ok {rounds['eos']['all_ok']}")
    # ---- padding (documentation: samplers are batch-1 and never pad)
    a, b = encode(tok, tests[0]["prompt"]), encode(tok, tests[60]["prompt"])
    n = max(len(a), len(b))
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    batch = torch.tensor([[pad] * (n - len(a)) + a, [pad] * (n - len(b)) + b], device=dev)
    mask = torch.tensor([[0] * (n - len(a)) + [1] * len(a), [0] * (n - len(b)) + [1] * len(b)], device=dev)
    with torch.no_grad():
        lb = torch.softmax(target(input_ids=batch, attention_mask=mask).logits[:, -1].float(), -1)
        la = torch.softmax(target(input_ids=torch.tensor([a], device=dev)).logits[0, -1].float(), -1)
        lbb = torch.softmax(target(input_ids=torch.tensor([b], device=dev)).logits[0, -1].float(), -1)
    res["padding"] = {"left_padded_batch_vs_single_max_abs_prob_diff": float(max((lb[0] - la).abs().max(), (lb[1] - lbb).abs().max())),
                      "note": "all samplers in this task run batch 1 without padding"}
    # ---- real-model distributional consistency (target-only vs speculative)
    rc = cfg["sampler_tests"]["real_consistency"]
    cons = []
    for j, p in enumerate(tests[:rc["prompts"] // 2] + tests[60:60 + rc["prompts"] - rc["prompts"] // 2]):
        ids = encode(tok, p["prompt"])
        A, B = [], []
        T, D = HFLM(target), HFLM(draft)
        for s in range(rc["samples"]):
            A.append(tuple(sample_plain(T, ids, rc["tokens"], set(), torch.Generator(device=dev).manual_seed(10_000 + s)).tokens))
            B.append(tuple(sample_speculative(T, D, ids, 4, rc["tokens"], set(), torch.Generator(device=dev).manual_seed(20_000 + s)).tokens))
        cons.append({"prompt": p["id"], "first_token": _chi_homogeneity([x[0] for x in A], [x[0] for x in B]),
                     "three_token_joint": _chi_homogeneity(A, B)})
    res["real_consistency"] = cons
    log("real-model consistency p-values: " + ", ".join(f"{c['prompt']} {c['first_token']['p_value']:.2f}/{c['three_token_joint']['p_value']:.2f}" for c in cons))
    res["seconds"] = time.perf_counter() - t0
    res["passed"] = bool(res["toy_chi2"]["passed"] and res["tokenizer"]["passed"] and rounds["eos"]["all_ok"]
                         and all(x["ok"] for x in trunc))
    save_json(out_path, res)
    return res
