"""Step 4: parallel sampling vs a single longer response vs sequential refinement (live, GPU).

Fixed 50-question MATH subset (first 50 evaluation ids of the seed-6304 permutation). Per question:
  single     1 response, max 3,072 new tokens                                  -> its boxed answer
  parallel   3 independent responses, max 1,024 each                           -> self-consistency over the 3
  refine     parallel response #0 (shared initial), then one critique-and-revision turn, max 2,048
                                                                               -> the revised boxed answer
Every call is independently seeded (SHA-256 of seed|question|call), batch size 1, FP16, no quantization,
temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0, native chat template with the system message
"You are a helpful assistant.". Gold answers are only used to grade afterwards.

Each finished call is appended to a JSONL checkpoint (resumable); `summarize_live` builds the results.
"""
import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .answers import NO_ANSWER, MathEquivalence, extract_last_boxed, group_math_answers
from .utils import task_path, load_json, atomic_json_dump, write_frame, stable_int, environment_info, ensure_dir

logger = logging.getLogger(__name__)


def build_live_questions(cfg) -> List[Dict[str, str]]:
    """The 50 live questions (id, original problem text, gold) from the supplied export, in permutation order."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    ids = load_json(task_path(cfg["paths"]["ids_dir"], "splits.json"))["live_math"]
    path = task_path(cfg["paths"]["data_dir"], cfg["datasets"]["math"]["folder"], "generations.parquet")
    t = pq.read_table(path, columns=["sample_id", "rollout_id", "question_text", "gold_answer"],
                      filters=[("rollout_id", "=", 0)])
    t = t.filter(pc.is_in(t["sample_id"], value_set=pa.array(ids)))
    by = {r["sample_id"]: r for r in t.to_pylist()}
    return [{"sample_id": q, "question": by[q]["question_text"], "gold": by[q]["gold_answer"]} for q in ids]


def live_questions(cfg) -> List[Dict[str, str]]:
    """Read data/ids/live_questions.json (written by `prepare`), so the GPU run does not need the export."""
    return load_json(task_path(cfg["paths"]["ids_dir"], "live_questions.json"))["questions"]


class Generator:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        lc = cfg["live"]
        self.torch, self.lc = torch, lc
        assert torch.cuda.is_available(), "live generation needs a GPU"
        self.device = torch.device("cuda:0")
        self.tok = AutoTokenizer.from_pretrained(lc["model"], revision=lc["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(lc["model"], revision=lc["revision"],
                                                          dtype=getattr(torch, lc["dtype"]),
                                                          attn_implementation=lc["attn_implementation"]).to(self.device)
        self.model.eval()
        self.eos = set(self.model.generation_config.eos_token_id if isinstance(
            self.model.generation_config.eos_token_id, list) else [self.model.generation_config.eos_token_id])
        self.eos |= {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.info = {"model": lc["model"], "revision": lc["revision"], "dtype": lc["dtype"],
                     "gpu": torch.cuda.get_device_name(0), "eos_ids": sorted(self.eos),
                     "load_peak_gb": torch.cuda.max_memory_allocated() / 2**30}

    def __call__(self, messages: List[Dict[str, str]], max_new_tokens: int, seed: int) -> Dict[str, Any]:
        torch, lc = self.torch, self.lc
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                      temperature=lc["temperature"], top_p=lc["top_p"], top_k=lc["top_k"],
                                      repetition_penalty=lc["repetition_penalty"], max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        gen = out[0, ids.shape[1]:].tolist()
        stopped = len(gen) > 0 and gen[-1] in self.eos
        kept = gen[:-1] if stopped else gen                       # stop token excluded (as in the export)
        text = self.tok.decode(kept, skip_special_tokens=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {"prompt": prompt, "input_tokens": int(ids.shape[1]), "generated_tokens": len(gen),
                "output_tokens": len(kept), "stopped_eos": bool(stopped),
                "truncated": bool(not stopped and len(gen) >= max_new_tokens), "max_new_tokens": max_new_tokens,
                "seed": int(seed), "time_s": dt, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                "response": text, "extracted": extract_last_boxed(text)}


def user_message(cfg, question: str) -> List[Dict[str, str]]:
    lc = cfg["live"]
    return [{"role": "system", "content": lc["system"]},
            {"role": "user", "content": question + lc["question_suffix_sep"] + lc["suffix"]}]


def call_plan(cfg, q: Dict[str, str]) -> List[Dict[str, Any]]:
    lc, s = cfg["live"], cfg["seed"]
    plan = [{"call": "single", "max_new_tokens": lc["single_max_new_tokens"],
             "seed": stable_int(s, q["sample_id"], "single", 0)}]
    plan += [{"call": f"parallel_{k}", "max_new_tokens": lc["parallel_max_new_tokens"],
              "seed": stable_int(s, q["sample_id"], "parallel", k)} for k in range(lc["parallel_n"])]
    plan += [{"call": "revision", "max_new_tokens": lc["revision_max_new_tokens"],
              "seed": stable_int(s, q["sample_id"], "revision", 0), "after": "parallel_0"}]
    return plan


def run_live(cfg: Dict[str, Any], ckpt_dir: str = "data/live_ckpt", max_questions: Optional[int] = None) -> None:
    ckpt = ensure_dir(task_path(ckpt_dir))
    path = os.path.join(ckpt, "live_calls.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[(r["sample_id"], r["call"])] = r
    qs = live_questions(cfg)[:max_questions]
    gen = None
    t_start = time.time()
    for qi, q in enumerate(qs):
        for c in call_plan(cfg, q):
            if (q["sample_id"], c["call"]) in done:
                continue
            if gen is None:
                gen = Generator(cfg)
                atomic_json_dump({**gen.info, "environment": environment_info()}, os.path.join(ckpt, "live_env.json"))
            msgs = user_message(cfg, q["question"])
            if c["call"] == "revision":
                init = done[(q["sample_id"], c["after"])]
                msgs = msgs + [{"role": "assistant", "content": init["response"]},
                               {"role": "user", "content": cfg["live"]["refine_instruction"]}]
            r = gen(msgs, c["max_new_tokens"], c["seed"])
            r.update({"sample_id": q["sample_id"], "call": c["call"], "question_index": qi})
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[(q["sample_id"], c["call"])] = r
            logger.info("[live] q%02d %s %s: %d out tokens, %.1f s, trunc=%s, boxed=%r", qi, q["sample_id"],
                        c["call"], r["output_tokens"], r["time_s"], r["truncated"], r["extracted"])
    logger.info("[live] all calls finished (%.0f min this session)", (time.time() - t_start) / 60)
    summarize_live(cfg, qs, done, ckpt)


def summarize_live(cfg, qs, done, ckpt) -> None:
    eq = MathEquivalence()
    rows, calls = [], []
    for q in qs:
        sid, gold = q["sample_id"], q["gold"]
        C = {c["call"]: done[(sid, c["call"])] for c in call_plan(cfg, q)}

        def ok(a):
            return a is not None and eq.equal(gold, a)
        for name, r in C.items():
            calls.append({k: v for k, v in r.items() if k not in ("prompt", "response")} | {"correct": ok(r["extracted"])})
        par = [C[f"parallel_{k}"] for k in range(cfg["live"]["parallel_n"])]
        gid, reps = group_math_answers([p["extracted"] for p in par], eq)
        counts = np.bincount(gid)
        win = next(g for g in gid if counts[g] == counts.max())               # tie -> earliest response
        par_ans = None if reps[win] == NO_ANSWER else reps[win]
        init, rev = C["parallel_0"], C["revision"]

        def agg(rs):
            return {"input_tokens": sum(r["input_tokens"] for r in rs), "output_tokens": sum(r["output_tokens"] for r in rs),
                    "time_s": sum(r["time_s"] for r in rs), "peak_gpu_gb": max(r["peak_gpu_gb"] for r in rs),
                    "n_truncated": sum(r["truncated"] for r in rs), "n_no_extraction": sum(r["extracted"] is None for r in rs),
                    "n_calls": len(rs)}
        for strat, ans, rs in (("single", C["single"]["extracted"], [C["single"]]), ("parallel", par_ans, par),
                               ("refine", rev["extracted"], [init, rev])):
            rows.append({"sample_id": sid, "strategy": strat, "answer": ans, "correct": ok(ans), **agg(rs)})
        rows[-1].update({"initial_answer": init["extracted"], "initial_correct": ok(init["extracted"]),
                         "revision_changed_answer": not (init["extracted"] is not None and rev["extracted"] is not None
                                                         and eq.equal(init["extracted"], rev["extracted"]))})
    import pandas as pd
    df = pd.DataFrame(rows)
    out = task_path(cfg["paths"]["results_dir"], "live")
    write_frame(df, os.path.join(out, "live_per_question.csv"))
    write_frame(pd.DataFrame(calls), os.path.join(out, "live_calls.csv"))
    # full prompts/responses (needed for the qualitative revision examples and reproducibility)
    with open(os.path.join(out, "live_calls_full.jsonl"), "w", encoding="utf-8") as f:
        for q in qs:
            for c in call_plan(cfg, q):
                f.write(json.dumps(done[(q["sample_id"], c["call"])], ensure_ascii=False) + "\n")
    B, seed = cfg["offline"]["bootstrap_resamples"], cfg["seed"]
    from .stats import boot_indices, mean_ci, paired_diff
    idx = boot_indices(len(qs), B, seed)
    S: Dict[str, Any] = {"n_questions": len(qs), "strategies": {}}
    piv = {s: df[df.strategy == s].set_index("sample_id").loc[[q["sample_id"] for q in qs]] for s in
           ("single", "parallel", "refine")}
    for s, d in piv.items():
        S["strategies"][s] = {"acc": mean_ci(d.correct.to_numpy(float), idx),
                              "mean_input_tokens": float(d.input_tokens.mean()),
                              "mean_output_tokens": float(d.output_tokens.mean()),
                              "total_output_tokens": int(d.output_tokens.sum()),
                              "mean_time_s": float(d.time_s.mean()), "total_time_s": float(d.time_s.sum()),
                              "peak_gpu_gb": float(d.peak_gpu_gb.max()),
                              "truncation_rate_calls": float(d.n_truncated.sum() / d.n_calls.sum()),
                              "no_extraction_rate_calls": float(d.n_no_extraction.sum() / d.n_calls.sum()),
                              "final_no_answer_rate": float(d.answer.isna().mean())}
    S["paired"] = {f"{a}-{b}": paired_diff(piv[a].correct.to_numpy(float), piv[b].correct.to_numpy(float), idx)
                   for a, b in (("parallel", "single"), ("refine", "single"), ("parallel", "refine"))}
    r = piv["refine"]
    S["revision_transitions"] = {"wrong_to_correct": int((~r.initial_correct & r.correct).sum()),
                                 "correct_to_wrong": int((r.initial_correct & ~r.correct).sum()),
                                 "correct_to_correct": int((r.initial_correct & r.correct).sum()),
                                 "wrong_to_wrong": int((~r.initial_correct & ~r.correct).sum()),
                                 "answer_changed": int(r.revision_changed_answer.sum()),
                                 "initial_acc": float(r.initial_correct.mean())}
    S["shared_initial_note"] = "parallel_0 is charged to both parallel and refine"
    if os.path.exists(os.path.join(ckpt, "live_env.json")):
        S["environment"] = load_json(os.path.join(ckpt, "live_env.json"))
    atomic_json_dump(S, os.path.join(out, "live_summary.json"))
    logger.info("[live] %s", {k: round(v["acc"][0], 3) for k, v in S["strategies"].items()})
