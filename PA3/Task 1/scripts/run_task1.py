"""Task 1 command-line orchestrator.

Steps (run from the `PA3/Task 1` directory):
  preflight    fast end-to-end checks before spending GPU hours (see step_preflight)
  data         curate MetaMathQA 40k + tokenize, and sample the activation probe set
  train        train one method (--method) or all (--method all); resumable; --max_hours budget
  merge        merge LoRA-4 / LoRA-16 / OPLoRA-16 into fresh FP16 backbones + logit checks
  benchmarks   lm-eval on the 7 conditions (or --conditions a,b)
  qualitative  24 prompts x 7 conditions (or --conditions a,b), grouped outputs + annotation sheet
  diagnostics  rho_k, eta, R(z) and supporting weight-space measurements
  report       tables + figures (scripts/compile_task1_report.py)
  status       what is finished / missing
"""
import os
import sys
import shutil
import logging
import argparse
from typing import Any, Dict, List, Optional

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_ROOT)
if TASK_ROOT not in sys.path:
    sys.path.insert(0, TASK_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from src.utils import (atomic_json_dump, ensure_dir, environment_info, free_cuda, load_config,  # noqa: E402
                       load_json, setup_logging)

logger = logging.getLogger("run_task1")

METHODS = ["full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]
CONDITIONS = ["baseline", "full_ft_mid", "full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]
RESULTS = "results"


# --------------------------------------------------------------------------------------------
# Path resolution (checkpoints may be spread over several Kaggle input datasets)
# --------------------------------------------------------------------------------------------
def find_run_dir(roots: List[str], method: str) -> Optional[str]:
    for r in roots:
        d = os.path.join(r, method)
        if os.path.exists(os.path.join(d, "TRAIN_DONE")):
            return d
    for r in roots:                       # unfinished run (for status / resume messages)
        d = os.path.join(r, method)
        if os.path.isdir(d):
            return d
    return None


def export_dir(roots: List[str], method: str, update: int) -> Optional[str]:
    d = find_run_dir(roots, method)
    if d is None:
        return None
    p = os.path.join(d, f"update_{update:04d}")
    return p if os.path.exists(os.path.join(p, "DONE")) else None


def condition_model_path(cond: str, cfg: Dict[str, Any], roots: List[str], merged_root: str) -> Optional[str]:
    t = cfg["train"]
    if cond == "baseline":
        return cfg["model_name"]
    if cond == "full_ft_mid":
        return export_dir(roots, "full_ft", t["midpoint_update"])
    if cond in ("full_ft", "l2sp"):
        return export_dir(roots, cond, t["final_update"])
    p = os.path.join(merged_root, cond)
    return p if os.path.exists(os.path.join(p, "DONE")) else None


def parse_list(s: Optional[str], allowed: List[str]) -> List[str]:
    if not s or s == "all":
        return list(allowed)
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise SystemExit(f"Unknown names {bad}; allowed: {allowed}")
    return out


# --------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------
def step_data(cfg):
    from src.data_prep import create_activation_probe_set, prepare_metamathqa_subset
    d = cfg["data"]
    if os.path.exists(d["probe_file"]):
        logger.info("Probe set exists: %s", d["probe_file"])
    else:
        create_activation_probe_set(cfg)
    if os.path.exists(d["train_file"]) and os.path.exists(d["token_file"]):
        logger.info("Training data exists: %s", d["train_file"])
    else:
        prepare_metamathqa_subset(cfg)


def step_train(cfg, methods, ckpt_root, max_hours, stop_after=None):
    from src.train import EXIT_PAUSED, TrainingPaused, train_method
    if not os.path.exists(cfg["data"]["token_file"]):
        raise SystemExit("Run `--step data` first (token file missing).")
    for m in methods:
        try:
            logs = train_method(m, cfg, ckpt_root, max_hours=max_hours, stop_after=stop_after)
            free_cuda()
        except TrainingPaused as e:
            logger.warning("%s paused at update %s; re-run the same command to resume.", m, e)
            sys.exit(EXIT_PAUSED)
        collect_training_metrics([ckpt_root], [m])
        logger.info("%s finished (%.2f h).", m, logs["train_compute_sec"] / 3600)


def collect_training_metrics(roots, methods=METHODS):
    """Copy the small machine-readable training logs into results/ (committed to git)."""
    for m in methods:
        d = find_run_dir(roots, m)
        if d and os.path.exists(os.path.join(d, "training_metrics.json")):
            ensure_dir(os.path.join(RESULTS, "training"))
            shutil.copy(os.path.join(d, "training_metrics.json"), os.path.join(RESULTS, "training", f"{m}.json"))
            for fn in os.listdir(d):
                if fn.startswith("run_manifest"):
                    shutil.copy(os.path.join(d, fn), os.path.join(RESULTS, "training", f"{m}_{fn}"))


def step_merge(cfg, roots, merged_root):
    from src.models import merge_and_verify
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in ("lora_r4", "lora_r16", "oplora_r16"):
        out = os.path.join(merged_root, m)
        if os.path.exists(os.path.join(out, "DONE")):
            logger.info("Merged %s exists.", m)
            continue
        src = export_dir(roots, m, cfg["train"]["final_update"])
        if src is None:
            logger.error("Final checkpoint for %s not found in %s; skipping.", m, roots)
            continue
        bases = os.path.join(find_run_dir(roots, m), "oplora_bases.pt") if m.startswith("oplora") else None
        rep = merge_and_verify(cfg, m, src, out, bases, device)
        atomic_json_dump(rep, os.path.join(RESULTS, "merge", f"{m}.json"))
        free_cuda()


def step_benchmarks(cfg, conds, roots, merged_root, limit=None, out_root=None):
    from src.evaluate import run_benchmarks
    out_root = out_root or os.path.join(RESULTS, "benchmarks")
    done = {}
    for c in conds:
        path = condition_model_path(c, cfg, roots, merged_root)
        if path is None:
            logger.error("Model for condition %s not found; skipping.", c)
            continue
        done[c] = run_benchmarks(c, path, cfg, out_root, limit=limit)
        free_cuda()
    if limit is None:
        allsum = {}
        for c in CONDITIONS:
            p = os.path.join(out_root, c, "summary.json")
            if os.path.exists(p):
                allsum[c] = load_json(p)
        atomic_json_dump(allsum, os.path.join(out_root, "all_conditions_summary.json"))
    return done


def step_qualitative(cfg, conds, roots, merged_root):
    from src.evaluate import generate_qualitative, write_qualitative_outputs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_dir = ensure_dir(os.path.join(RESULTS, "qualitative", "raw"))
    for c in conds:
        out = os.path.join(raw_dir, f"{c}.json")
        if os.path.exists(out):
            continue
        path = condition_model_path(c, cfg, roots, merged_root)
        if path is None:
            logger.error("Model for condition %s not found; skipping.", c)
            continue
        atomic_json_dump(generate_qualitative(path, cfg, device), out)
    have = {c: load_json(os.path.join(raw_dir, f"{c}.json")) for c in CONDITIONS
            if os.path.exists(os.path.join(raw_dir, f"{c}.json"))}
    if len(have) == len(CONDITIONS):
        write_qualitative_outputs(have, cfg, os.path.join(RESULTS, "qualitative"), CONDITIONS)
    else:
        logger.info("Qualitative raw outputs for %d/7 conditions; grouping waits for all 7.", len(have))


def step_diagnostics(cfg, roots, merged_root):
    from src.diagnostics import (activation_ratios, cache_probe_activations, oplora_constraint_checks,
                                 summarize, verify_probe_prompts_against_baseline, weight_metrics)
    from src.models import load_adapter_deltas, load_base_model, load_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers, k = cfg["diagnostics"]["layers"], cfg["diagnostics"]["k"]
    targets = tuple(cfg["adapters"]["target_modules"])
    out_dir = ensure_dir(os.path.join(RESULTS, "diagnostics"))
    t = cfg["train"]

    base = load_base_model(cfg, torch.float32, device).eval()
    tok = load_tokenizer(cfg)
    W0 = {l: base.get_submodule(f"model.layers.{l}.mlp.down_proj").weight.detach().float().cpu().clone()
          for l in layers}
    w0_norms = {n: float(m.weight.detach().float().norm()) for n, m in base.named_modules()
                if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in targets}

    # Same starting-weight bases as OPLoRA (saved by its run); recomputed identically if absent.
    op_dir = find_run_dir(roots, "oplora_r16")
    bases_src = "oplora_bases.pt"
    Uk, Vk = {}, {}
    if op_dir and os.path.exists(os.path.join(op_dir, "oplora_bases.pt")):
        saved = torch.load(os.path.join(op_dir, "oplora_bases.pt"), map_location="cpu")["bases"]
        for l in layers:
            Uk[l], Vk[l] = saved[f"model.layers.{l}.mlp.down_proj"]
        del saved
    else:
        bases_src = "recomputed_fp32_svd"
    S0 = {}
    for l in layers:
        U, S, Vh = torch.linalg.svd(W0[l].double(), full_matrices=False)
        S0[l] = S[:k].float()
        if l not in Uk:
            Uk[l], Vk[l] = U[:, :k].float(), Vh[:k].T.float()

    # Activation cache (starting model, FP32). Prompts are replaced by the exact lm-eval baseline
    # prompts when those are available, and the agreement is recorded.
    probe = load_json(cfg["data"]["probe_file"])
    verification = verify_probe_prompts_against_baseline(probe, os.path.join(RESULTS, "benchmarks", "baseline"))
    gpath = os.path.join(RESULTS, "benchmarks", "baseline", "gsm8k_per_example.json")
    if os.path.exists(gpath):
        lm_prompts = {r["doc_id"]: r["prompt"] for r in load_json(gpath)}
        for p in probe["gsm8k"]:
            p["prompt"] = lm_prompts.get(p["doc_id"], p["prompt"])
    verification["bases_source"] = bases_src
    atomic_json_dump(verification, os.path.join(out_dir, "probe_and_basis_verification.json"))
    acts = cache_probe_activations(base, tok, probe, layers, device)
    del base
    free_cuda()

    conds = {"full_ft_mid": ("full", "full_ft", t["midpoint_update"]),
             "full_ft": ("full", "full_ft", t["final_update"]), "l2sp": ("full", "l2sp", t["final_update"]),
             "lora_r4": ("lora", "lora_r4", t["final_update"]), "lora_r16": ("lora", "lora_r16", t["final_update"]),
             "oplora_r16": ("oplora", "oplora_r16", t["final_update"])}
    summary, per_probe, layerwise, constraints = {}, {}, {}, {}
    for cond, (mtype, method, upd) in conds.items():
        d = export_dir(roots, method, upd)
        if d is None:
            logger.error("Checkpoint for %s missing; skipping diagnostics.", cond)
            continue
        dW = {}
        if mtype == "full":
            diag = torch.load(os.path.join(d, "diag_down_proj_fp32.pt"), map_location="cpu")
            for l in layers:
                dW[l] = diag[f"model.layers.{l}.mlp.down_proj.weight"].float() - W0[l]
            norms = load_json(os.path.join(d, "module_update_norms.json"))
            layerwise[cond] = {n: v["delta_fro"] / v["w0_fro"] for n, v in norms.items()}
        else:
            deltas = load_adapter_deltas(mtype, d)
            for l in layers:
                B, A, s = deltas[f"model.layers.{l}.mlp.down_proj"]
                dW[l] = s * (B @ A)
            layerwise[cond] = {n: float((s * (B @ A)).norm()) / w0_norms[n] for n, (B, A, s) in deltas.items()}
        summary[cond], per_probe[cond] = {}, {}
        for l in layers:
            wm = weight_metrics(W0[l], dW[l], Uk[l], Vk[l], S0[l])
            r_math = activation_ratios(W0[l], dW[l], acts["gsm8k"], l)
            r_non = activation_ratios(W0[l], dW[l], acts["hellaswag"], l)
            sm, sn = summarize(r_math), summarize(r_non)
            summary[cond][str(l)] = {**wm, "R_math": sm, "R_nonmath": sn,
                                     "R_math_over_nonmath": sm["mean"] / max(sn["mean"], 1e-30)}
            per_probe[cond][str(l)] = {
                "gsm8k": [{"doc_id": a["doc_id"], "R": r} for a, r in zip(acts["gsm8k"], r_math)],
                "hellaswag": [{"doc_id": a["doc_id"], "R": r} for a, r in zip(acts["hellaswag"], r_non)]}
        if mtype == "oplora":
            merged_w = {}
            mpath = os.path.join(merged_root, "oplora_r16", "model.safetensors")
            if os.path.exists(mpath):
                from safetensors import safe_open
                with safe_open(mpath, framework="pt") as f:
                    for l in layers:
                        merged_w[l] = f.get_tensor(f"model.layers.{l}.mlp.down_proj.weight").float()
            constraints = {str(l): oplora_constraint_checks(dW[l], Uk[l], Vk[l], W0[l], merged_w.get(l))
                           for l in layers}
        logger.info("Diagnostics %s: %s", cond, {l: (round(v["rho_k"], 4), round(v["eta"], 5))
                                                  for l, v in summary[cond].items()})
    atomic_json_dump(summary, os.path.join(out_dir, "diagnostics_summary.json"))
    atomic_json_dump(per_probe, os.path.join(out_dir, "activation_ratios_per_probe.json"))
    atomic_json_dump(layerwise, os.path.join(out_dir, "layerwise_eta.json"))
    if constraints:
        atomic_json_dump(constraints, os.path.join(out_dir, "oplora_constraint_verification.json"))


def step_preflight(cfg, quick_train: bool = True):
    """~10-15 min on a T4. Fails loudly on anything that would otherwise waste a long run."""
    import math
    from src.evaluate import run_benchmarks
    from src.models import (assert_matched_init, build_training_model, load_tokenizer, merge_and_verify,
                            save_oplora_checkpoint)
    from src.train import collate, response_loss_sum
    from src.data_prep import load_lm_eval_docs
    from src.utils import NO_THINK_SUFFIX, apply_chat
    report: Dict[str, Any] = {"environment": environment_info()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Environment: %s", report["environment"])
    if device.type != "cuda":
        logger.warning("No GPU visible: preflight will be slow and memory checks are skipped.")

    # 1) chat template (non-thinking) + lm-eval split sizes
    tok = load_tokenizer(cfg)
    p = apply_chat(tok, [{"role": "user", "content": "hi"}], True)
    assert p.endswith(NO_THINK_SUFFIX), p
    counts = {}
    for t, n in cfg["eval"]["expected_docs"].items():
        counts[t] = len(load_lm_eval_docs(t)[1])
        assert counts[t] == n, f"{t}: {counts[t]} docs, expected {n}"
    report["doc_counts"] = counts
    logger.info("lm-eval split sizes OK: %s", counts)

    # 2) the manual evaluates in FP16: check the baseline is numerically sane in FP16 vs FP32
    from src.models import load_base_model
    prompts = [apply_chat(tok, [{"role": "user", "content": "Q: Tom has 3 apples and buys 5 more. How many?\nA:"}], True),
               "Roof shingle removal: A man is sitting on a roof. He"]
    with torch.no_grad():
        m32 = load_base_model(cfg, torch.float32, device).eval()
        enc = [tok(p_, return_tensors="pt", add_special_tokens=False).to(device) for p_ in prompts]
        l32 = [m32(**e).logits.float() for e in enc]
        del m32
        free_cuda()
        m16 = load_base_model(cfg, torch.float16, device).eval()
        l16 = [m16(**e).logits.float() for e in enc]
        del m16
        free_cuda()
    finite = all(bool(torch.isfinite(x).all()) for x in l16)
    agree = float(torch.cat([(a.argmax(-1) == b.argmax(-1)).flatten() for a, b in zip(l16, l32)]).float().mean())
    report["fp16_vs_fp32"] = {"finite": finite, "top1_agreement": agree,
                              "max_abs_logit_diff": max(float((a - b).abs().max()) for a, b in zip(l16, l32))}
    logger.info("FP16 vs FP32 baseline: %s", report["fp16_vs_fp32"])
    if not finite or agree < 0.9:
        raise RuntimeError(f"FP16 baseline is not numerically sane: {report['fp16_vs_fp32']}")

    # 3) lm-eval end-to-end on 8 docs per task (baseline) into a scratch folder
    pre_dir = os.path.join(RESULTS, "_preflight")
    s = run_benchmarks("baseline", cfg["model_name"], cfg, pre_dir, limit=8)
    report["lm_eval_limit8"] = {k: s[k] for k in ("gsm8k_flexible_em", "hellaswag_acc_norm", "boolq_acc",
                                                  "arc_easy_acc_norm")}
    rows = load_json(os.path.join(pre_dir, "baseline", "gsm8k_per_example.json"))
    report["gsm8k_truncation_source"] = s["gsm8k"]["truncation_source"]
    report["gsm8k_example"] = {k: rows[0][k] for k in ("extracted_flexible", "gold", "stop_reason",
                                                       "n_generated_tokens")}
    logger.info("lm-eval smoke test OK: %s", report["lm_eval_limit8"])
    free_cuda()

    # 4) one optimizer-free forward/backward per method type, matched init check, merge round trip
    tokdata = torch.load(cfg["data"]["token_file"], map_location="cpu") if os.path.exists(cfg["data"]["token_file"]) else None
    if tokdata is not None:
        lens = [len(x) for x in tokdata["input_ids"]]
        order = sorted(range(len(lens)), key=lambda i: -lens[i])[:2]       # the two longest examples
        longest = collate([tokdata["input_ids"][i] for i in order], [tokdata["prompt_lens"][i] for i in order],
                          tok.pad_token_id)
        report["longest_microbatch_tokens"] = [lens[i] for i in order]
    tmp_root = ensure_dir(os.path.join("checkpoints", "_preflight"))
    lora_model = None
    for method in (["full_ft", "lora_r16", "oplora_r16"] if quick_train else []):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, info = build_training_model(method, cfg, device,
                                           oplora_bases_path=os.path.join(tmp_root, "oplora_bases.pt"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if info["type"] != "full":
            model.enable_input_require_grads()
        model.train()
        if tokdata is not None and device.type == "cuda":
            ids, lab, attn = (x.to(device) for x in longest)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = response_loss_sum(model, ids, lab, attn, cfg["train"]["ce_chunk_tokens"])
            loss.backward()
            assert math.isfinite(float(loss)), "non-finite loss"
            extra = 0
            if info["type"] == "full":   # + AdamW states (2 x FP32) + grads already allocated
                extra = 2 * info["trainable_params"] * 4
            peak = (torch.cuda.max_memory_allocated() + extra) / 2**30
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            report[f"memory_{method}_longest_batch_gb_est"] = peak
            logger.info("%s longest microbatch fwd/bwd OK; est. peak %.2f / %.2f GB", method, peak, total)
            if peak > 0.95 * total:
                raise RuntimeError(f"{method}: estimated peak memory {peak:.1f} GB exceeds the GPU")
        report[f"trainable_params_{method}"] = info["trainable_params"]
        if method == "lora_r16":
            lora_model = model
            continue
        if method == "oplora_r16":
            report["matched_init_modules"] = assert_matched_init(lora_model, model)
            # merge round trip with non-zero B so the check is meaningful
            with torch.no_grad():
                for n, prm in model.named_parameters():
                    if n.endswith("lora_B"):
                        prm.normal_(0, 1e-3)
            ck = ensure_dir(os.path.join(tmp_root, "oplora_r16", "update_1250"))
            save_oplora_checkpoint(model, ck, export_projected=True)
            del model, lora_model
            free_cuda()
            rep = merge_and_verify(cfg, "oplora_r16", ck, os.path.join(tmp_root, "merged_oplora"),
                                   os.path.join(tmp_root, "oplora_bases.pt"), device)
            report["merge_roundtrip_oplora"] = rep
            continue
        del model
        free_cuda()
    shutil.rmtree(tmp_root, ignore_errors=True)
    atomic_json_dump(report, os.path.join(RESULTS, "preflight_report.json"))
    logger.info("PREFLIGHT PASSED. Report: %s", os.path.join(RESULTS, "preflight_report.json"))


def step_status(cfg, roots, merged_root):
    print("== training ==")
    for m in METHODS:
        d = find_run_dir(roots, m)
        state = "missing"
        if d:
            if os.path.exists(os.path.join(d, "TRAIN_DONE")):
                state = "done"
            elif os.path.exists(os.path.join(d, "recovery", "state.pt")):
                st = torch.load(os.path.join(d, "recovery", "state.pt"), map_location="cpu", weights_only=False)
                state = f"resumable at update {st['update']}"
            else:
                state = "started (no recovery state)"
        print(f"  {m:12s} {state}   {d or ''}")
    print("== conditions ==")
    for c in CONDITIONS:
        p = condition_model_path(c, cfg, roots, merged_root)
        b = os.path.exists(os.path.join(RESULTS, "benchmarks", c, "summary.json"))
        q = os.path.exists(os.path.join(RESULTS, "qualitative", "raw", f"{c}.json"))
        print(f"  {c:12s} model={'yes' if p else 'NO '} benchmarks={'yes' if b else 'no '} qualitative={'yes' if q else 'no'}")
    print("diagnostics:", os.path.exists(os.path.join(RESULTS, "diagnostics", "diagnostics_summary.json")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=["preflight", "data", "train", "merge", "benchmarks",
                                                      "qualitative", "diagnostics", "report", "status", "collect"])
    ap.add_argument("--method", default="all", help="training method(s), comma-separated, or 'all'")
    ap.add_argument("--conditions", default="all", help="evaluation condition(s), comma-separated, or 'all'")
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--ckpt_root", default="checkpoints",
                    help="where training writes; for later steps a comma-separated list of roots to search")
    ap.add_argument("--merged_root", default=None, help="where merged adapter models live (default <first root>/merged)")
    ap.add_argument("--max_hours", type=float, default=None, help="stop training cleanly after this many hours")
    ap.add_argument("--stop_after", type=int, default=None, help="pause training after this update (resumable)")
    ap.add_argument("--skip_train_check", action="store_true", help="preflight: skip model/memory checks")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    roots = [r for r in args.ckpt_root.split(",") if r]
    merged_root = args.merged_root or os.path.join(roots[0], "merged")

    if args.step == "preflight":
        step_preflight(cfg, quick_train=not args.skip_train_check)
    elif args.step == "data":
        step_data(cfg)
    elif args.step == "train":
        step_train(cfg, parse_list(args.method, METHODS), roots[0], args.max_hours, args.stop_after)
    elif args.step == "merge":
        step_merge(cfg, roots, merged_root)
    elif args.step == "benchmarks":
        step_benchmarks(cfg, parse_list(args.conditions, CONDITIONS), roots, merged_root)
    elif args.step == "qualitative":
        step_qualitative(cfg, parse_list(args.conditions, CONDITIONS), roots, merged_root)
    elif args.step == "diagnostics":
        step_diagnostics(cfg, roots, merged_root)
    elif args.step == "collect":
        collect_training_metrics(roots)
    elif args.step == "report":
        import subprocess
        collect_training_metrics(roots)
        subprocess.run([sys.executable, os.path.join("scripts", "compile_task1_report.py")], check=True)
    elif args.step == "status":
        step_status(cfg, roots, merged_root)


if __name__ == "__main__":
    main()
