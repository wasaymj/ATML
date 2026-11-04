"""Task 2 command-line orchestrator.

Steps (run from the `PA3/Task 2` directory):
  data        download annotations, select every split (seed 6304), save ids + stats, download the selected images,
              write the blind label-audit sheets
  features    cache SigLIP features (FP16) for every selected image, then release the encoder
  preflight   fast checks before spending GPU hours (see step_preflight)
  train       three-stage curriculum for --run main | compress49 | all (resumable; --max_hours budget)
  final_eval  four image conditions x two final-test suites for --run; prior baseline; qualitative panels
  collect     copy the small training logs / dev outputs from the checkpoint folder into results/
  report      tables + figures (scripts/compile_task2_report.py)
  status      what is finished / missing
"""
import os
import sys
import shutil
import logging
import argparse
from typing import Any, Dict, List

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_ROOT)
if TASK_ROOT not in sys.path:
    sys.path.insert(0, TASK_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from src.utils import (atomic_json_dump, ensure_dir, environment_info, free_cuda, load_config,  # noqa: E402
                       load_json, setup_logging, task_path)

logger = logging.getLogger("run_task2")


def results_dir(cfg: Dict[str, Any]) -> str:
    return task_path(cfg["paths"].get("results_dir", "results"))


def parse_runs(s: str, cfg: Dict[str, Any]) -> List[str]:
    allowed = list(cfg["runs"])
    if not s or s == "all":
        return allowed
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise SystemExit(f"Unknown runs {bad}; allowed: {allowed}")
    return out


def final_weights(cfg: Dict[str, Any], roots: List[str], run: str):
    last = cfg["train"]["stages"][-1]
    sub = f"stage{len(cfg['train']['stages'])}_{last['name']}"
    for r in roots:
        p = os.path.join(r, run, sub)
        if os.path.exists(os.path.join(p, "DONE")):
            return os.path.join(p, "weights.pt")
    return None


# --------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------
def step_data(cfg, skip_images: bool = False):
    from transformers import AutoTokenizer
    from src.data_prep import all_image_keys, download_images, label_audit_sheet, prepare_data
    R = results_dir(cfg)
    build = task_path(cfg["paths"]["build_dir"])
    if os.path.exists(os.path.join(build, "vqa_test.json")) and os.path.exists(os.path.join(R, "data", "data_stats.json")):
        logger.info("Selected data exists: %s", build)
    else:
        tok = AutoTokenizer.from_pretrained(cfg["models"]["language"], revision=cfg["models"]["revision"])
        stats = prepare_data(cfg, tok)
        atomic_json_dump(stats, os.path.join(R, "data", "data_stats.json"))
        logger.info("data selected: %s", stats["checks"])
    if skip_images:
        return
    keys = all_image_keys(build)
    info = download_images(cfg, keys["train2014"] + keys["val2014"])
    atomic_json_dump({**info, "by_split": {k: len(v) for k, v in keys.items()}},
                     os.path.join(R, "data", "images.json"))
    sheet_dir = task_path("label_audit")
    if not os.path.exists(os.path.join(sheet_dir, "label_audit_blind.csv")):
        rep = label_audit_sheet(cfg, sheet_dir)
        shutil.copy(os.path.join(sheet_dir, "label_audit_key.json"), ensure_dir(os.path.join(R, "label_audit")))
        logger.info("label audit sheets: %s", rep)


def step_features(cfg):
    from src.features import cache_features
    rep = cache_features(cfg)
    path = os.path.join(results_dir(cfg), "resources", "feature_cache.json")
    if rep["splits"]:                       # do not overwrite the first (real) caching report with a no-op
        atomic_json_dump(rep, path)
    logger.info("feature cache: %s", rep)


def step_train(cfg, runs, ckpt_root, max_hours, stop_after=None):
    from src.train import EXIT_PAUSED, TrainingPaused, train_run
    for r in runs:
        try:
            train_run(r, cfg, ckpt_root, max_hours=max_hours, stop_after=stop_after)
            free_cuda()
        except TrainingPaused as e:
            logger.warning("%s paused at update %s; re-run the same command to resume.", r, e)
            collect(cfg, [ckpt_root], [r])
            sys.exit(EXIT_PAUSED)
        collect(cfg, [ckpt_root], [r])


def collect(cfg, roots, runs):
    """Copy the small machine-readable training outputs into results/ (committed to git)."""
    R = results_dir(cfg)
    for run in runs:
        for root in roots:
            d = os.path.join(root, run)
            if not os.path.exists(os.path.join(d, "training_log.json")):
                continue
            out = ensure_dir(os.path.join(R, "training", run))
            for fn in os.listdir(d):
                if fn == "training_log.json" or fn == "schedule_summary.json" or fn.startswith("run_manifest"):
                    shutil.copy(os.path.join(d, fn), os.path.join(out, fn))
            if os.path.isdir(os.path.join(d, "dev")):
                shutil.copytree(os.path.join(d, "dev"), os.path.join(out, "dev"), dirs_exist_ok=True)
            for sub in sorted(os.listdir(d)):
                if sub.startswith("stage") and os.path.exists(os.path.join(d, sub, "info.json")):
                    shutil.copy(os.path.join(d, sub, "info.json"), os.path.join(out, f"{sub}_info.json"))
            break


def step_final_eval(cfg, runs, roots):
    from src.analysis import prior_baseline, render_qualitative, select_qualitative
    from src.data_prep import load_sets, locate_image
    from src.evaluate import final_evaluate
    from src.features import FeatureStore
    from src.utils import peak_gb, reset_peak
    from src.vlm import VLM
    R = results_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = load_sets(cfg)
    atomic_json_dump(prior_baseline(sets), os.path.join(R, "final", "prior_baseline.json"))
    for run in runs:
        out_dir = os.path.join(R, "final", run)
        if os.path.exists(os.path.join(out_dir, "summary.json")):
            logger.info("final evaluation of %s exists", run)
            continue
        w = final_weights(cfg, roots, run)
        if w is None:
            logger.error("final Stage 3 weights for %s not found in %s; skipping.", run, roots)
            continue
        store = FeatureStore(cfg, cfg["runs"][run]["visual_tokens"])
        vlm = VLM(cfg, cfg["runs"][run]["visual_tokens"], device)
        vlm.load_state(torch.load(w, map_location="cpu", weights_only=False))
        vlm.eval()
        reset_peak()
        rep = final_evaluate(vlm, store, sets, cfg, out_dir)
        rep["peak_gpu_gb"] = peak_gb()
        rep["weights"] = w
        rep["environment"] = environment_info()
        atomic_json_dump(rep, os.path.join(out_dir, "summary.json"))
        # qualitative panels (stated selection rule; needs the images, so done here)
        qual = {}
        for suite in ("grounding", "vqa"):
            rows = {c: load_json(os.path.join(out_dir, f"{suite}_{c}.json")) for c in cfg["final_eval"]["conditions"]}
            qual[suite] = select_qualitative(rows, cfg["final_eval"]["qualitative_per_category"], cfg["seed"])
            try:
                render_qualitative(qual[suite], lambda k: locate_image(cfg, k),
                                   os.path.join(ensure_dir(os.path.join(R, "figures")), f"F6_qualitative_{run}_{suite}.png"))
            except Exception as e:          # never lose the evaluation because of a plotting problem
                logger.error("qualitative figure failed: %s", e)
        atomic_json_dump(qual, os.path.join(out_dir, "qualitative.json"))
        del vlm, store
        free_cuda()


def step_preflight(cfg):
    """~5-10 min on a T4. Fails loudly on anything that would otherwise waste a long run."""
    import math
    from src.data_prep import load_sets, verify_splits
    from src.evaluate import generate_items, resolve_example, teacher_forced
    from src.features import FeatureStore, check_pooling
    from src.train import build_schedule, calibration_inputs, check_gradients, schedule_summary
    from src.utils import NO_THINK_SUFFIX, peak_gb, reset_peak
    from src.vlm import VLM
    R = results_dir(cfg)
    report: Dict[str, Any] = {"environment": environment_info()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No GPU visible: preflight is slow and memory checks are skipped.")

    # 1) data: split isolation, sizes, mismatch mapping, schedule mixture
    sets = load_sets(cfg)
    report["splits"] = verify_splits({k: v for k, v in sets.items() if k != "mismatch"}, sets["mismatch"], cfg)
    schedule = build_schedule(cfg, sets)
    report["schedule"] = schedule_summary(schedule, cfg)
    assert build_schedule(cfg, sets) == schedule, "schedule is not deterministic"
    fc = os.path.join(R, "resources", "feature_cache.json")
    if os.path.exists(fc):
        report["siglip_final_layernorm_check"] = load_json(fc).get("final_layernorm_check")
    report["pooling"] = check_pooling()
    assert report["pooling"]["pooling_order_ok"]

    for run, rc in cfg["runs"].items():
        n_vis = rc["visual_tokens"]
        store = FeatureStore(cfg, n_vis)
        x = store.get([sets["vqa_dev"][0]["items"][0]["image"]])
        assert x.shape == (1, n_vis, cfg["models"]["vision_dim"]), x.shape
        reset_peak()
        vlm = VLM(cfg, n_vis, device)
        vlm.enable_gradient_checkpointing()
        r: Dict[str, Any] = {}

        # 2) chat template + visual insertion + no-image rebuild
        e = vlm.codec.encode("qa", sets["vqa_dev"][0]["question"], sets["vqa_dev"][0]["items"][0]["answer"])
        joint = vlm.tokenizer(vlm.codec.prefix_text + vlm.tokenizer.decode(e.suffix), add_special_tokens=False)["input_ids"]
        assert joint == e.prefix + e.suffix, "segment tokenization differs from joint tokenization"
        assert vlm.tokenizer.decode(e.suffix).endswith(NO_THINK_SUFFIX)
        z = vlm.visual_embeddings(x)[0]
        b = vlm.build_batch([e, e], [z, None], with_target=True, pad_left=False, label_eos=True)
        lens = b["attention_mask"].sum(1).tolist()
        assert lens[0] - lens[1] == n_vis, lens
        assert int(b["visual_mask"][0].sum()) == n_vis and int(b["visual_mask"][1].sum()) == 0
        lab = b["labels"]
        assert int((lab[0] != -100).sum()) == len(e.target) + 1 == int((lab[1] != -100).sum())
        first_vis = int(b["visual_mask"][0].float().argmax())
        assert first_vis == len(e.prefix), "visual tokens must follow '<|im_start|>user\\n'"
        assert torch.equal(b["position_ids"][1, :lens[1]], torch.arange(lens[1], device=device)), "no-image positions"
        bl = vlm.build_batch([e, e], [z, None], with_target=False, pad_left=True, label_eos=False)
        assert int(bl["attention_mask"][1].sum()) + n_vis == int(bl["attention_mask"][0].sum())
        nb = vlm.build_batch([e], [z], with_target=True, pad_left=False, label_eos=False)
        assert int((nb["labels"] != -100).sum()) == len(e.target), "answer NLL must exclude termination"
        r["sequence"] = {"prefix_tokens": len(e.prefix), "visual_tokens": n_vis, "suffix_tokens": len(e.suffix),
                         "target_tokens": len(e.target), "with_image_len": lens[0], "no_image_len": lens[1]}

        # 3) calibration (training data only)
        xc, ids = calibration_inputs(cfg, sets, store, vlm.tokenizer)
        r["calibration"] = vlm.calibrate_scale(xc, ids)

        # 4) gradient flow, frozen sets, memory on the longest Stage 1-3 microbatches
        vlm.train()
        stage_first = {}
        acc = 0
        for si, st in enumerate(cfg["train"]["stages"]):
            stage_first[si] = acc
            acc += st["updates"]
        for si in range(len(cfg["train"]["stages"])):
            if cfg["train"]["stages"][si]["lora_lr"] is not None and not vlm.has_lora:
                r["lora_trainable"] = vlm.add_lora(cfg["seed"] + 1)
                vlm.enable_gradient_checkpointing()
            refs = schedule[stage_first[si]][:cfg["train"]["microbatch"]]
            exs = [resolve_example(sets, rf) for rf in refs]
            encs = [vlm.codec.encode(ex["kind_prompt"], ex["question"], ex["target"]) for ex in exs]
            vlm.zero_grad(set_to_none=True)
            with vlm.autocast():
                zz = vlm.visual_embeddings(store.get([ex["image"] for ex in exs]))
                bb = vlm.build_batch(encs, list(zz), with_target=True, pad_left=False, label_eos=True)
                per_ex, cnt = vlm.per_example_loss(bb)
            per_ex.sum().backward()
            assert all(math.isfinite(v) for v in per_ex.tolist())
            check_gradients(vlm, si)
            r[f"stage{si + 1}"] = {"loss": float(per_ex.detach().mean()), "params": vlm.parameter_report(),
                                   "tokens_per_example": cnt.tolist(), "seq_len": int(bb["attention_mask"].shape[1])}
        vlm.zero_grad(set_to_none=True)
        if device.type == "cuda":
            r["peak_gpu_gb_microbatch"] = peak_gb()
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            if r["peak_gpu_gb_microbatch"]["allocated"] > 0.9 * total:
                raise RuntimeError(f"{run}: peak memory {r['peak_gpu_gb_microbatch']} is too close to {total:.1f} GB")

        # 5) evaluation paths: NLL with/without image, greedy generation
        vlm.eval()
        q = sets["vqa_dev"][:2]
        items = [("qa", p["question"], it["answer"], it["image"]) for p in q for it in p["items"]]
        r["nll_with_image"] = teacher_forced(vlm, store, items, label_eos=False, batch_size=8)
        r["nll_no_image"] = teacher_forced(vlm, store, [(a, b_, c, None) for a, b_, c, _ in items], False, 8)
        gens = generate_items(vlm, store, [("qa", p["question"], it["image"]) for p in q for it in p["items"]],
                              cfg["final_eval"]["max_new_tokens"], 8)
        r["generation_example"] = gens[:4]
        # Batch composition must not change a teacher-forced NLL (padding, masks and positions are correct). In FP16 a
        # batch of one (no padding) and a padded batch use different attention kernels and rounding, so differences
        # of ~1e-2..1e-1 nats are expected and only recorded; the assertion runs in FP32, where a masking/position
        # bug would still show but kernel rounding does not.
        mixed = items + [(a, b_, c, None) for a, b_, c, _ in items[:2]]       # with- and without-image rows together
        batched16 = teacher_forced(vlm, store, mixed, False, len(mixed))
        single16 = [teacher_forced(vlm, store, [it], False, 1)[0] for it in mixed]
        r["fp16_batch_vs_single_max_abs_nll_diff"] = max(abs(a - b_) for a, b_ in zip(batched16, single16))
        dtypes = {n: p.dtype for n, p in vlm.lm.named_parameters()}
        vlm.lm.float()
        amp = vlm.use_amp
        vlm.use_amp = False
        try:
            batched32 = teacher_forced(vlm, store, mixed, False, len(mixed))
            single32 = [teacher_forced(vlm, store, [it], False, 1)[0] for it in mixed]
        finally:
            vlm.use_amp = amp
            with torch.no_grad():
                for n, p in vlm.lm.named_parameters():   # FP16 -> FP32 -> FP16 is exact, so the backbone is unchanged
                    p.data = p.data.to(dtypes[n])
        r["fp32_batch_vs_single_max_abs_nll_diff"] = max(abs(a - b_) for a, b_ in zip(batched32, single32))
        assert r["fp32_batch_vs_single_max_abs_nll_diff"] < 1e-3, (batched32, single32)
        assert next(p for n, p in vlm.lm.named_parameters() if "lora_" not in n).dtype == vlm.lm_dtype
        g1 = generate_items(vlm, store, [("qa", q[0]["question"], q[0]["items"][0]["image"])], 8, 1)[0]["text"]
        r["generation_batch_invariance"] = g1 == gens[0]["text"] or g1[:4] == gens[0]["text"][:4]
        report[run] = r
        logger.info("preflight %s OK: %s", run, {k: r[k] for k in ("sequence", "calibration", "peak_gpu_gb_microbatch",
                                                                   "fp16_batch_vs_single_max_abs_nll_diff",
                                                                   "fp32_batch_vs_single_max_abs_nll_diff") if k in r})
        del vlm, store
        free_cuda()
    atomic_json_dump(report, os.path.join(R, "preflight_report.json"))
    logger.info("PREFLIGHT PASSED. Report: %s", os.path.join(R, "preflight_report.json"))


def step_status(cfg, roots):
    import torch as _t
    R = results_dir(cfg)
    print("data selected:", os.path.exists(os.path.join(R, "data", "data_stats.json")))
    print("features:", {s: os.path.exists(os.path.join(task_path(cfg["paths"]["feature_dir"]), f"{s}.pt"))
                        for s in ("train2014", "val2014")})
    for run in cfg["runs"]:
        state = "missing"
        for r in roots:
            d = os.path.join(r, run)
            if os.path.exists(os.path.join(d, "TRAIN_DONE")):
                state = f"done ({d})"
                break
            if os.path.exists(os.path.join(d, "recovery", "state.pt")):
                st = _t.load(os.path.join(d, "recovery", "state.pt"), map_location="cpu", weights_only=False)
                state = f"resumable at update {st['update']} ({d})"
                break
        fe = os.path.exists(os.path.join(R, "final", run, "summary.json"))
        print(f"  {run:11s} train={state}  final_eval={'yes' if fe else 'no'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=["data", "features", "preflight", "train", "final_eval",
                                                      "collect", "report", "status"])
    ap.add_argument("--run", default="main", help="main, compress49, comma-separated, or 'all'")
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--ckpt_root", default="checkpoints",
                    help="where training writes; for later steps a comma-separated list of roots to search")
    ap.add_argument("--max_hours", type=float, default=None, help="stop training cleanly after this many hours")
    ap.add_argument("--stop_after", type=int, default=None, help="pause training after this update (resumable)")
    ap.add_argument("--skip_images", action="store_true", help="data: select splits only (no image download)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    roots = [r for r in args.ckpt_root.split(",") if r]
    if args.step == "data":
        step_data(cfg, args.skip_images)
    elif args.step == "features":
        step_features(cfg)
    elif args.step == "preflight":
        step_preflight(cfg)
    elif args.step == "train":
        step_train(cfg, parse_runs(args.run, cfg), roots[0], args.max_hours, args.stop_after)
    elif args.step == "final_eval":
        step_final_eval(cfg, parse_runs(args.run, cfg), roots)
    elif args.step == "collect":
        collect(cfg, roots, list(cfg["runs"]))
    elif args.step == "report":
        import subprocess
        collect(cfg, roots, list(cfg["runs"]))
        subprocess.run([sys.executable, os.path.join("scripts", "compile_task2_report.py"), "--config", args.config],
                       check=True)
    elif args.step == "status":
        step_status(cfg, roots)


if __name__ == "__main__":
    main()
