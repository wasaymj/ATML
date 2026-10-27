"""Builds every Task 1 report table (Markdown + LaTeX) and figure (PDF + PNG) from results/.

Run from `PA3/Task 1`:  python scripts/compile_task1_report.py
Every number in the outputs traces to a JSON file under results/ (see Implementation_Notes).
Missing inputs are skipped with a log message, so the script can be run at any stage.
"""
import os
import sys
import csv
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_ROOT)
sys.path.insert(0, TASK_ROOT)
from src.utils import load_config, load_json, setup_logging, write_text  # noqa: E402

_TCFG = load_config()["train"]
MID, FINAL = str(_TCFG["midpoint_update"]), str(_TCFG["final_update"])

logger = logging.getLogger("compile_report")
R = "results"
TAB = os.path.join(R, "tables")
FIG = os.path.join(R, "figures")
SEED = 6304
N_BOOT = 10000

CONDS = ["baseline", "full_ft_mid", "full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]
LABEL = {"baseline": "Baseline", "full_ft_mid": "Full FT (mid, 625)", "full_ft": "Full FT",
         "l2sp": "Full FT + L2-SP", "lora_r4": "LoRA-4", "lora_r16": "LoRA-16", "oplora_r16": "OPLoRA-16"}
SHORT = {"baseline": "Base", "full_ft_mid": "FullFT@625", "full_ft": "FullFT", "l2sp": "L2-SP",
         "lora_r4": "LoRA-4", "lora_r16": "LoRA-16", "oplora_r16": "OPLoRA-16"}
METHODS = ["full_ft", "l2sp", "lora_r4", "lora_r16", "oplora_r16"]
# Colour follows the method (fixed categorical order); the midpoint reuses Full FT's hue, hatched.
COLOR = {"full_ft": "#2a78d6", "full_ft_mid": "#2a78d6", "l2sp": "#eb6834", "lora_r4": "#1baf7a",
         "lora_r16": "#6250d6", "oplora_r16": "#e34948", "baseline": "#8c8c8c"}
MARKER = {"full_ft": "o", "full_ft_mid": "o", "l2sp": "s", "lora_r4": "^", "lora_r16": "D", "oplora_r16": "v"}
TASKS = [("gsm8k", "GSM8K", "gsm8k_flexible_em"), ("hellaswag", "HellaSwag", "hellaswag_acc_norm"),
         ("boolq", "BoolQ", "boolq_acc"), ("arc_easy", "ARC-Easy", "arc_easy_acc_norm")]
PAIRS = [("Rank", "lora_r4", "lora_r16"), ("Explicit regularization", "full_ft", "l2sp"),
         ("Orthogonal protection", "lora_r16", "oplora_r16"), ("Full FT vs LoRA-16", "full_ft", "lora_r16"),
         ("Full FT progression", "full_ft_mid", "full_ft")]


def opt(path):
    return load_json(path) if os.path.exists(path) else None


def save_table(name: str, md: List[str], tex: List[str]):
    write_text(os.path.join(TAB, name + ".md"), "\n".join(md) + "\n")
    write_text(os.path.join(TAB, name + ".tex"), "\n".join(tex) + "\n")
    logger.info("table %s", name)


def tex_table(caption: str, label: str, colspec: str, header: str, rows: List[str]) -> List[str]:
    return [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule",
            *[r + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def sgn(x: Optional[float], d: int = 2) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def f(x: Optional[float], d: int = 2) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# --------------------------------------------------------------------------------------------
# Per-example loading and paired statistics
# --------------------------------------------------------------------------------------------
def per_example(cond: str, task: str) -> Optional[np.ndarray]:
    """Per-doc correctness (primary metric) ordered by doc_id."""
    d = os.path.join(R, "benchmarks", cond)
    if task == "gsm8k":
        rows = opt(os.path.join(d, "gsm8k_per_example.json"))
        key = "correct"
    else:
        rows = opt(os.path.join(d, f"{task}_per_example.json"))
        key = "acc" if task == "boolq" else "acc_norm"
    if rows is None:
        return None
    rows = sorted(rows, key=lambda r: r["doc_id"])
    return np.array([float(r[key]) for r in rows])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float]:
    """Mean of (b - a) in percentage points with a 95% percentile interval over resampled docs."""
    diff = (b - a) * 100.0
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for s in range(0, n, 500):
        idx = rng.integers(0, len(diff), size=(min(500, n - s), len(diff)))
        means[s:s + idx.shape[0]] = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def binom_se(pct: Optional[float], arr: Optional[np.ndarray]) -> Optional[float]:
    """Standard error of an accuracy (in %), sqrt(p(1-p)/n)."""
    if pct is None or arr is None or len(arr) == 0:
        return None
    p = pct / 100.0
    return 100.0 * float(np.sqrt(p * (1 - p) / len(arr)))


def flips(a: np.ndarray, b: np.ndarray) -> Dict[str, int]:
    a, b = a > 0.5, b > 0.5
    return {"lost": int((a & ~b).sum()), "gained": int((~a & b).sum()), "n": int(a.size)}


def mcnemar_p(lost: int, gained: int) -> float:
    """Exact two-sided McNemar (binomial) p-value on discordant pairs."""
    from math import comb
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------
def table_benchmarks(S: Dict[str, Any]) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    base = S["baseline"]
    ci: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for c in CONDS[1:]:
        if c not in S:
            continue
        ci[c] = {}
        for t, _, _ in TASKS:
            a, b = per_example("baseline", t), per_example(c, t)
            if a is not None and b is not None and len(a) == len(b):
                ci[c][t] = paired_bootstrap(a, b)
    md = ["| Condition | GSM8K flex EM | Δ | HellaSwag acc_norm (acc) | Δ | BoolQ acc | Δ | ARC-E acc_norm (acc) | Δ | mean Δ retention |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tex_rows = []
    for c in CONDS:
        if c not in S:
            continue
        s = S[c]
        vals = [s["gsm8k_flexible_em"], s["hellaswag_acc_norm"], s["boolq_acc"], s["arc_easy_acc_norm"]]
        deltas = [None if c == "baseline" else v - b for v, b in zip(vals, [base["gsm8k_flexible_em"],
                  base["hellaswag_acc_norm"], base["boolq_acc"], base["arc_easy_acc_norm"]])]
        ret = None if c == "baseline" else float(np.mean(deltas[1:]))
        se = [binom_se(v, per_example(c, t)) for v, (t, _, _) in zip(vals, TASKS)]
        pm = lambda i: f"{f(vals[i])} ± {f(se[i])}"
        md.append(f"| {LABEL[c]} | {pm(0)} | {sgn(deltas[0])} | {pm(1)} ({f(s['hellaswag_acc'])}) | "
                  f"{sgn(deltas[1])} | {pm(2)} | {sgn(deltas[2])} | {pm(3)} ({f(s['arc_easy_acc'])}) | "
                  f"{sgn(deltas[3])} | {sgn(ret)} |")
        cell = lambda v, d: f(v) if d is None else f"{f(v)} ({sgn(d)})"
        tex_rows.append(f"{LABEL[c]} & {cell(vals[0], deltas[0])} & {cell(vals[1], deltas[1])} & "
                        f"{cell(vals[2], deltas[2])} & {cell(vals[3], deltas[3])} & {sgn(ret)}")
    md.append("\nPrimary metrics in % ± 1 binomial SE; Δ = change from the shared baseline in percentage points (pp). "
              "Raw (unnormalized) accuracy in parentheses where lm-eval provides it. "
              "Mean Δ retention = mean of the HellaSwag, BoolQ and ARC-Easy Δ.")
    md.append("\n**95% paired-bootstrap intervals for Δ (10,000 resamples of documents, seed 6304):**\n")
    md.append("| Condition | GSM8K | HellaSwag | BoolQ | ARC-E |\n|---|---|---|---|---|")
    for c, d in ci.items():
        md.append(f"| {LABEL[c]} | " + " | ".join(
            f"{sgn(d[t][0])} [{sgn(d[t][1])}, {sgn(d[t][2])}]" if t in d else "--" for t, _, _ in TASKS) + " |")
    save_table("T1_benchmarks", md, tex_table(
        "Zero-shot results for all seven evaluation conditions (\\%). Parentheses: change from baseline in "
        "percentage points. GSM8K: flexible-extract exact match (test, 1,319); HellaSwag and ARC-Easy: "
        "normalized accuracy; BoolQ: accuracy. Retention: mean change on the three non-math benchmarks.",
        "tab:t1_bench", "lccccc", "Condition & GSM8K & HellaSwag & BoolQ & ARC-Easy & Retention $\\Delta$",
        tex_rows))
    return ci


def table_controlled(S):
    md = ["| Comparison (B − A) | Benchmark | Δ pp [95% CI] | lost (A✓→B✗) | gained (A✗→B✓) | McNemar p |",
          "|---|---|---|---|---|---|"]
    tex_rows, out = [], {}
    for name, a, b in PAIRS:
        if a not in S or b not in S:
            continue
        for t, tl, _ in TASKS:
            x, y = per_example(a, t), per_example(b, t)
            if x is None or y is None:
                continue
            d, lo, hi = paired_bootstrap(x, y)
            fl = flips(x, y)
            p = mcnemar_p(fl["lost"], fl["gained"])
            out.setdefault(name, {})[t] = {"delta": d, "ci": [lo, hi], **fl, "mcnemar_p": p}
            md.append(f"| {name}: {SHORT[b]} − {SHORT[a]} | {tl} | {sgn(d)} [{sgn(lo)}, {sgn(hi)}] | {fl['lost']} | "
                      f"{fl['gained']} | {p:.3g} |")
            tex_rows.append(f"{name} ({SHORT[b]}$-${SHORT[a]}) & {tl} & {sgn(d)} [{sgn(lo)}, {sgn(hi)}] & "
                            f"{fl['lost']} / {fl['gained']} & {p:.2g}")
    md.append("\nAll comparisons use final checkpoints except the progression row. Paired over identical documents.")
    save_table("T4_controlled_comparisons", md, tex_table(
        "Controlled comparisons (B$-$A) on identical documents: mean difference with 95\\% paired-bootstrap "
        "interval, discordant documents (lost / gained) and exact McNemar $p$.", "tab:t1_pairs", "llccc",
        "Comparison & Benchmark & $\\Delta$ pp [95\\% CI] & lost / gained & $p$", tex_rows))
    json.dump(out, open(os.path.join(TAB, "T4_controlled_comparisons.json"), "w"), indent=2)


def table_forgetting(S):
    """Document-level forgetting vs the baseline: how much churn hides behind each net change."""
    md = ["| Condition | Benchmark | baseline-correct lost | baseline-wrong gained | net | churn % of docs |",
          "|---|---|---|---|---|---|"]
    for c in CONDS[1:]:
        if c not in S:
            continue
        for t, tl, _ in TASKS:
            a, b = per_example("baseline", t), per_example(c, t)
            if a is None or b is None:
                continue
            fl = flips(a, b)
            md.append(f"| {LABEL[c]} | {tl} | {fl['lost']} | {fl['gained']} | {fl['gained'] - fl['lost']:+d} | "
                      f"{100 * (fl['lost'] + fl['gained']) / fl['n']:.1f} |")
    save_table("T9_forgetting_events", md, [])


def table_mc_behaviour(S):
    """BoolQ 'yes' rate and likelihood margins: does fine-tuning shift calibration/bias?"""
    md = ["| Condition | BoolQ pred-yes % (gold 62.2%) | BoolQ mean margin ll(yes)−ll(no) | HellaSwag mean gold-vs-best-other margin | ARC-E mean gold-vs-best-other margin |",
          "|---|---|---|---|---|"]
    for c in CONDS:
        d = os.path.join(R, "benchmarks", c)
        bq, hs, ae = (opt(os.path.join(d, f"{t}_per_example.json")) for t in ("boolq", "hellaswag", "arc_easy"))
        if bq is None:
            continue
        yes = np.mean([r["pred"] == 1 for r in bq]) * 100
        gold_yes = np.mean([r["gold"] == 1 for r in bq]) * 100
        marg = np.mean([r["loglik"][1] - r["loglik"][0] for r in bq])

        def gm(rows):
            if not rows:
                return None
            vals = []
            for r in rows:
                ll = np.array(r["loglik"])
                g = int(r["gold"])
                vals.append(ll[g] - np.max(np.delete(ll, g)))
            return float(np.mean(vals))
        md.append(f"| {LABEL[c]} | {yes:.1f} (gold {gold_yes:.1f}) | {marg:+.3f} | {f(gm(hs), 3)} | {f(gm(ae), 3)} |")
    md.append("\nMargins use raw (unnormalized) summed log-likelihoods; positive = model prefers the gold choice.")
    save_table("T10_likelihood_behaviour", md, [])


def table_gsm8k(S):
    md = ["| Condition | flex EM | strict EM | flex invalid % | truncated % | EM non-truncated | format-aware EM* | flex✗ but format-aware✓ % | mean gen tokens | answer formats |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tex_rows = []
    for c in CONDS:
        if c not in S:
            continue
        g = S[c]["gsm8k"]
        fm = ", ".join(f"{k}:{v}" for k, v in sorted(g["answer_format_counts"].items(), key=lambda kv: -kv[1]))
        md.append(f"| {LABEL[c]} | {f(g['flexible_em'])} | {f(g['strict_em'])} | {f(g['flexible_invalid_pct'])} | "
                  f"{f(g['truncated_pct'])} | {f(g['em_non_truncated'])} | {f(g['format_aware_em'])} | "
                  f"{f(g['flex_wrong_but_format_aware_right_pct'])} | {f(g['mean_generated_tokens'], 0)} | {fm} |")
        tex_rows.append(f"{LABEL[c]} & {f(g['flexible_em'])} & {f(g['strict_em'])} & {f(g['flexible_invalid_pct'])} & "
                        f"{f(g['truncated_pct'])} & {f(g['format_aware_em'])} & {f(g['mean_generated_tokens'], 0)}")
    md.append("\n*Format-aware EM is a diagnostic re-extraction (boxed / 'The answer is' / '####' first, then last "
              "number); it is not the reported metric. It separates extraction failures of the stock filter from "
              "wrong answers. Truncated = hit the 1,024-token cap without a stop sequence (recorded during generation).")
    save_table("T6_gsm8k_accounting", md, tex_table(
        "GSM8K answer accounting. Strict-match expects ``The answer is N.''; flexible-extract takes the last "
        "number. Format-aware EM is a diagnostic re-extraction only.", "tab:t1_gsm", "lcccccc",
        "Condition & Flex EM & Strict EM & Invalid \\% & Trunc. \\% & Format-aware & Tokens", tex_rows))


def table_resources():
    md = ["| Method | trainable params | % of model | train compute (h) | resp. tok/s | peak GPU alloc (GB) | eval export (MB) | trainable state FP32 (MB) | recovery ckpt (MB) | skipped | clipped % | method overhead |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex_rows = []
    for m in METHODS:
        d = opt(os.path.join(R, "training", f"{m}.json"))
        if d is None:
            continue
        fin = d["checkpoints"].get(FINAL, {})
        rec = max((v.get("recovery_size_mb", 0) for v in d["checkpoints"].values()), default=0)
        if m == "l2sp":
            ov = (f"θ0 {d['l2sp_reference_storage_mb']:.0f} MB on {d['l2sp_reference_device']}; "
                  f"{d.get('l2sp_bytes_transferred_per_update_mb', 0):.0f} MB/update transfer "
                  f"({f(d.get('l2sp_one_transfer_sec'), 2)} s); regularizer {f(d.get('l2sp_regularizer_time_sec', 0) / 60, 1)} min")
        elif m == "oplora_r16":
            ov = (f"SVD {f(d.get('svd_preprocessing_time_sec'), 1)} s; bases {f(d.get('oplora_basis_storage_mb'), 0)} MB; "
                  f"export {f(d.get('oplora_export_time_sec'), 2)} s / {f(d.get('oplora_export_mb'), 1)} MB")
        elif m == "full_ft":
            ov = "θ0 snapshot for displacement only (not used in the objective)"
        else:
            ov = "--"
        pct = 100 * d["trainable_params"] / d["total_params_unique"]
        hours = d["train_compute_sec"] / 3600
        md.append(f"| {LABEL[m]} | {d['trainable_params']:,} | {pct:.2f} | {hours:.2f} | {d['response_tokens_per_sec']:.0f} | "
                  f"{d['peak_gpu_mem_allocated_gb']:.2f} | {fin.get('export_weights_mb', 0):.1f} | "
                  f"{d['trainable_state_fp32_mb']:.1f} | {rec:.0f} | {d['n_skipped_updates']} | "
                  f"{100 * d['frac_updates_clipped']:.1f} | {ov} |")
        tex_rows.append(f"{LABEL[m]} & {d['trainable_params'] / 1e6:.2f}M & {hours:.2f} & "
                        f"{d['peak_gpu_mem_allocated_gb']:.1f} & {fin.get('export_weights_mb', 0):.0f} & {rec:.0f}")
    md.append("\nCompute time excludes checkpoint I/O (logged separately). Eval export = FP16 model for Full FT "
              "variants, FP32 adapter tensors for adapter methods. Equal updates/examples do not imply equal compute.")
    save_table("T2_resources", md, tex_table(
        "Training resources (one epoch, 1,250 updates, same GPU type). Export: evaluation checkpoint size; "
        "recovery: optimizer-inclusive resumable state.", "tab:t1_res", "lccccc",
        "Method & Trainable & Time (h) & Peak mem (GB) & Export (MB) & Recovery (MB)", tex_rows))


def table_l2sp():
    a, b = opt(os.path.join(R, "training", "full_ft.json")), opt(os.path.join(R, "training", "l2sp.json"))
    if not a or not b:
        return
    ups = b["updates"]
    md = ["| Quantity | Full FT | Full FT + L2-SP |", "|---|---|---|",
          f"| ‖θ−θ0‖₂ at {MID} | {f(a.get('midpoint_displacement'), 4)} | {f(b['checkpoints'].get(MID, {}).get('displacement'), 4)} |",
          f"| ‖θ−θ0‖₂ at {FINAL} | {f(a.get('final_displacement'), 4)} | {f(b.get('final_displacement'), 4)} |"]
    w, m_, fi = _TCFG["warmup_updates"], int(MID), int(FINAL)
    for lo, hi in ((1, w), (w + 1, m_), (m_ + 1, fi)):
        seg = [u for u in ups if lo <= u["update"] <= hi and "omega" in u and u["sft_loss"] is not None]
        if not seg:
            continue
        om = np.mean([u["omega"] for u in seg])
        pen = np.mean([u["lambda_omega"] for u in seg])
        ratio = np.mean([u["lambda_omega"] / u["sft_loss"] for u in seg]) * 100
        gr = np.median([u["penalty_grad_norm"] / max(u["sft_grad_norm"], 1e-12) for u in seg])
        md.append(f"| updates {lo}–{hi}: mean Ω / λΩ / λΩ÷L_SFT / median ‖λ∇Ω‖÷‖∇L_SFT‖ | -- | "
                  f"{om:.3f} / {pen:.2e} / {ratio:.3f}% / {gr:.3f} |")
    last = ups[-1]
    md.append(f"| final update: SFT loss / total loss | {f(a['updates'][-1]['sft_loss'], 4)} / -- | "
              f"{f(last['sft_loss'], 4)} / {f(last.get('total_loss'), 4)} |")
    md.append("\nSFT loss is compared across methods; the L2-SP total objective is reported separately (manual).")
    save_table("T5_l2sp", md, [])


def table_diagnostics():
    D = opt(os.path.join(R, "diagnostics", "diagnostics_summary.json"))
    if not D:
        return
    md = ["| Condition | layer | ρ_k (ref k/d) | η | R(z) math mean | R(z) non-math mean | math/non-math | energy in top-16 | σ_top-k mean rel. change | Prop.2 residual |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tex_rows = []
    for c in CONDS[1:]:
        if c not in D:
            continue
        for l, v in D[c].items():
            md.append(f"| {LABEL[c]} | {l} | {v['rho_k']:.4f} ({v['rho_random_reference']:.3f}) | {v['eta']:.2e} | "
                      f"{v['R_math']['mean']:.2e} | {v['R_nonmath']['mean']:.2e} | {v['R_math_over_nonmath']:.2f} | "
                      f"{v.get('energy_top16', float('nan')):.3f} | {v.get('sigma_topk_rel_change_mean', float('nan')):.2e} | "
                      f"{v.get('prop2_residual', float('nan')):.2e} |")
            tex_rows.append(f"{LABEL[c]} & {l} & {v['rho_k']:.3f} & {v['eta']:.1e} & {v['R_math']['mean']:.1e} & "
                            f"{v['R_nonmath']['mean']:.1e}")
    save_table("T3_diagnostics", md, tex_table(
        "down\\_proj diagnostics (FP32). $\\rho_k$: fraction of $\\|\\Delta W\\|_F^2$ in the top-$k$ ($k=128$) "
        "output subspace of $W_0$ (isotropic reference $k/d_{out}=0.125$); $\\eta=\\|\\Delta W\\|_F/\\|W_0\\|_F$; "
        "$R(z)$: mean activation correction on 32 GSM8K and 32 HellaSwag probes.", "tab:t1_diag", "lccccc",
        "Condition & Layer & $\\rho_k$ & $\\eta$ & $R$ math & $R$ non-math", tex_rows))
    C = opt(os.path.join(R, "diagnostics", "oplora_constraint_verification.json"))
    md = ["| Check | " + " | ".join(f"layer {l}" for l in (C or {})) + " |", "|---" * (1 + len(C or {})) + "|"]
    if C:
        for key in ("rel_left_UkT_dW", "rel_right_dW_Vk", "merged_fp16_rel_left", "merged_fp16_eta"):
            md.append(f"| {key} | " + " | ".join(f"{C[l].get(key, float('nan')):.2e}" for l in C) + " |")
    for m in ("lora_r4", "lora_r16", "oplora_r16"):
        mv = opt(os.path.join(R, "merge", f"{m}.json"))
        if mv:
            md.append(f"\nMerge check {LABEL[m]}: FP32 max |Δlogit| {mv['fp32_max_abs_logit_diff']:.2e}; FP16 max "
                      f"{mv['fp16_max_abs_logit_diff']:.3f}, mean {mv['fp16_mean_abs_logit_diff']:.4f}, top-1 agreement "
                      f"{100 * mv['fp16_top1_agreement']:.2f}% over {mv['n_positions']} positions.")
    md.append("\nNumerical verification of OPLoRA's constraints is separate from evidence about benchmark behaviour.")
    save_table("T7_oplora_and_merge_checks", md, [])


def table_qualitative():
    M = opt(os.path.join(R, "qualitative", "auto_metrics_168.json"))
    if not M:
        return
    md = ["| Condition | mean tokens | truncated | JSON strict | JSON lenient | template OK | any math marker | repeated 4-gram frac | think tag |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in CONDS:
        rows = [r for r in M if r["condition"] == c]
        if not rows:
            continue
        js = [r for r in rows if r["format"] == "json"]
        tp = [r for r in rows if r["format"] == "template"]
        pct = lambda xs: 100 * np.mean(xs) if xs else float("nan")
        md.append(f"| {LABEL[c]} | {np.mean([r['n_generated_tokens'] for r in rows]):.0f} | "
                  f"{sum(r['truncated'] for r in rows)}/24 | {pct([r['json_strict'] for r in js]):.0f}% | "
                  f"{pct([r['json_lenient'] for r in js]):.0f}% | {pct([r['template_required_ok'] for r in tp]):.0f}% | "
                  f"{sum(r['any_math_marker'] for r in rows)}/24 | {np.mean([r['repeated_4gram_frac'] for r in rows]):.3f} | "
                  f"{sum(r['contains_think_tag'] for r in rows)} |")
    md.append("\nAutomatic measurements over all 168 responses; they support, not replace, the manual 42-response annotation.")
    save_table("T8_qualitative_auto", md, [])



def rows_of(cond: str, task: str) -> Optional[List[Dict[str, Any]]]:
    """Per-example rows of one condition and benchmark, sorted by doc_id."""
    rows = opt(os.path.join(R, "benchmarks", cond, f"{task}_per_example.json"))
    return sorted(rows, key=lambda r: r["doc_id"]) if rows else None


def mc_scores(r: Dict[str, Any], task: str) -> np.ndarray:
    """Choice scores the primary metric uses: raw log-likelihood (BoolQ) or per-character (acc_norm)."""
    ll = np.array(r["loglik"], float)
    if task != "boolq" and r.get("choice_chars"):
        ll = ll / np.array(r["choice_chars"], float)
    return ll


def mc_pred(r: Dict[str, Any], task: str) -> int:
    return int(np.argmax(mc_scores(r, task)))


def table_stability(S):
    """How much of each model's output is unchanged: identical GSM8K responses and identical MC choices."""
    md = ["| Comparison | GSM8K identical response % | GSM8K same extracted answer % | HellaSwag same choice % | BoolQ same choice % | ARC-E same choice % |",
          "|---|---|---|---|---|---|"]
    pairs = [("vs baseline", "baseline", c) for c in CONDS[1:]] + [(n, a, b) for n, a, b in PAIRS]
    for name, a, b in pairs:
        if a not in S or b not in S:
            continue
        cells = []
        ga, gb = rows_of(a, "gsm8k"), rows_of(b, "gsm8k")
        if ga and gb and len(ga) == len(gb):
            cells.append(f"{100 * np.mean([x['response'].strip() == y['response'].strip() for x, y in zip(ga, gb)]):.1f}")
            cells.append(f"{100 * np.mean([x['extracted_flexible'] == y['extracted_flexible'] for x, y in zip(ga, gb)]):.1f}")
        else:
            cells += ["--", "--"]
        for t in ("hellaswag", "boolq", "arc_easy"):
            ra, rb = rows_of(a, t), rows_of(b, t)
            cells.append(f"{100 * np.mean([mc_pred(x, t) == mc_pred(y, t) for x, y in zip(ra, rb)]):.1f}"
                         if ra and rb and len(ra) == len(rb) else "--")
        label = f"{LABEL[b]} vs baseline" if name == "vs baseline" else f"{name}: {SHORT[b]} vs {SHORT[a]}"
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("\nIdentical = byte-identical greedy response after stripping whitespace. A small net score change "
              "with few identical outputs means the model changed a lot but in offsetting directions; a regularizer "
              "that leaves outputs identical had nothing to constrain.")
    save_table("T13_output_stability", md, [])


def table_strata(S):
    """Where forgetting happens: flips by baseline confidence, HellaSwag by source, GSM8K by solution length."""
    out: Dict[str, Any] = {}
    md = ["**(a) Answer changes vs baseline by baseline confidence quintile** (Q1 = smallest margin between the "
          "baseline's top two choices, i.e. near-ties; Q5 = most confident). Cells: % of documents whose choice "
          "changed (lost / gained counts).\n",
          "| Condition | Benchmark | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|---|"]
    for t, tl, _ in TASKS[1:]:
        base = rows_of("baseline", t)
        if not base:
            continue
        marg = []
        for r in base:
            sc = np.sort(mc_scores(r, t))
            marg.append(sc[-1] - sc[-2])
        edges = np.quantile(marg, [0.2, 0.4, 0.6, 0.8])
        q = np.digitize(marg, edges)
        acc_b = np.array([float(mc_pred(r, t) == r["gold"]) for r in base])
        md.append(f"| Baseline accuracy per quintile | {tl} | " +
                  " | ".join(f"{100 * acc_b[q == i].mean():.1f}%" for i in range(5)) + " |")
        for c in CONDS[1:]:
            rc = rows_of(c, t)
            if c not in S or not rc or len(rc) != len(base):
                continue
            pb = np.array([mc_pred(r, t) for r in base])
            pc = np.array([mc_pred(r, t) for r in rc])
            acc_c = np.array([float(p == r["gold"]) for p, r in zip(pc, rc)])
            cells = []
            for i in range(5):
                m = q == i
                lost = int(((acc_b == 1) & (acc_c == 0) & m).sum())
                gained = int(((acc_b == 0) & (acc_c == 1) & m).sum())
                cells.append(f"{100 * (pb[m] != pc[m]).mean():.1f} ({lost}/{gained})")
                out.setdefault(c, {}).setdefault(t, []).append(
                    {"quintile": i + 1, "changed_pct": 100 * float((pb[m] != pc[m]).mean()), "lost": lost, "gained": gained})
            md.append(f"| {LABEL[c]} | {tl} | " + " | ".join(cells) + " |")

    base = rows_of("baseline", "hellaswag")
    if base and base[0].get("source"):
        srcs = sorted({r["source"] for r in base if r.get("source")})
        md += ["\n**(b) HellaSwag acc_norm by source** (Δ pp vs baseline).\n",
               "| Condition | " + " | ".join(f"{s_} (n={sum(r.get('source') == s_ for r in base)})" for s_ in srcs) + " |",
               "|---" * (1 + len(srcs)) + "|"]
        for c in CONDS:
            rc = rows_of(c, "hellaswag")
            if c not in S or not rc:
                continue
            cells = []
            for s_ in srcs:
                a = np.mean([r["acc_norm"] for r in rc if r.get("source") == s_]) * 100
                b0 = np.mean([r["acc_norm"] for r in base if r.get("source") == s_]) * 100
                cells.append(f"{a:.1f}" if c == "baseline" else f"{a:.1f} ({a - b0:+.1f})")
            md.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")

    gb = rows_of("baseline", "gsm8k")
    if gb and gb[0].get("gold_steps") is not None:
        bins = [(1, 2, "1–2"), (3, 3, "3"), (4, 4, "4"), (5, 99, "5+")]
        nb = [sum(lo <= (r["gold_steps"] or 0) <= hi for r in gb) for lo, hi, _ in bins]
        md += ["\n**(c) GSM8K flexible EM by reference-solution length** (lines before `####`; Δ pp vs baseline).\n",
               "| Condition | " + " | ".join(f"{lab} steps (n={n})" for (_, _, lab), n in zip(bins, nb)) + " |",
               "|---" * (1 + len(bins)) + "|"]
        for c in CONDS:
            rc = rows_of(c, "gsm8k")
            if c not in S or not rc:
                continue
            cells = []
            for lo, hi, _ in bins:
                sel = lambda rows: [r["correct"] for r in rows if lo <= (r.get("gold_steps") or 0) <= hi]
                a, b0 = 100 * np.mean(sel(rc)), 100 * np.mean(sel(gb))
                cells.append(f"{a:.1f}" if c == "baseline" else f"{a:.1f} ({a - b0:+.1f})")
            md.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    md.append("\nIf answer changes concentrate in Q1, much of the 'forgetting' is near-tie documents flipping; "
              "changes in Q5 mean confident baseline knowledge was overwritten.")
    save_table("T14_forgetting_strata", md, [])
    json.dump(out, open(os.path.join(TAB, "T14_forgetting_strata.json"), "w"), indent=2)
    return out


def table_gsm8k_extra(S):
    """Accuracy among extractable answers and length of correct vs wrong responses."""
    md = ["| Condition | answers with an extractable number % | EM among extractable | mean tokens (correct) | mean tokens (wrong) | EM among answers ≤ 256 tokens | EM among answers > 256 tokens |",
          "|---|---|---|---|---|---|---|"]
    for c in CONDS:
        rows = rows_of(c, "gsm8k")
        if c not in S or not rows:
            continue
        valid = [r for r in rows if not r["flexible_invalid"]]
        cor = [r["n_generated_tokens"] for r in rows if r["correct"] == 1]
        wr = [r["n_generated_tokens"] for r in rows if r["correct"] != 1]
        short = [r["correct"] for r in rows if r["n_generated_tokens"] <= 256]
        long_ = [r["correct"] for r in rows if r["n_generated_tokens"] > 256]
        mean = lambda xs, k=1: f(k * float(np.mean(xs)), 1) if xs else "--"
        md.append(f"| {LABEL[c]} | {100 * len(valid) / len(rows):.1f} | {mean([r['correct'] for r in valid], 100)} | "
                  f"{mean(cor)} | {mean(wr)} | {mean(short, 100)} (n={len(short)}) | {mean(long_, 100)} (n={len(long_)}) |")
    md.append("\nSeparates 'produces a usable final number' from 'the number is right', and checks whether score "
              "changes travel with response length (MetaMathQA answers are short).")
    save_table("T6b_gsm8k_extraction_and_length", md, [])


def write_extraction_audit(S):
    """Sample GSM8K responses where the stock flexible filter and the format-aware re-extraction disagree,
    plus a few where they agree, for a hand check of the automatic scorer (never overwritten)."""
    path = os.path.join(R, "benchmarks", "gsm8k_extraction_audit.csv")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(SEED)
    out = []
    for c in CONDS:
        rows = rows_of(c, "gsm8k")
        if c not in S or not rows:
            continue
        dis = [r for r in rows if r["correct"] != r["format_aware_correct"]]
        agr = [r for r in rows if r["correct"] == r["format_aware_correct"]]
        for group, k in ((dis, 6), (agr, 2)):
            for i in rng.permutation(len(group))[:k]:
                r = group[int(i)]
                out.append({"condition": c, "doc_id": r["doc_id"], "gold": r["gold"],
                            "response_tail": r["response"][-600:], "human_final_answer": "",
                            "human_correct_yes_no": "", "notes": ""})
    if not out:
        return None
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    logger.info("Wrote GSM8K extraction audit sheet %s (%d rows)", path, len(out))
    return path


def cohen_kappa(x: List[bool], y: List[bool]) -> Optional[float]:
    if not x:
        return None
    x, y = np.array(x, bool), np.array(y, bool)
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return None if pe == 1 else (po - pe) / (1 - pe)


def yes(v) -> bool:
    return str(v).strip().lower() in ("y", "yes", "1", "true")


def table_audits():
    """Hand labels vs automatic checks: the 42-response annotation and the GSM8K extraction audit."""
    qdir = os.path.join(R, "qualitative")
    sheet, key = os.path.join(qdir, "annotation_sheet_42_blind.csv"), os.path.join(qdir, "annotation_key_42.csv")
    if os.path.exists(sheet) and os.path.exists(key):
        lab = {r["item"]: r for r in csv.DictReader(open(sheet, encoding="utf-8"))}
        rows = [{**k, **lab[k["item"]]} for k in csv.DictReader(open(key, encoding="utf-8")) if k["item"] in lab]
        done = [r for r in rows if r.get("coherence_relevance_1to5", "").strip()]
        if done:
            md = ["| Condition | format compliance (0-2) | coherence/relevance (1-5) | repetition | unnecessary math framing | inappropriate | truncated | annotated |",
                  "|---|---|---|---|---|---|---|---|"]
            num = lambda xs: f(float(np.mean([float(x) for x in xs if str(x).strip()])), 2) if any(str(x).strip() for x in xs) else "--"
            for c in CONDS:
                rr = [r for r in done if r["condition"] == c]
                if not rr:
                    continue
                md.append(f"| {LABEL[c]} | {num([r['format_compliance_0to2'] for r in rr])} | "
                          f"{num([r['coherence_relevance_1to5'] for r in rr])} | {sum(yes(r['repetition_yes_no']) for r in rr)}/{len(rr)} | "
                          f"{sum(yes(r['unnecessary_math_framing_yes_no']) for r in rr)}/{len(rr)} | "
                          f"{sum(yes(r['inappropriate_content_yes_no']) for r in rr)}/{len(rr)} | "
                          f"{sum(yes(r['truncated']) for r in rr)}/{len(rr)} | {len(rr)}/6 |")
            md.append("\nBy format (all conditions): " + "; ".join(
                f"{fm}: compliance {num([r['format_compliance_0to2'] for r in done if r['format'] == fm])}"
                for fm in ("plain", "json", "template")))
            save_table("T11_annotation_summary", md, [])

            checks = [("JSON valid (auto strict) vs compliance = 2", "json",
                       lambda r: yes(r["auto_json_strict"]), lambda r: r["format_compliance_0to2"].strip() == "2"),
                      ("JSON valid (auto lenient) vs compliance ≥ 1", "json",
                       lambda r: yes(r["auto_json_lenient"]), lambda r: r["format_compliance_0to2"].strip() in ("1", "2")),
                      ("template keys filled (auto) vs compliance = 2", "template",
                       lambda r: yes(r["auto_template_ok"]), lambda r: r["format_compliance_0to2"].strip() == "2"),
                      ("any math marker (auto) vs unnecessary math framing", None,
                       lambda r: bool(r["auto_math_markers"].strip()), lambda r: yes(r["unnecessary_math_framing_yes_no"])),
                      ("repeated 4-gram frac > 0.2 (auto) vs repetition", None,
                       lambda r: float(r["auto_repeated_4gram_frac"] or 0) > 0.2, lambda r: yes(r["repetition_yes_no"]))]
            md = ["| Automatic check vs hand label | n | agreement % | Cohen's κ | auto yes / human yes |", "|---|---|---|---|---|"]
            for name, fmt, auto, human in checks:
                rr = [r for r in done if fmt is None or r["format"] == fmt]
                a, h = [auto(r) for r in rr], [human(r) for r in rr]
                k = cohen_kappa(a, h)
                md.append(f"| {name} | {len(rr)} | {100 * np.mean(np.array(a) == np.array(h)):.0f} | "
                          f"{f(k, 2)} | {sum(a)} / {sum(h)} |" if rr else f"| {name} | 0 | -- | -- | -- |")
            md.append("\nThe automatic measures in T8 cover all 168 responses; this table says how far they can be "
                      "trusted, using the blind hand labels as reference.")
            save_table("T15_auto_vs_human_audit", md, [])

    path = os.path.join(R, "benchmarks", "gsm8k_extraction_audit.csv")
    if os.path.exists(path):
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["human_correct_yes_no"].strip()]
        if rows:
            md = ["| Condition | audited | flexible-extract agrees with hand label | format-aware agrees with hand label |",
                  "|---|---|---|---|"]
            for c in CONDS + ["all"]:
                rr = rows if c == "all" else [r for r in rows if r["condition"] == c]
                if not rr:
                    continue
                fa = fx = 0
                for r in rr:
                    pe = {x["doc_id"]: x for x in rows_of(r["condition"], "gsm8k") or []}.get(int(r["doc_id"]))
                    if pe is None:
                        continue
                    hc = yes(r["human_correct_yes_no"])
                    fx += (pe["correct"] == 1) == hc
                    fa += (pe["format_aware_correct"] == 1) == hc
                md.append(f"| {'All' if c == 'all' else LABEL[c]} | {len(rr)} | {fx}/{len(rr)} | {fa}/{len(rr)} |")
            md.append("\nRows are mostly sampled where the two extractors disagree, so agreement here is not the "
                      "scorer's overall accuracy; it shows which extractor is right when they differ.")
            save_table("T16_gsm8k_extraction_audit", md, [])


def table_data():
    d = opt(os.path.join("data", "data_prep_stats.json"))
    if not d:
        return
    md = ["| Quantity | Value |", "|---|---|"]
    for k in ("examined", "selected", "excluded_empty", "excluded_gsm8k_test_match", "excluded_too_long",
              "excluded_template_error", "last_selected_source_idx", "tokens_total", "response_tokens_total"):
        md.append(f"| {k} | {d.get(k)} |")
    md.append(f"| tokens/example (mean, median, p95, max) | {d['tokens_per_example']} |")
    md.append(f"| MetaMathQA types | {d['metamath_type_counts']} |")
    save_table("T12_data_curation", md, [])


# --------------------------------------------------------------------------------------------
# Figures (sized for a 5.5-inch NeurIPS text width; 8-9 pt text at print size)
# --------------------------------------------------------------------------------------------
def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
                         "figure.constrained_layout.use": True})
    return plt


def logy(ax):
    """Log y-axis with major tick labels only (minor labels collide in small panels)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    if lo > 0 and np.log10(hi / lo) < 1.3:      # under ~one decade: label the 2x and 5x minor ticks
        ax.yaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0)))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())


def savefig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info("figure %s", name)


def smooth(y, w=25):
    y = np.array([np.nan if v is None else v for v in y], float)
    if np.isnan(y).any():                       # skipped/non-finite updates are logged as null
        idx = np.arange(len(y))
        ok = ~np.isnan(y)
        y = np.interp(idx, idx[ok], y[ok]) if ok.any() else y
    if len(y) < w:
        return y
    k = np.ones(w) / w
    out = np.convolve(y, k, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])


def fig_training(plt):
    T = {m: opt(os.path.join(R, "training", f"{m}.json")) for m in METHODS}
    T = {m: d for m, d in T.items() if d}
    if not T:
        return
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for m, d in T.items():
        u = [x["update"] for x in d["updates"]]
        y = [x["sft_loss"] for x in d["updates"]]
        ax[0].plot(u, [np.nan if v is None else v for v in y], color=COLOR[m], alpha=0.15, linewidth=0.6)
        ax[0].plot(u, smooth(y), color=COLOR[m], label=LABEL[m])
        ax[1].plot(u, [x["lr"] for x in d["updates"]], color=COLOR[m], label=LABEL[m])
    ax[0].axvline(int(MID), color="0.5", linestyle=":", linewidth=0.8)
    ax[0].set(xlabel="optimizer update", ylabel="SFT loss (response tokens)", title="(a) SFT loss")
    ax[0].legend(frameon=False, ncol=1)
    ax[1].set(xlabel="optimizer update", ylabel="learning rate", title="(b) schedule")
    logy(ax[1])
    savefig(fig, "F1_sft_loss")

    # grad norms (single axis, log scale)
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    for m, d in T.items():
        u = [x["update"] for x in d["updates"]]
        ax.plot(u, smooth([x["grad_norm_pre_clip"] for x in d["updates"]]), color=COLOR[m], label=LABEL[m])
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set(xlabel="optimizer update", ylabel="pre-clip gradient norm")
    logy(ax)
    ax.legend(frameon=False, ncol=3)
    savefig(fig, "F8_grad_norms")

    if "full_ft" in T and "l2sp" in T:
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2))
        l2 = T["l2sp"]["updates"]
        u = [x["update"] for x in l2 if "omega" in x]
        ax[0].plot(u, [np.nan if x["sft_loss"] is None else x["sft_loss"] for x in l2 if "omega" in x],
                   color=COLOR["l2sp"], alpha=0.4, label="SFT loss")
        ax[0].plot(u, [x["lambda_omega"] for x in l2 if "omega" in x], color="0.2", label=r"$\lambda\Omega$")
        ax[0].set(xlabel="optimizer update", ylabel="loss term", title="(a) L2-SP objective terms")
        logy(ax[0])
        ax[0].legend(frameon=False)
        for m in ("full_ft", "l2sp"):
            tr = T[m]["displacement_trace"]
            ax[1].plot([x["update"] for x in tr], [x["displacement"] for x in tr], color=COLOR[m], label=LABEL[m])
        ax[1].set(xlabel="optimizer update", ylabel=r"$\|\theta-\theta_0\|_2$", title="(b) displacement")
        ax[1].legend(frameon=False)
        savefig(fig, "F2_l2sp_displacement")


def fig_benchmarks(plt, S, ci):
    conds = [c for c in CONDS[1:] if c in S]
    if not conds:
        return
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.3), sharey=False)
    for ax, (t, tl, key) in zip(axes, TASKS):
        base = S["baseline"][key]
        y = np.arange(len(conds))
        vals = [S[c][key] - base for c in conds]
        err = np.array([[v - ci[c][t][1], ci[c][t][2] - v] if t in ci.get(c, {}) else [0, 0]
                        for c, v in zip(conds, vals)]).T
        bars = ax.barh(y, vals, xerr=err, color=[COLOR[c] for c in conds], height=0.7,
                       error_kw={"linewidth": 0.8, "ecolor": "0.25"})
        for b, c in zip(bars, conds):
            if c == "full_ft_mid":
                b.set_hatch("////")
                b.set_facecolor("white")
                b.set_edgecolor(COLOR[c])
        ax.axvline(0, color="0.3", linewidth=0.7)
        ax.set_title(tl)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[c] for c in conds] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Δ vs baseline (pp)")
    savefig(fig, "F3_benchmark_deltas")

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for c in conds:
        g = S[c]["gsm8k_flexible_em"] - S["baseline"]["gsm8k_flexible_em"]
        r = np.mean([S[c][k] - S["baseline"][k] for _, _, k in TASKS[1:]])
        ax.scatter(g, r, s=36, color=COLOR[c], marker=MARKER[c], edgecolor="white", linewidth=0.8, zorder=3,
                   facecolor="white" if c == "full_ft_mid" else COLOR[c])
        ax.annotate(SHORT[c], (g, r), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
    ax.axhline(0, color="0.5", linewidth=0.7)
    ax.axvline(0, color="0.5", linewidth=0.7)
    ax.set(xlabel="GSM8K Δ (pp)", ylabel="mean retention Δ (pp)")
    savefig(fig, "F4_tradeoff")


def fig_gsm8k(plt, S):
    conds = [c for c in CONDS if c in S]
    rows = {c: opt(os.path.join(R, "benchmarks", c, "gsm8k_per_example.json")) for c in conds}
    rows = {c: r for c, r in rows.items() if r}
    if not rows:
        return
    cats = [("correct", "#1baf7a"), ("wrong (extracted, complete)", "#eb6834"),
            ("unextractable (complete)", "#8c8c8c"), ("truncated at cap", "#2a3a4a")]
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
    left = np.zeros(len(rows))
    shares = []
    for rr in rows.values():
        n = len(rr)
        c1 = sum(r["correct"] > 0 for r in rr)
        tr = sum(r["correct"] == 0 and r["truncated"] for r in rr)
        inv = sum(r["correct"] == 0 and not r["truncated"] and r["flexible_invalid"] for r in rr)
        shares.append(np.array([c1, n - c1 - tr - inv, inv, tr]) / n * 100)
    shares = np.array(shares)
    y = np.arange(len(rows))
    for j, (name, col) in enumerate(cats):
        ax[0].barh(y, shares[:, j], left=left, color=col, height=0.7, label=name, edgecolor="white", linewidth=0.5)
        left += shares[:, j]
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([SHORT[c] for c in rows])
    ax[0].invert_yaxis()
    ax[0].set(xlabel="% of 1,319 test questions", xlim=(0, 100), title="(a) outcome")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax[0].grid(axis="y", visible=False)
    data = [[r["n_generated_tokens"] for r in rr] for rr in rows.values()]
    bp = ax[1].boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], rows):
        patch.set_facecolor(COLOR[c])
        patch.set_alpha(0.6)
    ax[1].set_yticks(np.arange(1, len(rows) + 1))
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].set(xlabel="generated tokens", title="(b) response length")
    savefig(fig, "F6_gsm8k_outcomes")


def fig_diagnostics(plt):
    D = opt(os.path.join(R, "diagnostics", "diagnostics_summary.json"))
    if not D:
        return
    conds = [c for c in CONDS[1:] if c in D]
    layers = list(next(iter(D.values())).keys())
    panels = [("rho_k", r"(a) $\rho_k$, top-128 output subspace", False), ("eta", r"(b) $\eta$", True),
              ("R_math", r"(c) $R(z)$, GSM8K probes", True), ("R_nonmath", r"(d) $R(z)$, HellaSwag probes", True)]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.9))
    x = np.arange(len(layers))
    w = 0.8 / len(conds)
    ref = D[conds[0]][layers[0]]["rho_random_reference"]
    for ax, (key, title, log) in zip(axes.flat, panels):
        for i, c in enumerate(conds):
            v = [D[c][l][key]["mean"] if isinstance(D[c][l][key], dict) else D[c][l][key] for l in layers]
            ax.bar(x - 0.4 + w * (i + 0.5), v, w * 0.9, color="white" if c == "full_ft_mid" else COLOR[c],
                   edgecolor=COLOR[c], hatch="////" if c == "full_ft_mid" else None, linewidth=0.6, label=SHORT[c])
        if key == "rho_k":
            ax.axhline(ref, color="0.3", linestyle="--", linewidth=0.8)
            ax.annotate(r"isotropic $k/d_{out}$", (x[-1] + 0.45, ref), xytext=(0, 3), textcoords="offset points",
                        ha="right", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {l}" for l in layers])
        ax.set_title(title, loc="left")
        if log:
            logy(ax)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=6, frameon=False)
    savefig(fig, "F5_diagnostics")

    L = opt(os.path.join(R, "diagnostics", "layerwise_eta.json"))
    if L:
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
        for c in [c for c in CONDS[1:] if c in L]:
            for j, mod in enumerate(("mlp.down_proj", "self_attn.q_proj")):
                ys = [(int(n.split(".")[2]), v) for n, v in L[c].items() if n.endswith(mod)]
                ys.sort()
                ax[j].plot([a for a, _ in ys], [b for _, b in ys], color=COLOR[c], marker=MARKER[c], markersize=2.5,
                           linestyle="--" if c == "full_ft_mid" else "-", label=SHORT[c])
                ax[j].set(title=mod, xlabel="layer")
                logy(ax[j])
        ax[0].set_ylabel(r"$\eta=\|\Delta W\|_F/\|W_0\|_F$")
        ax[1].legend(frameon=False, ncol=2, fontsize=6.5)
        savefig(fig, "F7_layerwise_eta")


def fig_strata(plt, strata):
    """F9: share of documents whose answer changed vs baseline, by baseline-confidence quintile."""
    tasks = [(t, tl) for t, tl, _ in TASKS[1:] if any(t in v for v in strata.values())]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (t, tl) in zip(axes, tasks):
        for c in CONDS[1:]:
            v = strata.get(c, {}).get(t)
            if not v:
                continue
            ax.plot([x["quintile"] for x in v], [x["changed_pct"] for x in v], color=COLOR[c], marker=MARKER[c],
                    markersize=3, linestyle="--" if c == "full_ft_mid" else "-", label=SHORT[c])
        ax.set(title=tl, xlabel="baseline confidence quintile", xticks=[1, 2, 3, 4, 5])
    axes[0].set_ylabel("answers changed vs baseline (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=6, frameon=False)
    savefig(fig, "F9_changes_by_confidence")


def main():
    setup_logging()
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    os.makedirs(TAB, exist_ok=True)
    S = opt(os.path.join(R, "benchmarks", "all_conditions_summary.json")) or {}
    ci = {}
    if "baseline" in S:
        ci = table_benchmarks(S)
        table_controlled(S)
        table_forgetting(S)
        table_mc_behaviour(S)
        table_gsm8k(S)
        table_gsm8k_extra(S)
        table_stability(S)
        strata = table_strata(S)
        write_extraction_audit(S)
    table_audits()
    table_resources()
    table_l2sp()
    table_diagnostics()
    table_qualitative()
    table_data()
    try:
        plt = mpl()
    except ImportError:
        logger.warning("matplotlib missing; figures skipped")
        return
    fig_training(plt)
    if "baseline" in S:
        fig_benchmarks(plt, S, ci)
        fig_gsm8k(plt, S)
        fig_strata(plt, strata)
    fig_diagnostics(plt)
    logger.info("Done: tables in %s, figures in %s", TAB, FIG)


if __name__ == "__main__":
    main()
