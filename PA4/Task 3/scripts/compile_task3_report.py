"""Compile Task 3 tables (results/tables), figures (results/figures) and analysis JSON (results/analysis).
CPU only; every table/figure is independent and skipped with a message when its inputs are missing.
  python scripts/compile_task3_report.py [--results results] [--config configs/task3.yaml]
Figures follow the PA3 house style (5.5-inch text width, 8 pt text, "(a)" panel titles, labelled axes/colour bars).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import RESULTS, load_config, load_json, save_json  # noqa: E402

DRAFTS = ["original", "off_policy_s6304", "off_policy_s6305", "on_policy_s6304", "on_policy_s6305"]
DLAB = {"original": "original", "off_policy_s6304": "off-policy s6304", "off_policy_s6305": "off-policy s6305",
        "on_policy_s6304": "on-policy s6304", "on_policy_s6305": "on-policy s6305", "target": "target (1.5B)"}
DCOL = {"original": "#7f7f7f", "off_policy_s6304": "#1f77b4", "off_policy_s6305": "#6baed6",
        "on_policy_s6304": "#d62728", "on_policy_s6305": "#fc9272"}
DSHORT = {"original": "orig.", "off_policy_s6304": "off 6304", "off_policy_s6305": "off 6305",
          "on_policy_s6304": "on 6304", "on_policy_s6305": "on 6305"}
WL = [("math", "mathematics"), ("short_familiar", "familiar extraction"), ("short_shifted", "shifted extraction")]
BINS = ["1-16", "17-32", "33-64", "65-192"]
BANKS = ["teacher", "exact_spec_original_gamma4", "original_draft", "student_off_policy_s6304",
         "student_off_policy_s6305", "student_on_policy_s6304", "student_on_policy_s6305"]
BLAB = {"teacher": "teacher (i)", "original_draft": "original draft (ii)", "exact_spec_original_gamma4": "exact spec. (iii)",
        **{f"student_{d}": f"own: {DLAB[d]}" for d in DRAFTS[1:]}}
METRICS = [("fkl", "forward KL"), ("rkl", "reverse KL"), ("tv", "TV"), ("exp_acc", "1 - TV"), ("agree", "greedy agree.")]


class R:
    def __init__(self, res: Path, cfg):
        self.res, self.cfg = res, cfg
        self.tab, self.fig, self.ana = res / "tables", res / "figures", res / "analysis"
        for d in (self.tab, self.fig, self.ana):
            d.mkdir(parents=True, exist_ok=True)
        self._c = {}

    def j(self, *p):
        f = self.res.joinpath(*p)
        return load_json(f) if f.exists() else None

    def align(self):
        if "align" not in self._c:
            f = self.res / "align" / "align_rows.jsonl.gz"
            self._c["align"] = [json.loads(x) for x in gzip.open(f, "rt")] if f.exists() else None
        return self._c["align"]

    def bench(self, name):
        k = ("bench", name)
        if k not in self._c:
            self._c[k] = self.j("bench", f"{name}.json.gz")
        return self._c[k]


def write_table(r, name, header, rows, note=""):
    with open(r.tab / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
            ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    (r.tab / f"{name}.md").write_text("\n".join(lines + (["", note] if note else [])) + "\n", encoding="utf-8")


def f3(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "lines.linewidth": 1.2, "lines.markersize": 3.5, "savefig.bbox": "tight", "savefig.dpi": 300,
                         "pdf.fonttype": 42, "figure.constrained_layout.use": True})
    return plt


def save_fig(r, fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(r.fig / f"{name}.pdf")
    fig.savefig(r.fig / f"{name}.png", dpi=300)
    plt.close(fig)


def title(ax, t):
    ax.set_title(t, loc="left")


def boot_ci(x, B=10000, seed=6304, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [np.nan, np.nan]
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    return np.percentile(stat(x[idx], axis=1), [2.5, 97.5]).tolist()


def boot_ratio_ci(num, den, B=10000, seed=6304):
    """CI of sum(num)/sum(den) with resampling of prompts (num/den are per-prompt sums)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (B, len(num)))
    return np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5]).tolist()


def per_prompt(rows) -> dict:
    """Collapse benchmark runs to prompts (the bootstrap unit prescribed by the manual): the repetitions of a prompt
    are summed (acceptance counts, times) or averaged (speedup ratio)."""
    g = defaultdict(list)
    for x in rows:
        g[x["prompt_id"]].append(x)
    ids = sorted(g)
    f = lambda fn: np.array([fn(g[i]) for i in ids], float)  # noqa: E731
    return {"ids": ids, "acc": f(lambda v: sum(x["spec"]["accepted"] for x in v)),
            "exam": f(lambda v: sum(x["spec"]["examined"] for x in v)),
            "ratio": f(lambda v: np.mean([x["target"]["wall"] / x["spec"]["wall"] for x in v])),
            "passes": f(lambda v: sum(x["spec"]["rounds"] for x in v)),
            "tokens": f(lambda v: sum(x["spec"]["n_tokens"] for x in v))}


def paired_prompt_diff(a: dict, b: dict, B=10000, seed=6304) -> dict:
    """Paired (same prompts) differences a - b with a prompt bootstrap: pooled acceptance (pp), mean per-prompt
    speedup, pooled target passes per output token."""
    common = sorted(set(a["ids"]) & set(b["ids"]))
    ia = np.array([a["ids"].index(i) for i in common])
    ib = np.array([b["ids"].index(i) for i in common])
    idx = np.random.default_rng(seed).integers(0, len(common), (B, len(common)))

    def stat(fa, fb):
        point = fa(np.arange(len(common))[None]) - fb(np.arange(len(common))[None])
        bs = fa(idx) - fb(idx)
        return float(point[0]), np.percentile(bs, [2.5, 97.5]).tolist()
    acc = stat(lambda I: 100 * a["acc"][ia][I].sum(1) / a["exam"][ia][I].sum(1),
               lambda I: 100 * b["acc"][ib][I].sum(1) / b["exam"][ib][I].sum(1))
    spd = stat(lambda I: a["ratio"][ia][I].mean(1), lambda I: b["ratio"][ib][I].mean(1))
    pas = stat(lambda I: a["passes"][ia][I].sum(1) / a["tokens"][ia][I].sum(1),
               lambda I: b["passes"][ib][I].sum(1) / b["tokens"][ib][I].sum(1))
    return {"n_prompts": len(common), "acceptance_pp": acc, "speedup": spd, "passes_per_token": pas}


def comparisons(names) -> list:
    """Main paired comparisons: every trained draft vs the original; on- vs off-policy at each seed."""
    out = [(d, "original") for d in names if d != "original" and "original" in names]
    out += [(f"on_policy_s{s}", f"off_policy_s{s}") for s in (6304, 6305)
            if f"on_policy_s{s}" in names and f"off_policy_s{s}" in names]
    return out


# ===================================================================================== sampler correctness
def t_sampler(r, plt):
    v = r.j("validation", "validation.json")
    if not v:
        return "no validation"
    toy = v["toy_chi2"]
    rows = [[k, x["gamma"], x["residual"], f"{x['chi2']:.1f}", x["df"], f"{x['p_value']:.3g}",
             "rejected" if x["rejected_at_alpha"] else "not rejected", f"{x['max_abs_joint_error']:.4f}",
             f"{x['acceptance_rate']:.3f}", x["bonus_rounds"]] for k, x in toy["runs"].items()]
    write_table(r, "T1_toy_chi2", ["Sampler", "gamma", "Residual", "Pearson chi2", "df", "p", f"Decision (alpha {toy['alpha']})",
                                   "max |joint error|", "acceptance", "bonus rounds"], rows,
                f"{toy['n']:,} fixed-seed length-3 sequences per sampler; 3-symbol second-order Markov target and draft "
                f"(the draft depends on cached history, so rollback errors would bias the joint); 27 joint categories, "
                f"minimum expected count {toy['min_expected']:.0f}. 'p' residual = deliberately incorrect sampler.")
    keys = list(toy["exact_joint"])
    ex = np.array([toy["exact_joint"][k] for k in keys])
    fig, ax = plt.subplots(2, 1, figsize=(5.5, 3.0), sharex=True)
    for a, run, lab in [(ax[0], "correct_gamma4", "(a) correct sampler, gamma = 4"),
                        (ax[1], "incorrect_residual_p_gamma2", "(b) incorrect sampler (residual replaced by p)")]:
        em = np.array([toy["runs"][run]["empirical_joint"][k] for k in keys])
        x = np.arange(len(keys))
        a.bar(x - 0.2, ex, 0.4, color="0.6", label="exact target joint")
        a.bar(x + 0.2, em, 0.4, color="#1f77b4" if run.startswith("correct") else "#d62728", label="empirical")
        a.set_ylabel("probability")
        title(a, f"{lab}: p = {toy['runs'][run]['p_value']:.2g}")
        a.grid(axis="x", visible=False)
    ax[1].set_xticks(range(len(keys)))
    ax[1].set_xticklabels(keys, rotation=90, fontsize=6)
    ax[1].set_xlabel("sequence (x1 x2 x3)")
    ax[0].legend(frameon=False, ncol=2)
    save_fig(r, fig, "F1_toy_joint")
    rd = v.get("rounds", {})
    chk = [["tokenizer: chat template / token IDs on all prompts", f"{v['tokenizer']['chat_template_mismatches']} / "
            f"{v['tokenizer']['token_id_mismatches']} mismatches ({v['tokenizer']['n_prompts']} prompts)"],
           ["split-prefill equivalence: max |dp|", f"{v['split_prefill']['max_abs_prob_diff']:.2e}"],
           ["cache consistency in the sampler: max |dp| target / draft",
            f"{v['cache_consistency']['target_max_err']:.2e} / {v['cache_consistency']['draft_max_err']:.2e}"],
           ["fully accepted rounds (draft = target): acceptance, bonus rounds",
            f"{rd['draft_equals_target']['acceptance']:.4f}, {rd['draft_equals_target']['bonus']}"],
           ["one-token rounds (flattened draft): rounds with 0 accepted", f"{rd['flat_draft']['zero_acceptance_rounds']} of "
            f"{rd['flat_draft']['rounds']}"],
           ["final-token truncation (caps 1, 2, 5, 7, 11; gamma 4)", str(all(x["ok"] for x in rd["final_token_truncation"]))],
           ["EOS handling (20 short-form runs)", str(rd["eos"]["all_ok"])],
           ["padding: left-padded batch vs single, max |dp|", f"{v['padding']['left_padded_batch_vs_single_max_abs_prob_diff']:.2e}"]]
    for c in v.get("real_consistency", []):
        chk.append([f"real model target vs speculative, {c['prompt']}: first token / 3-token joint p",
                    f"{c['first_token']['p_value']:.3f} / {c['three_token_joint']['p_value']:.3f}"])
    write_table(r, "T2_validation_checks", ["Check", "Result"], chk,
                "Real-model checks run before any timing. dp = absolute difference of next-token probabilities. The "
                "real-model consistency rows compare 400 target-only with 400 speculative samples per prompt (chi-square "
                "homogeneity; large p = no detectable difference).")


# ============================================================================================ distillation
def t_distill(r, plt):
    runs = {n: r.j("distill", n, "summary.json") for n in DRAFTS[1:]}
    runs = {n: s for n, s in runs.items() if s}
    if not runs:
        return "no distillation"
    merge = {n: r.j("merge", f"{n}.json") for n in runs}
    rows = []
    for n, s in runs.items():
        t = s["timings_s"]
        pool = s.get("teacher_pool") or {}
        m = merge.get(n) or {}
        rows.append([DLAB[n], s["updates"], s["skipped_updates"], s["scored_tokens"]["math"], s["scored_tokens"]["short"],
                     s["presentations"], s["truncated_by_quota"], f"{s['response_length_mean']['math']:.0f} / {s['response_length_mean']['short']:.1f}",
                     f"{s['final_kl']:.4f}", f"{pool.get('teacher_generation_seconds', 0) / 60:.1f}", f"{t['teacher_scoring_s'] / 60:.1f}",
                     f"{t['student_rollout_s'] / 60:.1f}", f"{t['student_update_s'] / 60:.1f}",
                     f"{m.get('merge_seconds', float('nan')):.1f}", f"{(s['peak_allocated_bytes'] or 0) / 2**30:.2f}"])
    s0 = next(iter(runs.values()))
    write_table(r, "T3_distillation_runs", ["Run", "Updates", "Skipped (overflow)", "Math tokens", "Short tokens", "Presentations",
                                            "Quota-truncated", "Mean response length math / short", "Final-update KL",
                                            "Teacher generation min", "Teacher scoring min", "Student rollout min",
                                            "Backward + update min", "Merge s", "Peak GiB"], rows,
                f"Each run: 40,000 teacher-scored tokens, one AdamW update per 256 (last 64). Teacher generation = the "
                f"frozen teacher-response pool (off-policy only; counted per seed). Student: {s0['params']['total_unique']:,} "
                f"parameters with LoRA, {s0['params']['trainable']:,} trainable (adapters, r = 8 on q/k/v/o).")
    mrows = [[DLAB[n], f"{m['merged_vs_unmerged_max_abs_logit']:.3e}", f"{m['merged_vs_unmerged_mean_kl']:.2e}",
              f"{m['merged_vs_unmerged_argmax_agreement']:.4f}", f"{m['merge_seconds']:.2f}"] for n, m in merge.items() if m]
    if mrows:
        write_table(r, "T4_merge_check", ["Run", "max |logit merged - unmerged|", "mean KL", "argmax agreement", "merge s"], mrows,
                    "Merged FP16 checkpoint vs the unmerged adapter on the same prompt batch (8 held-out prompts).")
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
    for a, seed in zip(ax, (6304, 6305)):
        for meth, col in (("off_policy", "#1f77b4"), ("on_policy", "#d62728")):
            u = r.j("distill", f"{meth}_s{seed}", "updates.json")
            if u:
                x = np.array([z["scored_total"] for z in u]) / 1000
                y = np.array([z["kl"] for z in u])
                k = 9
                if len(y) >= k:
                    a.plot(x, y, color=col, alpha=0.25, lw=0.6)
                    a.plot(x[k - 1:], np.convolve(y, np.ones(k) / k, "valid"), color=col, label=meth.replace("_", "-"))
                else:
                    a.plot(x, y, color=col, marker=".", label=meth.replace("_", "-"))
        a.set(xlabel="teacher-scored tokens (thousands)", yscale="log")
        title(a, f"({'ab'[seed - 6304]}) seed {seed}")
    ax[0].set_ylabel("forward KL per token (training)")
    ax[0].legend(frameon=False)
    save_fig(r, fig, "F2_distillation_curves")


# ===================================================================================== distribution alignment
def _agg(rows, d, key):
    v = [x[d][key] for x in rows if d in x]
    return float(np.mean(v)) if v else np.nan


def t_align(r, plt):
    A = r.align()
    if not A:
        return "no alignment"
    drafts = [d for d in DRAFTS if d in A[0]]
    by_bank = defaultdict(list)
    for x in A:
        by_bank[x["bank"]].append(x)
    rows, out = [], {}
    for b in [b for b in BANKS if b in by_bank]:
        for d in drafts:
            vals = [f"{_agg(by_bank[b], d, k):.4f}" for k, _ in METRICS]
            rows.append([BLAB[b], DLAB[d], len(by_bank[b])] + vals)
            out[f"{b}|{d}"] = {k: _agg(by_bank[b], d, k) for k, _ in METRICS}
    write_table(r, "T5_alignment_common_prefixes", ["Prefix bank", "Draft", "Positions"] + [l for _, l in METRICS], rows,
                "Token-weighted means over the same saved prefixes for every draft (full 151,936-token vocabulary, "
                "temperature 1). Banks (i) and (iii) sample the same target-prefix law; 'own' banks are each trained "
                "student's frozen rollouts, evaluated by every draft.")
    save_json(r.ana / "alignment_means.json", out)
    # complete grid (machine-readable): every metric x bank x draft x workload x position bin
    grid = {}
    for b in by_bank:
        for w, _ in WL:
            for bn in BINS:
                sub = [x for x in by_bank[b] if x["workload"] == w and x["bin"] == bn]
                if sub:
                    grid[f"{b}|{w}|{bn}"] = {"n": len(sub), "target_entropy": float(np.mean([x["target_entropy"] for x in sub])),
                                             **{d: {k: _agg(sub, d, k) for k, _ in METRICS} for d in drafts}}
    save_json(r.ana / "alignment_grid.json", grid)
    # paired comparisons on common prefixes (prompt bootstrap; a prompt's value = mean over its saved positions)
    rows, pc = [], {}
    B = r.cfg["eval"]["bootstrap"]
    for b in [b for b in ("teacher", "exact_spec_original_gamma4", "original_draft") if b in by_bank]:
        g = defaultdict(list)
        for x in by_bank[b]:
            g[x["prompt_id"]].append(x)
        ids = sorted(g)
        idx = np.random.default_rng(B["seed"]).integers(0, len(ids), (B["resamples"], len(ids)))
        for a_, b_ in comparisons(drafts):
            cells = [f"{DLAB[a_]} - {DLAB[b_]}", BLAB[b], len(ids)]
            for k, sc in (("exp_acc", 100), ("fkl", 1)):
                diff = np.array([np.mean([x[a_][k] - x[b_][k] for x in g[i]]) for i in ids]) * sc
                ci_ = np.percentile(diff[idx].mean(1), [2.5, 97.5])
                pc[f"{a_}-{b_}|{b}|{k}"] = [float(diff.mean()), ci_.tolist()]
                cells.append(f"{diff.mean():+.{2 if sc == 100 else 4}f} [{ci_[0]:+.{2 if sc == 100 else 4}f}, "
                             f"{ci_[1]:+.{2 if sc == 100 else 4}f}]")
            rows.append(cells)
    if rows:
        write_table(r, "T5b_paired_alignment_differences", ["Comparison (a - b)", "Prefix bank", "Prompts",
                                                            "1 - TV difference (pp) [95% CI]", "Forward KL difference [95% CI]"], rows,
                    "Same prefixes for both drafts; a prompt's value is the mean over its saved positions (prompts weighted "
                    "equally, unlike the token-weighted T5); 95% CIs from 10,000 bootstrap resamples of prompts.")
        save_json(r.ana / "paired_alignment_differences.json", pc)
    # seed dispersion for the two methods on the teacher bank
    rows = []
    for b in ("teacher", "exact_spec_original_gamma4", "original_draft"):
        if b not in by_bank:
            continue
        for meth in ("off_policy", "on_policy"):
            v = [out.get(f"{b}|{meth}_s{s}", {}).get("exp_acc") for s in (6304, 6305)]
            v = [x for x in v if x is not None and np.isfinite(x)]
            if v:
                rows.append([BLAB[b], meth.replace("_", "-"), f"{np.mean(v):.4f}", f"{(max(v) - min(v)) / 2:.4f}",
                             " / ".join(f"{x:.4f}" for x in v)])
    if rows:
        write_table(r, "T6_seed_dispersion", ["Bank", "Method", "Mean 1 - TV over seeds", "Half range", "Seeds 6304 / 6305"], rows)
    # workload x position (teacher bank + own banks)
    rows = []
    for b in ("teacher", "exact_spec_original_gamma4"):
        if b not in by_bank:
            continue
        for w, wl in WL:
            for bn in BINS:
                sub = [x for x in by_bank[b] if x["workload"] == w and x["bin"] == bn]
                if sub:
                    rows.append([BLAB[b], wl, bn, len(sub)] + [f"{_agg(sub, d, 'exp_acc'):.3f}" for d in drafts] +
                                [f"{np.mean([x['target_entropy'] for x in sub]):.2f}"])
    write_table(r, "T7_alignment_by_workload_position", ["Bank", "Workload", "Positions", "n"] + [f"1-TV {DLAB[d]}" for d in drafts] +
                ["target entropy (nats)"], rows,
                "Expected one-token acceptance 1 - TV by workload and response-position bin (65-192 = long-response bank).")
    # own-rollout exposure: every draft on every student's own bank
    own = [b for b in BANKS if b.startswith("student_") and b in by_bank]
    if own:
        rows = [[BLAB[b]] + [f"{_agg(by_bank[b], d, 'fkl'):.4f} / {_agg(by_bank[b], d, 'exp_acc'):.3f}" for d in drafts] for b in own]
        write_table(r, "T8_student_rollout_banks", ["Frozen rollout bank (producer)"] + [DLAB[d] for d in drafts], rows,
                    "Forward KL / expected acceptance of every draft on each trained student's own frozen rollouts "
                    "(paired: same prefixes for every draft). The diagonal is a draft on its own rollouts.")
    # aligned disagreements: largest TV improvement / degradation of the on-policy s6304 draft vs the original
    dd = "on_policy_s6304" if "on_policy_s6304" in drafts else (drafts[1] if len(drafts) > 1 else None)
    tok = None
    try:
        from transformers import AutoTokenizer
        from src.data import revisions
        tok = AutoTokenizer.from_pretrained(r.cfg["models"]["target"], revision=revisions(r.cfg)["target"])
    except Exception:
        pass

    def dec(i):
        return repr(tok.decode([i])) if tok is not None else str(i)
    if dd and "teacher" in by_bank:
        T = sorted(by_bank["teacher"], key=lambda x: x[dd]["tv"] - x["original"]["tv"])
        rows = []
        for x in T[:6] + T[-6:]:
            rows.append([x["prompt_id"], x["position"], f"{dec(x['p_argmax'])} ({x['p_argmax_prob']:.2f})",
                         f"{dec(x['original']['q_argmax'])} ({x['original']['q_argmax_prob']:.2f})", f"{x['original']['tv']:.3f}",
                         f"{dec(x[dd]['q_argmax'])} ({x[dd]['q_argmax_prob']:.2f})", f"{x[dd]['tv']:.3f}"])
        write_table(r, "T9_prefix_disagreements", ["Prompt", "Position", "Target top token (p)", "Original top token (q)", "TV original",
                                                   f"{DLAB[dd]} top token (q)", f"TV {DLAB[dd]}"], rows,
                    f"Teacher-bank prefixes with the largest TV improvement (first 6) and degradation (last 6) of "
                    f"{DLAB[dd]} relative to the original draft.")
    # consistency: teacher (i) vs exact speculative (iii)
    if "teacher" in by_bank and "exact_spec_original_gamma4" in by_bank:
        rows = []
        bk = {b: r.j("banks", f"{b}.json.gz") for b in ("teacher", "exact_spec_original_gamma4")}
        for key, lab in [("exp_acc", "1 - TV"), ("fkl", "forward KL"), ("tv", "TV")]:
            for d in drafts:
                per = {}
                for b in ("teacher", "exact_spec_original_gamma4"):
                    g = defaultdict(list)
                    for x in by_bank[b]:
                        g[x["prompt_id"]].append(x[d][key])
                    per[b] = np.array([np.mean(v) for v in g.values()])
                diff = per["exact_spec_original_gamma4"].mean() - per["teacher"].mean()
                rng = np.random.default_rng(6304)
                bs = [rng.choice(per["exact_spec_original_gamma4"], len(per["exact_spec_original_gamma4"])).mean() -
                      rng.choice(per["teacher"], len(per["teacher"])).mean() for _ in range(4000)]
                ci_ = np.percentile(bs, [2.5, 97.5])
                rows.append([lab, DLAB[d], f"{per['teacher'].mean():.4f}", f"{per['exact_spec_original_gamma4'].mean():.4f}",
                             f"{diff:+.4f} [{ci_[0]:+.4f}, {ci_[1]:+.4f}]"])
        for stat, f in [("response length", lambda b: [x["source_length"] for x in b["records"]]),
                        ("EOS-terminated share", lambda b: [x["termination"] == "eos" for x in b["records"]])]:
            a_, b_ = np.array(f(bk["teacher"]), float), np.array(f(bk["exact_spec_original_gamma4"]), float)
            rows.append([stat, "-", f"{a_.mean():.3f}", f"{b_.mean():.3f}", f"{b_.mean() - a_.mean():+.3f}"])
        rows.append(["target entropy at saved positions", "-", f"{np.mean([x['target_entropy'] for x in by_bank['teacher']]):.3f}",
                     f"{np.mean([x['target_entropy'] for x in by_bank['exact_spec_original_gamma4']]):.3f}", ""])
        write_table(r, "T10_target_vs_speculative_consistency", ["Statistic", "Draft", "Teacher bank (i)", "Speculative bank (iii)",
                                                                  "(iii) - (i) [unpaired 95% CI over prompts]"], rows,
                    "Both banks sample the target-prefix law (different seeds), so differences should be within sampling "
                    "uncertainty; a systematic gap would indicate an implementation or evaluation problem.")
    # figures: 1-TV by position and workload; draft x bank heat map
    if "teacher" in by_bank:
        fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.1), sharey=True)
        for a, (w, wl) in zip(ax, WL):
            for d in drafts:
                ys = [_agg([x for x in by_bank["teacher"] if x["workload"] == w and x["bin"] == bn], d, "exp_acc") for bn in BINS]
                a.plot(range(4), ys, marker="o", color=DCOL[d], label=DLAB[d])
            a.set(xticks=range(4), xlabel="response position")
            a.set_xticklabels(BINS, fontsize=6.5)
            title(a, f"({'abc'[WL.index((w, wl))]}) {wl}")
        ax[0].set_ylabel("expected acceptance 1 - TV")
        fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
        save_fig(r, fig, "F3_alignment_by_position")
        banks = [b for b in BANKS if b in by_bank]
        M = np.array([[_agg(by_bank[b], d, "exp_acc") for b in banks] for d in drafts])
        fig, ax = plt.subplots(figsize=(5.5, 2.0))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] < np.nanmean(M) else "black")
        ax.set(xticks=range(len(banks)), yticks=range(len(drafts)), yticklabels=[DLAB[d] for d in drafts])
        ax.set_xticklabels([BLAB[b] for b in banks], rotation=25, ha="right", fontsize=6.5)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("mean 1 - TV")
        cb.formatter.set_useOffset(False)
        cb.update_ticks()
        save_fig(r, fig, "F4_alignment_draft_by_bank")


# =================================================================================== speculative decoding
def _bench_rows(r, name):
    b = r.bench(name)
    return b["rows"] if b else []


def _spec_stats(rows):
    sp = [x["spec"] for x in rows]
    tg = [x["target"] for x in rows]
    acc, exa, pro = sum(s["accepted"] for s in sp), sum(s["examined"] for s in sp), sum(s["proposed"] for s in sp)
    out_tok = sum(s["n_tokens"] for s in sp)
    rounds = sum(s["rounds"] for s in sp)
    ratio = [t["wall"] / s["wall"] for s, t in zip(sp, tg)]                  # per-prompt speedup (time ratio)
    tps_s = [s["n_tokens"] / s["wall"] for s in sp]
    tps_t = [t["n_tokens"] / t["wall"] for t in tg]
    return {"acc_exam": acc / max(1, exa), "acc_prop": acc / max(1, pro), "acc_per_round": acc / max(1, rounds),
            "passes_per_tok": rounds / max(1, out_tok), "draft_fw_per_tok": sum(s["draft_forwards"] for s in sp) / max(1, out_tok),
            "tps_spec": float(np.median(tps_s)), "tps_target": float(np.median(tps_t)), "ratio": ratio,
            "speedup_median": float(np.median(ratio)), "speedup_total": sum(t["wall"] for t in tg) / sum(s["wall"] for s in sp),
            "lat_spec": float(np.median([s["wall"] for s in sp])), "lat_target": float(np.median([t["wall"] for t in tg])),
            "peak_spec": max((s["peak_allocated_bytes"] or 0) for s in sp), "peak_target": max((t["peak_allocated_bytes"] or 0) for t in tg),
            "eos_spec": float(np.mean([s["termination"] == "eos" for s in sp])),
            "eos_target": float(np.mean([t["termination"] == "eos" for t in tg])), "n": len(rows),
            "accepted": [s["accepted"] for s in sp], "examined": [s["examined"] for s in sp]}


def t_spec(r, plt):
    have = [d for d in DRAFTS if r.bench(d)]
    if not have:
        return "no benchmark"
    B = r.cfg["eval"]["bootstrap"]
    rows, stats = [], {}
    for d in have:
        for g in r.cfg["eval"]["gammas"]:
            rs = [x for x in _bench_rows(r, d) if x["gamma"] == g]
            if not rs:
                continue
            s = _spec_stats(rs)
            stats[(d, g)] = s
            pp = per_prompt(rs)
            s["prompt_speedup_median"] = float(np.median(pp["ratio"]))
            ci_s = boot_ci(pp["ratio"], B["resamples"], B["seed"], np.median)
            ci_a = boot_ratio_ci(pp["acc"], pp["exam"], B["resamples"], B["seed"])
            rows.append([DLAB[d], g, f"{len(pp['ids'])} x {s['n'] // max(1, len(pp['ids']))}",
                         f"{s['acc_exam']:.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}]", f"{s['acc_prop']:.3f}",
                         f"{s['acc_per_round']:.2f}", f"{s['passes_per_tok']:.3f}", f"{s['draft_fw_per_tok']:.2f}",
                         f"{s['tps_spec']:.1f}", f"{s['tps_target']:.1f}",
                         f"{s['prompt_speedup_median']:.3f} [{ci_s[0]:.3f}, {ci_s[1]:.3f}]", f"{s['lat_spec'] * 1000:.0f} / {s['lat_target'] * 1000:.0f}",
                         f"{s['peak_spec'] / 2**30:.2f} / {s['peak_target'] / 2**30:.2f}", f"{s['eos_spec']:.2f} / {s['eos_target']:.2f}"])
    write_table(r, "T11_speculative_main", ["Draft", "gamma", "Prompts x reps", "Accepted / examined [95% CI]", "Accepted / proposed",
                                            "Accepted per verification", "Target passes / output token", "Draft passes / token",
                                            "Tok/s spec (median run)", "Tok/s target-only (median run)",
                                            "Speedup, median over prompts [95% CI]",
                                            "Latency ms spec / target", "Peak GiB spec / target", "EOS share spec / target"], rows,
                "120 prompts x 2 paired repetitions, EOS-aware 64-token cap, temperature 1. Per-run speedup = adjacent "
                "target-only time / speculative time (same prompt, repetition and seed); a prompt's speedup is the mean "
                "over its repetitions. 95% CIs: bootstrap over prompts (10,000 resamples; repetitions stay together).")
    # paired comparisons between drafts (same prompts, repetitions and seeds): prompt bootstrap
    rows, pc = [], {}
    for g in r.cfg["eval"]["gammas"]:
        pps = {d: per_prompt([x for x in _bench_rows(r, d) if x["gamma"] == g]) for d in have}
        pps = {d: v for d, v in pps.items() if v["ids"]}
        for a_, b_ in comparisons(list(pps)):
            c = paired_prompt_diff(pps[a_], pps[b_], B["resamples"], B["seed"])
            pc[f"{a_}-{b_}|{g}"] = c
            rows.append([f"{DLAB[a_]} - {DLAB[b_]}", g, c["n_prompts"]] +
                        [f"{c[k][0]:+.{n}f} [{c[k][1][0]:+.{n}f}, {c[k][1][1]:+.{n}f}]"
                         for k, n in (("acceptance_pp", 1), ("speedup", 3), ("passes_per_token", 4))])
    if rows:
        write_table(r, "T11b_paired_draft_comparisons", ["Comparison (a - b)", "gamma", "Prompts", "Acceptance difference (pp) [95% CI]",
                                                         "Mean speedup difference [95% CI]",
                                                         "Target passes / token difference [95% CI]"], rows,
                    "Paired over the same prompts, repetitions and sampling seeds; 95% CIs from 10,000 bootstrap resamples "
                    "of prompts. Speedups are each draft's own adjacent-target ratios, so the comparison is not affected by "
                    "load differences between benchmark runs. Training seeds are reported separately.")
        save_json(r.ana / "paired_draft_comparisons.json", pc)
    # time decomposition and cost model
    iso = r.j("bench", "isolated_costs.json") or {}
    rows, rows2, cm = [], [], {}
    for (d, g), s in stats.items():
        rs = [x for x in _bench_rows(r, d) if x["gamma"] == g]
        ph = defaultdict(float)
        for x in rs:
            for k, v in x["spec"]["times"].items():
                ph[k] += v
        R_ = sum(x["spec"]["rounds"] for x in rs)
        wall = sum(x["spec"]["wall"] for x in rs)
        host = wall - sum(ph.values())
        rows.append([DLAB[d], g] + [f"{1000 * ph[k] / R_:.2f}" for k in ("draft", "verify", "accept", "rollback")] +
                    [f"{1000 * ph['prefill'] / len(rs):.1f}", f"{1000 * host / len(rs):.2f}",
                     f"{100 * ph['verify'] / wall:.0f}% / {100 * ph['draft'] / wall:.0f}%"])
        t_step = sum(x["target"]["times"].get("step_forward", 0) for x in rs) / max(1, sum(x["target"]["n_tokens"] for x in rs))
        d_fw = ph["draft"] / max(1, sum(x["spec"]["draft_forwards"] for x in rs))
        c_run, v_run = d_fw / t_step, (ph["verify"] / R_) / t_step
        a = s["acc_exam"]
        S_hat = (1 + sum(a ** j for j in range(1, g + 1))) / (g * c_run + v_run)
        grid = np.linspace(0, 1, 10001)
        ok = grid[(1 + sum(grid ** j for j in range(1, g + 1))) > g * c_run + v_run]
        be = float(ok[0]) if len(ok) else None
        cm[f"{d}|{g}"] = {"alpha": a, "c": c_run, "v": v_run, "S_hat": S_hat, "S_measured": s["speedup_total"], "break_even": be}
        rows2.append([DLAB[d], g, f"{a:.3f}", f"{c_run:.3f}", f"{v_run:.2f}", f"{S_hat:.3f}", f"{s['speedup_total']:.3f}",
                      f"{be:.3f}" if be is not None else "none (> 1)"])
    write_table(r, "T12_round_time_decomposition", ["Draft", "gamma", "Draft proposal ms / round", "Target verification ms / round",
                                                     "Accept + residual sampling ms / round", "Cache rollback ms / round",
                                                     "Prefill ms / request", "Host overhead ms / request", "Share of time: verify / draft"], rows,
                "CUDA-synchronised phase timers inside every round; host overhead = wall time - all timed phases.")
    rows = []
    for d in have:
        for wu in (r.bench(d) or {}).get("warmup", [])[:1]:                  # the first (cold) warm-up of the run
            rows.append([DLAB[d], wu["gamma"], wu["tokens"], f"{1000 * wu['spec_s'][0]:.0f}",
                         f"{1000 * np.median(wu['spec_s'][1:]):.0f}", f"{1000 * wu['target_s'][0]:.0f}",
                         f"{1000 * np.median(wu['target_s'][1:]):.0f}"])
    if rows:
        write_table(r, "T12b_warm_setup", ["Draft", "gamma", "Tokens", "Speculative 1st call ms", "Speculative warm ms",
                                           "Target-only 1st call ms", "Target-only warm ms"], rows,
                    "Three warm-up calls on the first prompt before each benchmark (reported only here); '1st call' includes one-off "
                    "CUDA / cuBLAS / allocator setup, 'warm' is the median of calls 2-3. Not part of any per-prompt row.")
    iso_note = "; ".join(f"L={L}: c={v['c']:.3f}, " + ", ".join(f"v{g}={v[f'v_{g}']:.2f}" for g in r.cfg["eval"]["gammas"])
                         for L, v in iso.items()) if iso else "isolated costs not available"
    write_table(r, "T13_cost_model", ["Draft", "gamma", "alpha (accepted/examined)", "c (measured in run)", "v_gamma (measured in run)",
                                      "Predicted S_hat", "Measured S (total time ratio)", "Break-even alpha"], rows2,
                "S_hat = (1 + sum_{j=1..gamma} alpha^j) / (gamma c + v_gamma); c = draft forward time / target decode-step "
                "time, v_gamma = verification time per round / target decode-step time, both from the same runs. Isolated "
                f"microbenchmark: {iso_note}.")
    save_json(r.ana / "cost_model.json", cm)
    # workload x output length
    rows = []
    for d in have:
        for g in r.cfg["eval"]["gammas"]:
            rs = [x for x in _bench_rows(r, d) if x["gamma"] == g]
            for w, wl in WL:
                sub = [x for x in rs if x["workload"] == w]
                if sub:
                    s = _spec_stats(sub)
                    rows.append([DLAB[d], g, wl, len(sub), f"{s['acc_exam']:.3f}", f"{s['passes_per_tok']:.3f}",
                                 f"{s['speedup_median']:.3f}", f"{s['eos_spec']:.2f}"])
            for lo, hi in ((1, 16), (17, 32), (33, 64)):
                sub = [x for x in rs if lo <= x["spec"]["n_tokens"] <= hi]
                if sub:
                    s = _spec_stats(sub)
                    rows.append([DLAB[d], g, f"output {lo}-{hi} tokens", len(sub), f"{s['acc_exam']:.3f}",
                                 f"{s['passes_per_tok']:.3f}", f"{s['speedup_median']:.3f}", f"{s['eos_spec']:.2f}"])
    write_table(r, "T14_speculative_by_workload_and_length", ["Draft", "gamma", "Subset", "Runs", "Accepted / examined",
                                                              "Target passes / token", "Speedup (median)", "EOS share"], rows)
    # expected (bank) vs empirical (trajectory) acceptance
    A = r.align()
    rows, calib = [], {}
    for d in have:
        bank_e = _agg([x for x in A if x["bank"] == "teacher"], d, "exp_acc") if A else np.nan
        for g in r.cfg["eval"]["gammas"]:
            rs = [x for x in _bench_rows(r, d) if x["gamma"] == g]
            P = np.array([p for x in rs for p in x.get("proposals", [])], float)
            if not len(P):
                continue
            traj_e = float(np.mean(1 - P[:, 5]))
            emp = float(P[:, 2].mean())
            first = P[P[:, 1] == 1]
            rows.append([DLAB[d], g, f"{bank_e:.3f}", f"{traj_e:.3f}", f"{emp:.3f}", f"{first[:, 2].mean():.3f}",
                         f"{P[P[:, 1] == g][:, 2].mean():.3f}" if (P[:, 1] == g).any() else "-"])
            edges = np.linspace(0, 1, 11)
            b = np.clip(np.digitize(1 - P[:, 5], edges) - 1, 0, 9)
            calib[f"{d}|{g}"] = {"bin_center": ((edges[:-1] + edges[1:]) / 2).tolist(),
                                 "empirical": [float(P[b == i, 2].mean()) if (b == i).any() else None for i in range(10)],
                                 "n": [int((b == i).sum()) for i in range(10)]}
    write_table(r, "T15_expected_vs_empirical_acceptance", ["Draft", "gamma", "Expected 1-TV, teacher bank", "Expected 1-TV along own "
                                                             "trajectories", "Empirical acceptance", "Empirical, 1st proposal",
                                                             "Empirical, last proposal"], rows,
                "The trajectory expectation averages 1 - TV(p, q) at every examined proposal; under exact sampling it is an "
                "unbiased estimate of the empirical acceptance. The bank expectation uses fixed teacher prefixes (other "
                "prompts' positions, not the visited ones).")
    save_json(r.ana / "acceptance_calibration.json", calib)
    # throughput diagnostic, unmerged wrapper
    rows = []
    for d in have:
        b = r.bench(f"throughput_{d}")
        if not b:
            continue
        for g in r.cfg["eval"]["gammas"]:
            rs = [x for x in b["rows"] if x["gamma"] == g]
            if rs:
                s = _spec_stats(rs)
                rows.append([DLAB[d], g, len(rs), f"{s['tps_spec']:.1f}", f"{s['tps_target']:.1f}", f"{s['speedup_median']:.3f}",
                             f"{s['acc_exam']:.3f}"])
    if rows:
        write_table(r, "T16_fixed64_throughput", ["Draft", "gamma", "Runs", "Tok/s spec", "Tok/s target", "Speedup (median)",
                                                  "Accepted / examined"], rows,
                    "24 fixed GSM8K prompts, exactly 64 new tokens with EOS ignored (timing diagnostic only; outputs after "
                    "an EOS are not answers).")
    u = r.bench("unmerged")
    if u:
        name = u["rows"][0]["draft"].replace("_unmerged", "")
        rs_u = u["rows"]
        rs_m = [x for x in _bench_rows(r, name) if x["gamma"] == rs_u[0]["gamma"] and x["rep"] == 0]
        su, sm = _spec_stats(rs_u), _spec_stats(rs_m) if rs_m else None
        rows = [["unmerged (peft wrapper)", f"{su['tps_spec']:.1f}", f"{su['speedup_median']:.3f}", f"{su['acc_exam']:.3f}"]]
        if sm:
            rows.append(["merged FP16", f"{sm['tps_spec']:.1f}", f"{sm['speedup_median']:.3f}", f"{sm['acc_exam']:.3f}"])
        write_table(r, "T17_merged_vs_unmerged", ["Draft execution", "Tok/s spec", "Speedup (median)", "Accepted / examined"], rows,
                    f"{DLAB.get(name, name)}, gamma {rs_u[0]['gamma']}, repetition 0 for both.")
    # figures: measured vs predicted speedup; acceptance calibration; time decomposition
    if cm:
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3))
        for k, v in cm.items():
            d, g = k.split("|")
            ax[0].plot(v["S_hat"], v["S_measured"], "o" if g == "2" else "s", color=DCOL[d], label=f"{DLAB[d]}, gamma {g}")
        lim = [min(min(v["S_hat"], v["S_measured"]) for v in cm.values()) * 0.9, max(max(v["S_hat"], v["S_measured"]) for v in cm.values()) * 1.1]
        ax[0].plot(lim, lim, "k--", lw=0.8)
        ax[0].axhline(1, color="0.5", lw=0.6, ls=":")
        ax[0].set(xlabel="predicted speedup S_hat", ylabel="measured speedup S")
        title(ax[0], "(a) cost model vs measurement")
        for k, v in calib.items():
            d, g = k.split("|")
            if g != str(max(r.cfg["eval"]["gammas"])):
                continue
            xs = [c for c, e, n in zip(v["bin_center"], v["empirical"], v["n"]) if e is not None and n >= 20]
            ys = [e for e, n in zip(v["empirical"], v["n"]) if e is not None and n >= 20]
            ax[1].plot(xs, ys, marker="o", color=DCOL[d], label=DLAB[d])
        ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[1].set(xlabel="predicted acceptance 1 - TV(p, q)", ylabel="empirical acceptance", xlim=(0, 1), ylim=(0, 1))
        title(ax[1], f"(b) calibration, gamma {max(r.cfg['eval']['gammas'])}")
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=6)
        save_fig(r, fig, "F5_speedup_and_calibration")
        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        keys = list(stats)
        parts = [("draft", "draft proposal", "#1f77b4"), ("verify", "target verification", "#d62728"),
                 ("accept", "accept / residual", "#2ca02c"), ("rollback", "cache rollback", "#9467bd")]
        bottom = np.zeros(len(keys))
        for k, lab, col in parts:
            vals = []
            for d, g in keys:
                rs = [x for x in _bench_rows(r, d) if x["gamma"] == g]
                vals.append(1000 * sum(x["spec"]["times"].get(k, 0) for x in rs) / max(1, sum(x["spec"]["n_tokens"] for x in rs)))
            ax.bar(range(len(keys)), vals, bottom=bottom, color=col, label=lab)
            bottom += np.array(vals)
        tgt = [1000 * sum(x["target"]["wall"] for x in _bench_rows(r, d) if x["gamma"] == g) /
               max(1, sum(x["target"]["n_tokens"] for x in _bench_rows(r, d) if x["gamma"] == g)) for d, g in keys]
        ax.plot(range(len(keys)), tgt, "k_", ms=14, mew=1.5, label="target-only (total)")
        ax.set(xticks=range(len(keys)), ylabel="ms per output token")
        ax.set_xticklabels([f"{DSHORT[d]}\n\u03b3={g}" for d, g in keys], fontsize=6)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
        save_fig(r, fig, "F6_time_per_token_decomposition")


# ================================================================================ task and systems analysis
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")


def answer_extracted(text: str, answer: str, family: int) -> bool:
    """Lenient correctness: the requested item, extracted from free text, equals the answer (format ignored).
    Month: the first month name; integer: the last integer; code: the first standalone two-letter capital code."""
    import re
    if family == 0:
        m = re.findall(rf"\b({_MONTHS})\b", text)
        return bool(m) and m[0] == answer
    if family == 1:
        m = re.findall(r"\d[\d,]*", text)
        return bool(m) and m[-1].replace(",", "") == answer
    m = re.findall(r"\b[A-Z]{2}\b", text)
    return bool(m) and m[0] == answer


def t_quality(r, plt):
    models = [m for m in ["target"] + DRAFTS if r.j("quality", f"{m}.json.gz")]
    if not models:
        return "no quality"
    from src.utils import data_repo, read_jsonl
    short = read_jsonl(data_repo(r.cfg) / r.cfg["data"]["short_test"])      # committed file (no GSM8K download needed)
    fam = {x["id"]: x["family"] for x in short}
    ans = {x["id"]: x["answer"] for x in short}
    rows = []
    for m in models:
        q = r.j("quality", f"{m}.json.gz")["rows"]
        mt = [x for x in q if x["workload"] == "math"]
        row = [DLAB[m], f"{100 * np.mean([x['correct'] for x in mt]):.1f}",
               f"{100 * np.mean([x['correct'] for x in mt if x['termination'] == 'eos']):.1f}",
               f"{100 * np.mean([x['truncated'] for x in mt]):.1f}", f"{np.mean([x['n_tokens'] for x in mt]):.0f}"]
        for w in ("short_familiar", "short_shifted"):
            s = [x for x in q if x["workload"] == w]
            lenient = [answer_extracted(x["text"], ans[x["prompt_id"]], fam[x["prompt_id"]]) for x in s] if fam and ans else []
            row += [f"{100 * np.mean(lenient):.1f}" if lenient else "-",
                    f"{100 * np.mean([x['correct'] for x in s]):.1f}"]
        rows.append(row)
    write_table(r, "T18_standalone_quality", ["Model", "Math correct %", "Math correct among EOS-ended %",
                                              "Math truncated at 512 %", "Math mean tokens",
                                              "Familiar answer correct %", "Familiar exact-format correct %",
                                              "Shifted answer correct %", "Shifted exact-format correct %"], rows,
                "Each model alone, temperature 1, caps 512 (math) / 32 (extraction), 2 repetitions x 120 prompts. Measures "
                "the draft as a model; speculative outputs follow the target and are not used here. Extraction: 'answer "
                "correct' = the requested item extracted from the free text (first month name / last integer / first "
                "two-letter code) equals the gold answer, format ignored; 'exact-format correct' = the whole response is "
                "exactly the answer (the format-compliance check). Math: last number vs gold; unextractable responses "
                "count as wrong.")
    # amortisation
    orig = r.bench("original")
    rows = []
    for d in DRAFTS[1:]:
        s = r.j("distill", d, "summary.json")
        b = r.bench(d)
        if not (s and b and orig):
            continue
        pool = s.get("teacher_pool") or {}
        m = r.j("merge", f"{d}.json") or {}
        cost = (pool.get("teacher_generation_seconds", 0) + sum(s["timings_s"].values()) + m.get("merge_seconds", 0))
        for g in r.cfg["eval"]["gammas"]:
            # per-request times on a common scale: the pooled target-only time divided by each draft's own paired
            # speedup (total-time ratio), so differences in background load between benchmark runs cancel
            ref = np.mean([x["target"]["wall"] for bb in (orig, b) for x in bb["rows"] if x["gamma"] == g])
            S_new = (sum(x["target"]["wall"] for x in b["rows"] if x["gamma"] == g) /
                     sum(x["spec"]["wall"] for x in b["rows"] if x["gamma"] == g))
            S_old = (sum(x["target"]["wall"] for x in orig["rows"] if x["gamma"] == g) /
                     sum(x["spec"]["wall"] for x in orig["rows"] if x["gamma"] == g))
            new, old, tgt = ref / S_new, ref / S_old, ref
            save_o, save_t = old - new, tgt - new
            cost_osd = cost - s["timings_s"]["teacher_scoring_s"]      # OSD-style: reuse verification logits as teacher
            rows.append([DLAB[d], g, f"{cost / 60:.1f}", f"{cost_osd / 60:.1f}", f"{1000 * save_o:+.1f}",
                         f"{cost / save_o:,.0f}" if save_o > 0 else "never",
                         f"{1000 * save_t:+.1f}", f"{cost / save_t:,.0f}" if save_t > 0 else "never"])
    if rows:
        write_table(r, "T19_amortised_cost", ["Draft", "gamma", "Distillation + merge cost (GPU min)",
                                              "... without teacher scoring (OSD-style reuse)",
                                              "Saving per request vs original draft (ms)", "Requests to break even (vs original)",
                                              "Saving per request vs target-only (ms)", "Requests to break even (vs target-only)"], rows,
                    "Cost = teacher-pool generation + teacher scoring + student rollouts + updates + merging (one T4). "
                    "Savings are per-request differences on the 120-prompt, 64-token benchmark: per-request time = mean "
                    "target-only time (pooled over the two benchmarks) / the draft's measured total-time speedup. These are "
                    "point estimates: where the paired speedup difference in T11b has a 95% CI that includes 0, the saving "
                    "and its break-even count are not supported by the data. The OSD column "
                    "(Liu et al., 2024) removes the teacher-scoring cost, which an online deployment gets for free from "
                    "the target's verification logits; it is a what-if, not a measured method.")
    # EOS / truncation in the speed benchmark
    rows = []
    for d in [d for d in DRAFTS if r.bench(d)]:
        for w, wl in WL:
            rs = [x for x in _bench_rows(r, d) if x["workload"] == w]
            if rs:
                rows.append([DLAB[d], wl, len(rs), f"{np.mean([x['spec']['termination'] == 'eos' for x in rs]):.2f}",
                             f"{np.mean([x['spec']['termination'] == 'cap' for x in rs]):.2f}",
                             f"{np.mean([x['target']['termination'] == 'cap' for x in rs]):.2f}",
                             f"{np.mean([x['spec']['n_tokens'] for x in rs]):.1f}"])
    if rows:
        write_table(r, "T20_eos_and_truncation", ["Draft", "Workload", "Runs", "EOS share (spec)", "64-cap share (spec)",
                                                  "64-cap share (target-only)", "Mean output tokens (spec)"], rows,
                    "Mathematical outputs truncated at 64 tokens are speed-test outputs, never completed solutions.")
    # qualitative continuations
    qual = r.cfg["eval"]["qualitative_prompts"]
    tok = None
    try:
        from transformers import AutoTokenizer
        from src.data import revisions
        tok = AutoTokenizer.from_pretrained(r.cfg["models"]["target"], revision=revisions(r.cfg)["target"])
    except Exception:
        pass
    rows = []
    for pid in qual:
        for m in models:
            q = r.j("quality", f"{m}.json.gz")["rows"]
            x = next((x for x in q if x["prompt_id"] == pid and x["rep"] == 0), None)
            if x:
                rows.append([pid, DLAB[m], repr(" ".join(x["text"].split())[:110]), x.get("correct"), x["termination"]])
    if rows:
        write_table(r, "T21_matched_continuations", ["Prompt", "Model (standalone)", "Response (first 110 chars)", "Correct",
                                                     "Termination"], rows,
                    "Fixed prompts chosen before results; repetition 0, temperature 1.")


def t_diag(r, plt):
    """Mechanism diagnostic (scripts/diagnose_task3.py): D1 launch-bound decode, D2 math degradation, D3 split-prefill."""
    made = []
    la = r.j("diagnostics", "launch.json")
    if la:
        rows = []
        for w, lab in (("target", "target 1.5B"), ("draft", "draft 0.5B")):
            x = la["models"][w]
            pr = x.get("profile", {})
            rows.append([lab, x["layers"], f"{x['params_per_layer'] / 1e6:.1f}", f"{x['batch_ms']['1']:.1f}",
                         f"{x['fit_ms']['per_layer']:.2f}", f"{x['fit_ms']['intercept']:.1f}", f"{x['fit_ms']['r2']:.3f}",
                         " / ".join(f"{x['batch_ms'][b]:.1f}" for b in x["batch_ms"]),
                         f3(pr.get("kernels_per_step"), 0), f3(pr.get("kernel_ms_per_step"), 1), f3(pr.get("gpu_busy_share"), 2)])
        s = la["summary"]
        write_table(r, "D1_launch_bound_decode", ["Model", "Layers", "Params / layer (M)", "Step ms (batch 1)",
                                                  "ms per layer (fit)", "Intercept ms", "R^2",
                                                  f"Step ms at batch {' / '.join(la['models']['target']['batch_ms'])}",
                                                  "Kernels / step", "GPU kernel ms / step", "GPU busy share"], rows,
                    f"One cached single-token decode step after {la['context_tokens']} context tokens, FP16, each model "
                    f"alone on a T4. Layer fit: step time of models truncated to their first k decoder layers. "
                    f"c (full models) = {s['c_full']:.3f}; layer ratio draft/target = {s['layer_ratio']:.3f}; per-layer slope "
                    f"ratio draft/target = {s['per_layer_slope_ratio']:.3f} while params per layer ratio = "
                    f"{s['params_per_layer_ratio']:.3f}. GPU busy share = profiled kernel time / wall time.")
        made.append("D1")
    mt = r.j("diagnostics", "math.json")
    if mt:
        rows = []
        cap_share = {}
        for n in DRAFTS[1:]:
            f = r.res / "distill" / n / "presentations.jsonl.gz"
            if f.exists():
                pr = [json.loads(x) for x in gzip.open(f, "rt")]
                m_ = [x for x in pr if x["category"] == "math"]
                cap_share[n] = float(np.mean([x["termination"] == "cap" for x in m_]))
        for n, x in mt["models"].items():
            q = r.j("quality", f"{n}.json.gz")
            t1 = 100 * np.mean([z["correct"] for z in q["rows"] if z["workload"] == "math"]) if q else np.nan
            tb, ow = x.get("teacher_bank", {}), x.get("own_rollouts", {})
            rows.append([DLAB.get(n, n), f"{t1:.1f}", f"{100 * x['greedy_accuracy']:.1f}", f"{100 * x['greedy_truncated']:.1f}",
                         f"{x['greedy_mean_tokens']:.0f}", f3(tb.get("entropy"), 3), f3(tb.get("top_prob"), 3),
                         f3(ow.get("entropy"), 3), f3(ow.get("top_prob"), 3),
                         f"{100 * cap_share[n]:.0f}" if n in cap_share else "-"])
        write_table(r, "D2_math_degradation", ["Model", "Math acc % (temperature 1, T18)", "Math acc % (greedy)",
                                               "Greedy truncated at 512 %", "Greedy mean tokens",
                                               "Entropy, teacher-bank math (nats)", "Top-token prob, teacher bank",
                                               "Entropy, own rollouts (nats)", "Top-token prob, own rollouts",
                                               "Training math responses cut at 192 %"], rows,
                    "60 GSM8K test prompts, cap 512, greedy decoding (one run, deterministic). Entropy / top-token "
                    "probability of each model's next-token distribution at the saved math positions of the teacher bank "
                    "(same prefixes for every draft) and of its own temperature-1 rollout bank. Last column: share of the "
                    "math training presentations (teacher pool or student rollouts) that ended at the 192-token cap.")
        made.append("D2")
    sp = r.j("diagnostics", "split.json")
    if sp:
        rows = []
        for dt, run in sp["runs"].items():
            for x in run["rows"]:
                rows.append([dt, x["model"], x["prompt"], f"{x['max_abs_prob_diff']:.2e}", x["argmax_position"], x["prompt_tokens"],
                             f"{x['max_in_shared_prefix']:.2e}", f"{x['max_in_rest_of_prompt']:.2e}",
                             f"{x['max_in_generated']:.2e}" if x["max_in_generated"] is not None else "-",
                             f"{x['argmax_agreement']:.4f}"])
        write_table(r, "D3_split_prefill_precision", ["Precision", "Model", "Prompt", "max |dp|", "At position", "Prompt tokens",
                                                      f"max |dp|, shared prefix (first {sp['shared_prefix_tokens']} tokens)",
                                                      "max |dp|, rest of prompt", "max |dp|, generated (cached decode)",
                                                      "argmax agreement"], rows,
                    "The validate check (one full forward vs prefill + cached token-by-token decode, 24 sampled tokens), "
                    "with the location of the maximum, in FP16 and FP32.")
        made.append("D3")
    if la:
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2))
        for w, col, lab in (("target", "#d62728", "target 1.5B"), ("draft", "#1f77b4", "draft 0.5B")):
            x = la["models"][w]
            ks = [int(k) for k in x["layers_ms"]]
            ax[0].plot(ks, list(x["layers_ms"].values()), "o", color=col, label=lab)
            kk = np.array([0, max(ks)])
            ax[0].plot(kk, x["fit_ms"]["intercept"] + x["fit_ms"]["per_layer"] * kk, "-", color=col, lw=0.8)
            bs = [int(b) for b in x["batch_ms"]]
            ax[1].plot(bs, list(x["batch_ms"].values()), "o-", color=col, label=lab)
        ax[0].set(xlabel="decoder layers kept", ylabel="ms per decode step", ylim=(0, None))
        title(ax[0], "(a) step time vs layers (batch 1)")
        ax[1].set(xlabel="batch size", ylabel="ms per decode step", xscale="log", ylim=(0, None))
        ax[1].set_xticks(bs)
        ax[1].set_xticklabels([str(b) for b in bs])
        title(ax[1], "(b) step time vs batch")
        ax[0].legend(frameon=False)
        save_fig(r, fig, "FD1_launch_bound_decode")
        made.append("FD1")
    return ", ".join(made) or "no diagnostic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    r = R(Path(a.results), load_config(a.config))
    plt = setup_mpl()
    failed = []
    for fn in (t_sampler, t_distill, t_align, t_spec, t_quality, t_diag):
        try:
            msg = fn(r, plt)
            print(f"{fn.__name__}: {'ok' if not msg else msg}")
        except Exception:
            failed.append(fn.__name__)
            print(f"{fn.__name__}: FAILED\n{traceback.format_exc()}")
    print(f"failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
