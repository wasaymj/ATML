"""Builds every Task 3 report table (Markdown + LaTeX) and figure (PDF + PNG) from results/.

Run from `PA3/Task 3`:  python scripts/compile_task3_report.py
Missing inputs are skipped with a log message, so the script can run at any stage (e.g. before the live run).
Headline numbers use the 800 held-out evaluation questions; dev (200) is used only to tune adaptive stopping.
"""
import os
import sys
import glob
import logging


import numpy as np

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_ROOT)
sys.path.insert(0, TASK_ROOT)
from src.utils import load_config, load_json, setup_logging, task_path, write_text, read_frame, stable_int  # noqa: E402

logger = logging.getLogger("compile_report")
CFG = load_config()
R = task_path(CFG["paths"]["results_dir"])
TAB, FIG = os.path.join(R, "tables"), os.path.join(R, "figures")
DS = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# fixed categorical slot per method (validated reference palette order; colour follows the method everywhere)
MCOLOR = {"self_consistency": "#2a78d6", "rwc_self_certainty": "#eb6834", "top1_self_certainty": "#1baf7a",
          "first_ans": "#eda100", "rwc_visconf": "#e87ba4", "top1_visconf": "#008300", "rwc_vision": "#6250d6",
          "top1_logprob": "#e34948"}
MLABEL = {"first_ans": "FirstAns", "self_consistency": "Self-consistency", "top1_logprob": "Top-1 mean logp",
          "top1_self_certainty": "Top-1 self-certainty", "rwc_self_certainty": "Self-certainty + RWC",
          "shortest": "Shortest rollout", "top1_vision": "Top-1 vision", "rwc_vision": "Vision + RWC",
          "top1_visconf": "Top-1 VisConf", "rwc_visconf": "VisConf + RWC", "top1_token_fusion": "Top-1 token fusion",
          "rwc_token_fusion": "Token fusion + RWC"}
SCORE_LABEL = {"L": "mean logp L", "K": "self-certainty K", "V": "vision V", "C": "calibrated conf. C",
               "Z": "VisConf Z", "Ztoken": "token fusion", "neg_len": "shortness (-T)", "cov": "Cov(c, v)"}


def opt(path):
    return load_json(path) if os.path.exists(path) else None


def pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}"


def ci(t, d=1, signed=False):
    if t is None:
        return "--"
    m, lo, hi = t
    s = "+" if signed else ""
    return f"{100 * m:{s}.{d}f} [{100 * lo:{s}.{d}f}, {100 * hi:{s}.{d}f}]"


def dci(t, d=1):
    return "--" if t is None else f"{100 * t['diff']:+.{d}f} [{100 * t['lo']:+.{d}f}, {100 * t['hi']:+.{d}f}]"


def save_table(name, caption, header, rows, colspec=None, note=""):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", note]
    write_text(os.path.join(TAB, name + ".md"), "\n".join(md) + "\n")
    esc = lambda s: str(s).replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")  # noqa: E731
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}",
           rf"\label{{tab:{name}}}", rf"\begin{{tabular}}{{{colspec or 'l' + 'r' * (len(header) - 1)}}}",
           r"\toprule", " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_text(os.path.join(TAB, name + ".tex"), "\n".join(tex) + "\n")
    logger.info("table %s", name)


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4,
                         "savefig.dpi": 300, "pdf.fonttype": 42, "figure.facecolor": "white"})
    return plt


def savefig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    logger.info("figure %s", name)


def summary(ds):
    return opt(os.path.join(R, "offline", f"{ds}_summary.json"))


# ------------------------------------------------------------------------------------------------- tables
def t1_data():
    pre = opt(os.path.join(R, "data", "prepare_summary.json"))
    if pre is None:
        return
    rows = []
    for ds in DS:
        if ds not in pre:
            continue
        s = load_json(os.path.join(R, "data", f"{ds}_scoring.json"))
        q = read_frame(os.path.join(R, "data", f"{ds}_questions.csv"))
        rows.append([DS[ds], s["n_questions"], f"{s['retained_tokens'] / s['n_rollouts']:.0f}", pct(s["cap_rate"]),
                     pct(s["no_answer_rate"]), pct(s["rollout_accuracy"]), pct(s["coverage32"]),
                     f"{q.n_groups.mean():.1f}", s["scorer_version"]])
    save_table("T1_data", "Supplied rollout pools (32 rollouts per question, all 1,000 questions; correctness computed "
               "with the fixed scorer).", ["Model / dataset", "Questions", "Mean tokens", "Capped %", "No answer %",
                                           "Rollout acc. %", "Coverage@32 %", "Answer groups@32", "Scorer"], rows)


def t2_opportunity():
    rows = []
    for ds in DS:
        S = summary(ds)
        if S is None:
            continue
        E = S["by_split"]["eval"]
        for N in S["budgets"]:
            b = E["budgets"][str(N)]
            sc = b["methods"]["self_consistency"]
            rows.append([DS[ds] if N == S["budgets"][0] else "", N, ci(b["methods"]["first_ans"]["acc"]),
                         ci(b["oracle_coverage"]), f"{b['n_answer_groups']:.2f}", ci(sc["acc"]),
                         pct(sc["fail_no_correct_candidate"]), pct(sc["fail_missed_available"]),
                         f"{sc['cost'] / 1000:.1f}k"])
    if rows:
        save_table("T2_sampling_opportunity", "Step 1 on the 800 held-out questions: FirstAns accuracy, oracle coverage "
                   "and answer diversity per budget, with the self-consistency failure decomposition (no correct "
                   "candidate vs. a correct candidate available but missed). Mean of 10 orders; 95% question "
                   "bootstrap CIs.", ["Pool", "N", "FirstAns %", "Coverage %", "Groups", "SC acc. %",
                                      "No correct %", "Missed %", "Tokens/q"], rows)


def t3_methods():
    for ds in DS:
        S = summary(ds)
        if S is None:
            continue
        E = S["by_split"]["eval"]["budgets"]
        Ns = [8, 16, 32]
        rows = []
        for m in S["methods"]:
            r = [MLABEL[m]] + [ci(E[str(N)]["methods"][m]["acc"]) for N in Ns]
            r += [dci(E["32"]["methods"][m]["vs_self_consistency"]) if m != "self_consistency" else "--"]
            rows.append(r)
        rows.append(["Oracle coverage"] + [ci(E[str(N)]["oracle_coverage"]) for N in Ns] + ["--"])
        save_table(f"T3_methods_{ds}", f"{DS[ds]}: selection accuracy (%) by method and budget on the 800 held-out "
                   "questions (mean over 10 orders, 95% CI), and the paired difference to self-consistency at N = 32 "
                   "(pp, 1,000 question bootstrap resamples).", ["Method", "N = 8", "N = 16", "N = 32", "vs SC @32 (pp)"],
                   rows)


def t4_contrasts():
    rows = []
    for ds in DS:
        S = summary(ds)
        if S is None:
            continue
        for N in (8, 32):
            for k, v in S["by_split"]["eval"]["budgets"][str(N)]["contrasts"].items():
                a, b = k.split("-")
                rows.append([DS[ds], N, f"{MLABEL[a]} - {MLABEL[b]}", dci(v), f"{v['q_better']}/{v['q_worse']}"])
    if rows:
        save_table("T4_controlled_contrasts", "Controlled contrasts on held-out questions: same score with a different "
                   "aggregation (value of consensus) or same aggregation with a different score (value of the "
                   "signal). Paired differences in pp with 95% CIs; questions better/worse.",
                   ["Pool", "N", "Contrast", "Diff (pp)", "Q better/worse"], rows, "llllr")


def t5_signals():
    rows = []
    for ds in DS:
        s = opt(os.path.join(R, "signals", f"{ds}_signals.json"))
        if s is None:
            continue
        A, M = s["eval"]["all_questions"], s["eval"]["mixed_questions"]
        for sc in ("L", "K", "V", "C", "Z", "Ztoken", "neg_len"):
            if f"qgap_{sc}" not in A:
                continue
            g, au = M.get(f"qgap_{sc}"), M.get(f"auc_{sc}")
            rl = A.get(f"rho_len_{sc}")
            rows.append([DS[ds], SCORE_LABEL[sc], f"{100 * g['mean']:+.1f} [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}]",
                         f"{au['mean']:.3f} [{au['lo']:.3f}, {au['hi']:.3f}]",
                         "--" if sc == "neg_len" or rl is None else f"{rl['mean']:+.2f}"])
        rc = A.get("rho_correct_T")
        if rc:
            rows.append([DS[ds], "length T vs correct (rho)", "--", "--", f"{rc['mean']:+.2f}"])
    if rows:
        save_table("T5_signal_discrimination", "What the signals capture at N = 32 (held-out questions with mixed "
                   "correctness): top-minus-bottom quartile correctness within a question (pp), within-question AUC, "
                   "and the mean within-question Spearman correlation of the score with rollout length.",
                   ["Pool", "Score", "Top-bottom quartile (pp)", "AUC", "rho(length, score)"], rows, "lllrr")


def t6_conf_vision():
    s = opt(os.path.join(R, "signals", "mathvista_signals.json"))
    S = summary("mathvista")
    if s is None:
        return
    A, M = s["eval"]["all_questions"], s["eval"]["mixed_questions"]
    td = s["token_corr_distribution"]
    rows = [["Token-level corr(c_t, v_t) within rollout", f"mean {td['mean']:+.3f}, median {td['median']:+.3f}, "
             f"IQR [{td['q25']:+.3f}, {td['q75']:+.3f}], {100 * td['frac_positive']:.0f}% > 0"]]
    if "traj_rho_CV" in A:
        t = A["traj_rho_CV"]
        rows.append(["Rollout-level Spearman(C_i, V_i) within question", f"{t['mean']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]"])
    for k in ("acc_hiC_hiV", "acc_hiC_loV", "acc_loC_hiV", "acc_loC_loV"):
        if k in M:
            rows.append([f"Rollout accuracy, {k[4:].replace('_', ' / ').replace('hi', 'high ').replace('lo', 'low ')}",
                         f"{100 * M[k]['mean']:.1f} [{100 * M[k]['lo']:.1f}, {100 * M[k]['hi']:.1f}]"])
    for k, lab in (("cov_correct_minus_incorrect", "Cov(c, v): correct - incorrect rollouts"),
                   ("tokcorr_correct_minus_incorrect", "corr(c, v): correct - incorrect rollouts"),
                   ("auc_cov", "AUC of Cov(c, v) for correctness")):
        if k in M:
            rows.append([lab, f"{M[k]['mean']:+.4f} [{M[k]['lo']:+.4f}, {M[k]['hi']:+.4f}]"])
    if S:
        for N in (8, 32):
            c = S["by_split"]["eval"]["budgets"][str(N)]["contrasts"]
            for k in ("top1_token_fusion-top1_visconf", "rwc_token_fusion-rwc_visconf"):
                if k in c:
                    a, b = k.split("-")
                    rows.append([f"Selection acc. {MLABEL[a]} - {MLABEL[b]}, N = {N} (pp)", dci(c[k])])
    save_table("T6_confidence_vision", "MathVista: confidence-vision relationship and token-wise vs trajectory-level "
               "fusion (held-out questions; rollout-level rows use questions with mixed correctness; median splits "
               "within each question).", ["Quantity", "Value [95% CI]"], rows, "ll")


def t7_live():
    s = opt(os.path.join(R, "live", "live_summary.json"))
    if s is None:
        return logger.info("skip T7 (live run not summarized yet)")
    rows = []
    for k, lab in (("single", "Single (1 x 3,072)"), ("parallel", "Parallel (3 x 1,024) + SC"),
                   ("refine", "Refine (1,024 + 2,048)")):
        v = s["strategies"][k]
        rows.append([lab, ci(v["acc"]), f"{v['mean_input_tokens']:.0f}", f"{v['mean_output_tokens']:.0f}",
                     f"{v['mean_time_s']:.1f}", pct(v["truncation_rate_calls"]), pct(v["no_extraction_rate_calls"]),
                     f"{v['peak_gpu_gb']:.2f}"])
    t = s["revision_transitions"]
    save_table("T7_live", f"Live MATH comparison on the fixed {s['n_questions']}-question subset (same 3,072-token output "
               "allowance; actual tokens and runtime per question on one T4, model loading excluded; the shared initial "
               f"response is charged to both parallel and refine). Revision transitions vs the shared initial answer: "
               f"wrong->correct {t['wrong_to_correct']}, correct->wrong {t['correct_to_wrong']}, unchanged-correct "
               f"{t['correct_to_correct']}, unchanged-wrong {t['wrong_to_wrong']}.",
               ["Strategy", "Acc. % [CI]", "In tok", "Out tok", "Time s", "Trunc. % calls", "No box % calls",
                "Peak GB"], rows)
    rows = [[k.replace("-", " - "), dci(v), f"{v['q_better']}/{v['q_worse']}"] for k, v in s["paired"].items()]
    save_table("T7b_live_paired", "Live strategies: paired accuracy differences (pp, question bootstrap).",
               ["Contrast", "Diff (pp)", "Q better/worse"], rows, "llr")


def t8_adaptive():
    rows, rows2 = [], []
    for ds in DS:
        a = opt(os.path.join(R, "adaptive", f"{ds}_adaptive.json"))
        if a is None:
            continue
        f32 = a["fixed_sc"]["32"]
        rows.append([DS[ds], "Fixed-32 SC", "--", ci(f32["acc"]), f"{f32['cost'] / 1000:.1f}k", "32.0", "--", "--"])
        for fam, p in a["policies"].items():
            s = p["setting"]
            lab = s["label"] if isinstance(s["label"], str) else ""
            dist = ", ".join(f"{k}:{100 * v:.0f}%" for k, v in p["stop_distribution"].items())
            rows.append([DS[ds], fam.capitalize(), lab, ci(p["acc"]), f"{p['cost'] / 1000:.1f}k",
                         f"{p['mean_rollouts']:.1f}", dci(p["vs_fixed32_sc"]),
                         f"N={p['nearest_cost_fixed_N']}: {dci(p['vs_nearest_cost_fixed_sc'])}"])
            rows2.append([DS[ds], fam.capitalize(), dist, f"{100 * p['early_stop_rate']:.1f}",
                          p["incorrect_early_stops"], p["incorrect_early_stops_recoverable_at_32"],
                          p["correct_early_stops_where_32_wrong"]])
    if rows:
        save_table("T8_adaptive", "Adaptive stopping on the 800 held-out questions with thresholds frozen on dev (cheapest "
                   "setting within 1 pp of fixed-32 self-consistency on dev). Accuracy %, mean tokens per question, "
                   "mean rollouts; paired differences (pp) to fixed-32 SC and to the fixed-N SC of most similar cost.",
                   ["Pool", "Policy", "Setting", "Acc. % [CI]", "Tokens/q", "Rollouts", "vs fixed-32 (pp)",
                    "vs similar-cost fixed N (pp)"], rows, "llllrrll")
        save_table("T9_stopping", "Stopping behaviour on held-out questions (counts over question x order runs): "
                   "stop-point distribution, early-stop rate, incorrect early stops and how many fixed-32 SC would "
                   "have recovered, and early stops that were right where fixed-32 was wrong.",
                   ["Pool", "Policy", "Stop distribution", "Early %", "Wrong early", "Recoverable@32",
                    "Right early, 32 wrong"], rows2, "lllrrrr")


def t10_qualitative():
    """Fixed rule: within each category, held-out questions ordered by SHA-256(seed:ds:sample_id); first two."""
    md = ["# Qualitative examples (stated selection rule)", "",
          "Rule: candidates in each category are held-out questions ordered by SHA-256(6304:dataset:sample_id); the "
          "first two are shown. Order 0 of the saved rollout orders is used.", ""]
    for ds in DS:
        p = os.path.join(R, "offline", f"{ds}_selections.parquet")
        if not os.path.exists(p):
            continue
        sel = read_frame(p)
        lab = read_frame(os.path.join(R, "data", f"{ds}_rollouts.csv"))
        qd = read_frame(os.path.join(R, "data", f"{ds}_questions.csv")).set_index("sample_id")
        best = "rwc_visconf" if ds == "mathvista" else "rwc_self_certainty"
        s32 = sel[(sel.N == 32) & (sel.order == 0) & (sel.split == "eval")]
        w = s32.pivot(index="sample_id", columns="method", values="correct")
        g = s32.pivot(index="sample_id", columns="method", values="group")
        cats = {"rescue (quality+RWC right, SC wrong)": w.index[w[best] & ~w["self_consistency"]],
                "regression (SC right, quality+RWC wrong)": w.index[~w[best] & w["self_consistency"]],
                "coverage without selection (correct candidate exists, both wrong)":
                    w.index[~w[best] & ~w["self_consistency"] & qd.loc[w.index, "coverage32"].astype(bool)]}
        md += [f"## {DS[ds]} (N = 32)", ""]
        for cat, ids in cats.items():
            ids = sorted(ids, key=lambda s: stable_int(6304, ds, s))
            md += [f"### {cat}: {len(ids)} held-out questions", ""]
            for sid in ids[:2]:
                L = lab[lab.sample_id == sid]
                counts = L.groupby("group_key").size().sort_values(ascending=False)
                md += [f"* **{sid}** gold `{qd.loc[sid, 'gold']}`; answer groups (key: votes, correct?): " +
                       "; ".join(f"`{k}`: {v} ({'✓' if L[L.group_key == k].correct.iloc[0] else '✗'})"
                                 for k, v in counts.head(5).items()),
                       f"  SC -> `{L[L.group == g.loc[sid, 'self_consistency']].group_key.iloc[0]}`, "
                       f"{MLABEL[best]} -> `{L[L.group == g.loc[sid, best]].group_key.iloc[0]}`", ""]
    for ds in DS:
        p = os.path.join(R, "adaptive", f"{ds}_eval_simulation.parquet")
        if not os.path.exists(p):
            continue
        sim = read_frame(p)
        bad = sim[(sim.stop_n < 32) & ~sim.correct & sim.fixed32_correct & (sim.order == 0)]
        ids = sorted(bad.sample_id.unique(), key=lambda s: stable_int(6304, ds, s))
        md += [f"## Premature stops, {DS[ds]}: {len(bad)} (family, question) cases at order 0 where fixed-32 was right",
               ""]
        for sid in ids[:2]:
            r = bad[bad.sample_id == sid].iloc[0]
            md += [f"* **{sid}** ({r.family}): stopped at n = {int(r.stop_n)} with vote share {r.sc_share_at_stop:.2f}; "
                   f"share of correct rollouts in the full pool {r.frac_correct_32:.2f}", ""]
    live = os.path.join(R, "live", "live_per_question.csv")
    if os.path.exists(live):
        d = read_frame(live)
        r = d[d.strategy == "refine"]
        md += ["## Revision examples (live)", ""]
        for cat, m in (("wrong -> correct", ~r.initial_correct.astype(bool) & r.correct.astype(bool)),
                       ("correct -> wrong", r.initial_correct.astype(bool) & ~r.correct.astype(bool))):
            ids = sorted(r[m].sample_id, key=lambda s: stable_int(6304, "live", s))
            md += [f"### {cat}: {int(m.sum())} questions", ""]
            for sid in ids[:2]:
                x = r[r.sample_id == sid].iloc[0]
                show = lambda a: "no answer (no box / truncated)" if not isinstance(a, str) or not a else f"`{a}`"  # noqa: E731
                md += [f"* **{sid}**: initial {show(x.initial_answer)} -> revised {show(x.answer)}", ""]
    write_text(os.path.join(TAB, "T10_qualitative.md"), "\n".join(md) + "\n")
    logger.info("table T10_qualitative")


# ------------------------------------------------------------------------------------------------ figures
def f1_scaling():
    S = {ds: summary(ds) for ds in DS}
    if not any(S.values()):
        return
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=False)
    for ax, ds in zip(axes, DS):
        s = S[ds]
        if s is None:
            continue
        E = s["by_split"]["eval"]["budgets"]
        Ns = s["budgets"]
        ax.plot(Ns, [100 * E[str(N)]["oracle_coverage"][0] for N in Ns], "--", color=INK, lw=1.2, label="Oracle coverage")
        show = ["first_ans", "self_consistency", "top1_self_certainty", "rwc_self_certainty"] + \
               (["top1_visconf", "rwc_visconf"] if ds == "mathvista" else [])
        for m in show:
            y = [100 * E[str(N)]["methods"][m]["acc"][0] for N in Ns]
            ax.plot(Ns, y, "-o", color=MCOLOR[m], label=MLABEL[m], ms=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("candidates N")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(DS[ds])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, "F1_accuracy_coverage_vs_N")
    plt.close(fig)


def f2_cost():
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    any_ = False
    for ax, ds in zip(axes, DS):
        s = summary(ds)
        if s is None:
            continue
        any_ = True
        E = s["by_split"]["eval"]["budgets"]
        Ns = s["budgets"]
        for m in ["self_consistency", "rwc_self_certainty"] + (["rwc_visconf"] if ds == "mathvista" else []):
            ax.plot([E[str(N)]["methods"][m]["cost"] / 1000 for N in Ns],
                    [100 * E[str(N)]["methods"][m]["acc"][0] for N in Ns], "-o", ms=3, color=MCOLOR[m], label=MLABEL[m])
        ax.plot([E[str(N)]["methods"]["self_consistency"]["cost"] / 1000 for N in Ns],
                [100 * E[str(N)]["oracle_coverage"][0] for N in Ns], "--", color=INK, lw=1.0, label="Oracle coverage")
        a = opt(os.path.join(R, "adaptive", f"{ds}_adaptive.json"))
        if a:
            for fam, mk in (("agreement", "D"), ("quality", "s")):
                p = a["policies"][fam]
                ax.plot(p["cost"] / 1000, 100 * p["acc"][0], mk, color=INK, ms=5, mfc="white", mew=1.2,
                        label=f"Adaptive ({fam})")
        ax.set_xscale("log")
        ticks = [t for t in (0.25, 0.5, 1, 2, 5, 10, 20) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("tokens per question (thousands, log)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(DS[ds])
    if not any_:
        return plt.close(fig)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig(fig, "F2_accuracy_vs_token_cost")
    plt.close(fig)


def f3_failures():
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    any_ = False
    for ax, ds in zip(axes, DS):
        s = summary(ds)
        if s is None:
            continue
        any_ = True
        E = s["by_split"]["eval"]["budgets"]
        Ns = s["budgets"]
        best = "rwc_visconf" if ds == "mathvista" else "rwc_self_certainty"
        x = np.arange(len(Ns))
        for k, (m, off) in enumerate((("self_consistency", -0.2), (best, 0.2))):
            ok = np.array([100 * E[str(N)]["methods"][m]["acc"][0] for N in Ns])
            miss = np.array([100 * E[str(N)]["methods"][m]["fail_missed_available"] for N in Ns])
            none = np.array([100 * E[str(N)]["methods"][m]["fail_no_correct_candidate"] for N in Ns])
            ax.bar(x + off, ok, 0.38, color=MCOLOR[m], edgecolor="white", lw=0.8, label=f"correct ({MLABEL[m]})")
            ax.bar(x + off, miss, 0.38, bottom=ok, color=MCOLOR[m], alpha=0.35, edgecolor="white", lw=0.8,
                   label=f"missed, available ({MLABEL[m]})")
            ax.bar(x + off, none, 0.38, bottom=ok + miss, color="#d6d5d0", edgecolor="white", lw=0.8,
                   label="no correct candidate" if k == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("candidates N")
        ax.set_title(DS[ds])
    if not any_:
        return plt.close(fig)
    axes[0].set_ylabel("% of held-out questions")
    seen, h, l = set(), [], []
    for ax in axes:                                     # both panels: each has its own quality+RWC method
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in seen:
                seen.add(ll)
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.16), fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    savefig(fig, "F3_sampling_vs_selection_failures")
    plt.close(fig)


def f4_gain_by_difficulty():
    """Gain of quality+RWC over SC by the share of correct rollouts in the N = 32 pool (retrospective)."""
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    any_ = False
    bins = [(0, 0), (1, 4), (5, 8), (9, 12), (13, 15), (16, 20), (21, 31), (32, 32)]
    for ax, ds in zip(axes, DS):
        p = os.path.join(R, "offline", f"{ds}_per_question.csv")
        if not os.path.exists(p):
            continue
        any_ = True
        d = read_frame(p)
        q = read_frame(os.path.join(R, "data", f"{ds}_questions.csv")).set_index("sample_id")
        d = d[(d.N == 32) & (d.split == "eval")]
        best = "rwc_visconf" if ds == "mathvista" else "rwc_self_certainty"
        w = d.pivot(index="sample_id", columns="method", values="acc")
        nc = q.loc[w.index, "n_correct"]
        labels, gain, n = [], [], []
        for lo, hi in bins:
            m = (nc >= lo) & (nc <= hi)
            labels.append(f"{lo}" if lo == hi else f"{lo}-{hi}")
            gain.append(100 * (w.loc[m, best] - w.loc[m, "self_consistency"]).mean() if m.any() else 0)
            n.append(int(m.sum()))
        x = np.arange(len(bins))
        ax.bar(x, gain, 0.7, color=MCOLOR[best], edgecolor="white")
        for xi, (gv, nv) in enumerate(zip(gain, n)):
            ax.text(xi, gv + (0.3 if gv >= 0 else -0.3), f"n={nv}", ha="center", va="bottom" if gv >= 0 else "top",
                    fontsize=5.5, color=MUTED)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("correct rollouts in the 32-rollout pool")
        lo_, hi_ = min(gain + [0]), max(gain + [0])
        ax.set_ylim(lo_ - 1.5, hi_ + 1.8)                 # room for the n= labels
        ax.set_ylabel(f"{MLABEL[best]} - SC (pp)")
        ax.set_title(DS[ds])
    if not any_:
        return plt.close(fig)
    fig.tight_layout()
    savefig(fig, "F4_gain_by_correct_share")
    plt.close(fig)


def f5_signal_gaps():
    plt = plt_setup()
    labels, vals, los, his, cols = [], [], [], [], []
    for ds in DS:
        s = opt(os.path.join(R, "signals", f"{ds}_signals.json"))
        if s is None:
            continue
        M = s["eval"]["mixed_questions"]
        for sc in ("neg_len", "L", "K", "V", "C", "Z", "Ztoken"):
            k = f"qgap_{sc}"
            if k in M:
                labels.append(f"{'MATH' if ds == 'math' else 'MathVista'}: {SCORE_LABEL[sc]}")
                vals.append(100 * M[k]["mean"])
                los.append(100 * (M[k]["mean"] - M[k]["lo"]))
                his.append(100 * (M[k]["hi"] - M[k]["mean"]))
                cols.append("#2a78d6" if ds == "math" else "#e87ba4")
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(5.5, 0.22 * len(vals) + 0.6))
    y = np.arange(len(vals))[::-1]                     # horizontal bars: long labels stay readable
    ax.barh(y, vals, 0.7, color=cols, edgecolor="white")
    ax.errorbar(vals, y, xerr=[los, his], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("top-quartile minus bottom-quartile correctness within a question (pp, N = 32)")
    fig.tight_layout()
    savefig(fig, "F5_signal_quartile_gaps")
    plt.close(fig)


def f6_conf_vision():
    p = os.path.join(R, "signals", "mathvista_signal_rollouts.csv")
    s = opt(os.path.join(R, "signals", "mathvista_signals.json"))
    if not os.path.exists(p) or s is None:
        return
    plt = plt_setup()
    d = read_frame(p)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1, 1.3]})
    M = s["eval"]["mixed_questions"]
    grid = np.array([[M["acc_loC_hiV"]["mean"], M["acc_hiC_hiV"]["mean"]],
                     [M["acc_loC_loV"]["mean"], M["acc_hiC_loV"]["mean"]]]) * 100
    ax = axes[0]
    ax.imshow(grid, cmap="Blues", vmin=grid.min() - 5, vmax=grid.max() + 2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:.1f}%", ha="center", va="center", color=INK, fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["low C", "high C"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["high V", "low V"])
    ax.grid(False)
    ax.set_title("rollout accuracy by median split")
    ax = axes[1]
    x = d.tokcorr_cv.dropna().to_numpy()
    ax.hist(x, bins=60, color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(np.median(x), color=INK, lw=1.0, ls="--", label=f"median {np.median(x):+.2f}")
    ax.set_xlabel(f"within-rollout corr(c_t, v_t), rollouts >= {s['token_corr_distribution']['min_tokens']} tokens")
    ax.set_ylabel("rollouts")
    ax.legend(frameon=False)
    ax.set_title("token-level confidence-vision correlation")
    fig.tight_layout()
    savefig(fig, "F6_confidence_vision")
    plt.close(fig)


def f7_trajectories():
    p = os.path.join(R, "signals", "mathvista_trajectories.json")
    if not os.path.exists(p):
        return
    tr = load_json(p)[:3]
    plt = plt_setup()
    # one axis per quantity (no dual axes): confidence row above vision row, sharing x, per trajectory
    fig, axes = plt.subplots(2, len(tr), figsize=(5.5, 2.6), sharex="col")
    axes = np.atleast_2d(axes)

    def smooth(x, w=25):
        x = np.asarray(x, float)
        return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) > w else x
    for j, t in enumerate(tr):
        axes[0, j].plot(smooth(t["c"]), color="#2a78d6", lw=1.0)
        axes[1, j].plot(smooth(t["v"]), color="#eb6834", lw=1.0)
        axes[0, j].set_title(f"{t['sample_id']} r{t['rollout_id']}: {t['category'].replace('_', ' ')}\n"
                             f"{'correct' if t['correct'] else 'incorrect'}, C={t['C']:.2f}, V={t['V']:.3f}",
                             fontsize=6)
        axes[1, j].set_xlabel("generated token")
    axes[0, 0].set_ylabel("confidence c_t\n(25-tok avg)")
    axes[1, 0].set_ylabel("vision v_t\n(25-tok avg)")
    fig.tight_layout()
    savefig(fig, "F7_illustrative_trajectories")
    plt.close(fig)


def f8_adaptive():
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    any_ = False
    for ax, ds in zip(axes, DS):
        g = os.path.join(R, "adaptive", f"{ds}_dev_grid.csv")
        a = opt(os.path.join(R, "adaptive", f"{ds}_adaptive.json"))
        if not os.path.exists(g) or a is None:
            continue
        any_ = True
        d = read_frame(g)
        for fam, col, mk in (("agreement", "#2a78d6", "o"), ("quality", "#eb6834", "s")):
            x = d[d.family == fam]
            ax.scatter(x.cost / 1000, 100 * x.acc, s=10, color=col, marker=mk, alpha=0.7, label=f"dev grid: {fam}",
                       edgecolor="white", linewidth=0.3)
            c = a["frozen"]["chosen"][fam]
            ax.scatter([c["cost"] / 1000], [100 * c["acc"]], s=40, facecolor="white", edgecolor=col, marker=mk,
                       linewidth=1.4, label=f"chosen ({fam})", zorder=5)
        ax.axhline(100 * a["frozen"]["fixed32_sc_dev_acc"], color=INK, lw=0.8, ls="--", label="fixed-32 SC (dev)")
        ax.axhline(100 * (a["frozen"]["fixed32_sc_dev_acc"] - 0.01), color=MUTED, lw=0.6, ls=":", label="-1 pp")
        ax.set_xlabel("dev tokens per question (thousands)")
        ax.set_ylabel("dev accuracy (%)")
        ax.set_title(DS[ds])
    if not any_:
        return plt.close(fig)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    savefig(fig, "F8_adaptive_dev_grid")
    plt.close(fig)


def f9_stopping():
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
    any_ = False
    for ax, ds in zip(axes, DS):
        p = os.path.join(R, "adaptive", f"{ds}_eval_simulation.parquet")
        if not os.path.exists(p):
            continue
        any_ = True
        sim = read_frame(p)
        sim = sim[sim.family == "quality"]
        bins = np.array([0, 0.25, 0.5, 0.75, 1.0001])
        labs = ["<25%", "25-50%", "50-75%", ">=75%"]
        b = np.digitize(sim.frac_correct_32, bins) - 1
        bottom = np.zeros(4)
        for n, col in zip((4, 8, 16, 32), ("#2a78d6", "#1baf7a", "#eda100", "#d6d5d0")):
            share = np.array([100 * ((sim.stop_n == n) & (b == k)).sum() / max((b == k).sum(), 1) for k in range(4)])
            ax.bar(np.arange(4), share, 0.7, bottom=bottom, color=col, edgecolor="white", lw=0.8, label=f"stop at {n}")
            bottom += share
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labs)
        ax.set_xlabel("correct share of the 32-rollout pool\n(retrospective)")
        ax.set_title(f"{DS[ds]}, quality-aware")
    if not any_:
        return plt.close(fig)
    axes[0].set_ylabel("% of runs")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, "F9_stopping_vs_pool_quality")
    plt.close(fig)


def f10_live():
    s = opt(os.path.join(R, "live", "live_summary.json"))
    if s is None:
        return
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    names = ["single", "parallel", "refine"]
    labs = ["Single", "Parallel", "Refine"]                 # allowances (1x3,072 / 3x1,024 / 1,024+2,048) in caption
    cols = ["#eda100", "#2a78d6", "#1baf7a"]
    for ax, (key, ylab) in zip(axes, (("acc", "accuracy (%)"), ("mean_output_tokens", "output tokens / question"),
                                      ("mean_time_s", "seconds / question"))):
        v = [s["strategies"][n][key] for n in names]
        if key == "acc":
            m = [100 * x[0] for x in v]
            err = [[100 * (x[0] - x[1]) for x in v], [100 * (x[2] - x[0]) for x in v]]
            ax.bar(range(3), m, 0.65, color=cols, edgecolor="white")
            ax.errorbar(range(3), m, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        else:
            ax.bar(range(3), v, 0.65, color=cols, edgecolor="white")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labs, fontsize=5.5)
        ax.set_ylabel(ylab)
    fig.tight_layout()
    savefig(fig, "F10_live_strategies")
    plt.close(fig)


def main():
    setup_logging()
    steps = [t1_data, t2_opportunity, t3_methods, t4_contrasts, t5_signals, t6_conf_vision, t7_live, t8_adaptive,
             t10_qualitative, f1_scaling, f2_cost, f3_failures, f4_gain_by_difficulty, f5_signal_gaps, f6_conf_vision,
             f7_trajectories, f8_adaptive, f9_stopping, f10_live]
    for fn in steps:
        try:
            fn()
        except Exception as e:                        # one missing input must not hide the other outputs
            logger.exception("report step %s failed: %s", fn.__name__, e)
    logger.info("tables: %d, figures: %d", len(glob.glob(os.path.join(TAB, "*.md"))),
                len(glob.glob(os.path.join(FIG, "*.png"))))


if __name__ == "__main__":
    main()
