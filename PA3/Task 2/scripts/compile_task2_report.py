"""Builds every Task 2 report table (Markdown + LaTeX) and figure (PDF + PNG) from results/.

Run from `PA3/Task 2`:  python scripts/compile_task2_report.py
Every number in the outputs traces to a JSON/CSV file under results/ (see Implementation_Notes/Task2_04).
Missing inputs are skipped with a log message, so the script can be run at any stage.
"""
import os
import sys
import csv
import glob
import logging
import argparse
from typing import Any, Dict, List, Optional

import numpy as np

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_ROOT)
sys.path.insert(0, TASK_ROOT)
from src.utils import load_config, load_json, setup_logging, task_path, write_text  # noqa: E402
from src.vqa_eval import normalize_answer  # noqa: E402

logger = logging.getLogger("compile_report")

CONDS = ["correct", "complementary", "mismatched", "no_image"]
CLABEL = {"correct": "Correct", "complementary": "Compl. swap", "mismatched": "Mismatched", "no_image": "No image"}
# Reference categorical palette (validated order; slots 1-4 for the four conditions, 1 and 3 for the two runs).
CCOLOR = {"correct": "#2a78d6", "complementary": "#eb6834", "mismatched": "#1baf7a", "no_image": "#eda100"}
RCOLOR = {"main": "#2a78d6", "compress49": "#1baf7a"}
RLABEL = {"main": "196 tokens (main)", "compress49": "49 tokens"}
SUITES = [("grounding", "Annotation-derived grounding"), ("vqa", "Natural VQAv2")]
KLABEL = {"presence": "Presence", "counting": "Counting", "yes/no": "Yes/No", "number": "Number", "other": "Other",
          "all": "All"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

R = TAB = FIG = ""
CFG: Dict[str, Any] = {}


def opt(path: str):
    return load_json(path) if os.path.exists(path) else None


def save_table(name: str, md: List[str], tex: List[str]):
    write_text(os.path.join(TAB, name + ".md"), "\n".join(md) + "\n")
    write_text(os.path.join(TAB, name + ".tex"), "\n".join(tex) + "\n")
    logger.info("table %s", name)


def tex_table(caption: str, label: str, colspec: str, header: str, rows: List[str]) -> List[str]:
    return [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule",
            *[r + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def f(x: Optional[float], d: int = 1) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def pct_ci(c: Optional[Dict], d: int = 1, signed: bool = False) -> str:
    """CI dict of a [0,1] quantity -> 'xx.x [lo, hi]' in percent."""
    if not c or not np.isfinite(c.get("mean", np.nan)):
        return "--"
    fmt = f"{{:+.{d}f}}" if signed else f"{{:.{d}f}}"
    return f"{fmt.format(100 * c['mean'])} [{fmt.format(100 * c['lo'])}, {fmt.format(100 * c['hi'])}]"


def num_ci(c: Optional[Dict], d: int = 3, signed: bool = False) -> str:
    if not c or not np.isfinite(c.get("mean", np.nan)):
        return "--"
    fmt = f"{{:+.{d}f}}" if signed else f"{{:.{d}f}}"
    return f"{fmt.format(c['mean'])} [{fmt.format(c['lo'])}, {fmt.format(c['hi'])}]"


def texesc(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


def md_and_tex(name, caption, header: List[str], rows: List[List[str]], colspec: Optional[str] = None):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    tex = tex_table(texesc(caption), f"tab:t2_{name.lower()}", colspec or ("l" + "r" * (len(header) - 1)),
                    " & ".join(texesc(h) for h in header), [" & ".join(texesc(c) for c in r) for r in rows])
    save_table(name, md, tex)


# --------------------------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------------------------
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 1.6, "axes.titlesize": 7.5, "axes.titleweight": "bold", "legend.fontsize": 6.5,
                         "lines.markersize": 3.5, "axes.titlepad": 11})
    return plt


def savefig(fig, name: str):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=170, bbox_inches="tight")
    logger.info("figure %s", name)


def stage_lines(ax, ends: List[int]):
    for e in ends[:-1]:
        ax.axvline(e, color=MUTED, lw=0.8, ls="--", zorder=0)


def stage_labels(ax, cfg):
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    start = 0
    for i, st in enumerate(cfg["train"]["stages"]):
        ax.text(start + st["updates"] / 2, 1.01, f"S{i + 1}", ha="center", va="bottom", fontsize=6, color=MUTED,
                transform=trans)
        start += st["updates"]


def rolling(y: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------
def t1_data():
    s = opt(os.path.join(R, "data", "data_stats.json"))
    if s is None:
        return logger.info("skip T1 (no data_stats.json)")
    rows = []
    order = ["captions_train", "captions_dev", "grounding_pool_presence", "grounding_pool_counting", "grounding_dev",
             "grounding_test", "vqa_train", "vqa_dev", "vqa_test"]
    for n in order:
        z = s["sizes"].get(n)
        if z is None:
            continue
        types = s["types"].get(n, {})
        rows.append([n, str(z.get("pairs", "--")), str(z.get("questions", z.get("examples"))), str(z["images"]),
                     ", ".join(f"{k} {v}" for k, v in types.items()) or "--"])
    md_and_tex("T1_data", "Selected data (seed 6304). Training: train2014; dev/test: val2014, image-disjoint across "
               "all evaluation sets and from training.", ["Set", "Pairs", "Questions/examples", "Images", "Types"],
               rows, "lrrrl")
    c = s["checks"]
    lines = [f"* eval suites image-disjoint: {c['eval_suites_image_disjoint']}; train/eval image overlap: "
             f"{c['train_eval_image_overlap']}; mismatched images exclude own pair: {c['mismatch_excludes_own_pair']}",
             f"* VQAv2 val filter: {s['vqa_val_filter']}", f"* VQAv2 train filter: {s['vqa_train_filter']}",
             f"* eligible pairs by type (val / train): {s['vqa_val_eligible_by_type']} / {s['vqa_train_eligible_by_type']}",
             f"* grounding label rules: {s['grounding_label_rules']}; not countable: {s['not_countable_categories']}",
             f"* max tokens: {s['max_tokens']} (caps {CFG['data']['caps']})",
             "* top answers: " + "; ".join(f"{k}: {v['top10'][:5]}" for k, v in s["answers"].items())]
    write_text(os.path.join(TAB, "T1_data_details.md"), "\n".join(lines) + "\n")


def final_summary(run: str) -> Optional[Dict]:
    return opt(os.path.join(R, "final", run, "summary.json"))


def t2_final(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return logger.info("skip T2 (no final summary for %s)", run)
    for suite, title in SUITES:
        s = S[suite]
        kinds = s["kinds"]
        header = ["Condition", "Acc"] + [KLABEL[k] for k in kinds] + ["Pair acc", "Identical", "Empty", "Ref. NLL",
                                                                       "s/answer"]
        rows = []
        for c in CONDS:
            x = s["conditions"].get(c)
            if x is None:
                continue
            rows.append([CLABEL[c], pct_ci(x["acc"])] + [pct_ci(x["acc_by_kind"][k]) for k in kinds] +
                        [pct_ci(x["pair_acc"]), pct_ci(x["identical"]), f(100 * x["empty"]["mean"]),
                         num_ci(x["nll"], 2), f(x["gen_sec_per_answer"], 3)])
        metric = "normalized exact match" if suite == "grounding" else "VQA soft accuracy (local subset)"
        md_and_tex(f"T2_final_{suite}_{run}",
                   f"{title}, final test ({s['n_pairs']} pairs / {s['n_questions']} questions), {RLABEL[run]}. Accuracy = "
                   f"{metric} in %, 95% pair-bootstrap CI (4,000 resamples); pair acc = both answers exactly match their "
                   f"canonical references; NLL = mean reference-answer NLL (termination excluded).", header, rows)


def t3_advantage(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    rows = []
    for suite, title in SUITES:
        s = S[suite]
        for other in ["complementary", "mismatched", "no_image"]:
            adv = s["advantage"].get(other)
            if not adv:
                continue
            for name in ["all"] + s["kinds"]:
                a = adv[name]
                rows.append([title if name == "all" else "", f"vs {CLABEL[other]}" if name == "all" else "",
                             KLABEL[name], pct_ci(a["d_acc"], signed=True), pct_ci(a["d_pair_acc"], signed=True),
                             num_ci(a["d_nll"], 2, signed=True)])
    md_and_tex(f"T3_correct_image_advantage_{run}",
               f"Correct-image advantage ({RLABEL[run]}): accuracy and pair accuracy of the correct image minus the "
               "control (percentage points), and NLL(control) - NLL(correct) of the original reference (> 0: the "
               "correct image makes the reference more likely). 95% CIs from matched pair resamples.",
               ["Suite", "Control", "Type", "d Acc (pp)", "d Pair acc (pp)", "d NLL"], rows, "lllrrr")


def t4_swap(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    rows = []
    for suite, title in SUITES:
        sw = S[suite]["swap"].get("complementary")
        if not sw:
            continue
        def row(label, t):
            return [title if label == "All" else "", label, str(t.get("n", 0)), f(100 * t.get("changed", np.nan)),
                    f(100 * t.get("follows_shown_image", np.nan)), f(100 * t.get("unchanged", np.nan)),
                    f(100 * t.get("changed_other", np.nan))]
        rows.append(row("All", sw["all"]))
        for k, t in sw["by_kind"].items():
            rows.append(row(KLABEL[k], t))
        rows.append(row("right w/ own image", sw["given_correct_under_original"]))
        rows.append(row("wrong w/ own image", sw["given_wrong_under_original"]))
        fs = S[suite].get("follows_shown_image", {}).get("all")
        rows.append(["", "pairs: both follow shown image", "", f(100 * sw.get("pairs_both_follow_shown", np.nan)),
                     f"score vs shown refs {pct_ci(fs)}", "", ""])
    md_and_tex(f"T4_swap_decomposition_{run}",
               f"What happens when the partner image is shown ({RLABEL[run]}), in % of questions. Changed = prediction "
               "differs from the correct-image prediction (image sensitivity); follows = changed to the shown "
               "(partner) image's answer; unchanged = same answer as with the own image; other = changed to an "
               "unsupported answer. 'Score vs shown refs' scores swapped answers against the partner's references.", ["Suite", "Subset", "n", "Changed", "Follows shown", "Unchanged", "Other"],
               rows, "llrrrrr")


def load_logs(run: str) -> Optional[Dict]:
    return opt(os.path.join(R, "training", run, "training_log.json"))


def t5_dev(run: str = "main"):
    L = load_logs(run)
    if L is None:
        return logger.info("skip T5 (no training log for %s)", run)
    ends = [0] + list(np.cumsum([s["updates"] for s in CFG["train"]["stages"]]))
    rows = []
    for d in sorted(L["dev"], key=lambda x: x["update"]):
        if d["update"] not in ends:
            continue
        r = [str(d["update"]), f(d["caption_loss"]["correct"], 3), f(d["caption_loss"]["no_image"], 3)]
        for suite, _ in SUITES:
            n = d["answer_nll"][suite]
            r += [f(n["correct"], 3), f(n["delta_swap"], 3, ), f(n["delta_no_image"], 3)]
            g = d.get("generation", {}).get(suite)
            r += [f(100 * g["correct"]["acc"]) if g else "--", f(100 * g["complementary"]["acc"]) if g else "--",
                  f(100 * g["correct"]["pair_acc"]) if g else "--"]
        r.append(f(d.get("visual_norm_ratio"), 2))
        rows.append(r)
    hdr = ["Update", "Cap loss", "Cap loss (no img)"]
    for suite, _ in SUITES:
        hdr += [f"{suite} NLL", f"{suite} dNLL swap", f"{suite} dNLL no-img", f"{suite} acc", f"{suite} acc swap",
                f"{suite} pair acc"]
    hdr.append("|z|/|e|")
    md_and_tex(f"T5_dev_by_stage_{run}",
               f"Development metrics before training and at the stage endpoints ({RLABEL[run]}). dNLL = NLL(control) - "
               "NLL(correct image); generated accuracy in %; |z|/|e| = mean visual-token norm relative to the text-"
               "embedding norm.", hdr, rows)


def t6_resources():
    fc = opt(os.path.join(R, "resources", "feature_cache.json"))
    rows = []
    if fc:
        n = sum(v["n_images"] for v in fc["splits"].values())
        rows.append(["Feature caching (SigLIP, frozen)", "--", "0", f(fc["total_sec"] / 60, 1),
                     f(fc["peak_gpu_gb"]["allocated"], 2), f"{n} images, {f(sum(v['file_mb'] for v in fc['splits'].values()), 0)} MB FP16"])
    for run in CFG["runs"]:
        L = load_logs(run)
        if L is None:
            continue
        for name, st in L["stages"].items():
            per = st["train_sec"] / max(st["updates"], 1)
            rows.append([f"{RLABEL[run]}: {name}", str(st["updates"]), f"{st['trainable_params']:,}",
                         f(st["train_sec"] / 60, 1), f(st["peak_gpu_gb"], 2), f"{f(per, 2)} s/update"])
        rows.append([f"{RLABEL[run]}: dev evaluation ({len(L['dev'])} points)", "--", "--", f(L["dev_eval_sec"] / 60, 1),
                     f(L["dev_peak_gpu_gb"], 2), f"checkpoint I/O {f(L['checkpoint_io_sec'] / 60, 1)} min"])
        S = final_summary(run)
        if S:
            gen = np.mean([S[s]["conditions"][c]["gen_sec_per_answer"] for s, _ in SUITES for c in CONDS
                           if c in S[s]["conditions"]])
            wall = sum(v["wall_sec"] for v in S["timing"].values())
            rows.append([f"{RLABEL[run]}: final evaluation (2 suites x 4 conditions)", "--", "--", f(wall / 60, 1),
                         f(S.get("peak_gpu_gb", {}).get("allocated"), 2), f"{f(1000 * gen, 1)} ms/answer (greedy, batch 8)"])
    if rows:
        md_and_tex("T6_resources", "Compute and memory (Kaggle T4). Training time excludes development evaluation and "
                   "checkpoint I/O; feature caching is separate from training; peak = max allocated GPU memory.",
                   ["Phase", "Updates", "Trainable params", "Time (min)", "Peak GPU (GB)", "Notes"], rows, "lrrrrl")


def t7_consistency(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    rows = []
    for suite, title in SUITES:
        c = S[suite].get("consistency")
        if not c:
            continue
        t = c["table"]
        rows.append([title, str(c["n"]), f(c["spearman_dnll_dscore"], 2), f(100 * c["frac_lik_prefers_correct"]),
                     f(100 * c["frac_gen_better_with_correct"]), f(100 * c["frac_gen_tie"]),
                     f(100 * c["agreement_when_gen_differs"]),
                     f"{t['lik_prefers_correct&gen_better']}/{t['lik_prefers_correct&gen_tie']}/{t['lik_prefers_correct&gen_worse']}",
                     f"{t['lik_not&gen_better']}/{t['lik_not&gen_tie']}/{t['lik_not&gen_worse']}"])
    md_and_tex(f"T7_likelihood_vs_generation_{run}",
               f"Do teacher-forced likelihood and free generation agree about image use ({RLABEL[run]})? Per question: "
               "dNLL = NLL(swap) - NLL(correct), d score = score(correct) - score(swap). Counts are better/tie/worse "
               "generation outcomes when likelihood does / does not prefer the correct image.",
               ["Suite", "n", "Spearman", "Lik. prefers correct %", "Gen. better %", "Gen. tie %",
                "Agreement % (gen differs)", "Lik+: b/t/w", "Lik-: b/t/w"], rows)


def t8_prior_bias(run: str = "main"):
    S = final_summary(run)
    P = opt(os.path.join(R, "final", "prior_baseline.json"))
    if S is None:
        return
    rows = []
    for suite, title in SUITES:
        s = S[suite]
        p = (P or {}).get(suite, {})
        for k in ["all"] + s["kinds"]:
            pr = p.get("acc") if k == "all" else p.get("acc_by_kind", {}).get(k)
            accs = [(s["conditions"][c]["acc"] if k == "all" else s["conditions"][c]["acc_by_kind"][k])["mean"]
                    for c in CONDS if c in s["conditions"]]
            rows.append([title if k == "all" else "", KLABEL[k], f(100 * pr) if pr is not None else "--",
                         *[f(100 * a) for a in accs]])
        yr = [f(100 * s["conditions"][c].get("yes_rate", np.nan)) for c in CONDS if c in s["conditions"]]
        rows.append(["", "yes-rate (yes/no questions)", "--", *yr])
    md_and_tex(f"T8_prior_and_bias_{run}",
               f"Language-prior baseline (most frequent training answer per question type; text only, pair accuracy 0 "
               f"by construction) vs the VLM ({RLABEL[run]}) under each image condition; accuracy in %, plus the "
               "share of 'yes' answers on yes/no-type questions (references are balanced 50/50 by construction).",
               ["Suite", "Type", "Prior", *[CLABEL[c] for c in CONDS]], rows, "llrrrrr")


def t9_count_confusion(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    md = []
    for suite, title in SUITES:
        cc = S[suite].get("count_confusion")
        if not cc:
            continue
        for c in ("correct", "no_image"):
            m = cc.get(c, {})
            preds = sorted({p for v in m.values() for p in v}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 99, x))
            md += [f"**{title} - {CLABEL[c]}** (rows: reference count, columns: prediction)", "",
                   "| ref \\ pred | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
            md += [f"| {r} | " + " | ".join(str(m[r].get(p, 0)) for p in preds) + " |" for r in sorted(m, key=lambda x: int(x) if x.isdigit() else 99)]
            md.append("")
    if md:
        write_text(os.path.join(TAB, f"T9_count_confusion_{run}.md"), "\n".join(md))


def t10_compression():
    A, B = final_summary("main"), final_summary("compress49")
    if A is None or B is None:
        return logger.info("skip T10 (compression run not evaluated)")
    rows = []
    for suite, title in SUITES:
        for c in CONDS:
            a, b = A[suite]["conditions"][c], B[suite]["conditions"][c]
            rows.append([title if c == "correct" else "", CLABEL[c], pct_ci(a["acc"]), pct_ci(b["acc"]),
                         pct_ci(a["pair_acc"]), pct_ci(b["pair_acc"]), f(a["nll"]["mean"], 2), f(b["nll"]["mean"], 2),
                         f(1000 * a["gen_sec_per_answer"], 1), f(1000 * b["gen_sec_per_answer"], 1)])
        da = A[suite]["advantage"]["complementary"]["all"]
        db = B[suite]["advantage"]["complementary"]["all"]
        rows.append(["", "d vs swap", pct_ci(da["d_acc"], signed=True), pct_ci(db["d_acc"], signed=True),
                     pct_ci(da["d_pair_acc"], signed=True), pct_ci(db["d_pair_acc"], signed=True),
                     num_ci(da["d_nll"], 2, True), num_ci(db["d_nll"], 2, True), "", ""])
    La, Lb = load_logs("main"), load_logs("compress49")
    if La and Lb:
        rows.append(["Training", "time (min) / peak GB", f(La.get("train_sec_total", 0) / 60, 1),
                     f(Lb.get("train_sec_total", 0) / 60, 1),
                     f(max(s["peak_gpu_gb"] for s in La["stages"].values()), 2),
                     f(max(s["peak_gpu_gb"] for s in Lb["stages"].values()), 2), "", "", "", ""])
    md_and_tex("T10_compression", "Optional visual-token compression: 196 tokens (main) vs 49 tokens (2 x 2 average "
               "pooling before a newly initialized connector and adapters; same seed, data, budgets and evaluation).",
               ["Suite", "Condition", "Acc 196", "Acc 49", "Pair 196", "Pair 49", "NLL 196", "NLL 49",
                "ms/ans 196", "ms/ans 49"], rows, "llrrrrrrrr")


def t11_label_audit():
    key = opt(os.path.join(R, "label_audit", "label_audit_key.json"))
    filled = os.path.join(R, "label_audit", "label_audit_filled.csv")
    if key is None or not os.path.exists(filled):
        return logger.info("skip T11 (label audit not filled: %s)", filled)
    hand = {}
    with open(filled, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].isdigit() and len(row) >= 3 and row[2].strip():
                hand[row[0].zfill(3)] = row[2].strip().lower()
    by_kind: Dict[str, Dict[str, int]] = {}
    disagree = []
    for k in key:
        h = hand.get(k["audit_id"])
        if h is None:
            continue
        d = by_kind.setdefault(k["kind"], {"n": 0, "agree": 0, "ambiguous": 0})
        d["n"] += 1
        if h == "?":
            d["ambiguous"] += 1
        elif normalize_answer(h) == normalize_answer(k["label"]):
            d["agree"] += 1
        else:
            disagree.append({**k, "hand": h})
    rows = [[KLABEL[k], str(v["n"]), f(100 * v["agree"] / max(v["n"], 1)), f(100 * v["ambiguous"] / max(v["n"], 1))]
            for k, v in by_kind.items()]
    md_and_tex("T11_label_audit", "Blind hand check of the annotation-derived final-test labels (image + question only, "
               "no label or prediction shown).", ["Type", "Checked", "Agree with COCO-derived label %", "Ambiguous %"],
               rows)
    write_text(os.path.join(TAB, "T11_label_audit_disagreements.md"),
               "\n".join(f"* {d['qid']} ({d['kind']}): label {d['label']!r}, hand {d['hand']!r}" for d in disagree) + "\n")


def t12_qualitative(run: str = "main"):
    Q = opt(os.path.join(R, "final", run, "qualitative.json"))
    if Q is None:
        return
    from src.analysis import QUAL_CATEGORIES
    md = [f"# Qualitative complementary-pair examples ({RLABEL[run]})", "",
          "Selection rule: every final-test pair is assigned to the first matching category below using only the "
          "correct-image and swapped-image generations; within a category pairs are ordered by SHA-256(6304:pair_id) and "
          f"the first {CFG['final_eval']['qualitative_per_category']} are shown. Category sizes are given so the "
          "examples can be read against how common each behaviour is.", ""]
    md += [f"* **{k}**: {v}" for k, v in QUAL_CATEGORIES.items()]
    for suite, title in SUITES:
        md += ["", f"## {title}", ""]
        for ex in Q.get(suite, []):
            p = ex["predictions"]
            md += [f"**[{ex['category']} - {ex['category_size']} pairs] {ex['kind']}**: {ex['question']}  ",
                   f"images {ex['images'][0]} / {ex['images'][1]}; references {ex['references'][0]!r} / {ex['references'][1]!r}  "]
            for c in CONDS:
                if c in p:
                    md.append(f"- {CLABEL[c]}: {p[c][0]!r} / {p[c][1]!r} (NLL {ex['nll'][c][0]:.2f} / {ex['nll'][c][1]:.2f})")
            md.append("")
    write_text(os.path.join(TAB, f"T12_qualitative_{run}.md"), "\n".join(md))


def t13_mixture(run: str = "main"):
    sch = opt(os.path.join(R, "training", run, "schedule_summary.json"))
    if sch is None:
        return
    rows = []
    for st in CFG["train"]["stages"]:
        s = sch[st["name"]]
        target = st["mixture"]
        rows.append([st["name"], str(s["examples"]),
                     ", ".join(f"{k} {100 * s['draw_fraction'].get(k, 0):.1f}% (target {100 * v:.0f}%)" for k, v in target.items()),
                     ", ".join(f"{k} {v}" for k, v in s["unique_pool_items"].items())])
    md_and_tex(f"T13_training_mixture_{run}", "Realized sampling mixture (seeded sampler, no quotas): share of draws per "
               "component and number of distinct pool items presented.", ["Stage", "Examples", "Draws", "Distinct items"],
               rows, "lrll")


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
def f1_learning_curves(run: str = "main"):
    L = load_logs(run)
    if L is None:
        return
    plt = plt_setup()
    ends = list(np.cumsum([s["updates"] for s in CFG["train"]["stages"]]))
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.4))
    ax = axes[0, 0]
    ups = np.array([u["update"] for u in L["updates"]])
    comp_color = {"caption": "#2a78d6", "presence": "#eb6834", "counting": "#1baf7a", "vqa": "#eda100"}
    for comp, col in comp_color.items():
        y = np.array([u["component_loss"].get(comp, np.nan) if u["component_loss"] else np.nan for u in L["updates"]],
                     dtype=float)
        if np.isfinite(y).any():
            ax.plot(ups, rolling(y, 25), color=col, lw=1.4, label=comp)
    ax.set_title("(a) training loss (25-update mean)")
    ax.set_xlabel("update")
    ax.set_ylabel("per-example CE")
    ax.legend()
    dev = sorted(L["dev"], key=lambda d: d["update"])
    du = [d["update"] for d in dev]
    ax = axes[0, 1]
    ax.plot(du, [d["caption_loss"]["correct"] for d in dev], "-o", ms=4, color=CCOLOR["correct"], label="with image")
    ax.plot(du, [d["caption_loss"]["no_image"] for d in dev], "-o", ms=4, color=CCOLOR["no_image"], label="no image")
    ax.set_title("(b) dev caption reference loss")
    ax.set_xlabel("update")
    ax.legend()
    for j, (suite, title) in enumerate(SUITES):
        ax = axes[1, 0] if j == 0 else axes[1, 1]
        for c in ("correct", "complementary", "no_image"):
            ax.plot(du, [d["answer_nll"][suite][c] for d in dev], "-o", ms=4, color=CCOLOR[c], label=CLABEL[c])
        ax.set_title(f"({'c' if j == 0 else 'd'}) dev answer NLL, {suite}")
        ax.set_xlabel("update")
        ax.legend()
    ax = axes[2, 0]
    for suite, col, mk in (("grounding", "#2a78d6", "o"), ("vqa", "#eb6834", "s")):
        ax.plot(du, [d["answer_nll"][suite]["delta_swap"] for d in dev], "-" + mk, ms=4, color=col,
                label=f"{suite}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("(e) swap penalty NLL(partner) - NLL(correct)")
    ax.set_xlabel("update")
    ax.legend()
    ax = axes[2, 1]
    if "generation" in dev[0]:
        for suite, col, mk in (("grounding", "#2a78d6", "o"), ("vqa", "#eb6834", "s")):
            ax.plot(du, [100 * d["generation"][suite]["correct"]["acc"] for d in dev], "-" + mk, ms=4, color=col,
                    label=f"{suite}, correct image")
            ax.plot(du, [100 * d["generation"][suite]["no_image"]["acc"] for d in dev], ":" + mk, ms=3, color=col,
                    lw=1.4, label=f"{suite}, no image")
        ax.set_ylabel("accuracy (%)")
        ax.legend()
    ax.set_title("(f) dev generated accuracy")
    ax.set_xlabel("update")
    for a in axes.flat:
        stage_lines(a, ends)
        stage_labels(a, CFG)
    fig.tight_layout()
    savefig(fig, f"F1_learning_curves_{run}")
    plt.close(fig)


def f2_final_accuracy(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), gridspec_kw={"width_ratios": [3, 4]})
    for ax, (suite, title) in zip(axes, SUITES):
        s = S[suite]
        groups = ["all"] + s["kinds"]
        w = 0.19
        for i, c in enumerate(CONDS):
            if c not in s["conditions"]:
                continue
            vals, lo, hi = [], [], []
            for g in groups:
                x = s["conditions"][c]["acc"] if g == "all" else s["conditions"][c]["acc_by_kind"][g]
                vals.append(100 * x["mean"])
                lo.append(100 * (x["mean"] - x["lo"]))
                hi.append(100 * (x["hi"] - x["mean"]))
            xs = np.arange(len(groups)) + (i - 1.5) * (w + 0.01)
            ax.bar(xs, vals, w, color=CCOLOR[c], label=CLABEL[c], edgecolor="white", linewidth=0.8)
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([KLABEL[g] for g in groups])
        ax.set_ylim(0, 100)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(title)
    # legend above the panels so it never covers the tall correct-image bars
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, f"F2_final_accuracy_{run}")
    plt.close(fig)


def f3_swap(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    plt = plt_setup()
    labels, parts = [], []
    for suite, title in SUITES:
        sw = S[suite]["swap"].get("complementary")
        if not sw:
            continue
        for k, t in [("all", sw["all"])] + list(sw["by_kind"].items()):
            labels.append(f"{'Grounding' if suite == 'grounding' else 'VQAv2'}: {KLABEL[k]}")
            parts.append([t.get("follows_shown_image", 0), t.get("unchanged", 0), t.get("changed_other", 0)])
    if not parts:
        return
    parts = 100 * np.array(parts)
    fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(labels) + 0.9))
    cols = ["#2a78d6", "#8c8c8c", "#eb6834"]
    names = ["changed to shown image's answer", "unchanged answer", "changed to unsupported answer"]
    left = np.zeros(len(labels))
    y = np.arange(len(labels))[::-1]
    for j in range(3):
        ax.barh(y, parts[:, j], left=left, color=cols[j], edgecolor="white", linewidth=2, label=names[j], height=0.62)
        for yi, l, v in zip(y, left, parts[:, j]):
            if v >= 8:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white")
        left += parts[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of questions after the complementary swap")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    savefig(fig, f"F3_swap_outcomes_{run}")
    plt.close(fig)


def f4_consistency(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    rng = np.random.default_rng(0)
    for ax, (suite, title) in zip(axes, SUITES):
        c = S[suite].get("consistency")
        if not c:
            continue
        pq = c["per_question"]
        x = np.array([p["d_nll"] for p in pq])
        y = np.array([p["d_score"] for p in pq]) + rng.uniform(-0.04, 0.04, len(pq))
        ax.scatter(x, y, s=14, color="#2a78d6", alpha=0.6, edgecolors="white", linewidths=0.6)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("NLL(swap) - NLL(correct)")
        ax.set_ylabel("score(correct) - score(swap)")
        ax.set_title(f"{'grounding' if suite == 'grounding' else 'VQAv2'}: Spearman {c['spearman_dnll_dscore']:.2f}")
    fig.tight_layout()
    savefig(fig, f"F4_likelihood_vs_generation_{run}")
    plt.close(fig)


def f5_confusion(run: str = "main"):
    S = final_summary(run)
    if S is None:
        return
    plt = plt_setup()
    panels = []
    for suite, title in SUITES:
        cc = S[suite].get("count_confusion")
        if cc:
            for c in ("correct", "no_image"):
                panels.append((f"{'Counting' if suite == 'grounding' else 'VQAv2 number'}, {CLABEL[c].lower()}", cc[c]))
    if not panels:
        return
    nrow = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(5.5, 2.3 * nrow))
    for ax in np.atleast_1d(axes).flat[len(panels):]:
        ax.axis("off")
    for ax, (t, m) in zip(np.atleast_1d(axes).flat, panels):
        refs = sorted(m, key=lambda x: int(x) if x.isdigit() else 99)
        preds = [str(i) for i in range(0, 7)] + ["other", "empty"]
        M = np.array([[m[r].get(p, 0) for p in preds] for r in refs], dtype=float)
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(refs)):
            for j in range(len(preds)):
                if M[i, j]:
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=6.5,
                            color="white" if Mn[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels(preds, fontsize=6)
        ax.set_yticks(range(len(refs)))
        ax.set_yticklabels(refs)
        ax.set_xlabel("prediction")
        ax.set_ylabel("reference")
        ax.set_title(t)
        ax.grid(False)
    fig.tight_layout()
    savefig(fig, f"F5_count_confusion_{run}")
    plt.close(fig)


def f7_norms():
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    any_ = False
    for run in CFG["runs"]:
        L = load_logs(run)
        if L is None:
            continue
        dev = sorted(L["dev"], key=lambda d: d["update"])
        ax.plot([d["update"] for d in dev], [d["visual_norm_ratio"] for d in dev], "-o", ms=4, color=RCOLOR[run],
                label=RLABEL[run])
        any_ = True
    if not any_:
        plt.close(fig)
        return
    stage_lines(ax, list(np.cumsum([s["updates"] for s in CFG["train"]["stages"]])))
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xlabel("update")
    ax.set_ylabel("visual / text embedding norm")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "F7_visual_token_norm")
    plt.close(fig)


def f8_compression():
    A, B = final_summary("main"), final_summary("compress49")
    if A is None or B is None:
        return
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    for ax, (suite, title) in zip(axes, SUITES):
        w = 0.36
        for i, (run, S) in enumerate((("main", A), ("compress49", B))):
            vals = [100 * S[suite]["conditions"][c]["acc"]["mean"] for c in CONDS]
            lo = [100 * (S[suite]["conditions"][c]["acc"]["mean"] - S[suite]["conditions"][c]["acc"]["lo"]) for c in CONDS]
            hi = [100 * (S[suite]["conditions"][c]["acc"]["hi"] - S[suite]["conditions"][c]["acc"]["mean"]) for c in CONDS]
            xs = np.arange(len(CONDS)) + (i - 0.5) * (w + 0.02)
            ax.bar(xs, vals, w, color=RCOLOR[run], label=RLABEL[run], edgecolor="white", linewidth=0.8)
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([CLABEL[c].replace(" ", "\n") for c in CONDS])   # two-line labels: no overlap at 5.5 in
        ax.set_ylim(0, 100)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, "F8_compression")
    plt.close(fig)


def main():
    global R, TAB, FIG, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    args = ap.parse_args()
    setup_logging()
    CFG = load_config(args.config)
    R = task_path(CFG["paths"].get("results_dir", "results"))
    TAB, FIG = os.path.join(R, "tables"), os.path.join(R, "figures")
    steps = [t1_data, t6_resources, t10_compression, t11_label_audit, f7_norms, f8_compression]
    for run in CFG["runs"]:
        steps += [lambda r=run, fn=fn: fn(r) for fn in (t2_final, t3_advantage, t4_swap, t5_dev, t7_consistency,
                                                         t8_prior_bias, t9_count_confusion, t12_qualitative,
                                                         t13_mixture, f1_learning_curves, f2_final_accuracy, f3_swap,
                                                         f4_consistency, f5_confusion)]
    for fn in steps:
        try:
            fn()
        except Exception as e:              # one broken input must not hide every other table
            logger.exception("report step failed: %s", e)
    logger.info("tables: %d, figures: %d", len(glob.glob(os.path.join(TAB, "*.md"))),
                len(glob.glob(os.path.join(FIG, "*.png"))))


if __name__ == "__main__":
    main()
