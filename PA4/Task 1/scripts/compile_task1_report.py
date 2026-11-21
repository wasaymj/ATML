"""Compile Task 1 tables (results/tables/*.md + .csv), figures (results/figures/*.pdf + .png) and analysis JSON
(results/analysis/*.json) from the saved per-condition results. CPU only; safe to re-run; every table/figure is
independent and is skipped (with a message) if its inputs do not exist yet.

  python scripts/compile_task1_report.py [--results results] [--config configs/task1.yaml]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats import acc_paired, ppl_paired, spearman  # noqa: E402
from src.utils import RESULTS, load_config, load_json, save_json  # noqa: E402

COND = ["fp16", "P1_mag_s25", "P1_mag_s50", "P1_mag_s75", "P2_wanda_s25", "P2_wanda_s50", "P2_wanda_s75",
        "P3_s50", "P3_s75", "P2_wanda24_s50", "Q1_rtn_b8", "Q1_rtn_b4", "Q1_rtn_b3", "Q2_awq_b8", "Q2_awq_b4",
        "Q2_awq_b3", "Q3_mix_b4budget", "Q3_mix_b3budget"]
LABEL = {"fp16": "FP16", "P1_mag_s25": "P1 mag 25%", "P1_mag_s50": "P1 mag 50%", "P1_mag_s75": "P1 mag 75%",
         "P2_wanda_s25": "P2 Wanda 25%", "P2_wanda_s50": "P2 Wanda 50%", "P2_wanda_s75": "P2 Wanda 75%",
         "P3_s50": "P3 nonuni. 50%", "P3_s75": "P3 nonuni. 75%", "P2_wanda24_s50": "Wanda 2:4",
         "Q1_rtn_b8": "Q1 RTN INT8", "Q1_rtn_b4": "Q1 RTN INT4", "Q1_rtn_b3": "Q1 RTN INT3",
         "Q2_awq_b8": "Q2 AWQ INT8", "Q2_awq_b4": "Q2 AWQ INT4", "Q2_awq_b3": "Q2 AWQ INT3",
         "Q3_mix_b4budget": "Q3 mixed (INT4 budget)", "Q3_mix_b3budget": "Q3 mixed (INT3 budget)"}
PAIRS_MAIN = [("P2_wanda_s50", "P3_s50"), ("P2_wanda_s75", "P3_s75"), ("Q2_awq_b4", "Q3_mix_b4budget"),
              ("Q2_awq_b3", "Q3_mix_b3budget")]
PAIRS_EXTRA = [("P1_mag_s25", "P2_wanda_s25"), ("P1_mag_s50", "P2_wanda_s50"), ("P1_mag_s75", "P2_wanda_s75"),
               ("P2_wanda_s50", "P2_wanda24_s50"), ("Q1_rtn_b8", "Q2_awq_b8"), ("Q1_rtn_b4", "Q2_awq_b4"),
               ("Q1_rtn_b3", "Q2_awq_b3")]
SUITES = [("wikitext2", "WikiText-2"), ("c4val", "C4 val")]
PROJ = ["q", "k", "v", "o", "gate", "up", "down"]
GROUPS = ["qkv", "o", "gateup", "down"]
COLORS = {"P1": "#1f77b4", "P2": "#d62728", "P3": "#2ca02c", "NM": "#9467bd", "Q1": "#1f77b4", "Q2": "#d62728",
          "Q3": "#2ca02c", "baseline": "black"}


# ------------------------------------------------------------------------------------------------ I/O
class R:
    def __init__(self, res: Path, cfg):
        self.res, self.cfg = res, cfg
        self.tab, self.fig, self.ana = res / "tables", res / "figures", res / "analysis"
        for d in (self.tab, self.fig, self.ana):
            d.mkdir(parents=True, exist_ok=True)
        self.S = {}
        for c in COND:
            p = res / "conditions" / c / "summary.json"
            if p.exists():
                self.S[c] = load_json(p)
        self._cache = {}

    def has(self, *c):
        return all(x in self.S for x in c)

    def ppl(self, c):
        k = ("ppl", c)
        if k not in self._cache:
            self._cache[k] = dict(np.load(self.res / "conditions" / c / "ppl.npz"))
        return self._cache[k]

    def hs(self, c):
        k = ("hs", c)
        if k not in self._cache:
            with gzip.open(self.res / "conditions" / c / "hellaswag.jsonl.gz", "rt") as f:
                self._cache[k] = [json.loads(x) for x in f]
        return self._cache[k]

    def j(self, *parts):
        p = self.res.joinpath(*parts)
        return load_json(p) if p.exists() else None


def write_table(r: R, name, header, rows, note=""):
    with open(r.tab / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    if note:
        lines += ["", note]
    (r.tab / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def f3(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(c, d=2):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.3, "pdf.fonttype": 42,
                         "figure.constrained_layout.use": True})
    return plt


def save_fig(r: R, fig, name):
    fig.savefig(r.fig / f"{name}.pdf")
    fig.savefig(r.fig / f"{name}.png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------------------- paired stats
def paired(r: R, a, b):
    cfg = r.cfg["bootstrap"]
    out = {"a": a, "b": b}
    for s, _ in SUITES:
        pa, pb = r.ppl(a), r.ppl(b)
        out[s] = ppl_paired(pa[f"{s}_win_sum"], pa[f"{s}_win_n"], pb[f"{s}_win_sum"], pb[f"{s}_win_n"],
                            cfg["resamples"], cfg["seed"])
    ha, hb = r.hs(a), r.hs(b)
    n = min(len(ha), len(hb))
    assert [x["order"] for x in ha[:n]] == [x["order"] for x in hb[:n]]
    out["hellaswag_n"] = n
    out["acc"] = acc_paired([x["correct"] for x in ha[:n]], [x["correct"] for x in hb[:n]], cfg["resamples"], cfg["seed"])
    out["acc_norm"] = acc_paired([x["correct_norm"] for x in ha[:n]], [x["correct_norm"] for x in hb[:n]],
                                 cfg["resamples"], cfg["seed"])
    return out


# ============================================================================================ tables
def t_quality(r: R):
    if "fp16" not in r.S:
        return "baseline missing"
    rows, anal = [], {}
    for c in COND:
        if c not in r.S:
            continue
        s = r.S[c]
        st = s["storage"]
        setting = (f"{s['achieved_sparsity'] * 100:.2f}% zeros" if "achieved_sparsity" in s else
                   f"{st['effective_bits_per_eligible_weight']:.3f} bits/w" if "bits_map" in s else "16 bits/w")
        row = [LABEL[c], setting, f"{st['total_bytes'] / 2**20:,.1f}"]
        if c == "fp16":
            for k, _ in SUITES:
                row += [f3(s[k]["ppl"]), "-"]
            row += [s["hellaswag"]["n"], f3(100 * s["hellaswag"]["acc"], 2), "-", f3(100 * s["hellaswag"]["acc_norm"], 2), "-"]
        else:
            p = paired(r, "fp16", c)
            anal[c] = p
            for k, _ in SUITES:
                row += [f3(s[k]["ppl"]), f"{100 * p[k]['rel_ppl']:+.1f}% {ci([100 * x for x in p[k]['rel_ppl_ci']], 1)}"]
            row += [p["hellaswag_n"], f3(p["acc"]["acc_b"], 2), f"{p['acc']['d_pp']:+.2f} {ci(p['acc']['d_pp_ci'])}",
                    f3(p["acc_norm"]["acc_b"], 2), f"{p['acc_norm']['d_pp']:+.2f} {ci(p['acc_norm']['d_pp_ci'])}"]
        row += [f3(s["dev"]["ppl"]), f"{s['calibration']['kl']:.4f}"]
        rows.append(row)
    s0 = r.S["fp16"]
    write_table(r, "T1_quality", ["Condition", "Budget", "Total MB", "Wiki ppl", "Wiki rel. change [95% CI]",
                                  "C4 ppl", "C4 rel. change [95% CI]", "HS n", "HS acc %", "d acc pp [CI]",
                                  "HS acc_norm %", "d acc_norm pp [CI]", "Dev ppl", "Calib KL"], rows,
                f"Paired against FP16 on identical windows/examples; perplexity CIs resample whole windows "
                f"(B={r.cfg['bootstrap']['resamples']}). N scored tokens: WikiText-2 {s0['wikitext2']['N']}, "
                f"C4 {s0['c4val']['N']}; mean NLL FP16: {s0['wikitext2']['mean_nll']:.4f} / {s0['c4val']['mean_nll']:.4f}. "
                f"HellaSwag n=2000 for the manual's full set, otherwise the first 512 IDs. Total MB includes all "
                f"metadata and the uncompressed embedding/norms/biases (tied LM head counted once).")
    save_json(r.ana / "paired_vs_fp16.json", anal)


def t_nll(r: R):
    """Manual: report N and the mean NLL as well as perplexity, for every condition and both suites (+ dev)."""
    rows = []
    for c in COND:
        if c not in r.S:
            continue
        s = r.S[c]
        row = [LABEL[c]]
        for k in ("wikitext2", "c4val", "dev"):
            row += [s[k]["N"], f"{s[k]['mean_nll']:.5f}", f"{s[k]['ppl']:.4f}"]
        rows.append(row)
    if rows:
        write_table(r, "T1b_nll_and_token_counts", ["Condition", "Wiki N", "Wiki mean NLL", "Wiki ppl", "C4 N",
                                                    "C4 mean NLL", "C4 ppl", "Dev N", "Dev mean NLL", "Dev ppl"], rows,
                    "PPL = exp(mean NLL) over N scored target tokens; the first token of every window is never scored. "
                    "Dev = the 32 x 512 C4-train development sequences (selection set, not a test set).")


def t_pairs(r: R):
    rows, anal = [], {}
    for a, b in PAIRS_MAIN + PAIRS_EXTRA:
        if not r.has(a, b):
            continue
        p = paired(r, a, b)
        anal[f"{b}-{a}"] = p
        row = [f"{LABEL[b]} - {LABEL[a]}", "main" if (a, b) in PAIRS_MAIN else "extra"]
        for k, _ in SUITES:
            row += [f"{p[k]['d_ppl']:+.3f} {ci(p[k]['d_ppl_ci'], 3)}", f"{p[k]['d_nll']:+.4f} {ci(p[k]['d_nll_ci'], 4)}"]
        row += [p["hellaswag_n"], f"{p['acc']['d_pp']:+.2f} {ci(p['acc']['d_pp_ci'])}",
                f"{p['acc_norm']['d_pp']:+.2f} {ci(p['acc_norm']['d_pp_ci'])}", f"{p['acc_norm']['a_only']}/{p['acc_norm']['b_only']}"]
        rows.append(row)
    if rows:
        write_table(r, "T2_paired_differences", ["Difference (B - A)", "Type", "d Wiki ppl [95% CI]", "d Wiki NLL [CI]",
                                                 "d C4 ppl [CI]", "d C4 NLL [CI]", "HS n", "d acc pp [CI]",
                                                 "d acc_norm pp [CI]", "acc_norm A-only/B-only"], rows,
                    "Paired 95% bootstrap intervals: perplexity resamples whole evaluation windows and recomputes the pooled "
                    "token NLL before exponentiating; HellaSwag resamples examples (common subset: 2,000 if both "
                    "conditions have it, else the first 512).")
        save_json(r.ana / "paired_differences.json", anal)


def t_budget(r: R):
    rows = []
    for a, b in [("P2_wanda_s50", "P3_s50"), ("P2_wanda_s75", "P3_s75"), ("P2_wanda_s50", "P2_wanda24_s50"),
                 ("P1_mag_s50", "P2_wanda_s50")]:
        if r.has(a, b):
            za, zb = r.S[a]["zeros"], r.S[b]["zeros"]
            rows.append([f"{LABEL[b]} vs {LABEL[a]}", "zeroed eligible weights", f"{zb:,}", f"{za:,}", zb - za,
                         "OK" if zb == za else "MISMATCH"])
    for a, b in [("Q2_awq_b4", "Q3_mix_b4budget"), ("Q2_awq_b3", "Q3_mix_b3budget"), ("Q1_rtn_b4", "Q2_awq_b4"),
                 ("Q1_rtn_b3", "Q2_awq_b3")]:
        if r.has(a, b):
            ba, bb = r.S[a]["storage"]["total_bytes"], r.S[b]["storage"]["total_bytes"]
            rows.append([f"{LABEL[b]} vs {LABEL[a]}", "total packed bytes incl. metadata + FP16 rest", f"{bb:,}", f"{ba:,}",
                         bb - ba, "OK" if bb <= ba else "OVER BUDGET"])
    if rows:
        write_table(r, "T3_matched_budget_checks", ["Comparison", "Quantity", "Method", "Reference", "Difference",
                                                    "Check"], rows)


def t_resources(r: R):
    rows = []
    for c in COND:
        if c not in r.S:
            continue
        s = r.S[c]
        st, lat, t = s["storage"], s["latency"], s["timings_s"]
        res = r.j("conditions", c, "resources.json") or {}
        rows.append([LABEL[c], f"{st['total_bytes'] / 2**20:,.1f}",
                     f"{(st.get('dense_fp16_with_zeros_total_bytes') or st['fp16_total_bytes']) / 2**20:,.1f}",
                     f"{st['serialized_file_bytes'] / 2**20:,.1f}" if st.get("serialized_file_bytes") else "-",
                     f"{st.get('csr_int64_total_bytes', 0) / 2**20:,.1f}" if "csr_int64_total_bytes" in st else "-",
                     f"{t['compression_s']:.1f}",
                     f"{(s.get('compression_peak_allocated_bytes') or 0) / 2**30:.2f}",
                     f3(res.get("prefill_512", {}).get("median_ms"), 1), f3(res.get("prefill_2048", {}).get("median_ms"), 1),
                     f3(res.get("decode_64", {}).get("median_ms"), 1),
                     f"{res.get('prefill_2048', {}).get('peak_allocated_bytes', 0) / 2**30:.2f}",
                     f"{res.get('weights_allocated_bytes', 0) / 2**30:.2f}"])
    if rows:
        reps = r.cfg["resources"]["reps"]
        write_table(r, "T4_resources", ["Condition", "Logical MB (sparse bitmap / packed)", "As executed MB (dense FP16)",
                                        "Serialized file MB", "CSR int64 MB", "Compression s", "Compression peak GiB",
                                        "Prefill 512 ms", "Prefill 2048 ms", "Decode 64 tok ms",
                                        "Peak alloc prefill-2048 GiB", "Weights alloc GiB"], rows,
                    f"Latency: batch 1, median of {reps} CUDA-synchronised repetitions after warm-up; decode = 64 greedy "
                    "steps after a 512-token prefill. Every compressed model executes as dense FP16 (zeros / fake-quantized "
                    "weights), so its GPU allocation and latency are those of FP16; logical/packed bytes are storage only.")


def t_microbench(r: R):
    mb = r.j("resources", "microbench.json")
    if not mb:
        return "no microbench"
    rows = []
    dense = {(x["matrix"], x["M"]): x["median_ms"] for x in mb["results"] if x.get("kind") == "dense_fp16" and "median_ms" in x}
    for x in mb["results"]:
        base = dense.get((x["matrix"], x["M"]))
        val = x.get("median_ms")
        rows.append([x["matrix"], x["M"], x["kind"], x.get("sparsity", "-"),
                     f3(val, 4) if val is not None else "-", f"{base / val:.2f}x" if val and base else "-",
                     x.get("error", "")[:90]])
    write_table(r, "T5_kernel_microbench", ["Matrix", "M (tokens)", "Kernel", "Sparsity", "Median ms",
                                            "Speed-up vs dense", "Error / unsupported"], rows,
                f"Block {mb['block']} real FP16 weights, {mb['reps']} reps after {mb['warmup']} warm-up, CUDA events. "
                "Storage of the CSR conversions is in analysis/microbench_storage.json.")
    save_json(r.ana / "microbench_storage.json", mb.get("storage", []))
    sb = r.j("resources", "sdpa_backends.json")
    if sb:
        write_table(r, "T5b_sdpa_backends", ["Backend", "Status"], [[k, str(v)] for k, v in sb.items()])


def t_p3(r: R):
    cand = r.j("sensitivity", "p3_candidates.json")
    sel = r.j("sensitivity", "p3_selection.json")
    if not cand or not sel:
        return "no p3 selection"
    tr = r.j("transfer", "p3_transfer.json") or {}
    rows = []
    for cid, c in sorted(cand.items(), key=lambda kv: (kv[1]["S"], kv[1]["measure"], str(kv[1]["M"]), kv[1]["lam"])):
        chosen = sel["selected"].get(str(c["S"]), {})
        is_sel = c["measure"] == chosen.get("measure") and c["M"] == chosen.get("M") and c["lam"] == chosen.get("lam")
        t = tr.get(cid, {})
        rows.append([c["S"], c["measure"], c["M"] or "-", c["lam"], f"{min(c['allocation']):.3f}-{max(c['allocation']):.3f}",
                     f3(c["dev_ppl"]), "SELECTED" if is_sel else "", f3(t.get("wikitext2_ppl")), f3(t.get("c4val_ppl")),
                     f"{c['seconds']:.0f}"])
    write_table(r, "T6_p3_candidates", ["S", "Measure", "M", "lambda", "Block sparsity range", "Dev ppl", "",
                                        "Wiki ppl (post-hoc)", "C4 ppl (post-hoc)", "Build+dev s"], rows,
                f"Selection on the 32-sequence development set only: {sel['rule']}. Measure scores (mean dev NLL): "
                + ", ".join(f"{k} {v:.4f}" for k, v in sel["measure_scores_mean_dev_nll"].items())
                + ". Held-out columns were computed after the selection was frozen and never used for it.")
    anal = {}
    for S_ in sorted({c["S"] for c in cand.values()}):
        ids = [k for k, c in cand.items() if c["S"] == S_ and k in tr]
        if len(ids) >= 3:
            anal[str(S_)] = {s: spearman([cand[k]["dev_nll"] for k in ids], [tr[k][f"{s}_nll"] for k in ids])
                             for s, _ in SUITES}
    save_json(r.ana / "p3_dev_vs_heldout_transfer.json", anal)


def t_q3(r: R):
    cand = r.j("sensitivity", "q3_candidates.json")
    sel = r.j("sensitivity", "q3_selection.json")
    if not cand or not sel:
        return "no q3 selection"
    tr = r.j("transfer", "q3_transfer.json") or {}
    rows = []
    for cid, c in cand.items():
        hist = {}
        for v in c["allocation"].values():
            hist[v] = hist.get(v, 0) + 1
        is_sel = c["measure"] == sel["selected_measure"]
        t = tr.get(cid, {})
        rows.append([f"INT{c['budget_bits']}", c["measure"], " ".join(f"{b}b:{n}" for b, n in sorted(hist.items())),
                     f"{c['eligible_packed_bytes']:,}", f3(c["dev_ppl"]), "SELECTED" if is_sel and c["measure"] != "uniform" else "",
                     f3(t.get("wikitext2_ppl")), f3(t.get("c4val_ppl"))])
    write_table(r, "T7_q3_candidates", ["Budget", "Measure", "Units per bit width (of 112)", "Eligible packed bytes",
                                        "Dev ppl", "", "Wiki ppl (post-hoc)", "C4 ppl (post-hoc)"], rows,
                f"Allocation unit: one AWQ scaling group per block (q/k/v, o, gate/up, down; 28 x 4 = 112 units), the "
                f"finest unit whose quantizer is exactly Q2's at its bit width (q/k/v and gate/up share one AWQ scale). "
                f"Allocation: exact knapsack under the Q2 uniform budget. {sel['rule']}. Scores: "
                + ", ".join(f"{k} {v:.4f}" for k, v in sel["measure_scores_mean_dev_nll"].items()))


def t_calib_vs_heldout(r: R):
    if "fp16" not in r.S:
        return
    names = [c for c in COND if c in r.S and c != "fp16"]
    base = r.S["fp16"]
    x = {"calib_kl": [r.S[c]["calibration"]["kl"] for c in names],
         "calib_dnll": [r.S[c]["calibration"]["nll_cmp"] - r.S[c]["calibration"]["nll_ref"] for c in names],
         "dev_dnll": [r.S[c]["dev"]["mean_nll"] - base["dev"]["mean_nll"] for c in names],
         "drift_last_block": [r.S[c]["calibration"]["drift_last_block"] for c in names]}
    y = {"wiki_dnll": [r.S[c]["wikitext2"]["mean_nll"] - base["wikitext2"]["mean_nll"] for c in names],
         "c4_dnll": [r.S[c]["c4val"]["mean_nll"] - base["c4val"]["mean_nll"] for c in names],
         "hs512_d_acc_norm": [100 * (r.S[c]["hellaswag"]["acc_norm_first512"] - base["hellaswag"]["acc_norm_first512"]) for c in names]}
    rows, anal = [], {}
    for xk, xv in x.items():
        for yk, yv in y.items():
            sp = spearman(xv, yv)
            anal[f"{xk}~{yk}"] = sp
            rows.append([xk, yk, sp["n"], f3(sp["rho"]), f"{sp['p']:.2g}" if sp["p"] is not None else "-", f3(sp.get("pearson_r"))])
    for fam, members in [("pruning", [c for c in names if c.startswith(("P", "Q")) is False or c.startswith("P")]),
                         ("quant", [c for c in names if c.startswith("Q")])]:
        idx = [names.index(c) for c in members]
        if len(idx) >= 4:
            sp = spearman([x["calib_kl"][i] for i in idx], [y["wiki_dnll"][i] for i in idx])
            anal[f"{fam}:calib_kl~wiki_dnll"] = sp
            rows.append([f"calib_kl ({fam} only)", "wiki_dnll", sp["n"], f3(sp["rho"]), f"{sp['p']:.2g}", f3(sp.get("pearson_r"))])
    write_table(r, "T8_calibration_vs_heldout", ["Calibration / dev criterion", "Held-out outcome", "n conditions",
                                                 "Spearman rho", "p", "Pearson r"], rows,
                "Across all compressed conditions (n small: descriptive). dNLL = condition minus FP16 mean NLL.")
    save_json(r.ana / "calibration_vs_heldout.json", {"conditions": names, "x": x, "y": y, "correlations": anal})


def t_local_errors(r: R):
    rows = []
    agg = {}
    for c in COND:
        cal = r.j("conditions", c, "calibration.json") if c in r.S else None
        if not cal:
            continue
        for k in PROJ:
            vals = [v for key, v in cal["per_proj"].items() if key.split(".")[1] == k]
            rel = np.mean([v["rel_err"] for v in vals])
            ratio = np.median([v["rel_err"] / v["diag_pred_rel_err"] for v in vals if v["diag_pred_rel_err"] > 0] or [np.nan])
            agg[(c, k)] = (rel, ratio)
    for c in COND:
        if (c, "q") in agg and c != "fp16":
            rows.append([LABEL[c]] + [f"{agg[(c, k)][0]:.2e}" for k in PROJ] + [f3(np.nanmedian([agg[(c, k)][1] for k in PROJ]), 2)])
    if rows:
        write_table(r, "T9_layer_output_error", ["Condition"] + [f"{k} rel. err" for k in PROJ] + ["actual / diagonal pred."],
                    rows, "Mean over blocks of ||W_hat x - W x||^2 / ||W x||^2 on calibration activations entering each "
                    "projection in the compressed model (original coordinates for AWQ). The last column: median of measured "
                    "error / diagonal prediction sum_ij dW_ij^2 ||x_j||^2 (>1: correlated input channels add error, <1: cancel).")


def t_rq1(r: R):
    rows = []
    out = {}
    for s in r.cfg["pruning"]["sparsities"]:
        c = f"P2_wanda_s{int(round(s * 100))}"
        info = r.j("conditions", c, "compress_info.json") if c in r.S else None
        if not info:
            continue
        cm = f"P1_mag_s{int(round(s * 100))}"
        calm = r.j("conditions", cm, "calibration.json") if cm in r.S else None
        calw = r.j("conditions", c, "calibration.json")
        for k in PROJ:
            pp = [v for key, v in info["per_proj"].items() if key.split(".")[1] == k]
            ov = np.mean([v["overlap_with_magnitude"] for v in pp])
            mw = np.mean([v["score_mass_removed"] for v in pp])
            mm = np.mean([v["magnitude_score_mass_removed"] for v in pp])
            dr = np.median([v["magnitude_diag_pred_err"] / max(v["diag_pred_err"], 1e-30) for v in pp])
            rho = np.mean([v["summary"].get("spearman_absW_vs_score", np.nan) for v in pp])
            em = np.mean([v["rel_err"] for key, v in calm["per_proj"].items() if key.split(".")[1] == k]) if calm else np.nan
            ew = np.mean([v["rel_err"] for key, v in calw["per_proj"].items() if key.split(".")[1] == k])
            an = [v for key, v in info.get("act_norms", {}).items() if key.split(".")[1] == k]
            amax = np.mean([v["max_over_median"] for v in an]) if an else np.nan
            rows.append([f"{int(s * 100)}%", k, f3(rho), f3(ov), f3(mw, 4), f3(mm, 4), f3(dr, 2), f"{em:.2e}", f"{ew:.2e}",
                         f3(em / ew, 2) if ew > 0 else "-", f3(amax, 1)])
            out[f"{s}.{k}"] = {"rho": rho, "overlap": ov, "mass_w": mw, "mass_m": mm, "diag_ratio": dr, "err_m": em, "err_w": ew}
    if rows:
        write_table(r, "T10_magnitude_vs_wanda", ["Sparsity", "Proj", "rho(|W|, Wanda)", "Mask overlap", "Wanda-score mass removed (Wanda)",
                                                  "... removed (magnitude)", "Diag. pred. err mag/Wanda", "Layer err P1", "Layer err P2",
                                                  "P1/P2 err", "max/median ||X_j||"], rows,
                    "Statistics with the P2 run's own (sequential) calibration inputs; overlap = |M_mag & M_wanda| / |M_wanda| "
                    "at identical per-row counts. Diagonal prediction = removal error if input channels were uncorrelated.")
        save_json(r.ana / "rq1_magnitude_vs_wanda.json", out)


def t_awq(r: R):
    a = r.j("sensitivity", "awq_search.json")
    if not a:
        return
    rows = []
    for b in a["bits"]:
        for g in GROUPS:
            al = [a["table"][l][g][str(b)]["alpha"] for l in a["table"]]
            ratio = [a["table"][l][g][str(b)]["loss"] / max(a["table"][l][g][str(b)]["loss_rtn"], 1e-30) for l in a["table"]]
            rows.append([b, g, f3(np.mean(al), 2), f3(np.median(al), 2), f"{min(al):.2f}-{max(al):.2f}", f3(np.mean(ratio), 3),
                         f3(min(ratio), 3)])
    fc = a.get("fold_checks", {})
    write_table(r, "T11_awq_search", ["Bits", "Group", "Mean alpha", "Median alpha", "Range", "Mean loss AWQ/RTN", "Min ratio"], rows,
                f"Grid alpha in {{0, 0.05, ..., 0.95}} ({len(a['grid'])} points), loss = MSE of the inspected module output on the "
                f"128 calibration sequences. Search time {a['timings_s']['awq_search_s'] / 60:.1f} min. Function-preservation check "
                "(scales folded, no rounding, 4 dev sequences): " + "; ".join(
                    f"b{b}: max|d logit| {v['max_abs_logit_diff']:.3g}, top-1 agree {v['top1_agreement']:.4f}" for b, v in fc.items()))


def t_selection_effort(r: R):
    rows = []
    for fam in ("p3", "q3"):
        s = r.j("sensitivity", f"{fam}_sensitivity.json")
        sel = r.j("sensitivity", f"{fam}_selection.json")
        if s:
            for k, v in s["timings_s"].items():
                rows.append([fam.upper(), k, f"{v / 60:.2f}"])
            rows.append([fam.upper(), f"isolated probes (count {s['n_probes']})", f"{s['probe_seconds'] / 60:.2f}"])
        if sel:
            rows.append([fam.upper(), "dev candidate builds + dev ppl (all)", f"{sel['total_candidate_seconds'] / 60:.2f}"])
    a = r.j("sensitivity", "awq_search.json")
    if a:
        rows.append(["Q2/Q3", "AWQ scale search (all bit widths)", f"{a['timings_s']['awq_search_s'] / 60:.2f}"])
    if rows:
        write_table(r, "T12_selection_effort", ["Family", "Component", "Minutes (GPU)"], rows)


def t_tokens(r: R):
    """Where does the held-out NLL increase come from? Bins of baseline token NLL and token frequency."""
    if "fp16" not in r.S:
        return
    sel = r.res / "data_selection"
    win = {}
    for s, fn in [("wikitext2", "wikitext2_test_windows_locked.jsonl.gz"), ("c4val", "c4_validation_windows_locked.jsonl.gz")]:
        p = sel / fn
        if p.exists():
            with gzip.open(p, "rt") as f:
                win[s] = np.concatenate([np.array(json.loads(x)["token_ids"][1:]) for x in f])
    cal = sel / "c4_calibration_locked.jsonl.gz"
    freq = None
    if cal.exists():
        with gzip.open(cal, "rt") as f:
            toks = np.concatenate([np.array(json.loads(x)["token_ids"]) for x in f])
        freq = np.bincount(toks, minlength=160000)
    base = r.ppl("fp16")
    rows, out = [], {}
    keys = [c for c in ["P1_mag_s50", "P2_wanda_s50", "P3_s50", "P2_wanda_s75", "P3_s75", "Q1_rtn_b3", "Q2_awq_b3",
                        "Q3_mix_b3budget", "Q2_awq_b4"] if c in r.S]
    for s, _ in SUITES:
        b = base[f"{s}_token_nll"]
        qs = np.quantile(b, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(b, qs)
        fb = None
        if freq is not None and s in win and len(win[s]) == len(b):
            fq = freq[win[s]]
            fb = np.digitize(fq, [0.5, 10, 100, 1000])        # unseen, rare, ..., frequent in calibration text
        for c in keys:
            d = r.ppl(c)[f"{s}_token_nll"] - b
            per_bin = [float(d[bins == i].mean()) for i in range(5)]
            share = [float(d[bins == i].sum() / d.sum()) if d.sum() != 0 else np.nan for i in range(5)]
            row = [s, LABEL[c]] + [f"{x:+.3f}" for x in per_bin] + [f"{share[-1] * 100:.0f}%"]
            if fb is not None:
                row += [f"{float(d[fb == i].mean()):+.3f}" if (fb == i).any() else "-" for i in range(5)]
            rows.append(row)
            out[f"{s}.{c}"] = {"dnll_by_baseline_quintile": per_bin, "share_by_quintile": share,
                               "dnll_by_freq_bin": [float(d[fb == i].mean()) if fb is not None and (fb == i).any() else None for i in range(5)],
                               "frac_tokens_improved": float((d < 0).mean())}
    if rows:
        write_table(r, "T13_token_level_nll", ["Suite", "Condition", "dNLL Q1 (easiest)", "Q2", "Q3", "Q4", "Q5 (hardest)",
                                               "Share of total from Q5", "freq 0", "1-9", "10-99", "100-999", ">=1000"], rows,
                    "Per-token NLL change vs FP16 by quintile of the FP16 token NLL, and by the target token's count in the "
                    "128 x 512 calibration tokens.")
        save_json(r.ana / "token_level_nll.json", out)


def t_hs_margin(r: R):
    if "fp16" not in r.S:
        return
    b = r.hs("fp16")
    rows, out = [], {}
    def margin(x):
        ll = np.array(x["ll"]) / np.array(x["char_len"])
        g = x["gold"]
        return ll[g] - np.max(np.delete(ll, g))
    mb = np.array([margin(x) for x in b])
    for c in COND:
        if c == "fp16" or c not in r.S:
            continue
        h = r.hs(c)
        n = len(h)
        m0, m1 = mb[:n], np.array([margin(x) for x in h])
        cb = np.array([x["correct_norm"] for x in b[:n]])
        cc = np.array([x["correct_norm"] for x in h])
        q = np.quantile(np.abs(m0), [1 / 3, 2 / 3])
        tier = np.digitize(np.abs(m0), q)
        flips = [int(((cb != cc) & (tier == i)).sum()) for i in range(3)]
        acc_vs_norm = float(np.mean([x["pred"] != x["pred_norm"] for x in h]))
        corr = spearman(m0, m1 - m0)
        rows.append([LABEL[c], n, f"{np.mean(m1 - m0):+.4f}", f"{np.mean(np.abs(m1 - m0)):.4f}", *flips,
                     f3(acc_vs_norm, 3), f3(corr["rho"])])
        out[c] = {"mean_margin_change": float(np.mean(m1 - m0)), "flips_by_confidence_tercile": flips,
                  "acc_vs_norm_disagreement": acc_vs_norm}
    write_table(r, "T14_hellaswag_margins", ["Condition", "n", "Mean d margin", "Mean |d margin|", "Flips low-conf.",
                                             "Flips mid", "Flips high-conf.", "pred != pred_norm", "rho(margin, d margin)"], rows,
                "Margin = per-character log-likelihood of the gold ending minus the best wrong ending (acc_norm scale). "
                "Flips = examples whose acc_norm correctness differs from FP16, by tercile of FP16 |margin|.")
    save_json(r.ana / "hellaswag_margins.json", out)


def t_qualitative(r: R):
    if "fp16" not in r.S:
        return
    keys = [c for c in ["fp16", "P1_mag_s50", "P2_wanda_s50", "P3_s50", "P2_wanda_s75", "Q1_rtn_b3", "Q2_awq_b3", "Q3_mix_b3budget"]
            if c in r.S]
    hs = {c: {x["order"]: x for x in r.hs(c)} for c in keys}
    n_diag = min(len(v) for v in hs.values())                # 512 (diagnostic subset shared by every condition)
    sel_orders = list(range(0, n_diag, 64))                  # fixed a-priori selection: every 64th diagnostic ID
    # rule-based failure cases: FP16 correct with the largest margin, wrong under the most compressed configurations
    b = hs["fp16"]
    cand = []
    for o in range(n_diag):
        x = b[o]
        if not x["correct_norm"]:
            continue
        wrong = sum(1 for c in keys[1:] if not hs[c][o]["correct_norm"])
        ll = np.array(x["ll"]) / np.array(x["char_len"])
        cand.append((wrong, float(ll[x["gold"]] - np.max(np.delete(ll, x["gold"]))), o))
    cand.sort(key=lambda t: (-t[0], -t[1]))
    fail = [o for w, _, o in cand[:4] if w > 0]
    rows = []
    for o in sel_orders + fail:
        row = [o, "fixed" if o in sel_orders else "failure rule", hs["fp16"][o]["gold"]]
        for c in keys:
            x = hs[c][o]
            row.append(f"{x['pred_norm']}{'*' if x['correct_norm'] else ''}")
        rows.append(row)
    write_table(r, "T15_qualitative_hellaswag", ["Order", "Selection", "Gold"] + [LABEL[c] for c in keys], rows,
                "Predicted ending (acc_norm), * = correct. Fixed cases: diagnostic orders 0, 64, ..., 448 (chosen before "
                "results). Failure rule: FP16-correct cases wrong under the most compressed configurations, ties broken by "
                "the largest FP16 margin. Full candidate scores in analysis/qualitative_cases.json.")
    save_json(r.ana / "qualitative_cases.json", {str(o): {c: hs[c][o] for c in keys} for o in sel_orders + fail})
    cont = {c: r.j("conditions", c, "continuations.json") for c in keys}
    save_json(r.ana / "continuations_aligned.json", cont)


# =========================================================================================== figures
def f_tradeoff(r: R, plt):
    if "fp16" not in r.S:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    b = r.S["fp16"]
    for fam, mk in [("P1", "o"), ("P2", "s"), ("P3", "^")]:
        cs = [c for c in COND if c in r.S and r.S[c].get("family") == fam]
        if not cs:
            continue
        x = [100 * r.S[c]["achieved_sparsity"] for c in cs]
        for k, (s, lab) in enumerate(SUITES):
            ax[0].plot(x, [r.S[c][s]["ppl"] for c in cs], mk + ("-" if k == 0 else "--"), color=COLORS[fam],
                       label=f"{fam} {lab}")
    if "P2_wanda24_s50" in r.S:
        for k, (s, lab) in enumerate(SUITES):
            ax[0].plot([50], [r.S["P2_wanda24_s50"][s]["ppl"]], "D", color=COLORS["NM"], label=f"2:4 {lab}" if k == 0 else None)
    for k, (s, lab) in enumerate(SUITES):
        ax[0].axhline(b[s]["ppl"], color="gray", ls=":" if k else "-", lw=1)
    ax[0].set(xlabel="Achieved eligible-weight sparsity (%)", ylabel="Perplexity (log)", yscale="log",
              title="Pruning: perplexity vs sparsity")
    ax[0].legend(ncol=2)
    for fam, mk in [("Q1", "o"), ("Q2", "s"), ("Q3", "^")]:
        cs = [c for c in COND if c in r.S and r.S[c].get("family") == fam]
        if not cs:
            continue
        x = [r.S[c]["storage"]["total_bytes"] / 2**20 for c in cs]
        for k, (s, lab) in enumerate(SUITES):
            ax[1].plot(x, [r.S[c][s]["ppl"] for c in cs], mk + ("-" if k == 0 else "--"), color=COLORS[fam], label=f"{fam} {lab}")
    for k, (s, lab) in enumerate(SUITES):
        ax[1].plot([b["storage"]["total_bytes"] / 2**20], [b[s]["ppl"]], "*", color="k", ms=10, label="FP16" if k == 0 else None)
    ax[1].set(xlabel="Total packed model size incl. metadata (MiB)", ylabel="Perplexity (log)", yscale="log",
              title="Quantization: perplexity vs packed bytes")
    ax[1].legend(ncol=2)
    for c in COND:
        if c not in r.S or c == "fp16":
            continue
        fam = r.S[c]["family"]
        x = r.S[c]["storage"]["total_bytes"] / 2**20
        y = 100 * r.S[c]["hellaswag"]["acc_norm_first512"]
        ax[2].plot([x], [y], "o" if fam.startswith("P") or fam == "NM" else "s", color=COLORS.get(fam, "gray"))
        ax[2].annotate(LABEL[c].replace("nonuni. ", ""), (x, y), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax[2].axhline(100 * b["hellaswag"]["acc_norm_first512"], color="k", lw=1, label="FP16")
    ax[2].set(xlabel="Logical size (MiB; sparse = bitmap + FP16 nonzeros)", ylabel="HellaSwag acc_norm (%, first 512)",
              title="Downstream (paired 512-ID subset)")
    ax[2].legend()
    save_fig(r, fig, "F1_quality_vs_budget")


def f_p3(r: R, plt):
    s = r.j("sensitivity", "p3_sensitivity.json")
    sel = r.j("sensitivity", "p3_selection.json")
    if not s:
        return
    L = len(s["block_influence"])
    xs = np.arange(L)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    def nz(v):
        v = np.asarray(v, float)
        return (v - v.min()) / (np.ptp(v) or 1)
    curves = {"OWL LOD (M=5)": s["owl_lod"]["5"], "OWL LOD (M=7)": s["owl_lod"]["7"],
              "Block influence": s["block_influence"],
              "log Fisher-Taylor @50%": np.log(np.maximum(s["fisher_taylor"]["0.5"]["block"], 1e-30)),
              "log probe KL @50%": np.log(np.maximum([x["kl"] for x in s["probe_kl"]["0.5"]], 1e-12)),
              "log probe KL @75%": np.log(np.maximum([x["kl"] for x in s["probe_kl"]["0.75"]], 1e-12))}
    for k, v in curves.items():
        ax[0].plot(xs, nz(v), marker=".", label=k)
    ax[0].set(xlabel="Transformer block", ylabel="Sensitivity (min-max scaled)", title="P3 candidate sensitivity measures")
    ax[0].legend(ncol=2)
    if sel:
        w = 0.4
        for i, S_ in enumerate(sorted(sel["selected"])):
            a = sel["selected"][S_]["allocation"]
            ax[1].bar(xs + (i - 0.5) * w, np.array(a) * 100, w, label=f"S={float(S_):.0%} ({sel['selected'][S_]['measure']}, "
                      f"lambda={sel['selected'][S_]['lam']})")
            ax[1].axhline(float(S_) * 100, color="gray", lw=0.8)
        ax[1].set(xlabel="Transformer block", ylabel="Assigned sparsity (%)", title="Final P3 sparsity maps", ylim=(0, 90))
        ax[1].legend()
    save_fig(r, fig, "F2_p3_sensitivity_and_allocation")
    # component probes
    comp = s.get("component_probe_kl", {})
    if comp:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for i, sp in enumerate(["0.5", "0.75"]):
            v = [comp.get(f"{k}@{sp}", {}).get("kl", np.nan) for k in PROJ]
            ax.bar(np.arange(7) + (i - 0.5) * 0.4, v, 0.4, label=f"{float(sp):.0%} in every block")
        ax.set(xticks=range(7), xticklabels=PROJ, yscale="log", ylabel="Calibration KL vs FP16",
               title="Pruning one projection type in all blocks (Wanda)")
        ax.legend()
        save_fig(r, fig, "F2b_component_probes_pruning")


def f_q3(r: R, plt):
    s = r.j("sensitivity", "q3_sensitivity.json")
    sel = r.j("sensitivity", "q3_selection.json")
    if not s:
        return
    meas = sel["selected_measure"] if sel else "probe_kl"
    L = max(int(k.split(".")[0]) for k in s["unit_bytes"]) + 1
    def D(m, b):
        out = np.zeros((4, L))
        for l in range(L):
            for gi, g in enumerate(GROUPS):
                v = s[m][f"{l}.{g}"][str(b)]
                out[gi, l] = v["kl"] if m == "probe_kl" else v
        return out
    bits = [2, 3, 4]
    fig, ax = plt.subplots(len(bits) + (2 if sel else 0), 1, figsize=(11, 1.6 * (len(bits) + 2)), sharex=True)
    for i, b in enumerate(bits):
        im = ax[i].imshow(np.log10(np.maximum(D(meas, b), 1e-12)), aspect="auto", cmap="viridis")
        ax[i].set(yticks=range(4), yticklabels=GROUPS, title=f"log10 {meas} damage at {b} bits")
        fig.colorbar(im, ax=ax[i], pad=0.01)
    if sel:
        for j, bb in enumerate(sorted(sel["selected"])):
            alloc = sel["selected"][bb]["allocation"]
            M = np.array([[alloc[f"{l}.{g}"] for l in range(L)] for g in GROUPS])
            k = len(bits) + j
            im = ax[k].imshow(M, aspect="auto", cmap="RdYlGn", vmin=2, vmax=8)
            for gi in range(4):
                for l in range(L):
                    ax[k].text(l, gi, M[gi, l], ha="center", va="center", fontsize=7)
            ax[k].set(yticks=range(4), yticklabels=GROUPS, title=f"Final bit map, INT{bb} budget")
            fig.colorbar(im, ax=ax[k], pad=0.01)
    ax[-1].set_xlabel("Transformer block")
    save_fig(r, fig, "F3_q3_sensitivity_and_bitmaps")


def f_drift(r: R, plt):
    keys = [c for c in ["P1_mag_s50", "P2_wanda_s50", "P3_s50", "P2_wanda_s75", "P3_s75", "P2_wanda24_s50",
                        "Q1_rtn_b3", "Q2_awq_b3", "Q3_mix_b3budget", "Q1_rtn_b4", "Q2_awq_b4"] if c in r.S]
    if not keys:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for c in keys:
        d = r.j("conditions", c, "calibration.json")["drift"]
        a = ax[0] if c.startswith("P") else ax[1]
        a.plot([x["block"] for x in d], [x["rel_mse"] for x in d], marker=".", label=LABEL[c])
    for a, t in zip(ax, ["Pruning", "Quantization"]):
        a.set(yscale="log", xlabel="Block (output)", title=f"{t}: hidden-state drift vs FP16 (calibration)")
        a.legend(ncol=2)
    ax[0].set_ylabel("||h_comp - h_FP16||^2 / ||h_FP16||^2")
    save_fig(r, fig, "F4_block_drift")


def f_local_err(r: R, plt):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    for a, fams in [(ax[0], [("P1_mag", "P2_wanda")]), (ax[1], [("Q1_rtn", "Q2_awq")])]:
        pass
    groups = [(ax[0], [f"P1_mag_s{x}" for x in (25, 50, 75)] + [f"P2_wanda_s{x}" for x in (25, 50, 75)]),
              (ax[1], [f"Q1_rtn_b{x}" for x in (8, 4, 3)] + [f"Q2_awq_b{x}" for x in (8, 4, 3)])]
    for a, cs in groups:
        cs = [c for c in cs if c in r.S]
        w = 0.8 / max(1, len(cs))
        for i, c in enumerate(cs):
            cal = r.j("conditions", c, "calibration.json")
            v = [np.mean([x["rel_err"] for k, x in cal["per_proj"].items() if k.split(".")[1] == p]) for p in PROJ]
            a.bar(np.arange(7) + i * w - 0.4, v, w, label=LABEL[c])
        a.set(xticks=range(7), xticklabels=PROJ, yscale="log", ylabel="Mean layer-output rel. error")
        a.legend(ncol=2, fontsize=7.5)
    ax[0].set_title("Pruning: layer-output error on calibration inputs")
    ax[1].set_title("Quantization: layer-output error on calibration inputs")
    save_fig(r, fig, "F5_layer_output_error")


def f_calib_vs_heldout(r: R, plt):
    a = r.j("analysis", "calibration_vs_heldout.json")
    if not a:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for c, xk, yk in zip(a["conditions"], a["x"]["calib_kl"], a["y"]["wiki_dnll"]):
        fam = r.S[c]["family"]
        ax[0].scatter(xk, yk, color=COLORS.get(fam, "gray"), marker="o" if fam.startswith("P") or fam == "NM" else "s")
        ax[0].annotate(LABEL[c].replace("nonuni. ", ""), (xk, yk), fontsize=7, xytext=(3, 2), textcoords="offset points")
    sp = a["correlations"]["calib_kl~wiki_dnll"]
    ax[0].set(xscale="log", yscale="symlog", xlabel="Calibration KL vs FP16 (nats/token)", ylabel="WikiText-2 dNLL vs FP16",
              title=f"Calibration criterion vs held-out (Spearman {sp['rho']:.2f}, n={sp['n']})")
    tr = r.j("transfer", "p3_transfer.json")
    cand = r.j("sensitivity", "p3_candidates.json")
    if tr and cand:
        for S_, mk in [(0.5, "o"), (0.75, "s")]:
            ids = [k for k in tr if cand[k]["S"] == S_]
            ax[1].scatter([cand[k]["dev_ppl"] for k in ids], [tr[k]["wikitext2_ppl"] for k in ids], marker=mk,
                          label=f"P3 candidates S={S_:.0%}")
        ax[1].set(xscale="log", yscale="log", xlabel="Development ppl (selection criterion)", ylabel="WikiText-2 ppl (post-hoc)",
                  title="Does the dev ranking transfer?")
        ax[1].legend()
    save_fig(r, fig, "F6_calibration_vs_heldout")


def f_rq1(r: R, plt):
    info = r.j("conditions", "P2_wanda_s50", "compress_info.json") if "P2_wanda_s50" in r.S else None
    if not info:
        return
    L = max(int(k.split(".")[0]) for k in info["per_proj"]) + 1
    ov = np.array([[info["per_proj"][f"{l}.{p}"]["overlap_with_magnitude"] for l in range(L)] for p in PROJ])
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    im = ax[0].imshow(ov, aspect="auto", cmap="magma")
    ax[0].set(yticks=range(7), yticklabels=PROJ, title="Share of the Wanda 50% mask also chosen by magnitude (same per-row counts)")
    fig.colorbar(im, ax=ax[0], pad=0.01)
    d = r.j("sensitivity", "p3_sensitivity.json")
    if d:
        an = d["dense_act_norm"]
        A = np.array([[an[f"{l}.{p}"]["max_over_median"] for l in range(L)] for p in PROJ])
        im = ax[1].imshow(np.log10(A), aspect="auto", cmap="viridis")
        ax[1].set(yticks=range(7), yticklabels=PROJ, title="log10 max/median input-channel norm ||X_j|| (dense calibration)")
        fig.colorbar(im, ax=ax[1], pad=0.01)
    ax[1].set_xlabel("Transformer block")
    save_fig(r, fig, "F7_rq1_mask_overlap_activation_outliers")
    # sparsity dependence of overlap / removed score mass
    rq = r.j("analysis", "rq1_magnitude_vs_wanda.json")
    if rq:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
        for p in PROJ:
            sps = sorted({float(k.split(".")[0] + "." + k.split(".")[1]) for k in rq if k.endswith("." + p)})
            ax[0].plot([100 * s for s in sps], [rq[f"{s}.{p}"]["overlap"] for s in sps], marker="o", label=p)
            ax[1].plot([100 * s for s in sps], [rq[f"{s}.{p}"]["err_m"] / rq[f"{s}.{p}"]["err_w"] for s in sps], marker="o", label=p)
        ax[0].set(xlabel="Sparsity (%)", ylabel="Mask overlap", title="Magnitude and Wanda agree more as sparsity grows?")
        ax[1].set(xlabel="Sparsity (%)", ylabel="Layer error P1 / P2", yscale="log", title="Error cost of ignoring activations")
        ax[0].legend(ncol=4)
        save_fig(r, fig, "F7b_rq1_vs_sparsity")


def f_awq(r: R, plt):
    a = r.j("sensitivity", "awq_search.json")
    if not a:
        return
    L = len(a["table"])
    bits = [b for b in (4, 3, 2) if b in a["bits"]]
    fig, ax = plt.subplots(1, len(bits) + 1, figsize=(4 * (len(bits) + 1), 3.2))
    for i, b in enumerate(bits):
        A = np.array([[a["table"][str(l)][g][str(b)]["alpha"] for l in range(L)] for g in GROUPS])
        im = ax[i].imshow(A, aspect="auto", cmap="cividis", vmin=0, vmax=0.95)
        ax[i].set(yticks=range(4), yticklabels=GROUPS, xlabel="Block", title=f"Selected alpha, INT{b}")
    fig.colorbar(im, ax=ax[len(bits) - 1], pad=0.01)
    grid = a["grid"]
    for g in GROUPS:
        cur = np.mean([np.array(a["table"][str(l)][g]["3"]["losses"]) / a["table"][str(l)][g]["3"]["loss_rtn"] for l in range(L)], 0)
        ax[-1].plot(grid, cur, marker=".", label=g)
    ax[-1].set(xlabel="alpha", ylabel="Search loss / RTN loss (mean over blocks)", title="INT3 search curves")
    ax[-1].legend()
    save_fig(r, fig, "F8_awq_alpha")


def f_tokens(r: R, plt):
    t = r.j("analysis", "token_level_nll.json")
    if not t:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3), sharey=False)
    for i, (s, lab) in enumerate(SUITES):
        for k, v in t.items():
            if k.startswith(s + "."):
                ax[i].plot(range(1, 6), v["dnll_by_baseline_quintile"], marker="o", label=LABEL[k.split(".", 1)[1]])
        ax[i].set(xlabel="Quintile of FP16 token NLL (1 = easiest)", ylabel="Mean dNLL (nats)", title=lab, xticks=range(1, 6))
    ax[0].legend(ncol=2, fontsize=7.5)
    save_fig(r, fig, "F9_token_level_nll")


def f_latency(r: R, plt):
    cs = [c for c in COND if c in r.S]
    if not cs:
        return
    res = {c: r.j("conditions", c, "resources.json") for c in cs}
    cs = [c for c in cs if res[c] and "prefill_512" in res[c]]
    if not cs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, k in zip(ax, ["prefill_512", "prefill_2048", "decode_64"]):
        a.bar(range(len(cs)), [res[c][k]["median_ms"] for c in cs], yerr=[res[c][k]["std_ms"] for c in cs], capsize=2)
        a.set(xticks=range(len(cs)), ylabel="ms (median, +/- 1 sd)", title=k.replace("_", " "))
        a.set_xticklabels([LABEL[c] for c in cs], rotation=75, fontsize=7)
    save_fig(r, fig, "F10_latency")


def f_additivity(r: R, plt):
    s3 = r.j("sensitivity", "q3_sensitivity.json")
    sel3 = r.j("sensitivity", "q3_selection.json")
    s1 = r.j("sensitivity", "p3_sensitivity.json")
    sel1 = r.j("sensitivity", "p3_selection.json")
    pts = []
    if s3 and sel3:
        for bb, c in [("4", "Q3_mix_b4budget"), ("3", "Q3_mix_b3budget")]:
            if c in r.S:
                alloc = sel3["selected"][bb]["allocation"]
                pred = sum(s3["probe_kl"][u][str(b)]["kl"] for u, b in alloc.items())
                pts.append((LABEL[c], pred, r.S[c]["calibration"]["kl"]))
        for b, c in [(4, "Q2_awq_b4"), (3, "Q2_awq_b3"), (8, "Q2_awq_b8")]:
            if c in r.S:
                pred = sum(v[str(b)]["kl"] for v in s3["probe_kl"].values())
                pts.append((LABEL[c], pred, r.S[c]["calibration"]["kl"]))
    if s1:
        for sp, c in [("0.5", "P2_wanda_s50"), ("0.75", "P2_wanda_s75")]:
            if c in r.S:
                pts.append((LABEL[c], sum(x["kl"] for x in s1["probe_kl"][sp]), r.S[c]["calibration"]["kl"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lab, x, y in pts:
        ax.scatter(x, y)
        ax.annotate(lab, (x, y), fontsize=7.5, xytext=(3, 2), textcoords="offset points")
    lim = [min(min(p[1], p[2]) for p in pts) * 0.7, max(max(p[1], p[2]) for p in pts) * 1.4]
    ax.plot(lim, lim, "k--", lw=1, label="additive (y = x)")
    ax.set(xscale="log", yscale="log", xlabel="Sum of isolated probe KLs", ylabel="Measured calibration KL (all units compressed)",
           title="Do isolated sensitivities add up?")
    ax.legend()
    save_fig(r, fig, "F11_additivity")
    save_json(r.ana / "additivity.json", [{"condition": a, "sum_isolated_kl": b, "measured_kl": c} for a, b, c in pts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    r = R(Path(a.results), load_config(a.config))
    plt = setup_mpl()
    jobs = [t_quality, t_nll, t_pairs, t_budget, t_resources, t_microbench, t_p3, t_q3, t_calib_vs_heldout, t_local_errors,
            t_rq1, t_awq, t_selection_effort, t_tokens, t_hs_margin, t_qualitative]
    figs = [f_tradeoff, f_p3, f_q3, f_drift, f_local_err, f_calib_vs_heldout, f_rq1, f_awq, f_tokens, f_latency, f_additivity]
    failed = []
    for fn in jobs + figs:
        try:
            msg = fn(r, plt) if fn in figs else fn(r)
            print(f"{fn.__name__}: {'ok' if not msg else msg}")
        except Exception:
            failed.append(fn.__name__)
            print(f"{fn.__name__}: FAILED\n{traceback.format_exc()}")
    print(f"conditions found: {len(r.S)}/{len(COND)}; failed: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
