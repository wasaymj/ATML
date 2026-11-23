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
T8LAB = {"calib_kl": "calibration KL vs FP16", "calib_dnll": "calibration dNLL", "dev_dnll": "dev-set dNLL",
         "drift_last_block": "hidden-state drift, last block", "wiki_dnll": "WikiText-2 dNLL", "c4_dnll": "C4 dNLL",
         "hs512_d_acc_norm": "HellaSwag d acc_norm (first 512 IDs)"}
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
COLORS = {"P1": "#1f77b4", "P2": "#d62728", "P3": "#2ca02c", "NM": "#e377c2", "Q1": "#8c564b", "Q2": "#ff7f0e",
          "Q3": "#9467bd", "baseline": "black"}            # one colour per method across all figures


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
    esc = lambda x: str(x).replace("|", "\\|")              # a literal '|' inside a cell would split the Markdown column
    lines = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(x) for x in row) + " |" for row in rows]
    if note:
        lines += ["", note]
    (r.tab / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def f3(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(c, d=2):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def setup_mpl():
    """PA3 house style: sized for the 5.5-inch NeurIPS text width, 8-9 pt text at print size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "lines.linewidth": 1.2, "lines.markersize": 3.5, "savefig.bbox": "tight", "savefig.dpi": 300,
                         "pdf.fonttype": 42, "figure.constrained_layout.use": True})
    return plt


def save_fig(r: R, fig, name):
    fig.savefig(r.fig / f"{name}.pdf")
    fig.savefig(r.fig / f"{name}.png", dpi=300)
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
        setting = f"{s['achieved_sparsity'] * 100:.2f}% zeros" if "achieved_sparsity" in s else "-"
        row = [LABEL[c], setting, f"{st['effective_bits_per_eligible_weight']:.3f}", f"{st['total_bytes'] / 2**20:,.1f}"]
        if c == "fp16":
            for k, _ in SUITES:
                row += [f3(s[k]["ppl"]), "-"]
            row += [s["hellaswag"]["n"], f3(100 * s["hellaswag"]["acc"], 2), "-", f3(100 * s["hellaswag"]["acc_norm"], 2), "-"]
            row += [f3(s["dev"]["ppl"]), f"{s['calibration']['kl']:.4f}"]
            rows.append(row)
            nd = r.cfg["data"]["hellaswag_diag"]
            rows.append([f"FP16 (first {nd} IDs)", "-", "-", "-", "-", "-", "-", "-", nd,
                         f3(100 * s["hellaswag"]["acc_first512"], 2), "-", f3(100 * s["hellaswag"]["acc_norm_first512"], 2),
                         "-", "-", "-"])
            continue
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
    write_table(r, "T1_quality", ["Condition", "Sparsity", "Bits / eligible weight", "Total MiB", "Wiki ppl",
                                  "Wiki rel. change [95% CI]",
                                  "C4 ppl", "C4 rel. change [95% CI]", "HS n", "HS acc %", "d acc pp [CI]",
                                  "HS acc_norm %", "d acc_norm pp [CI]", "Dev ppl", "Calib KL"], rows,
                f"Paired against FP16 on identical windows/examples; perplexity CIs resample whole windows "
                f"(B={r.cfg['bootstrap']['resamples']}). N scored tokens: WikiText-2 {s0['wikitext2']['N']}, "
                f"C4 {s0['c4val']['N']}; mean NLL FP16: {s0['wikitext2']['mean_nll']:.4f} / {s0['c4val']['mean_nll']:.4f}. "
                f"HellaSwag n=2000 for the manual's full set, otherwise the first 512 IDs; the d columns are always "
                f"paired with FP16 on the same IDs, so a 512-ID row compares with the 'FP16 (first 512 IDs)' row, not "
                f"with the 2,000-ID FP16 row (the first 512 IDs are harder). Bits / eligible weight = stored bits per "
                f"eligible weight incl. all metadata (pruned: 1 mask bit + 16 bits per nonzero; quantized: codes + "
                f"zero-points + FP16 group scales). Total MiB (2^20 bytes) includes all metadata and the uncompressed "
                f"embedding/norms/biases (tied LM head counted once).")
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
                    "conditions have it, else the first 512). Each row is B - A, with A = the condition after the minus sign; "
                    "'A-only/B-only' = HellaSwag examples that only A / only B gets right under acc_norm (their difference "
                    "/ n gives the acc_norm change).")
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
                                                    "Check"], rows,
                    "Budget matching of every compared pair. Pruning: identical counts of zeroed eligible weights (exact "
                    "equality required). Quantization: total serialized bytes incl. codes, zero-points, FP16 group scales "
                    "and the uncompressed FP16 rest; the method must not exceed the reference (negative difference = the "
                    "mixed-precision knapsack leaves part of the budget unused).")


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
        write_table(r, "T4_resources", ["Condition", "Logical MiB (sparse bitmap / packed)", "As executed MiB (dense FP16)",
                                        "Serialized file MiB", "CSR int64 MiB", "Compression s", "Compression peak GiB",
                                        "Prefill 512 ms", "Prefill 2048 ms", "Decode 64 tok ms",
                                        "Peak alloc prefill-2048 GiB", "Weights alloc GiB"], rows,
                    f"Latency: batch 1, median of {reps} CUDA-synchronised repetitions after warm-up; decode = 64 greedy "
                    "steps after a 512-token prefill. Every compressed model executes as dense FP16 (zeros / fake-quantized "
                    "weights), so its GPU allocation and latency are those of FP16; logical/packed bytes are storage only. "
                    "MiB = 2^20 bytes.")
        t_theory_vs_measured(r)


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
        write_table(r, "T5b_sdpa_backends", ["Backend", "Status"], [[k, str(v)] for k, v in sb.items()],
                    "Attention backend of the Qwen model on the T4: availability of each SDPA backend when forced. "
                    "FlashAttention and cuDNN attention need a newer GPU than the T4 (compute capability 7.5), so "
                    "unforced calls use the memory-efficient kernel.")


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
                f"Selection rule: {sel['rule']} under the Q2 uniform packed-byte budget. Mean dev NLL per measure: "
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
            rows.append([T8LAB[xk], T8LAB[yk], sp["n"], f3(sp["rho"]), f"{sp['p']:.2g}" if sp["p"] is not None else "-", f3(sp.get("pearson_r"))])
    for fam, members in [("pruning", [c for c in names if c.startswith(("P", "Q")) is False or c.startswith("P")]),
                         ("quant", [c for c in names if c.startswith("Q")])]:
        idx = [names.index(c) for c in members]
        if len(idx) >= 4:
            sp = spearman([x["calib_kl"][i] for i in idx], [y["wiki_dnll"][i] for i in idx])
            anal[f"{fam}:calib_kl~wiki_dnll"] = sp
            rows.append([f"{T8LAB['calib_kl']} ({'pruning' if fam == 'pruning' else 'quantization'} only)", T8LAB["wiki_dnll"],
                         sp["n"], f3(sp["rho"]), f"{sp['p']:.2g}", f3(sp.get("pearson_r"))])
    write_table(r, "T8_calibration_vs_heldout", ["Calibration / dev criterion", "Held-out outcome", "n conditions",
                                                 "Spearman rho", "p", "Pearson r"], rows,
                "Across all compressed conditions (n small: descriptive). dNLL = condition minus FP16 mean NLL. The "
                "conditions span orders of magnitude of damage, so the rank correlations mainly show that calibration "
                "measures order the conditions correctly; p = 0 means below floating-point resolution (perfect ranking).")
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
            src_k = {"k": "q", "v": "q", "up": "gate"}.get(k, k)          # k/v share q's input, up shares gate's
            an = [v for key, v in info.get("act_norms", {}).items() if key.split(".")[1] == src_k]
            amax = np.mean([v["max_over_median"] for v in an]) if an else np.nan
            rows.append([f"{int(s * 100)}%", k, f3(rho), f3(ov), f3(mw, 4), f3(mm, 4), f3(dr, 2), f"{em:.2e}", f"{ew:.2e}",
                         f3(em / ew, 2) if ew > 0 else "-", f3(amax, 1)])
            out[f"{s}.{k}"] = {"rho": rho, "overlap": ov, "mass_w": mw, "mass_m": mm, "diag_ratio": dr, "err_m": em, "err_w": ew}
    if rows:
        write_table(r, "T10_magnitude_vs_wanda", ["Sparsity", "Proj", "rho(|W|, Wanda)", "Mask overlap", "Wanda-score mass removed (Wanda)",
                                                  "... removed (magnitude)", "Diag. pred. err mag/Wanda", "Layer err P1", "Layer err P2",
                                                  "P1/P2 err", "max/median ||X_j||"], rows,
                    "Statistics with the P2 run's own (sequential) calibration inputs; overlap = |M_mag & M_wanda| / |M_wanda| "
                    "at identical per-row counts. Diagonal prediction = removal error if input channels were uncorrelated. "
                    "max/median ||X_j||: k and v share q's input, up shares gate's input, so they repeat that value.")
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


# ============================================================================ mechanism analyses (saved data)
def _q3_unit_table(r: R):
    """Per Q3 unit (block, AWQ group): 2-bit probe KL next to every local quantity that could explain it."""
    q = r.j("sensitivity", "q3_sensitivity.json")
    a = r.j("sensitivity", "awq_search.json")
    p = r.j("sensitivity", "p3_sensitivity.json")
    if not (q and a):
        return None
    L = len(a["table"])
    act_key = {"qkv": "q", "o": "o", "gateup": "gate", "down": "down"}
    rows = []
    for l in range(L):
        t = a["table"][str(l)]
        for g in GROUPS:
            u = f"{l}.{g}"
            loss = t[g]["2"]["loss"]
            rows.append({"unit": u, "block": l, "group": g, "probe_kl2": q["probe_kl"][u]["2"]["kl"],
                         "probe_kl3": q["probe_kl"][u]["3"]["kl"], "loss2": loss, "ref_ms": t[g]["ref_ms"],
                         "rel_out_err2": loss / max(t[g]["ref_ms"], 1e-30), "resid_ms": t["resid_ms"],
                         "local_err2": q["local_err"][u]["2"], "fisher2": q["fisher_taylor"][u]["2"],
                         "act_max_over_median": p["dense_act_norm"][f"{l}.{act_key[g]}"]["max_over_median"] if p else np.nan,
                         "act_kurtosis": p["dense_act_norm"][f"{l}.{act_key[g]}"]["kurtosis"] if p else np.nan})
    for x in rows:
        x["kl_per_abs_err"] = x["probe_kl2"] / max(x["loss2"], 1e-30)       # amplification of absolute output error
        x["kl_per_rel_err"] = x["probe_kl2"] / max(x["rel_out_err2"], 1e-30)
    return rows


def t_mech_residual(r: R):
    """Where is the massive residual activation written and erased, and does it coincide with the fragile units?"""
    p = r.j("sensitivity", "p3_sensitivity.json")
    rows_u = _q3_unit_table(r)
    if not (p and rows_u):
        return "no sensitivity files"
    L = len(p["resid_rms"])
    U = {x["unit"]: x for x in rows_u}
    p3 = p["probe_kl"]
    rows, out = [], []
    for l in range(L):
        nxt_max = p["resid_max"][l + 1] if l + 1 < L else np.nan          # input of block l+1 = output of block l
        rec = {"block": l, "resid_rms_in": p["resid_rms"][l], "resid_max_in": p["resid_max"][l], "resid_max_out": nxt_max,
               "max_over_rms_in": p["resid_max"][l] / max(p["resid_rms"][l], 1e-30), "block_influence": p["block_influence"][l],
               **{f"ref_ms_{g}": U[f"{l}.{g}"]["ref_ms"] for g in GROUPS},    # qkv == o (attention out), gateup == down (MLP out)
               **{f"q3_kl2_{g}": U[f"{l}.{g}"]["probe_kl2"] for g in GROUPS},
               "p3_kl50": p3["0.5"][l]["kl"], "p3_kl75": p3["0.75"][l]["kl"]}
        out.append(rec)
        rows.append([l, f"{rec['resid_rms_in']:.2f}", f"{rec['resid_max_in']:.3g}", f"{nxt_max:.3g}" if np.isfinite(nxt_max) else "-",
                     f"{rec['max_over_rms_in']:.0f}", f3(rec["block_influence"]),
                     f"{rec['ref_ms_qkv']:.3g}", f"{rec['ref_ms_gateup']:.3g}", *[f"{rec[f'q3_kl2_{g}']:.3g}" for g in GROUPS],
                     f"{rec['p3_kl50']:.3g}", f"{rec['p3_kl75']:.3g}"])
    write_table(r, "T16_residual_stream_and_fragility",
                ["Block", "RMS(h_in)", "max|h_in|", "max|h_out|", "max/RMS in", "Block infl.",
                 "attention out MS", "MLP out MS", *[f"KL@2b {g}" for g in GROUPS], "P3 probe KL 50%", "P3 probe KL 75%"],
                rows, "Dense FP16 statistics on the 128 calibration sequences (saved by the main run). max|h| = largest "
                "absolute entry of the residual stream entering (in) / leaving (out) the block; out MS = mean square of the "
                "block's attention output (after o_proj) and MLP output, on dense calibration inputs. "
                "KL@2b = isolated 2-bit probe KL of that group; P3 probe KL = whole block pruned with Wanda at 50 / 75%.")
    # correlations of fragility with each candidate explanation (over all 112 units / 28 blocks)
    corr = {}
    y = [x["probe_kl2"] for x in rows_u]
    for k in ["loss2", "rel_out_err2", "local_err2", "fisher2", "ref_ms", "act_max_over_median", "act_kurtosis"]:
        corr[f"probe_kl2~{k}"] = spearman([x[k] for x in rows_u], y)
    corr["p3_kl50~block_influence"] = spearman([x["block_influence"] for x in out], [x["p3_kl50"] for x in out])
    corr["p3_kl50~max_over_rms_in"] = spearman([x["max_over_rms_in"] for x in out], [x["p3_kl50"] for x in out])
    save_json(r.ana / "mechanism_residual_stream.json", {"per_block": out, "per_unit": rows_u, "spearman": corr})
    T17LAB = {"loss2": "output MSE at 2 bits (absolute)", "rel_out_err2": "output MSE at 2 bits / output mean square",
              "local_err2": "output MSE at 2 bits / residual mean square (Q3 local_err)",
              "fisher2": "Fisher-Taylor damage at 2 bits", "ref_ms": "mean square of the group's dense output",
              "act_max_over_median": "input-channel norm, max / median", "act_kurtosis": "input-channel norm, kurtosis",
              "block_influence": "Block Influence (P3 measure)", "max_over_rms_in": "residual max / RMS entering the block"}
    crow = [[T17LAB.get(k.split("~")[1], k.split("~")[1]), v["n"], f3(v["rho"]), f"{v['p']:.2g}" if v["p"] is not None else "-"]
            for k, v in corr.items()]
    write_table(r, "T17_fragility_correlates", ["Candidate explanation of probe KL", "n", "Spearman rho", "p"], crow,
                "Rows 1-7: over the 112 Q3 units, against the isolated 2-bit probe KL. loss2 = absolute MSE of the "
                "group's output at 2 bits (AWQ search); rel_out_err2 = loss2 / output mean square; local_err2 = loss2 / "
                "residual mean square (the Q3 local_err measure); act_* = outlier statistics of the input-channel norms. "
                "Last two rows: over the 28 blocks, against the P3 50% block probe KL.")


def t_mech_local_alloc(r: R):
    """Why did the local-error allocation fail? Where it spent 2 bits, and what those units cost by probe KL."""
    cand = r.j("sensitivity", "q3_candidates.json")
    rows_u = _q3_unit_table(r)
    tr = r.j("transfer", "q3_transfer.json") or {}
    if not (cand and rows_u):
        return "no q3 candidates"
    U = {x["unit"]: x for x in rows_u}
    q = r.j("sensitivity", "q3_sensitivity.json")
    rows, out = [], {}
    for cid, c in cand.items():
        al = c["allocation"]
        two = [u for u, b in al.items() if b == 2]
        eight = [u for u, b in al.items() if b == 8]
        kl_sum = sum(q["probe_kl"][u][str(b)]["kl"] for u, b in al.items())
        kl_two = sum(q["probe_kl"][u]["2"]["kl"] for u in two)
        blocks2 = sorted({U[u]["block"] for u in two})
        mid = sum(1 for u in two if 3 <= U[u]["block"] <= 25)
        out[cid] = {"two_bit_units": two, "eight_bit_units": eight, "sum_probe_kl_at_alloc": kl_sum,
                    "sum_probe_kl_two_bit": kl_two, "dev_ppl": c["dev_ppl"]}
        rows.append([cid.replace("|budget=", " @ "), len(two), mid, len(eight), ",".join(map(str, blocks2[:12])) + ("..." if len(blocks2) > 12 else ""),
                     f"{kl_two:.3f}", f"{kl_sum:.3f}", f3(c["dev_ppl"]), f3(tr.get(cid, {}).get("wikitext2_ppl"))])
    write_table(r, "T18_q3_allocation_anatomy", ["Candidate", "# units at 2b", "... in blocks 3-25", "# units at 8b",
                                                 "Blocks holding 2-bit units", "Sum probe KL of 2-bit units",
                                                 "Sum probe KL, whole allocation", "Dev ppl", "Wiki ppl (post-hoc)"], rows,
                "Sum probe KL = sum of the isolated probe KLs of the units at their allocated widths (the additive "
                "prediction of the damage). local_err normalises the output MSE by the residual-stream mean square, "
                "which from block 3 on is dominated by the massive activation (T16), so mid-network units look cheap.")
    # a direct test of the normalisation explanation: rank of mid-block units under each normaliser
    save_json(r.ana / "q3_allocation_anatomy.json", out)


def t_mech_nm(r: R):
    """2:4 versus unstructured 50%: how much more Wanda-score mass the fixed pattern forces out."""
    a = r.j("conditions", "P2_wanda_s50", "compress_info.json") if "P2_wanda_s50" in r.S else None
    b = r.j("conditions", "P2_wanda24_s50", "compress_info.json") if "P2_wanda24_s50" in r.S else None
    if not (a and b):
        return "missing 2:4 or 50% Wanda"
    ca = r.j("conditions", "P2_wanda_s50", "calibration.json")
    cb = r.j("conditions", "P2_wanda24_s50", "calibration.json")
    rows, out = [], {}
    for k in PROJ:
        ma = np.mean([v["score_mass_removed"] for kk, v in a["per_proj"].items() if kk.split(".")[1] == k])
        mb = np.mean([v["score_mass_removed"] for kk, v in b["per_proj"].items() if kk.split(".")[1] == k])
        ea = np.mean([v["rel_err"] for kk, v in ca["per_proj"].items() if kk.split(".")[1] == k])
        eb = np.mean([v["rel_err"] for kk, v in cb["per_proj"].items() if kk.split(".")[1] == k])
        out[k] = {"mass_unstructured": ma, "mass_24": mb, "err_unstructured": ea, "err_24": eb}
        rows.append([k, f"{ma:.4f}", f"{mb:.4f}", f"{mb / ma:.2f}", f"{ea:.2e}", f"{eb:.2e}", f"{eb / ea:.2f}"])
    write_table(r, "T19_nm_vs_unstructured", ["Proj", "Score mass removed (50% unstr.)", "... (2:4)", "ratio",
                                              "Layer err (50% unstr.)", "Layer err (2:4)", "ratio"], rows,
                "Same sparsity and the same Wanda score; 2:4 must drop 2 of every 4 consecutive weights in a row, so it "
                "cannot keep clustered important weights. Score mass = removed share of sum |W_ij| ||X_j||.")
    save_json(r.ana / "nm_vs_unstructured.json", out)


def t_diagnostics(r: R):
    """Tables from scripts/diagnose_task1.py (supplementary mechanism diagnostic)."""
    d = r.j("diagnostics", "mechanism_diagnostic.json")
    if not d:
        return "no diagnostics/mechanism_diagnostic.json (run scripts/diagnose_task1.py)"
    n = d["meta"]["n_seq"]
    B = d["dense"]["boundaries"]
    a0 = d["dense"]["attn_mass_on_token0"]["per_layer_mean"]
    W = {w["block"]: w for w in d["dense"]["writers"]}
    rows = []
    for x in B:
        l = x["boundary"]
        w = W.get(l)
        rows.append([l, f"{x['max_over_rms']:.0f}", x["n_massive"], f3(x["massive_frac_at_pos0"], 2),
                     ",".join(map(str, list(x["massive_channels"])[:4])) or "-", f3(x["massive_share_of_sq_norm"], 3),
                     f"{x['tok0_norm_over_rest']:.1f}", f3(a0[l], 3),
                     f3(w["mlp_share_of_increment"], 2) if w else "-"])
    write_table(r, "T20_massive_activations", ["Boundary (input of block)", "max/RMS", "# massive entries",
                                               "share at token 0", "channels", "share of ||h||^2", "||h_0|| / mean ||h_t>0||",
                                               "attn mass on token 0 (block)", "MLP share of block's write (to next)"], rows,
                f"Dense FP16, first {n} calibration sequences, SDPA (token-0 attention mass in FP32). Massive = |h| > {d['meta']['massive_ratio']:.0f} x "
                "RMS of the boundary. Attention mass = mean over heads and queries t >= 1 of the weight on key 0 in that "
                "block. Last column: share of the increment h_(l+1) - h_l at the next boundary's massive entries written "
                f"by the MLP of block l (rest: attention); it is a signed share, so it leaves [0, 1] when the two "
                f"sublayers write in opposite directions (e.g. -0.30: attention adds more than the total, the MLP "
                f"removes part of it). Sanity: dense vs dense KL = {d['dense']['dense_vs_dense_kl']:.1e}.")
    wd = d["weight_difficulty"]
    tr, pt = d.get("trace", {}), d.get("patch", {})
    rows = []
    for u, t in tr.items():
        w = wd.get(u, {}).get("2")
        p = pt.get(u) or {}
        def g3(x, fmt):
            return "-" if x is None else format(x, fmt)

        def kl(k):
            return f"{p[k]['kl']:.3g}" if k in p else "-"
        L = len(t["drift"])
        last = L - 1
        rows.append([u, f"{t['kl']:.3g}", f3(t["saved_probe_kl_128seq_sdpa"], 4), g3(w and w["weight_rel_err"], ".3f"),
                     g3(w and w["act_weighted_rel_err"], ".3f"), g3(t["own_rel_err"], ".3g"), g3(t["own_err_share_at_pos0"], ".2f"),
                     f"{t['drift'][last]:.3g}", f"{t['drift_excl_massive'][last]:.3g}",
                     f"{max(abs(x) for x in t['attn0_change']):.3f}",
                     p.get("boundary", "-"), kl("token0"), kl("all_but_token0"), p.get("massive_boundary", "-"),
                     kl("massive_entries"), kl("random_entries_same_count")])
    if rows:
        write_table(r, "T21_propagation_and_patching",
                    ["Unit (2 bits; l.p75 = block pruned 75%)", "KL", "saved probe KL", "weight rel. err", "act-weighted rel. err", "own sublayer rel. err",
                     "own err share at token 0", "drift entering last block", "... excl. massive", "max |d attn to token 0|",
                     "patch boundary", "KL, token 0 restored", "KL, all but token 0 restored", "massive boundary",
                     "KL, massive entries restored", "KL, random entries restored"], rows,
                    "One unit at 2 bits (AWQ scales of the main run), or one whole block pruned with Wanda (l.p75 = "
                    "the P3 block probe), everything else FP16. Saved probe KL = the main run's value (all 128 "
                    "calibration sequences; this run: the first 32). Restored = activation patching "
                    "at the boundary right after the unit (massive / random: at the first boundary holding massive entries): "
                    "those residual entries are set back to their dense values and the rest of the network runs on. If "
                    "restoring token 0 removes most of the KL, the damage travels through the first token's state; if "
                    "restoring only the massive entries does, it travels through the massive activation specifically.")
    rows = []
    for u, p in d.get("protect", {}).items():
        rnd = [x["kl"] for x in p["random_input_channels"]]
        rn = [x["kl"] for x in p.get("random_neurons", [])]
        rows.append([u, f"{p['none']['kl']:.3g}", f"{p['top_input_channels']['kl']:.3g}",
                     f"{p['top_input_channels_requant']['kl']:.3g}" if "top_input_channels_requant" in p else "-",
                     f"{np.mean(rnd):.3g} +/- {np.std(rnd):.2g}",
                     f"{p['top_token0_neurons']['kl']:.3g}" if "top_token0_neurons" in p else "-",
                     f"{np.mean(rn):.3g}" if rn else "-"])
    if rows:
        write_table(r, "T22_weight_protection", ["Unit @2b", "KL, nothing kept", f"KL, top-{d['meta']['k']} input channels FP16",
                                                 "... also removed from group ranges",
                                                 "... random channels (mean +/- sd)", f"KL, top-{d['meta']['k']} token-0 neurons FP16",
                                                 "... random neurons"], rows,
                    "Causal test of where the damage enters: the same unit at 2 bits, with k input channels (largest dense "
                    "input norm) or k MLP neurons (largest |activation| at token 0; gate/up rows or down columns) kept in "
                    "FP16 (third column: the kept channels are also left out when the quantizer sets each group's range, "
                    "which tests whether those outliers coarsen the grid of the other weights), compared with "
                    f"{d['meta']['n_random']} random draws of the same size. Diagnostic only, not a method.")
    rows = []
    for u, w in wd.items():
        rows.append([u, *[f"{w[str(b)]['weight_rel_err']:.3f}" for b in (2, 3, 4)], f"{w['2']['act_weighted_rel_err']:.3f}",
                     f"{w['2']['share_of_act_err_on_top8_channels']:.2f}", f"{w['group_range_over_std_mean']:.2f}",
                     f"{w['input_norm_max_over_median']:.1f}"])
    write_table(r, "T23_weight_difficulty", ["Unit", "weight rel. err 2b", "3b", "4b", "act-weighted rel. err 2b",
                                             "share on top-8 channels", "group range / std", "input norm max/median"], rows,
                "Local difficulty of each Q3 unit with its AWQ scales: ||W_hat - W||^2 / ||W||^2, the same weighted by "
                "dense input-channel energy (diagonal prediction of relative output error), and the spread of the scaled "
                "weights inside 128-groups (larger = coarser grid for the bulk of weights).")


# ============================================================================================ figures
# PA3 house style: drawn at the 5.5-inch NeurIPS text width (3.3 in for single panels), 8-9 pt text at print size,
# panels titled "(a) ...", frameless legends, few series per panel. Main-text figures are F*, detailed versions
# kept for the appendix are FA*.
FAM_LABEL = {"P1": "magnitude", "P2": "Wanda", "P3": "nonuniform Wanda (P3)", "NM": "Wanda 2:4",
             "Q1": "RTN", "Q2": "AWQ", "Q3": "mixed precision (Q3)"}
SHORT = {"P1_mag_s25": "mag 25%", "P1_mag_s50": "mag 50%", "P1_mag_s75": "mag 75%", "P2_wanda_s25": "Wanda 25%",
         "P2_wanda_s50": "Wanda 50%", "P2_wanda_s75": "Wanda 75%", "P3_s50": "P3 50%", "P3_s75": "P3 75%",
         "P2_wanda24_s50": "2:4", "Q1_rtn_b8": "RTN 8b", "Q1_rtn_b4": "RTN 4b", "Q1_rtn_b3": "RTN 3b",
         "Q2_awq_b8": "AWQ 8b", "Q2_awq_b4": "AWQ 4b", "Q2_awq_b3": "AWQ 3b", "Q3_mix_b4budget": "Q3 4b",
         "Q3_mix_b3budget": "Q3 3b"}
GCOL = {"qkv": "#1f77b4", "o": "#ff7f0e", "gateup": "#2ca02c", "down": "#d62728"}
GLAB = {"qkv": "q/k/v", "o": "o", "gateup": "gate/up", "down": "down"}
PCOL = dict(zip(PROJ, ["#1f77b4", "#6baed6", "#9ecae1", "#ff7f0e", "#2ca02c", "#74c476", "#d62728"]))


def logy(ax):
    """Log y-axis (PA3 helper): major labels only, but label the 2x / 5x ticks when the range is under ~1 decade."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    if lo > 0 and np.log10(hi / lo) < 1.3:
        ax.yaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0)))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())


UNIT_TICK = {"0.qkv": "0\nq/k/v", "0.gateup": "0\ngate/up", "1.qkv": "1\nq/k/v", "1.gateup": "1\ngate/up",
             "1.down": "1\ndown", "2.gateup": "2\ngate/up", "2.down": "2\ndown", "4.gateup": "4\ngate/up",
             "13.qkv": "13\nq/k/v", "13.gateup": "13\ngate/up", "26.gateup": "26\ngate/up", "27.gateup": "27\ngate/up",
             "27.down": "27\ndown", "0.p75": "0\npruned", "13.p75": "13\npruned 75%", "27.p75": "27\npruned 75%"}
MEAS = {"fisher_taylor": "Fisher-Taylor", "probe_kl": "probe-KL", "local_err": "local-error",
        "block_influence": "Block Influence", "owl_lod": "OWL LOD"}


def title(ax, t):
    ax.set_title(t, loc="left")


def _fam(r, c):
    return r.S[c].get("family", c[:2])


def _quality_panels(r, ax_p, ax_q, suite, lab, letters):
    b = r.S["fp16"]
    for fam, mk in (("P1", "o"), ("P2", "s"), ("P3", "^")):
        cs = [c for c in COND if c in r.S and _fam(r, c) == fam]
        if cs:
            ax_p.plot([100 * r.S[c]["achieved_sparsity"] for c in cs], [r.S[c][suite]["ppl"] for c in cs], marker="o",
                      ms=3.5, color=COLORS[fam], label=FAM_LABEL[fam])
    if "P2_wanda24_s50" in r.S:
        ax_p.plot([50], [r.S["P2_wanda24_s50"][suite]["ppl"]], "D", ms=4, color=COLORS["NM"], label=FAM_LABEL["NM"])
    ax_p.axhline(b[suite]["ppl"], color="0.4", ls=":", lw=0.8, label="FP16")
    ax_p.set(xlabel="eligible-weight sparsity (%)", ylabel=f"{lab} perplexity")
    logy(ax_p)
    title(ax_p, f"({letters[0]}) pruning")
    for fam, mk in (("Q1", "o"), ("Q2", "s"), ("Q3", "^")):
        cs = [c for c in COND if c in r.S and _fam(r, c) == fam]
        if cs:
            ax_q.plot([r.S[c]["storage"]["total_bytes"] / 2**20 for c in cs], [r.S[c][suite]["ppl"] for c in cs],
                      marker="s", ms=3.5, color=COLORS[fam], label=FAM_LABEL[fam])
    ax_q.axhline(b[suite]["ppl"], color="0.4", ls=":", lw=0.8)
    ax_q.set(xlabel="packed model size (MiB)", ylabel=f"{lab} perplexity")
    logy(ax_q)
    title(ax_q, f"({letters[1]}) quantization")


def _quality_legend(fig, axes):
    h, l = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False, fontsize=6.5)


def f_tradeoff(r: R, plt):
    if "fp16" not in r.S:
        return
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.45))
    _quality_panels(r, ax[0], ax[1], "wikitext2", "WikiText-2", "ab")
    for c in COND:
        if c not in r.S or c == "fp16":
            continue
        fam = _fam(r, c)
        x, y = r.S[c]["storage"]["total_bytes"] / 2**20, 100 * r.S[c]["hellaswag"]["acc_norm_first512"]
        ax[2].plot([x], [y], "o" if fam in ("P1", "P2", "P3", "NM") else "s", ms=3.5, color=COLORS.get(fam, "gray"))
    ax[2].axhline(100 * r.S["fp16"]["hellaswag"]["acc_norm_first512"], color="0.4", ls=":", lw=0.8)
    ax[2].set(xlabel="stored size (MiB)", ylabel="HellaSwag acc_norm (%)")
    title(ax[2], "(c) HellaSwag, first 512 IDs")
    _quality_legend(fig, ax[:2])
    save_fig(r, fig, "F1_quality_vs_budget")
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3))
    _quality_panels(r, ax[0], ax[1], "c4val", "C4 validation", "ab")
    _quality_legend(fig, ax)
    save_fig(r, fig, "FA1_quality_vs_budget_c4")


def f_p3(r: R, plt):
    s = r.j("sensitivity", "p3_sensitivity.json")
    sel = r.j("sensitivity", "p3_selection.json")
    if not s:
        return
    L = len(s["block_influence"])
    xs = np.arange(L)

    def nz(v):
        v = np.asarray(v, float)
        return (v - v.min()) / (np.ptp(v) or 1)
    lk = {sp: np.log(np.maximum([x["kl"] for x in s["probe_kl"][sp]], 1e-12)) for sp in ("0.5", "0.75")}
    if sel:
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.1))
        meas = sorted({v["measure"] for v in sel["selected"].values()})
        ax[0].plot(xs, nz(s["block_influence"]), color="k", marker=".", ms=3, label="Block Influence (selected)")
        ax[0].plot(xs, nz(lk["0.5"]), color=COLORS["P2"], label="log probe KL, 50%")
        ax[0].plot(xs, nz(lk["0.75"]), color=COLORS["P2"], ls="--", label="log probe KL, 75%")
        ax[0].set(xlabel="block", ylabel="sensitivity (min-max scaled)")
        title(ax[0], "(a) block sensitivity")
        ax[0].legend(frameon=False, fontsize=6.5)
        w = 0.4
        for i, S_ in enumerate(sorted(sel["selected"])):
            a = sel["selected"][S_]["allocation"]
            ax[1].bar(xs + (i - 0.5) * w, np.array(a) * 100, w, color=["#74c476", COLORS["P3"]][i],
                      label=f"target {float(S_):.0%}")
            ax[1].axhline(float(S_) * 100, color="0.4", lw=0.7, ls=":")
        ax[1].set(xlabel="block", ylabel="assigned sparsity (%)", ylim=(0, 90))
        title(ax[1], "(b) P3 allocation (" + ", ".join(MEAS.get(m, m) for m in meas) + ")")
        ax[1].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax[1].grid(axis="x", visible=False)
        save_fig(r, fig, "F2_p3_sensitivity_and_allocation")
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    curves = [("OWL LOD (M=5)", s["owl_lod"]["5"], "-"), ("OWL LOD (M=7)", s["owl_lod"]["7"], "--"),
              ("Block Influence", s["block_influence"], "-"),
              ("log Fisher-Taylor, 50%", np.log(np.maximum(s["fisher_taylor"]["0.5"]["block"], 1e-30)), "-"),
              ("log probe KL, 50%", lk["0.5"], "-"), ("log probe KL, 75%", lk["0.75"], "--")]
    for (k, v, ls), col in zip(curves, ["#9467bd", "#9467bd", "k", "#ff7f0e", COLORS["P2"], COLORS["P2"]]):
        ax.plot(xs, nz(v), ls=ls, color=col, label=k)
    ax.set(xlabel="block", ylabel="sensitivity (min-max scaled)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save_fig(r, fig, "FA2_p3_all_measures")
    comp = s.get("component_probe_kl", {})
    if comp:
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        for i, sp in enumerate(["0.5", "0.75"]):
            v = [comp.get(f"{k}@{sp}", {}).get("kl", np.nan) for k in PROJ]
            ax.bar(np.arange(7) + (i - 0.5) * 0.4, v, 0.4, color=["#74c476", COLORS["P3"]][i],
                   label=f"{float(sp):.0%} in every block")
        ax.set(xticks=range(7), xticklabels=PROJ, ylabel="calibration KL vs FP16")
        logy(ax)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=6.5)
        save_fig(r, fig, "FA2b_component_probes_pruning")


def _bitmap(ax, M, fig, label=True):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["#d7301f", "#fc8d59", "#fdcc8a", "#1a9850"])
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=BoundaryNorm([1.5, 2.5, 3.5, 6, 9], 4))
    if label:
        for gi in range(M.shape[0]):
            for l in range(M.shape[1]):
                ax.text(l, gi, M[gi, l], ha="center", va="center", fontsize=5)
    cb = fig.colorbar(im, ax=ax, pad=0.01, ticks=[2, 3, 4.75, 7.5])
    cb.ax.set_yticklabels(["2", "3", "4", "8"])
    cb.set_label("bits")
    ax.grid(False)


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

    def heat(ax, b):
        im = ax.imshow(np.log10(np.maximum(D(meas, b), 1e-12)), aspect="auto", cmap="viridis")
        ax.set(yticks=range(4), yticklabels=[GLAB[g] for g in GROUPS])
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.01).set_label("log10 damage")

    def bmap(ax, bb):
        alloc = sel["selected"][bb]["allocation"]
        _bitmap(ax, np.array([[alloc[f"{l}.{g}"] for l in range(L)] for g in GROUPS]), fig)
        ax.set(yticks=range(4), yticklabels=[GLAB[g] for g in GROUPS])

    if sel:
        fig, ax = plt.subplots(2, 1, figsize=(5.5, 2.8), sharex=True)
        heat(ax[0], 2)
        title(ax[0], f"(a) {MEAS.get(meas, meas)} damage of one unit at 2 bits")
        bmap(ax[1], "3")
        title(ax[1], "(b) selected bit widths, INT3 budget")
        ax[1].set_xlabel("block")
        save_fig(r, fig, "F3_q3_sensitivity_and_bitmap")
    rows = [("heat", 2), ("heat", 3), ("heat", 4)] + ([("map", bb) for bb in sorted(sel["selected"])] if sel else [])
    fig, ax = plt.subplots(len(rows), 1, figsize=(5.5, 1.25 * len(rows)), sharex=True)
    for i, (kind, v) in enumerate(rows):
        if kind == "heat":
            heat(ax[i], v)
            title(ax[i], f"({'abcde'[i]}) {MEAS.get(meas, meas)} damage at {v} bits")
        else:
            bmap(ax[i], v)
            title(ax[i], f"({'abcde'[i]}) selected bit widths, INT{v} budget")
    ax[-1].set_xlabel("block")
    save_fig(r, fig, "FA3_q3_all_bits_and_bitmaps")


def f_drift(r: R, plt):
    groups = [[("P1_mag_s50", "-"), ("P2_wanda_s50", "-"), ("P3_s50", "-"), ("P2_wanda_s75", "--"), ("P3_s75", "--")],
              [("Q1_rtn_b3", "-"), ("Q2_awq_b3", "-"), ("Q3_mix_b3budget", "-"), ("Q2_awq_b4", "--")]]
    if not any(c in r.S for g in groups for c, _ in g):
        return
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
    for a, g, t in zip(ax, groups, ["(a) pruning", "(b) quantization"]):
        for c, ls in g:
            if c not in r.S:
                continue
            d = r.j("conditions", c, "calibration.json")["drift"]
            a.plot([x["block"] for x in d], [x["rel_mse"] for x in d], ls=ls, color=COLORS[_fam(r, c)], label=SHORT[c])
        a.set(xlabel="block (output)")
        logy(a)
        title(a, t)
        a.legend(frameon=False, fontsize=6.5, ncol=2)
    ax[0].set_ylabel("relative drift vs FP16")
    save_fig(r, fig, "F4_block_drift")


def f_local_err(r: R, plt):
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.7), sharey=False)
    groups = [(ax[0], "P1_mag", "P2_wanda", "s", (25, 50, 75), "%", "(a) pruning"),
              (ax[1], "Q1_rtn", "Q2_awq", "b", (8, 4, 3), "b", "(b) quantization")]
    for a, m1, m2, key, lv, unit, t in groups:
        cs = [(f"{m}_{key}{x}", m, x) for m in (m1, m2) for x in lv]
        cs = [c for c in cs if c[0] in r.S]
        w = 0.8 / max(1, len(cs))
        for i, (c, m, x) in enumerate(cs):
            cal = r.j("conditions", c, "calibration.json")
            v = [np.mean([y["rel_err"] for k, y in cal["per_proj"].items() if k.split(".")[1] == p]) for p in PROJ]
            a.bar(np.arange(7) + i * w - 0.4 + w / 2, v, w, color=COLORS[m[:2]], alpha=[0.35, 0.65, 1.0][lv.index(x)],
                  label=f"{FAM_LABEL[m[:2]]} {x}{unit}")
        a.set(xticks=range(7), xticklabels=PROJ)
        logy(a)
        a.grid(axis="x", visible=False)
        title(a, t)
        a.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax[0].set_ylabel("layer-output relative error")
    save_fig(r, fig, "FA5_layer_output_error")


def f_calib_vs_heldout(r: R, plt):
    a = r.j("analysis", "calibration_vs_heldout.json")
    if not a:
        return
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3))
    seen = set()
    for c, xk, yk in zip(a["conditions"], a["x"]["calib_kl"], a["y"]["wiki_dnll"]):
        fam = r.S[c]["family"]
        ax[0].plot([xk], [yk], "o" if fam in ("P1", "P2", "P3", "NM") else "s", ms=3.5, color=COLORS.get(fam, "gray"),
                   label=None if fam in seen else FAM_LABEL.get(fam, fam))
        seen.add(fam)
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    sp = a["correlations"]["calib_kl~wiki_dnll"]
    ax[0].set(xscale="log", yscale="symlog", xlabel="calibration KL vs FP16 (nats/token)", ylabel="WikiText-2 dNLL vs FP16")
    title(ax[0], f"(a) all conditions (Spearman {sp['rho']:.3f})")
    tr = r.j("transfer", "p3_transfer.json")
    cand = r.j("sensitivity", "p3_candidates.json")
    if tr and cand:
        for S_, mk, col in [(0.5, "o", "#74c476"), (0.75, "s", COLORS["P3"])]:
            ids = [k for k in tr if cand[k]["S"] == S_]
            ax[1].plot([cand[k]["dev_ppl"] for k in ids], [tr[k]["wikitext2_ppl"] for k in ids], mk, ms=3.5, color=col,
                       label=f"target {S_:.0%}")
        ax[1].set(xscale="log", yscale="log", xlabel="development ppl (selection)", ylabel="WikiText-2 ppl (post hoc)")
        title(ax[1], "(b) P3 candidates")
        ax[1].legend(frameon=False, fontsize=6.5)
    save_fig(r, fig, "F6_calibration_vs_heldout")


def f_rq1(r: R, plt):
    rq = r.j("analysis", "rq1_magnitude_vs_wanda.json")
    if rq:
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3))
        for p in PROJ:
            sps = sorted({float(k.rsplit(".", 1)[0]) for k in rq if k.endswith("." + p)})
            ax[0].plot([100 * s for s in sps], [rq[f"{s}.{p}"]["overlap"] for s in sps], marker="o", ms=3, color=PCOL[p], label=p)
            ax[1].plot([100 * s for s in sps], [rq[f"{s}.{p}"]["err_m"] / rq[f"{s}.{p}"]["err_w"] for s in sps], marker="o",
                       ms=3, color=PCOL[p], label=p)
        ax[0].set(xlabel="sparsity (%)", ylabel="share of Wanda mask")
        title(ax[0], "(a) mask overlap with magnitude")
        ax[1].set(xlabel="sparsity (%)", ylabel="error ratio, mag. / Wanda")
        logy(ax[1])
        title(ax[1], "(b) layer-error ratio")
        fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=7, frameon=False)
        save_fig(r, fig, "F7_rq1_vs_sparsity")
    info = r.j("conditions", "P2_wanda_s50", "compress_info.json") if "P2_wanda_s50" in r.S else None
    d = r.j("sensitivity", "p3_sensitivity.json")
    if not info:
        return
    L = max(int(k.split(".")[0]) for k in info["per_proj"]) + 1
    ov = np.array([[info["per_proj"][f"{l}.{p}"]["overlap_with_magnitude"] for l in range(L)] for p in PROJ])
    fig, ax = plt.subplots(2, 1, figsize=(5.5, 3.0), sharex=True)
    im = ax[0].imshow(ov, aspect="auto", cmap="magma")
    ax[0].set(yticks=range(7), yticklabels=PROJ)
    ax[0].grid(False)
    title(ax[0], "(a) share of the Wanda 50% mask also chosen by magnitude")
    fig.colorbar(im, ax=ax[0], pad=0.01).set_label("share")
    if d:
        an = d["dense_act_norm"]
        A = np.array([[an[f"{l}.{p}"]["max_over_median"] for l in range(L)] for p in ("q", "o", "gate", "down")])
        im = ax[1].imshow(np.log10(A), aspect="auto", cmap="viridis")
        ax[1].set(yticks=range(4), yticklabels=["q/k/v", "o", "gate/up", "down"])
        ax[1].grid(False)
        title(ax[1], "(b) log10 max/median input-channel norm")
        fig.colorbar(im, ax=ax[1], pad=0.01).set_label("log10 ratio")
    ax[1].set_xlabel("block")
    save_fig(r, fig, "FA7_mask_overlap_and_activation_outliers")


def f_awq(r: R, plt):
    a = r.j("sensitivity", "awq_search.json")
    if not a:
        return
    L = len(a["table"])
    bits = [b for b in (4, 3, 2) if b in a["bits"]]
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 3.4))
    for i, (ax, b) in enumerate(zip(axs.flat, bits)):
        A = np.array([[a["table"][str(l)][g][str(b)]["alpha"] for l in range(L)] for g in GROUPS])
        im = ax.imshow(A, aspect="auto", cmap="cividis", vmin=0, vmax=0.95)
        ax.set(yticks=range(4), yticklabels=[GLAB[g] for g in GROUPS], xlabel="block")
        ax.grid(False)
        title(ax, f"({'abc'[i]}) selected alpha, INT{b}")
    fig.colorbar(im, ax=axs[0, 1], pad=0.02).set_label(r"$\alpha$ (panels a-c)")
    ax = axs.flat[len(bits)]
    for g in GROUPS:
        cur = np.mean([np.array(a["table"][str(l)][g]["3"]["losses"]) / a["table"][str(l)][g]["3"]["loss_rtn"] for l in range(L)], 0)
        ax.plot(a["grid"], cur, color=GCOL[g], label=GLAB[g])
    ax.set(xlabel="alpha", ylabel="search loss / RTN loss")
    title(ax, f"({'abcd'[len(bits)]}) INT3 search curves")
    ax.legend(frameon=False, fontsize=6.5)
    save_fig(r, fig, "FA8_awq_alpha")


def f_tokens(r: R, plt):
    t = r.j("analysis", "token_level_nll.json")
    if not t:
        return
    keys = ["P1_mag_s50", "P2_wanda_s50", "P3_s50", "Q2_awq_b3", "Q3_mix_b3budget"]
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
    for i, (s, lab) in enumerate(SUITES):
        for c in keys:
            v = t.get(f"{s}.{c}")
            if v:
                ax[i].plot(range(1, 6), v["dnll_by_baseline_quintile"], marker="o", ms=3, color=COLORS[c[:2]],
                           ls="--" if c.startswith("Q") else "-", label=SHORT[c])
        ax[i].set(xlabel="FP16 token-NLL quintile (1 = easiest)", xticks=range(1, 6))
        title(ax[i], f"({'ab'[i]}) {lab}")
    ax[0].set_ylabel("mean dNLL vs FP16 (nats)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    save_fig(r, fig, "F9_token_level_nll")


def f_additivity(r: R, plt):
    s3 = r.j("sensitivity", "q3_sensitivity.json")
    sel3 = r.j("sensitivity", "q3_selection.json")
    s1 = r.j("sensitivity", "p3_sensitivity.json")
    pts = []
    if s3 and sel3:
        for bb, c in [("4", "Q3_mix_b4budget"), ("3", "Q3_mix_b3budget")]:
            if c in r.S:
                alloc = sel3["selected"][bb]["allocation"]
                pts.append((c, sum(s3["probe_kl"][u][str(b)]["kl"] for u, b in alloc.items()), r.S[c]["calibration"]["kl"]))
        for b, c in [(8, "Q2_awq_b8"), (4, "Q2_awq_b4"), (3, "Q2_awq_b3")]:
            if c in r.S:
                pts.append((c, sum(v[str(b)]["kl"] for v in s3["probe_kl"].values()), r.S[c]["calibration"]["kl"]))
    if s1:
        for sp, c in [("0.5", "P2_wanda_s50"), ("0.75", "P2_wanda_s75")]:
            if c in r.S:
                pts.append((c, sum(x["kl"] for x in s1["probe_kl"][sp]), r.S[c]["calibration"]["kl"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    marks = {"Q2_awq_b8": "v", "Q2_awq_b4": "s", "Q2_awq_b3": "D", "Q3_mix_b4budget": "s", "Q3_mix_b3budget": "D",
             "P2_wanda_s50": "o", "P2_wanda_s75": "^"}
    for c, x, y in pts:
        ax.plot([x], [y], marks.get(c, "o"), ms=4.5, color=COLORS[c[:2]], label=SHORT[c])
    lim = [min(min(p[1], p[2]) for p in pts) * 0.5, max(max(p[1], p[2]) for p in pts) * 2]
    ax.plot(lim, lim, color="0.4", ls="--", lw=0.8, label="additive (y = x)")
    ax.set(xscale="log", yscale="log", xlabel="sum of isolated probe KLs (nats/token)",
           ylabel="measured calibration KL (nats/token)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    save_fig(r, fig, "F11_additivity")
    save_json(r.ana / "additivity.json", [{"condition": LABEL[c], "sum_isolated_kl": x, "measured_kl": y} for c, x, y in pts])


def f_mech(r: R, plt):
    m = r.j("analysis", "mechanism_residual_stream.json")
    if not m:
        return
    B = m["per_block"]
    x = [b["block"] for b in B]
    fig, ax = plt.subplots(3, 1, figsize=(5.5, 3.9), sharex=True)
    ax[0].plot(x, np.maximum([b["resid_max_in"] for b in B], 1e-2), marker=".", ms=3, color="k", label="max |h|")
    ax[0].plot(x, np.maximum([b["resid_rms_in"] for b in B], 1e-2), color="0.55", label="RMS(h)")
    logy(ax[0])
    ax[0].set(ylabel="|h| entering block")
    title(ax[0], "(a) residual stream of the FP16 model")
    ax[0].legend(frameon=False, fontsize=6.5, loc="center right")
    ax[1].plot(x, [b["ref_ms_qkv"] for b in B], color=GCOL["qkv"], label="attention output")
    ax[1].plot(x, [b["ref_ms_gateup"] for b in B], color=GCOL["down"], label="MLP output")
    logy(ax[1])
    ax[1].set(ylabel="mean square")
    title(ax[1], "(b) what each block writes")
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2)
    for g in GROUPS:
        ax[2].plot(x, [b[f"q3_kl2_{g}"] for b in B], color=GCOL[g], label=f"{GLAB[g]} at 2 bits")
    ax[2].plot(x, [b["p3_kl50"] for b in B], "k--", lw=1, label="block pruned 50%")
    logy(ax[2])
    ax[2].set(xlabel="block", ylabel="probe KL")
    title(ax[2], "(c) isolated sensitivity")
    ax[2].legend(frameon=False, fontsize=6, ncol=3, loc="upper center")
    save_fig(r, fig, "F12_residual_stream_vs_fragility")


def f_diagnostics(r: R, plt):
    d = r.j("diagnostics", "mechanism_diagnostic.json")
    if not d or not d.get("trace"):
        return
    tr, pt, pr = d["trace"], d.get("patch", {}), d.get("protect", {})
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
    units = [u for u in ["0.qkv", "1.gateup", "1.down", "2.down", "26.gateup", "0.p75"] if u in tr and pt.get(u)]
    bars = [("compressed", lambda u: tr[u]["kl"], "0.25"),
            ("token 0 restored", lambda u: pt[u]["token0"]["kl"], "#1f77b4"),
            ("all but token 0 restored", lambda u: pt[u]["all_but_token0"]["kl"], "#9ecae1"),
            ("massive entries restored", lambda u: pt[u]["massive_entries"]["kl"] if "massive_entries" in pt[u] else np.nan,
             "#ff7f0e")]
    w = 0.8 / len(bars)
    for i, (lab, f, col) in enumerate(bars):
        ax[0].bar(np.arange(len(units)) - 0.4 + w * (i + 0.5), [f(u) for u in units], w, color=col, label=lab)
    ax[0].set(xticks=range(len(units)), xticklabels=[UNIT_TICK.get(u, u) for u in units], ylabel="final KL vs FP16")
    ax[0].tick_params(axis="x", labelsize=6)
    logy(ax[0])
    ax[0].grid(axis="x", visible=False)
    title(ax[0], "(a) activation patching")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    units = [u for u in ["0.qkv", "1.gateup", "26.gateup", "27.gateup", "27.down"] if u in pr]
    k = d["meta"]["k"]
    bars = [("nothing kept", lambda u: pr[u]["none"]["kl"], "0.25"),
            (f"top-{k} input channels in FP16", lambda u: pr[u]["top_input_channels"]["kl"], "#2ca02c"),
            (f"{k} random channels in FP16", lambda u: np.mean([x["kl"] for x in pr[u]["random_input_channels"]]), "#bdbdbd"),
            (f"top-{k} token-0 neurons in FP16", lambda u: pr[u]["top_token0_neurons"]["kl"] if "top_token0_neurons" in pr[u] else np.nan,
             "#9467bd")]
    w = 0.8 / len(bars)
    for i, (lab, f, col) in enumerate(bars):
        ax[1].bar(np.arange(len(units)) - 0.4 + w * (i + 0.5), [f(u) for u in units], w, color=col, label=lab)
    ax[1].set(xticks=range(len(units)), xticklabels=[UNIT_TICK.get(u, u) for u in units], ylabel="final KL vs FP16")
    ax[1].tick_params(axis="x", labelsize=6)
    logy(ax[1])
    ax[1].grid(axis="x", visible=False)
    title(ax[1], "(b) weight protection at 2 bits")
    ax[1].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    save_fig(r, fig, "F13_patching_and_protection")
    keep = [u for u in ["0.qkv", "1.gateup", "13.gateup", "26.gateup", "0.p75"] if u in tr]
    cols = dict(zip(keep, ["#1f77b4", "#d62728", "#7f7f7f", "#2ca02c", "#9467bd"]))
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.1))
    for u in keep:
        t = tr[u]
        xs = range(len(t["drift"]))
        blk, grp = u.split(".")
        lab = f"block {blk} pruned 75%" if grp == "p75" else f"block {blk} {GLAB.get(grp, grp)} @2b"
        ax[0].plot(xs, [v if v > 0 else np.nan for v in t["drift_excl_massive"]], color=cols[u], marker=".", ms=2.5, label=lab)
        ax[1].plot(xs, [np.nan if (v is None or v <= 0) else v for v in t["massive_rel_change"]], color=cols[u], label=lab)
        ax[2].plot(xs, t["attn0_change"], color=cols[u], label=lab)
    logy(ax[0])
    logy(ax[1])
    ax[0].set(xlabel="block (input)", ylabel="relative drift")
    title(ax[0], "(a) drift, massive excluded")
    ax[1].set(xlabel="block (input)", ylabel="relative change")
    title(ax[1], "(b) massive entries")
    ax[2].set(xlabel="block", ylabel="change in mass")
    title(ax[2], "(c) attention to token 0")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save_fig(r, fig, "FA13_propagation_traces")


# ================================================================== additions from the results audit (2026-11-22)
def t_theory_vs_measured(r: R):
    """RQ4 / manual: a concise comparison of theoretical versus measured savings, relative to FP16."""
    if "fp16" not in r.S:
        return
    f = r.S["fp16"]
    rf = r.j("conditions", "fp16", "resources.json") or {}
    rows = []
    for c in COND:
        if c not in r.S or c == "fp16":
            continue
        s, st = r.S[c], r.S[c]["storage"]
        res = r.j("conditions", c, "resources.json") or {}

        def ratio(key, sub="median_ms"):
            a, b = res.get(key, {}).get(sub), rf.get(key, {}).get(sub)
            return f"{a / b:.2f}x" if a and b else "-"
        w = res.get("weights_allocated_bytes")
        rows.append([LABEL[c], f"{st['effective_bits_per_eligible_weight']:.2f}",
                     f"{f['storage']['total_bytes'] / st['total_bytes']:.2f}x",
                     f"{100 * (1 - st['total_bytes'] / f['storage']['total_bytes']):.0f}%",
                     f"{w / rf['weights_allocated_bytes']:.2f}x" if w and rf.get("weights_allocated_bytes") else "-",
                     ratio("prefill_2048", "peak_allocated_bytes"), ratio("prefill_512"), ratio("prefill_2048"),
                     ratio("decode_64"), st.get("executed_as", "-")])
    write_table(r, "T4b_theory_vs_measured", ["Condition", "Stored bits / eligible weight", "Storage reduction (FP16 / method)",
                                              "Theoretical memory saving", "Measured weight allocation (method / FP16)",
                                              "Measured peak alloc, prefill 2048 (method / FP16)", "Prefill 512 time (method / FP16)",
                                              "Prefill 2048 time", "Decode 64 time", "Executed as"], rows,
                "Theoretical = serialized storage incl. all metadata (pruned: bitmap + FP16 nonzeros; quantized: packed codes "
                "+ zero-points + FP16 scales). Measured = the same model as executed on the T4 (dense FP16 GEMMs with zeros or "
                "fake-quantized weights): ratios near 1.00x mean no realised saving. Kernel-level evidence (CSR SpMM, 2:4 "
                "cuSPARSELt, INT4 unpack-then-GEMM) is in T5.")


STEP_NAME = {"dense_pass_s": "dense statistics pass (input norms, Block Influence, residual stats)",
             "owl_lod_s": "OWL outlier ratios", "fisher_s": "Fisher pass (squared gradients on the calibration set)",
             "probes_block_s": "isolated block probes (28 blocks x 2 sparsities)",
             "probes_component_s": "component probes (one projection type everywhere)",
             "dense_final_s": "dense reference pass for probes",
             "probes_unit_s": "isolated unit probes (112 units x 4 bit widths)"}


def t_selection_effort(r: R):
    rows = []
    for fam in ("p3", "q3"):
        s = r.j("sensitivity", f"{fam}_sensitivity.json")
        sel = r.j("sensitivity", f"{fam}_selection.json")
        tot = 0.0
        if s:
            for k, v in s["timings_s"].items():
                rows.append([fam.upper(), STEP_NAME.get(k, k), f"{v / 60:.2f}"])
                tot += v
        if sel:
            rows.append([fam.upper(), "development-set candidate builds + dev perplexity (all candidates)",
                         f"{sel['total_candidate_seconds'] / 60:.2f}"])
            tot += sel["total_candidate_seconds"]
        if tot:
            rows.append([fam.upper(), "**total selection effort**", f"**{tot / 60:.2f}**"])
    a = r.j("sensitivity", "awq_search.json")
    if a:
        rows.append(["Q2/Q3", "AWQ scale search (bits 2/3/4/8, shared by Q2 and Q3)", f"{a['timings_s']['awq_search_s'] / 60:.2f}"])
    if rows:
        write_table(r, "T12_selection_effort", ["Family", "Component", "Minutes (one T4)"], rows,
                    "GPU time of the sensitivity measurement and development-set selection, excluding the final builds "
                    "and evaluations (T4). The isolated probes are the dominant cost.")


def _clip(t, n):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n - 1] + "…"


def t_qualitative_detail(r: R):
    """Manual: 'a fixed selection of HellaSwag or model-continuation cases showing the same input and candidate scores
    across relevant configurations, including any observed failure or unexpected change'."""
    q = r.j("analysis", "qualitative_cases.json")
    txt = (r.j("analysis", "hellaswag_qualitative_text.json") or {}).get("cases", {})
    if q:
        keys = [c for c in ["fp16", "P1_mag_s50", "P2_wanda_s50", "P3_s50", "P2_wanda_s75", "Q1_rtn_b3", "Q2_awq_b3",
                            "Q3_mix_b3budget"] if c in next(iter(q.values()))]
        rows_ctx, rows_sc = [], []
        for o, case in q.items():
            t = txt.get(o)
            g = case["fp16"]["gold"]
            if t:
                rows_ctx.append([o, t["row"], _clip(t["context"], 150), f"{g}: " + _clip(t["endings"][g], 70)])
        fail = [o for o in q if int(o) not in range(0, 512, 64)]
        for o in fail:
            case = q[o]
            g = case["fp16"]["gold"]
            for e in range(4):
                cells = []
                for c in keys:
                    x = case[c]
                    v = x["ll"][e] / x["char_len"][e]
                    cells.append(f"{v:.3f}" + (" <" if x["pred_norm"] == e else ""))
                end = txt.get(o, {}).get("endings", [None] * 4)[e]
                rows_sc.append([o if e == 0 else "", f"{e}{' (gold)' if e == g else ''}",
                                _clip(end, 45) if end else "-"] + cells)
        if rows_ctx:
            write_table(r, "T15b_qualitative_inputs", ["Order", "Dataset row", "Context (as scored)", "Gold ending"], rows_ctx,
                        "Rowan/hellaswag validation rows of the qualitative cases (text verified against the saved candidate "
                        "character lengths and labels; analysis/hellaswag_qualitative_text.json).")
        if rows_sc:
            write_table(r, "T15c_candidate_scores", ["Order", "Ending", "Ending text"] + [LABEL[c] for c in keys], rows_sc,
                        "Per-character log-likelihood of each candidate ending (the acc_norm score); '<' marks the chosen "
                        "ending. Failure-rule cases of T15 (FP16 correct, most compressed configurations wrong).")
    cont = r.j("analysis", "continuations_aligned.json")
    if cont:
        rows = []
        n = len(next(iter(cont.values())))
        for i in range(n):
            first = next(iter(cont.values()))[i]
            rows.append([i, "prompt", _clip(first["prompt"], 120), "-"])
            for c, lst in cont.items():
                x = lst[i]
                rows.append(["", LABEL.get(c, c), _clip(x["continuation"], 120), f"{x.get('distinct2', float('nan')):.2f}"])
        write_table(r, "T15d_continuations", ["Prompt", "Model", "Greedy continuation (40 tokens)", "distinct-2"], rows,
                    "First 48 tokens of development sequences 0-3 as prompts (fixed before results); greedy decoding. "
                    "distinct-2 = share of distinct token bigrams (low values = repetition).")


def f_paired(r: R, plt):
    d = r.j("analysis", "paired_differences.json")
    if not d:
        return
    pairs = [(f"{b}-{a}", lab) for (a, b), lab in zip(PAIRS_MAIN, ["P3 - Wanda, 50%", "P3 - Wanda, 75%",
                                                                   "Q3 - AWQ, INT4 budget", "Q3 - AWQ, INT3 budget"])
             if f"{b}-{a}" in d]
    if not pairs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(pairs))
    for k, ((s, lab), col, off) in enumerate(zip(SUITES, ["#1f77b4", "#ff7f0e"], (-0.12, 0.12))):
        v = [d[key][s]["d_nll"] for key, _ in pairs]
        lo = [d[key][s]["d_nll_ci"][0] for key, _ in pairs]
        hi = [d[key][s]["d_nll_ci"][1] for key, _ in pairs]
        ax[0].errorbar(v, y + off, xerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="o", ms=3.5, color=col,
                       capsize=2, lw=1, label=lab)
    ax[0].axvline(0, color="0.4", lw=0.8)
    ax[0].set_xscale("symlog", linthresh=0.01)
    lo_all = min(min(d[key][s]["d_nll_ci"][0] for key, _ in pairs) for s, _ in SUITES)
    ax[0].set_xlim(lo_all * 1.6, 0.05)
    ax[0].set_xticks([t for t in (-1, -0.1, -0.01, 0) if t >= lo_all * 1.6])
    ax[0].set_xticklabels([f"{t:g}" for t in (-1, -0.1, -0.01, 0) if t >= lo_all * 1.6])
    ax[0].set(yticks=y, yticklabels=[lab for _, lab in pairs], xlabel="change in mean NLL (nats)")
    ax[0].invert_yaxis()
    ax[0].grid(axis="y", visible=False)
    title(ax[0], "(a) held-out likelihood")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    v = [d[key]["acc_norm"]["d_pp"] for key, _ in pairs]
    lo = [d[key]["acc_norm"]["d_pp_ci"][0] for key, _ in pairs]
    hi = [d[key]["acc_norm"]["d_pp_ci"][1] for key, _ in pairs]
    ax[1].errorbar(v, y, xerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="s", ms=3.5, color="k", capsize=2, lw=1)
    for yy, key in zip(y, [k for k, _ in pairs]):
        ax[1].annotate(f"n={d[key]['hellaswag_n']}", (hi[list(y).index(yy)], yy), xytext=(4, -2.5),
                       textcoords="offset points", fontsize=6)
    ax[1].axvline(0, color="0.4", lw=0.8)
    ax[1].set(yticks=y, yticklabels=[], xlabel="change in acc_norm (pp)")
    ax[1].invert_yaxis()
    ax[1].grid(axis="y", visible=False)
    title(ax[1], "(b) HellaSwag")
    save_fig(r, fig, "F5_paired_differences")


FIGURE_FILES = {"F1_quality_vs_budget", "F5_paired_differences", "FA1_quality_vs_budget_c4", "F2_p3_sensitivity_and_allocation",
                "FA2_p3_all_measures", "FA2b_component_probes_pruning", "F3_q3_sensitivity_and_bitmap",
                "FA3_q3_all_bits_and_bitmaps", "F4_block_drift", "FA5_layer_output_error", "F6_calibration_vs_heldout",
                "F7_rq1_vs_sparsity", "FA7_mask_overlap_and_activation_outliers", "FA8_awq_alpha", "F9_token_level_nll",
                "F11_additivity", "F12_residual_stream_vs_fragility", "F13_patching_and_protection",
                "FA13_propagation_traces"}


def prune_stale_figures(r: R):
    """Remove figure files this compiler no longer produces (renamed or dropped in the restyle)."""
    for f in r.fig.glob("*.*"):
        if f.stem not in FIGURE_FILES and f.suffix in (".png", ".pdf"):
            f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    r = R(Path(a.results), load_config(a.config))
    plt = setup_mpl()
    prune_stale_figures(r)
    jobs = [t_quality, t_nll, t_pairs, t_budget, t_resources, t_microbench, t_p3, t_q3, t_calib_vs_heldout, t_local_errors,
            t_rq1, t_awq, t_selection_effort, t_tokens, t_hs_margin, t_qualitative, t_qualitative_detail,
            t_mech_residual, t_mech_local_alloc, t_mech_nm, t_diagnostics]
    figs = [f_tradeoff, f_p3, f_q3, f_drift, f_local_err, f_calib_vs_heldout, f_rq1, f_awq, f_tokens, f_additivity,
            f_mech, f_diagnostics, f_paired]
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
