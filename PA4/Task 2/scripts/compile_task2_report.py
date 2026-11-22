"""Compile Task 2 tables (results/tables), figures (results/figures) and analysis JSON (results/analysis).
CPU only; every table/figure is independent and skipped with a message if its inputs are missing.
  python scripts/compile_task2_report.py [--results results] [--config configs/task2.yaml]
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

POL = ["fp16", "sink_recent", "question_aware", "int4_channelK_tokenV", "int4_tokenwise", "hybrid"]
PLAB = {"fp16": "FP16 full cache", "sink_recent": "Evict: sink+recent", "question_aware": "Evict: question-aware",
        "int4_channelK_tokenV": "INT4 channel-K/token-V", "int4_tokenwise": "INT4 token-wise K/V",
        "hybrid": "Hybrid FP16 + INT2"}
MECH = {"dense": "Dense", "window": "Window W=64", "linear": "Linear (ELU+1)", "dilated": "Local/dilated",
        "delta": "Delta-rule (B3)"}
BINS = [(0, 32), (32, 64), (64, 128), (128, 256)]


class R:
    def __init__(self, res: Path, cfg):
        self.res, self.cfg = res, cfg
        self.tab, self.fig, self.ana = res / "tables", res / "figures", res / "analysis"
        for d in (self.tab, self.fig, self.ana):
            d.mkdir(parents=True, exist_ok=True)

    def j(self, *p):
        f = self.res.joinpath(*p)
        return load_json(f) if f.exists() else None


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
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8.5, "savefig.bbox": "tight", "axes.grid": True,
                         "grid.alpha": 0.3, "pdf.fonttype": 42, "figure.constrained_layout.use": True})
    return plt


def save_fig(r, fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(r.fig / f"{name}.pdf")
    fig.savefig(r.fig / f"{name}.png", dpi=200)
    plt.close(fig)


def boot_diff(a, b, B=10000, seed=6304):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    diff = b - a
    # drawn in 500-row chunks from the same generator: the same index stream as one (B, n) draw, bounded memory
    d = np.concatenate([diff[rng.integers(0, len(a), (min(500, B - i), len(a)))].mean(1) for i in range(0, B, 500)])
    return float(b.mean() - a.mean()), np.percentile(d, [2.5, 97.5]).tolist()


# ============================================================================================ Part A
def a_tables(r, plt):
    c = r.j("part_a", "part_a_correctness.json")
    if not c:
        return "no part A"
    rows = []
    for x in c["vs_fp32_dense"]:
        if "out" in x:
            rows.append([x["L"], x["dtype"], x["impl"], f"{x['out']['max_abs']:.2e}", f"{x['out']['rel_l2']:.2e}",
                         f"{x['dq']['rel_l2']:.2e}", f"{x['dk']['rel_l2']:.2e}", f"{x['dv']['rel_l2']:.2e}",
                         x["out"]["nonfinite"] + x["dq"]["nonfinite"]])
        else:
            rows.append([x["L"], x.get("dtype", "-"), x.get("impl", "reference"), x.get("error", x.get("reference")), "", "", "", "", ""])
    write_table(r, "A1_errors_vs_fp32_dense", ["L", "dtype", "Impl", "Out max abs", "Out rel L2", "dQ rel L2", "dK rel L2",
                                               "dV rel L2", "Non-finite"], rows,
                "Reference: dense FP32 attention and its autograd gradients on identical Q, K, V and upstream gradient "
                "(B=2, H=4, d=64). FP32 reference itself vs FP64 in A1b.")
    write_table(r, "A1b_fp32_reference_vs_fp64", ["L", "Out rel L2", "dQ rel L2", "dK rel L2", "dV rel L2"],
                [[x["L"]] + [f"{x[n]['rel_l2']:.2e}" for n in ("out", "dq", "dk", "dv")] for x in c["reference_vs_fp64"]])
    write_table(r, "A1c_tile_sizes", ["L", "dtype", "Tile", "Impl", "Out rel L2", "dQ rel L2", "dK rel L2", "dV rel L2"],
                [[x["L"], x["dtype"], x["tile"], x["impl"]] + [f"{x[n]['rel_l2']:.2e}" for n in ("out", "dq", "dk", "dv")]
                 for x in c["tile_sweep"]])
    write_table(r, "A1d_causality", ["dtype", "Impl", "Max |diff| outputs <= t0 after perturbing keys/values > t0",
                                     "Later outputs changed", "max |dL/dK_future|", "max |dL/dV_future|"],
                [[x["dtype"], x["impl"], f"{x['max_diff_prefix_outputs']:.1e}", x["suffix_outputs_changed"],
                  f"{x['grad_future_keys_absmax']:.1e}", f"{x['grad_future_values_absmax']:.1e}"] for x in c["causality"]])
    write_table(r, "A1e_edge_cases", ["Case", "Result"], [[k, json.dumps(v)[:160]] for k, v in c["edge_cases"].items()])
    # figure: error vs L
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for dt, mk in (("float16", "o"), ("float32", "s")):
        for impl in ("dense", "tiled_autograd", "tiled_recompute", "sdpa_auto"):
            pts = [(x["L"], x["out"]["rel_l2"], x["dq"]["rel_l2"]) for x in c["vs_fp32_dense"]
                   if x.get("impl") == impl and x.get("dtype") == dt and "out" in x]
            if pts:
                L, o, g = zip(*pts)
                ax[0].plot(L, o, marker=mk, label=f"{impl} {dt}")
                ax[1].plot(L, g, marker=mk, label=f"{impl} {dt}")
    for a, t in zip(ax, ["Output", "dQ"]):
        a.set(xscale="log", yscale="log", xlabel="Sequence length L", ylabel="Relative L2 error vs dense FP32",
              title=f"{t} error vs dense FP32 reference")
    ax[0].legend(ncol=2, fontsize=7.5)
    save_fig(r, fig, "FA1_error_vs_length")


def a_bench(r, plt):
    b = r.j("part_a", "part_a_benchmark.json")
    if not b:
        return "no benchmark (CPU run)"
    rows = []
    for x in b["rows"]:
        rows.append([x["impl"], x["dtype"], x["L"], x["mode"], f3(x.get("median_ms"), 2), f3(x.get("std_ms"), 2),
                     f"{x['peak_allocated_bytes'] / 2**20:.0f}" if "peak_allocated_bytes" in x else "-",
                     f"{x['peak_reserved_bytes'] / 2**20:.0f}" if "peak_reserved_bytes" in x else "-", x.get("error", "")])
    write_table(r, "A2_benchmark", ["Impl", "dtype", "L", "Mode", "Median ms", "SD ms", "Peak alloc MiB (above inputs)",
                                    "Peak reserved MiB", "Error / unsupported"], rows,
                f"{b['reps']} CUDA-event timed repetitions after {b['warmup']} warm-ups; forward-only under no_grad vs "
                "complete forward+backward. tiled_* are Python loops (not fused kernels); tiled_autograd stores every "
                "tile's intermediates for autograd, tiled_recompute uses a FlashAttention-2-style recomputation backward.")
    disp = [[x["dtype"], x["L"], ",".join(x.get("forward", {}).get("backends", [])) or x.get("error", "-"),
             ",".join(x.get("backward", {}).get("backends", []))] for x in b["dispatch"]]
    write_table(r, "A3_sdpa_dispatch", ["dtype", "L", "Unforced SDPA forward ran", "Unforced SDPA backward ran"], disp,
                "Detected from the profiler's ATen op names, not assumed.")
    sup = defaultdict(set)
    for x in b["rows"]:
        if x.get("error"):
            sup[(x["impl"], x["dtype"])].add(f"L={x['L']} {x['mode']}: {x['error'][:60]}")
    write_table(r, "A4_unsupported_or_oom", ["Impl", "dtype", "Conditions"],
                [[i, d, "; ".join(sorted(v))[:400]] for (i, d), v in sorted(sup.items())])
    for metric, ylab, fname in (("median_ms", "Median latency (ms)", "FA2_latency_vs_length"),
                                ("peak_allocated_bytes", "Peak allocated above inputs (MiB)", "FA3_memory_vs_length")):
        fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for i, dt in enumerate(("float16", "float32")):
            for j, mode in enumerate(("forward", "forward_backward")):
                a = ax[i, j]
                for impl in ("dense", "tiled_autograd", "tiled_recompute", "sdpa_auto", "sdpa_math", "sdpa_efficient"):
                    pts = [(x["L"], x[metric] / (2**20 if metric != "median_ms" else 1)) for x in b["rows"]
                           if x["impl"] == impl and x["dtype"] == dt and x["mode"] == mode and metric in x]
                    if pts:
                        a.plot(*zip(*pts), marker="o", label=impl)
                a.set(xscale="log", yscale="log", title=f"{dt}, {mode.replace('_', '+')}", ylabel=ylab)
                if i == 1:
                    a.set_xlabel("Sequence length L")
        ax[0, 0].legend(fontsize=7.5)
        save_fig(r, fig, fname)


# ============================================================================================ Part B
def _runs(r):
    rd = r.res / "part_b" / "runs"
    out = {}
    if rd.exists():
        for p in sorted(rd.iterdir()):
            if (p / "summary.json").exists():
                out[p.name] = load_json(p / "summary.json")
    return out


def _qt(r):
    f = r.res / "part_b" / "test_query_table.npz"
    return dict(np.load(f)) if f.exists() else None


def _group(runs):
    g = defaultdict(list)
    for name, s in runs.items():
        g[(s["mech"], s["train_K"], s["train_regime"])].append((name, s))
    return g


def b_tables(r, plt):
    runs, qt = _runs(r), _qt(r)
    if not runs or qt is None:
        return "no part B"
    g = _group(runs)
    rows, anal = [], {}
    for (mech, K, reg), lst in sorted(g.items(), key=lambda kv: (list(MECH).index(kv[0][0]), kv[0][1], kv[0][2])):
        accs = defaultdict(list)
        for name, s in lst:
            c = np.load(r.res / "part_b" / "runs" / name / "test_predictions.npz")["correct"].astype(bool)
            for k in (4, 8, 16):
                for rg in ("near", "far", "mixed"):
                    m = (qt["K"] == k) & (qt["regime"] == rg)
                    accs[(k, rg)].append(c[m].mean())
        row = [MECH[mech], K, reg, len(lst), f"{np.mean([s['params'] for _, s in lst]):,.0f}"]
        for k in (4, 8, 16):
            for rg in ("near", "far", "mixed"):
                v = accs[(k, rg)]
                row.append(f"{100 * np.mean(v):.1f}±{100 * np.std(v):.1f}")
        fail = [n for n, s in lst if s["test_acc_all"] < 2 / 64 or s.get("diverged_at")]
        row.append(", ".join(fail) if fail else "")
        rows.append(row)
        anal[f"{mech}_K{K}_{reg}"] = {f"K{k}.{rg}": [float(x) for x in v] for (k, rg), v in accs.items()}
    hdr = ["Mechanism", "Train K", "Train regime", "Seeds", "Params"] + [f"K{k} {rg}" for k in (4, 8, 16) for rg in ("near", "far", "mixed")] + ["Failed to learn (<2x chance) / diverged"]
    write_table(r, "B1_test_accuracy", hdr, rows,
                "Test accuracy (%) mean±SD over seeds 6304-6306 on the frozen 1,024 sequences per (K, regime), pooled over "
                "the supervised query positions. Chance: label space 1/64 = 1.6 %; random choice among the K values of "
                "the example: 25 % (K=4), 12.5 % (K=8), 6.25 % (K=16).")
    save_json(r.ana / "b_accuracy_per_seed.json", anal)
    # distance bins x K, per model (mean over seeds)
    rows2 = []
    heat = {}
    for (mech, K, reg), lst in g.items():
        M = np.zeros((3, len(BINS)))
        for name, s in lst:
            c = np.load(r.res / "part_b" / "runs" / name / "test_predictions.npz")["correct"].astype(bool)
            for i, k in enumerate((4, 8, 16)):
                for j, (lo, hi) in enumerate(BINS):
                    m = (qt["K"] == k) & (qt["distance"] >= lo) & (qt["distance"] < hi)
                    M[i, j] += c[m].mean() / len(lst) if m.any() else np.nan
        heat[f"{mech}|{K}|{reg}"] = M.tolist()
        for i, k in enumerate((4, 8, 16)):
            rows2.append([MECH[mech], K, reg, k] + [f"{100 * x:.1f}" for x in M[i]])
    write_table(r, "B2_distance_by_load", ["Mechanism", "Train K", "Train regime", "Eval K"] +
                [f"[{lo},{hi})" for lo, hi in BINS], rows2,
                "Accuracy (%) by the recorded source-query distance (all regimes pooled) and evaluation K. For W = 64 "
                "two-layer models, distances > 126 are outside the receptive field by construction (an access limit, "
                "not a training failure).")
    save_json(r.ana / "b_distance_by_load.json", heat)
    _b_heatmap(r, plt, heat)
    _b_controls(r, runs, qt)
    _b_alignment(r, runs, qt)
    _b_intervention(r, runs, qt)


def _b_heatmap(r, plt, heat):
    keys = sorted(heat, key=lambda k: (list(MECH).index(k.split("|")[0]), int(k.split("|")[1]), k.split("|")[2]))
    n = len(keys)
    cols = 5
    rows = -(-n // cols)
    fig, ax = plt.subplots(rows, cols, figsize=(2.9 * cols, 2.4 * rows), squeeze=False)
    for a in ax.flat:
        a.axis("off")
    for a, k in zip(ax.flat, keys):
        mech, K, reg = k.split("|")
        a.axis("on")
        im = a.imshow(np.array(heat[k]) * 100, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        for i in range(3):
            for j in range(len(BINS)):
                v = heat[k][i][j]
                a.text(j, i, "-" if np.isnan(v) else f"{100 * v:.0f}", ha="center", va="center", fontsize=8,
                       color="white" if (np.isnan(v) or v < 0.6) else "black")
        a.set(xticks=range(len(BINS)), xticklabels=["0-31", "32-63", "64-127", "128+"], yticks=range(3),
              yticklabels=["K=4", "K=8", "K=16"], title=f"{MECH[mech]}\ntrained K={K}, {reg}")
        a.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Test accuracy (%)")
    save_fig(r, fig, "FB2_distance_by_load_heatmaps")


def _b_controls(r, runs, qt):
    rows = []
    for name, s in sorted(runs.items()):
        if "controls" not in s:
            continue
        base = s["test_cells"]
        for ctrl, v in s["controls"].items():
            rows.append([name, ctrl] + [f"{100 * base[f'K{k}.{rg}']:.1f} -> {100 * v['cells'][f'K{k}.{rg}']:.1f}"
                                        for k in (4, 16) for rg in ("near", "far", "mixed")])
    win = [(n, s) for n, s in runs.items() if s["mech"] == "window"]
    for n, s in sorted(win):
        rows.append([n, "trained with W=64"] + [f"{100 * s['test_cells'][f'K{k}.{rg}']:.1f}" for k in (4, 16)
                                                 for rg in ("near", "far", "mixed")])
    if rows:
        write_table(r, "B3_train_vs_test_mask_controls", ["Model", "Control"] + [f"K{k} {rg}" for k in (4, 16) for rg in ("near", "far", "mixed")],
                    rows, "Dense models evaluated with a W = 64 window or top-k content selection imposed at test time "
                    "only (dense -> controlled), next to the separately trained window models.")


def _b_alignment(r, runs, qt):
    rows = []
    for name, s in sorted(runs.items()):
        if s["mech"] not in ("dilated", "window", "dense"):
            continue
        c = np.load(r.res / "part_b" / "runs" / name / "test_predictions.npz")["correct"].astype(bool)
        far = qt["distance"] >= 128
        row = [name]
        for res_ in range(4):
            m = far & ((qt["distance"] - 1) % 4 == res_)
            row.append(f"{100 * c[m].mean():.1f} (n={m.sum()})")
        rows.append(row)
    if rows:
        write_table(r, "B4_far_accuracy_by_alignment", ["Model", "(d-1) mod 4 = 0", "= 1", "= 2", "= 3"], rows,
                    "Far queries (distance >= 128) split by the alignment of the VALUE token (at distance d-1 from the "
                    "query) with the dilated offsets {0,4,...,252}: a dilated head reaches the value directly only when "
                    "(d-1) mod 4 = 0.")


def _b_intervention(r, runs, qt):
    pb = r.cfg["part_b"]
    iv, cmp_ = pb["intervention"]["name"], pb["intervention"]["comparator"]
    rows, out = [], {}
    for seed in pb["seeds"]:
        a, b = f"{cmp_}_K16_mixed_s{seed}", f"{iv}_K16_mixed_s{seed}"
        if a not in runs or b not in runs:
            continue
        ca = np.load(r.res / "part_b" / "runs" / a / "test_predictions.npz")["correct"]
        cb = np.load(r.res / "part_b" / "runs" / b / "test_predictions.npz")["correct"]
        m = qt["K"] == 16
        d, ci = boot_diff(ca[m], cb[m])
        bins = []
        for lo, hi in BINS:
            mm = m & (qt["distance"] >= lo) & (qt["distance"] < hi)
            bins.append(f"{100 * ca[mm].mean():.1f}->{100 * cb[mm].mean():.1f}")
        rows.append([seed, runs[a]["params"], runs[b]["params"], f"{100 * ca[m].mean():.1f}", f"{100 * cb[m].mean():.1f}",
                     f"{100 * d:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"] + bins)
        out[seed] = {"diff": d, "ci": ci}
    th = r.j("part_b", "theory.json") or {}
    if rows:
        write_table(r, "B5_intervention", ["Seed", f"Params {cmp_}", f"Params {iv}", f"{cmp_} K16 acc %", f"{iv} K16 acc %",
                                           "Diff pp [95% CI, queries]"] + [f"[{lo},{hi}) {cmp_}->{iv}" for lo, hi in BINS], rows,
                    f"Prediction stated before evaluation: {pb['intervention']['prediction']} State per layer: "
                    f"linear {th.get('linear', {}).get('state_numbers_per_layer')} numbers, delta "
                    f"{th.get('delta', {}).get('state_numbers_per_layer')} numbers. CIs resample test queries within a seed.")
        save_json(r.ana / "b_intervention.json", out)


def b_figs(r, plt):
    runs = _runs(r)
    if not runs:
        return
    g = _group(runs)
    keys = sorted(g, key=lambda k: (list(MECH).index(k[0]), k[1], k[2]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for key in keys:
        lst = g[key]
        lab = f"{MECH[key[0]]} K={key[1]} {key[2]}"
        u = [x["update"] for x in lst[0][1]["train_curve"]]
        L = np.array([[x["loss"] for x in s["train_curve"]][:len(u)] for _, s in lst if len(s["train_curve"]) >= len(u)])
        if len(L):
            line, = ax[0].plot(u, L.mean(0), label=lab)
            ax[0].fill_between(u, L.min(0), L.max(0), alpha=0.15, color=line.get_color())
        du = [x["update"] for x in lst[0][1]["dev_curve"]]
        D = np.array([[x["dev_acc"] for x in s["dev_curve"]][:len(du)] for _, s in lst if len(s["dev_curve"]) >= len(du)])
        if len(D):
            ax[1].plot(du, 100 * D.mean(0), marker="o", label=lab)
    ax[0].set(xlabel="Optimizer update", ylabel="Training loss (64-way CE)", title="Training loss (mean, min-max over seeds)", yscale="log")
    ax[1].set(xlabel="Optimizer update", ylabel="Development accuracy (%)", title="Development accuracy at the checkpoints")
    ax[1].axhline(100 / 64, color="k", ls=":", lw=1)
    ax[1].legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_fig(r, fig, "FB1_learning_curves")


def b_resources(r, plt):
    th, sw = r.j("part_b", "theory.json"), r.j("part_b", "inference_sweep.json")
    runs = _runs(r)
    if th:
        rows = []
        for mech in ("dense", "window", "dilated"):
            t = th[mech]
            rows.append([MECH[mech], t["pairs_per_layer"], "/".join(f"{x:.0f}" for x in t["mean_positions_per_query"]),
                         "/".join(str(x) for x in t["max_lookback"]), t["kv_cache_bytes_per_layer_fp32"],
                         t.get("max_two_layer_span", "-"), f3(t.get("frac_distances_1_255_reachable_two_layers"), 3)])
        for mech in ("linear", "delta"):
            t = th[mech]
            rows.append([MECH[mech], "-", "all (compressed)", "unbounded", t["state_bytes_per_layer_fp32"], "-", "-"])
        ms = defaultdict(list)
        for s in runs.values():
            ms[s["mech"]].append(s["ms_per_update"])
        for row in rows:
            key = [k for k, v in MECH.items() if v == row[0]][0]
            row.append(f"{np.mean(ms[key]):.1f}" if ms.get(key) else "-")
        write_table(r, "B6_theoretical_vs_measured", ["Mechanism", "Attended pairs / layer (T=256)", "Mean positions/query per head",
                                                      "Max look-back per head", "Persistent bytes / layer (FP32; KV cache or state)",
                                                      "Max 2-layer span", "Share of distances 1-255 reachable in 2 layers",
                                                      "Measured ms / training update"], rows,
                    "Theoretical counts at T = 256. Softmax variants are trained with explicit dense masks, so their measured "
                    "time does not shrink with fewer pairs; the delta model uses a Python time loop.")
    if sw:
        rows = [[x["impl"], x["T"], f3(x.get("median_ms"), 2), f"{x['peak_allocated_bytes'] / 2**20:.0f}" if "peak_allocated_bytes" in x else "-",
                 x["materialises"], x.get("error", "")] for x in sw["rows"]]
        write_table(r, "B7_inference_sweep", ["Implementation", "T", "Median ms", "Peak alloc MiB", "Materialises", "Error"], rows, sw["note"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        for impl in sorted({x["impl"] for x in sw["rows"]}):
            pts = [(x["T"], x["median_ms"], x.get("peak_allocated_bytes", np.nan) / 2**20) for x in sw["rows"]
                   if x["impl"] == impl and "median_ms" in x]
            if pts:
                T, t, m = zip(*pts)
                ax[0].plot(T, t, marker="o", label=impl)
                ax[1].plot(T, m, marker="o", label=impl)
        ax[0].set(xscale="log", yscale="log", xlabel="T", ylabel="Forward ms (median)", title="Forward-only inference sweep")
        has_mem = any(np.isfinite(l.get_ydata()).any() for l in ax[1].get_lines())
        ax[1].set(xscale="log", yscale="log" if has_mem else "linear", xlabel="T", ylabel="Peak allocated (MiB)",
                  title="Memory" if has_mem else "Memory (not measured on CPU)")
        ax[0].legend(fontsize=7.5)
        save_fig(r, fig, "FB3_inference_sweep")


# ============================================================================================ Part C
def _c_rows(r):
    f = r.res / "part_c" / "per_prompt.jsonl.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        return [json.loads(x) for x in fh]


def c_tables(r, plt):
    rows = _c_rows(r)
    if not rows:
        return "no part C"
    B = r.cfg["part_c"]["bootstrap"]
    ref_ok = np.array([x["policies"]["fp16"]["exact_match"] for x in rows], bool)
    out = []
    for p in POL:
        v = [x["policies"][p] for x in rows]
        em = np.array([y["exact_match"] for y in v])
        out.append([PLAB[p], len(v), f"{100 * em.mean():.1f}", f"{100 * em[ref_ok].mean():.1f}" if ref_ok.any() else "-",
                    f"{np.mean([y['gold_nll'] for y in v]):.3f}",
                    f3(np.mean([y.get("post_prefill_kl", 0) for y in v]), 4), f3(np.mean([y.get("teacher_forced_kl", 0) for y in v]), 4),
                    f3(np.mean([y.get("evidence_retained_mean", y.get("evidence_fp16_mean", np.nan)) for y in v]), 3),
                    f"{np.mean([y['frac_fp16_bytes'] for y in v]):.4f} (max {np.max([y['frac_fp16_bytes'] for y in v]):.4f})",
                    str(all(y.get("within_budget", True) for y in v))])
    write_table(r, "C1_matched_budget", ["Cache", "n", "Exact match %", "EM % given FP16 correct", "Gold-answer NLL",
                                         "Post-prefill KL vs FP16", "Teacher-forced KL", "Evidence retained (FP16)",
                                         "Bytes / FP16 context", "All within budget"], out,
                "Budget per prompt = serialized bytes of the uniform INT4 channel-K/token-V context cache (64-element groups, "
                "FP16 scale + FP16 offset, FP16 trailing L mod 64 tokens). Evidence retained: share of gold-evidence tokens "
                "kept (eviction) or kept in FP16 (hybrid), averaged over layers and KV heads. Shared suffix/generated FP16 "
                "entries are reported separately in C4.")
    # breakdowns
    br = []
    for fac in ("target_tokens", "position", "distractors"):
        for lev in sorted({x[fac] for x in rows}, key=str):
            sub = [x for x in rows if x[fac] == lev]
            br.append([fac, lev, len(sub)] + [f"{100 * np.mean([x['policies'][p]['exact_match'] for x in sub]):.0f}" for p in POL])
    write_table(r, "C2_breakdown", ["Factor", "Level", "n"] + [PLAB[p] for p in POL], br,
                "Exact match (%). Cells of 10 prompts are descriptive only, not separately powered comparisons.")
    # paired bootstrap
    pairs = [("fp16", p) for p in POL[1:]] + [("sink_recent", "question_aware"), ("int4_tokenwise", "int4_channelK_tokenV"),
                                               ("question_aware", "hybrid"), ("int4_channelK_tokenV", "hybrid")]
    bt, anal = [], {}
    for a, b in pairs:
        ea = [x["policies"][a]["exact_match"] for x in rows]
        eb = [x["policies"][b]["exact_match"] for x in rows]
        d, ci = boot_diff(ea, eb, B["resamples"], B["seed"])
        na = [x["policies"][a]["gold_nll"] for x in rows]
        nb = [x["policies"][b]["gold_nll"] for x in rows]
        dn, cin = boot_diff(na, nb, B["resamples"], B["seed"])
        bt.append([f"{PLAB[b]} - {PLAB[a]}", f"{100 * d:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]",
                   f"{dn:+.3f} [{cin[0]:+.3f}, {cin[1]:+.3f}]", sum(1 for x, y in zip(ea, eb) if x > y), sum(1 for x, y in zip(ea, eb) if y > x)])
        anal[f"{b}-{a}"] = {"em_diff": d, "em_ci": ci, "nll_diff": dn, "nll_ci": cin}
    write_table(r, "C3_paired_bootstrap", ["Difference", "Exact match pp [95% CI]", "Gold NLL [95% CI]", "A only correct", "B only correct"],
                bt, f"Paired bootstrap over the {len(rows)} prompts, {B['resamples']} resamples, seed {B['seed']}.")
    save_json(r.ana / "c_paired_bootstrap.json", anal)
    # systems
    sysrows = [[PLAB[p], f"{np.mean([x['prefill_s'] for x in rows]) * 1000:.0f}",
                f"{np.mean([x['policies'][p].get('scoring_s', 0) for x in rows]) * 1000:.0f}",
                f"{np.mean([x['policies'][p].get('compress_s', 0) for x in rows]) * 1000:.0f}",
                f"{np.mean([x['policies'][p]['suffix_s'] for x in rows]) * 1000:.0f}",
                f"{np.mean([x['policies'][p]['decode_s'] / max(1, x['policies'][p]['decode_tokens']) for x in rows]) * 1000:.1f}",
                f"{np.mean([x['policies'][p]['bytes'] for x in rows]) / 2**20:.1f}",
                f"{np.mean([x['policies'][p].get('decode_time_cache_bytes_beyond_context', x['policies'][p]['shared_suffix_and_generated_fp16_bytes']) for x in rows]) / 2**20:.2f}",
                f"{np.mean([x['policies'][p].get('peak_allocated_policy') or 0 for x in rows]) / 2**30:.2f}"] for p in POL]
    write_table(r, "C4_systems", ["Cache", "Context prefill ms", "Question scoring ms", "Compression ms", "Suffix ms",
                                  "Decode ms / token", "Context cache MiB (serialized)", "Decode-time cache beyond context MiB",
                                  "Peak allocated GiB"], sysrows,
                "Means over prompts. Quantized caches are fake-quantized (dequantized FP16 in GPU memory) before ordinary "
                "SDPA, so allocated memory is not reduced; eviction happens after a full prefill, so the prefill peak is "
                "unchanged (see prefill_peak_allocated in per_prompt.jsonl.gz).")
    # reconstruction errors
    rec = []
    for p in ("int4_channelK_tokenV", "int4_tokenwise", "hybrid"):
        k = np.array([x["policies"][p]["k_rel_mse"] for x in rows])
        v = np.array([x["policies"][p]["v_rel_mse"] for x in rows])
        rec.append([PLAB[p], f"{k.mean():.2e}", f"{v.mean():.2e}", f"{k.mean(0).max():.2e} (layer {int(k.mean(0).argmax())})",
                    f"{v.mean(0).max():.2e} (layer {int(v.mean(0).argmax())})"])
    write_table(r, "C5_reconstruction", ["Cache", "K rel. MSE", "V rel. MSE", "Worst layer K", "Worst layer V"], rec,
                "Relative MSE of the dequantized context cache, mean over layers and prompts.")
    val = r.j("part_c", "validation.json")
    if val:
        cell = defaultdict(list)
        for x in rows:
            cell[(x["target_tokens"], x["position"], x["distractors"])].append(x["policies"]["fp16"]["exact_match"])
        write_table(r, "C6_validation_and_reference", ["Check / cell", "Value"],
                    [[k, v] for k, v in val["summary"].items()] +
                    [[f"FP16 accuracy {k}", f"{100 * np.mean(v):.0f}% (n={len(v)})"] for k, v in sorted(cell.items(), key=str)] +
                    [["encoded context length (2,048 / 4,096 targets)",
                      " / ".join(f"{np.mean([x['L'] for x in rows if x['target_tokens'] == t]):.0f}" for t in (2048, 4096))]])
    _c_bench(r)
    _c_figs(r, plt, rows)


def _c_bench(r):
    b = r.j("part_c", "bench_packed.json")
    if not b:
        return
    rows = []
    for x in b["rows"]:
        if "error" in x:
            rows.append([x["L"], x["batch"], "error", x["error"], "", "", "", ""])
            continue
        for key in ("channelK_tokenV_int4", "tokenwise_int4", "channelK_tokenV_int2"):
            rows.append([x["L"], x["batch"], key, f"{x[f'{key}_serialized_bytes'] / 2**20:.2f} / {x['fp16_bytes'] / 2**20:.2f}",
                         f"{x[f'{key}_encode']['median_ms']:.2f}", f"{x[f'{key}_decode']['median_ms']:.2f}",
                         f"{x['decode_step_fp16']['median_ms']:.2f}", f"{x[f'{key}_decode_step_with_unpack']['median_ms']:.2f}"])
    write_table(r, "C7_packed_cache_benchmark", ["L", "Batch", "Layout", "Serialized MiB / FP16 MiB", "Encode ms (all layers)",
                                                 "Unpack+dequantize ms", "Decode step FP16 ms", "Decode step with unpack ms"], rows,
                b["note"] + f"; median of {r.cfg['part_c']['bench']['reps']} CUDA-synchronised repetitions.")


def _c_figs(r, plt, rows):
    pos, dis = ["start", "middle", "end"], sorted({x["distractors"] for x in rows})
    fig, ax = plt.subplots(2, len(POL), figsize=(2.4 * len(POL), 4.6), squeeze=False)
    for i, t in enumerate((2048, 4096)):
        for j, p in enumerate(POL):
            M = np.full((3, len(dis)), np.nan)
            for a, ps in enumerate(pos):
                for b, dd in enumerate(dis):
                    sub = [x["policies"][p]["exact_match"] for x in rows if x["target_tokens"] == t and x["position"] == ps and x["distractors"] == dd]
                    if sub:
                        M[a, b] = np.mean(sub)
            im = ax[i, j].imshow(100 * M, vmin=0, vmax=100, cmap="viridis")
            for a in range(3):
                for b in range(len(dis)):
                    if np.isfinite(M[a, b]):
                        ax[i, j].text(b, a, f"{100 * M[a, b]:.0f}", ha="center", va="center", fontsize=8,
                                      color="white" if M[a, b] < 0.6 else "black")
            ax[i, j].set(xticks=range(len(dis)), xticklabels=dis, yticks=range(3), yticklabels=pos if j == 0 else [],
                         title=f"{PLAB[p]}\n~{t} tokens" if i == 0 else f"~{t} tokens")
            ax[i, j].grid(False)
            if i == 1:
                ax[i, j].set_xlabel("Distractors")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Exact match (%)")
    save_fig(r, fig, "FC1_accuracy_by_cell")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for p, mk in (("sink_recent", "o"), ("question_aware", "s"), ("hybrid", "^")):
        ev = [x["policies"][p].get("evidence_retained_mean", x["policies"][p].get("evidence_fp16_mean")) for x in rows]
        em = [x["policies"][p]["exact_match"] for x in rows]
        jit = np.random.default_rng(0).uniform(-0.04, 0.04, len(em))
        ax.scatter(ev, np.array(em) + jit, marker=mk, alpha=0.5, label=PLAB[p], s=14)
    ax.set(xlabel="Share of evidence tokens kept in FP16", ylabel="Exact match (jittered)", title="Evidence retention vs answer")
    ax.legend()
    save_fig(r, fig, "FC2_evidence_vs_accuracy")
    q = r.res / "part_c" / "qualitative"
    files = sorted(q.glob("*.npz")) if q.exists() else []
    if files:
        fig, ax = plt.subplots(len(files), 1, figsize=(11, 1.5 * len(files)), squeeze=False)
        for a, f in zip(ax[:, 0], files):
            z = np.load(f)
            L = int(z["evidence"].max()) + 1 if len(z["evidence"]) else 1
            for i, key in enumerate(("mask_sink_recent", "mask_question_aware", "mask_hybrid")):
                if key not in z.files:
                    continue
                m = z[key]
                if m.dtype != bool:
                    full = np.zeros((m.shape[0], m.shape[1], max(L, int(m.max()) + 1)), bool)
                    np.put_along_axis(full, m.astype(int), True, 2)
                    m = full
                frac = m.mean((0, 1))
                a.plot(np.arange(len(frac)), frac * 0.9 + i, lw=0.6)
            for e in z["evidence"]:
                a.axvline(e, color="r", lw=0.3, alpha=0.5)
            a.set(yticks=[0.45, 1.45, 2.45], yticklabels=["sink+recent", "question-aware", "hybrid FP16"], title=f.stem,
                  xlabel="Context position (red = gold evidence tokens)")
            a.grid(False)
        save_fig(r, fig, "FC3_selection_masks")
    meta = r.j("part_c", "prompts_meta.json")
    qual = {}
    for x in rows:
        if x["id"] in set(r.cfg["part_c"]["qualitative_ids"]):
            qual[x["id"]] = {"code": x["code"], **{p: x["policies"][p]["answer_text"] for p in POL}}
    if qual:
        write_table(r, "C8_qualitative_generations", ["Prompt", "Gold"] + [PLAB[p] for p in POL],
                    [[k, v["code"]] + [repr(v[p])[:40] for p in POL] for k, v in qual.items()],
                    "Fixed prompts chosen before results (config part_c.qualitative_ids); masks in figures/FC3.")


# ============================================================================ mechanism diagnostic (D*)
# Supplementary run (scripts/diagnose_task2.py). Figures FD* use the PA3 style (5.5 in wide, 8 pt text).
DCOL = {"dense": "#1f77b4", "window": "#2ca02c", "dilated": "#8c564b", "linear": "#d62728", "delta": "#9467bd"}
DARMS = {"fp16": "FP16 (reference)", "k_channel": "K channel-wise INT4 (main)", "k_token": "K token-wise INT4 (main)",
         "k_token_v_fp16": "K token-wise, V FP16", "v_token_k_fp16": "K FP16, V token-wise INT4",
         "k_token_int8": "K token-wise INT8", "k_token_center": "K token-wise, channel mean removed",
         "k_token_outlier2": "K token-wise, top-2 channels FP16", "k_token_outlier8": "K token-wise, top-8 channels FP16",
         "k_token_sink8": "K token-wise, first 8 tokens FP16", "k_token_layers01": "K token-wise in layers 0-1 only",
         "k_token_except01": "K token-wise except layers 0-1"}


def _pa3(plt):
    return plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                           "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False, "axes.grid": False,
                           "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
                           "savefig.dpi": 300, "figure.constrained_layout.use": True})


def _dlabel(name):
    mech, k, reg, s = name.split("_")
    return f"{MECH[mech]} {k} {reg}", s[1:]


def _dgroup(names):
    g = defaultdict(list)
    for n in names:
        g[_dlabel(n)[0]].append(n)
    return g


def d_b_tables(r, plt):
    d = r.j("diagnostics", "part_b_diagnostic.json")
    if not d:
        return "no part B diagnostic"
    tx, dy = d.get("taxonomy", {}), d.get("dynamics", {})
    if tx:
        rows = []
        for lab, ns in _dgroup(tx).items():
            accs = [tx[n]["acc"] for n in ns]
            split = np.std(accs) > 0.2                     # bimodal seeds -> one row per seed
            for sub in ([[n] for n in ns] if split else [ns]):
                t = [tx[n] for n in sub]
                sl = np.array([x["slot_acc_K16"] for x in t]).mean(0)
                seeds = "/".join(_dlabel(n)[1][-2:] for n in sub)
                rows.append([lab, seeds, f3(100 * np.mean([x["acc"] for x in t]), 1),
                             f3(100 * np.mean([x["wrong_in_context"] for x in t]), 1),
                             f3(100 * np.mean([x["not_in_context"] for x in t]), 1),
                             f3(100 * sl[:4].mean(), 1), f3(100 * sl[4:8].mean(), 1), f3(100 * sl[8:].mean(), 1),
                             f3(np.mean([x.get("wrong_slot_entropy_norm_K16", np.nan) for x in t]), 2)])
        write_table(r, "D1_failure_taxonomy", ["Model", "Seeds (63xx)", "Test acc %", "Wrong: other value of the sequence %",
                                               "Wrong: value not in the sequence %", "K16 slots 0-3 acc %", "K16 slots 4-7 acc %",
                                               "K16 slots 8-15 acc %", "Wrong-slot entropy (norm.)"], rows,
                    "All 9,216 test sequences (query-weighted). Slot = index j of the queried pair (key at 2j, value at 2j+1); "
                    "slot accuracy on the K = 16 test sequences, so every slot is measured under the same load of 16 pairs. "
                    "Wrong-slot entropy: entropy of the slot whose value a wrong answer copies, / ln 16 (1 = uniform over the "
                    "16 pairs). Configurations whose seeds differ by > 20 pp are listed per seed.")
    if dy:
        rows = []
        for lab, ns in _dgroup(dy).items():
            for n in ns:
                x = dy[n]
                la = x["loss_at"]
                rows.append([lab, _dlabel(n)[1], f3(x["ln_K_train"], 2), f3(la.get("500"), 2), f3(la.get("1000"), 2),
                             f3(la.get("2000"), 2), f3(la.get("4500"), 3), x["escape_update_acc50"] or "never",
                             f3(x["clipped_frac_mean"], 2)])
        write_table(r, "D2_training_dynamics", ["Model", "Seed", "ln K_train", "Train loss @500", "@1000", "@2000", "@4500",
                                                "First update with train acc >= 50%", "Share of updates clipped"], rows,
                    "Running training loss (50-update windows). Cross-entropy of a uniform guess over the K_train values of the "
                    "sequence = ln K_train; over all 64 values = ln 64 = 4.16. A run whose loss sits at ln K_train is "
                    "copying a value of the sequence without binding it to its key.")
    ci, ab = d.get("circuit", {}), d.get("ablation", {})
    if ci:
        rows = []
        for n, x in ci.items():
            b, ret, oth, fil = (np.array(x[k]) for k in ("bind", "ret", "other", "filler"))
            hb = int(b[0].argmax())
            bsl = np.array(x.get("bind_by_slot", np.full((2, 4, 16), np.nan)))[0, hb]
            rows.append([*_dlabel(n), f3(100 * tx.get(n, {}).get("acc", np.nan), 1), f"{f3(b[0].max(), 2)} (H{hb})",
                         " / ".join(f3(bsl[a:z].mean(), 2) for a, z in ((0, 4), (4, 8), (8, 16))),
                         f3(b[0].mean(), 2), f3(ret[1].mean(), 2), f3(oth[1].mean(), 3), f3(fil[1].mean(), 2)])
        write_table(r, "D3_attention_circuit", ["Model", "Seed", "Test acc %", "Layer-1 binding, best head",
                                                "Best head by slot 0-3 / 4-7 / 8-15", "Layer-1 binding, mean",
                                                "Layer-2 query -> matching value", "Layer-2 query -> each other value",
                                                "Layer-2 query -> filler (total)"], rows,
                    "Attention weights on K = 16 test sequences (ELU+1 models: the normalised kernel weights "
                    "phi(q)phi(k)/sum). Binding = weight from a value position to its own key (uniform causal attention would "
                    "give ~1/(2j+2)). Layer-2 entries are means over the four heads. Window/dilated models: queries with "
                    "distance <= 63 only. Delta-rule models have no attention matrix (see D5).")
    if ab:
        rows = []
        for n, x in ab.items():
            l1 = {k: v["acc"] for k, v in x.items() if k.startswith("L1H")}
            l2 = {k: v["acc"] for k, v in x.items() if k.startswith("L2H")}
            k1, k2 = min(l1, key=l1.get), min(l2, key=l2.get)
            bh = f"H{int(np.argmax(ci[n]['bind'][0]))}" if n in ci else "-"
            rows.append([*_dlabel(n), f3(100 * x["none"]["acc"], 1), f"{k1[2:]}: {f3(100 * l1[k1], 1)}", bh,
                         f3(100 * np.median(list(l1.values())), 1), f"{k2[2:]}: {f3(100 * l2[k2], 1)}",
                         f3(100 * x["L1_all"]["acc"], 1), f3(100 * x["L2_all"]["acc"], 1),
                         f3(100 * x["L1_all"]["wrong_in_context"], 1)])
        write_table(r, "D4_head_ablation", ["Model", "Seed", "Acc %", "Worst single layer-1 head: acc %",
                                            "Best binding head (D3)", "Median layer-1 head ablated: acc %",
                                            "Worst single layer-2 head: acc %", "All layer-1 heads: acc %",
                                            "All layer-2 heads: acc %", "All layer-1 heads: wrong but in-context %"], rows,
                    "Zero-ablation of one head's output (before the output projection) at test time; K = 16 test sequences "
                    "(window: near regime). Chance among the 16 values of a sequence = 6.25 %.")
    g = d.get("gates", {})
    if g.get("delta"):
        rows = []
        for n, x in g["delta"].items():
            bt = {k: np.array(v) for k, v in x["beta_by_type"].items()}
            a = x["acc"]
            rows.append([_dlabel(n)[1]] + [f"{f3(bt[t][0].mean(), 2)} / {f3(bt[t][1].mean(), 2)}"
                                           for t in ("key", "value", "query", "filler")] +
                        [f3(100 * a[k], 1) for k in ("none", "fillers_as_pairs", "uniform", "one", "pairs_as_fillers")])
        write_table(r, "D5_delta_gates", ["Seed", "beta key (L1 / L2)", "beta value", "beta query", "beta filler",
                                          "Acc %", "Fillers write like pairs", "Uniform beta", "beta = 1", "Pairs write like fillers"],
                    rows, "Delta rule, K = 16 test sequences. beta = write strength (sigmoid gate), mean over heads per layer. "
                    "Interventions replace beta at test time only, per head and sequence: fillers get the mean beta of the "
                    "key/value positions; every position gets the sequence mean; beta = 1 everywhere; key/value positions "
                    "get the mean filler beta.")
    if g.get("linear_filler_mask"):
        rows = [[*_dlabel(n), f3(100 * x["none"], 1), f3(100 * x["fillers_not_written"], 1)]
                for n, x in g["linear_filler_mask"].items()]
        write_table(r, "D5b_linear_filler_control", ["Model", "Seed", "Acc %", "Acc % with filler tokens not written"], rows,
                    "ELU+1 linear attention, K = 16 test sequences: at test time filler positions contribute nothing to the "
                    "state or the normaliser (both layers). If failure were interference from fillers, accuracy would rise.")
    rel = d.get("relocation", {})
    if rel:
        rows = []
        shifts = sorted({int(s) for x in rel.values() for s in x["by_shift"]})
        for lab, ns in _dgroup(rel).items():
            accs = np.array([[rel[n]["by_shift"][str(s)]["acc"] for s in shifts] for n in ns])
            split = np.std(accs[:, 0]) > 0.2
            for sub in ([[i] for i in range(len(ns))] if split else [list(range(len(ns)))]):
                rows.append([lab, "/".join(_dlabel(ns[i])[1][-2:] for i in sub)] + [f3(100 * accs[sub, j].mean(), 1)
                                                                                    for j in range(len(shifts))])
        write_table(r, "D6_pair_block_relocation", ["Model", "Seeds (63xx)"] + [f"slots {s}-{s + 3}" for s in shifts], rows,
                    "K = 4 test sequences with the block of 4 pairs moved from slots 0-3 to slots s..s+3 (positions before "
                    "the block become filler; queries that would fall inside or before the block are dropped). The number "
                    "of pairs stays 4; only where the block sits changes, and the layout leaves the training distribution "
                    "(filler before the pairs). Window models: one-hop-reachable queries only.")
    rt = d.get("retrain", {})
    if rt.get("arms"):
        tx_all = tx
        ref = {"window_mixed_reach63": "window_K16_mixed", "window_near_thinned": "window_K16_near",
               "dilated_near": "dilated_K16_mixed", "dilated_mixed_reach63": "dilated_K16_mixed",
               "linear_mixed_extra_seeds": "linear_K16_mixed"}
        rows = []
        for arm, x in rt["arms"].items():
            runs_ = x["runs"]
            base = [n for n in tx_all if n.startswith(ref[arm])]
            rows.append([arm, f"{x['mech']} K{x['K']} {x['regime']}", x["rule"], f3(x["supervised_queries_per_seq"], 2),
                         " / ".join(f3(100 * v["test_acc_all"], 1) for v in runs_.values()),
                         " / ".join(f3(100 * v["acc_reachable_d_le_63"], 1) for v in runs_.values()),
                         " / ".join(str(v["escape_update_acc50"] or "never") for v in runs_.values()),
                         " / ".join(f3(100 * tx_all[n].get("acc_reachable_d_le_63", tx_all[n]["acc"]), 1) for n in base)])
        write_table(r, "D7_retraining_arms", ["Arm", "Training data", "Supervised queries", "Supervised per sequence",
                                              "Test acc % per seed", "Acc % on queries with d <= 63", "Escape update",
                                              "Main-run reference (d <= 63 for window/dilated)"], rows,
                    f"Same pools, recipe (4,500 updates) and seeds as the main run (linear: five new seeds). reach = only "
                    f"queries with distance <= 63 supervised; thin = each query kept with probability "
                    f"{f3(rt.get('reach_fraction_mixed_K16'), 3)} (the supervised share under 'reach' on mixed data). "
                    f"Reference column: the main-run model the arm is compared with ({', '.join(sorted(set(ref.values())))}).")
    return "ok"


def d_c_tables(r, plt):
    d = r.j("diagnostics", "part_c_diagnostic.json")
    if not d:
        return "no part C diagnostic"
    rows = []
    for arm, s in d["arms"].items():
        at = s.get("attn_by_layer")
        rows.append([DARMS.get(arm, arm), f3(100 * s["exact_match"], 1), f3(s["gold_nll_median"], 3), f3(s.get("post_prefill_kl"), 4),
                     f"{s['k_rel_mse']:.2e}", f3(s["extra_fp16_bytes_mean"] / 2 ** 20, 2),
                     f3(np.mean(at["kl"]), 3) if at else "-", f3(np.mean(at["evidence_mass"]), 3) if at else "-",
                     f3(np.mean(at["token0_mass"]), 3) if at else "-"])
    fp = d["arms"]["k_channel"].get("attn_by_layer")
    write_table(r, "D8_tokenwise_key_interventions", ["Context cache", "Exact match %", "Gold NLL (median)", "Post-prefill KL",
                                                      "K rel. MSE", "Extra FP16 MiB", "Attention KL vs FP16 (mean layer)",
                                                      "Evidence mass", "Token-0 mass"], rows,
                f"{d['n_prompts']} prompts; only the context cache differs (suffix and generated tokens FP16; greedy cap "
                f"{d['max_new_tokens']} tokens, FP16 answers take 6). V is token-wise INT4 unless stated. Attention columns: "
                f"FP16 suffix queries against the arm's context keys, mean over layers, heads and suffix positions (FP16: "
                f"evidence mass {f3(np.mean(fp['evidence_mass_fp16']), 3) if fp else '-'}, token-0 mass "
                f"{f3(np.mean(fp['token0_mass_fp16']), 3) if fp else '-'}). Extra FP16 MiB: bytes the arm keeps beyond the "
                f"INT4 layout (diagnostic arms are not budget-matched).")
    ks = d["key_stats_by_layer"]
    write_table(r, "D9_key_channel_statistics", ["Layer", "max |channel mean| / median channel SD",
                                                 "Token-wise INT4 step / median channel SD",
                                                 "Ordinary channels with SD < step/2 %", "Top-8 mean |K| / median",
                                                 "Top-8 channel overlap with prompt 1 (Jaccard)"],
                [[l, f3(ks["offset_ratio"][l], 1), f3(ks["step_over_sd"][l], 2), f3(100 * ks["collapsed_share"][l], 1),
                  f3(ks["top8_mean_abs_over_median"][l], 1), f3(ks["top8_jaccard_vs_first_prompt"][l], 2)]
                 for l in range(len(ks["offset_ratio"]))],
                "Context keys after RoPE, FP16 model, mean over KV heads and prompts. Step = (max - min)/15 of a 64-channel "
                "token group (median over tokens and groups). A channel whose spread is below half a step is mapped to <= 2 "
                "codes, i.e. its token-to-token variation is erased.")
    return "ok"


def d_figs(r, plt):
    d = r.j("diagnostics", "part_b_diagnostic.json") or {}
    dc = r.j("diagnostics", "part_c_diagnostic.json")
    made = []
    tx, runs = d.get("taxonomy", {}), _runs(r)
    with _pa3(plt):
        if tx and runs:
            fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3))
            for (mech, K), ls in [(("dense", 4), "-"), (("dense", 8), "-"), (("dense", 16), "-"), (("linear", 4), "--")]:
                ns = [n for n in tx if n.startswith(f"{mech}_K{K}_mixed")]
                if not ns:
                    continue
                sl = 100 * np.array([tx[n]["slot_acc_K16"] for n in ns]).mean(0)
                ax[0].plot(range(16), sl, ls, marker="o", ms=2.5, color={4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c"}[K],
                           label=f"{'Dense' if mech == 'dense' else 'Linear'}, K_train={K}")
            ax[0].axhline(100 / 16, color="0.5", lw=0.8, ls=":")
            ax[0].set(xlabel="Slot of the queried pair (K = 16 test sequences)", ylabel="Test accuracy (%)",
                      title="(a) Accuracy by pair slot", ylim=(0, 105), xticks=range(0, 16, 2))
            ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.5))
            for n, c in [("linear_K16_mixed_s6304", "#d62728"), ("linear_K16_mixed_s6305", "#ff9896"),
                         ("linear_K16_mixed_s6306", "#7f0000"), ("delta_K16_mixed_s6304", "#9467bd"),
                         ("dense_K16_mixed_s6304", "#1f77b4")]:
                if n in runs:
                    tc = runs[n]["train_curve"]
                    ax[1].plot([t["update"] for t in tc], [t["loss"] for t in tc], color=c,
                               label=f"{MECH[n.split('_')[0]]} s{n[-4:]}")
            ax[1].axhline(np.log(16), color="0.3", lw=0.8, ls="--")
            ax[1].text(4500, np.log(16) + 0.1, "ln 16", ha="right", va="bottom", fontsize=7)
            ax[1].set(xlabel="Optimizer update", ylabel="Training loss (nats)", title="(b) K = 16 mixed training", ylim=(0, 4.4))
            ax[1].legend(loc="lower left")
            save_fig(r, fig, "FD1_slot_accuracy_and_plateau")
            made.append("FD1")
        ci, ab = d.get("circuit", {}), d.get("ablation", {})
        if ci and ab:
            fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
            for n, x in ci.items():
                mech = n.split("_")[0]
                ax[0].scatter(max(x["bind"][0]), 100 * tx.get(n, {}).get("acc", np.nan), s=10, color=DCOL[mech],
                              label=MECH[mech])
            h, l = ax[0].get_legend_handles_labels()
            u = dict(zip(l, h))
            ax[0].legend(u.values(), u.keys(), loc="lower right")
            ax[0].set(xlabel="Best layer-1 binding weight", ylabel="Test accuracy (%)",
                      title="(a) Binding head vs accuracy (42 runs)", xlim=(0, 1), ylim=(0, 105))
            names = list(ab)
            xs = np.arange(len(names))
            none = [100 * ab[n]["none"]["acc"] for n in names]
            worst1 = [100 * min(v["acc"] for k, v in ab[n].items() if k.startswith("L1H")) for n in names]
            med1 = [100 * np.median([v["acc"] for k, v in ab[n].items() if k.startswith("L1H")]) for n in names]
            all2 = [100 * ab[n]["L2_all"]["acc"] for n in names]
            for off, vals, c, lab in [(-0.27, med1, "#2ca02c", "median layer-1 head"),
                                      (0.0, worst1, "#d62728", "worst layer-1 head"), (0.27, all2, "#1f77b4", "all layer-2 heads")]:
                ax[1].bar(xs + off, vals, 0.27, color=c, label=lab)
            ax[1].axhline(100 / 16, color="0.5", lw=0.8, ls=":")
            groups = defaultdict(list)
            for i, n in enumerate(names):
                groups[n.split("_")[0] + (" near" if n.startswith("dense_K16_near") else "")].append(i)
            short = {"dense": "dense", "dense near": "d. near", "window": "window", "linear": "lin.", "delta": "delta"}
            for gname, idx in groups.items():
                ax[1].text(np.mean(idx), -0.16, short.get(gname, gname), transform=ax[1].get_xaxis_transform(),
                           ha="center", va="top", fontsize=6)
            ax[1].set_xticks(xs)
            ax[1].set_xticklabels([n[-2:] for n in names], fontsize=6)
            ax[1].set(ylabel="Test accuracy (%)", title="(b) Zero-ablation of heads (K = 16)", ylim=(0, 130), yticks=range(0, 101, 20))
            ax[1].legend(loc="upper center", ncol=3, handlelength=1, columnspacing=0.8)
            save_fig(r, fig, "FD2_circuit_and_ablation")
            made.append("FD2")
        rt, g = d.get("retrain", {}), d.get("gates", {})
        if rt.get("arms") or g.get("delta"):
            fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3))
            if rt.get("arms"):
                labs = []
                for i, (arm, x) in enumerate(rt["arms"].items()):
                    v = [100 * y["acc_reachable_d_le_63"] if x["mech"] in ("window", "dilated") else 100 * y["test_acc_all"]
                         for y in x["runs"].values()]
                    ax[0].scatter([i] * len(v), v, s=12, color=DCOL[x["mech"]], zorder=3)
                    labs.append(arm.replace("_", " ").replace("extra seeds", "5 new seeds"))
                base = {"window": "window_K16_mixed", "dilated": "dilated_K16_mixed", "linear": "linear_K16_mixed"}
                for i, (arm, x) in enumerate(rt["arms"].items()):
                    ns = [n for n in tx if n.startswith(base.get(x["mech"], "~"))]
                    if ns and "near" not in arm:
                        v = [100 * tx[n].get("acc_reachable_d_le_63", tx[n]["acc"]) for n in ns]
                        ax[0].scatter([i + 0.2] * len(v), v, s=10, marker="x", color="0.4", zorder=3)
                ax[0].set_xticks(range(len(labs)))
                ax[0].set_xticklabels(labs, rotation=25, ha="right", fontsize=6)
                ax[0].set(ylabel="Test accuracy (%)", title="(a) Retraining arms (x = main run)", ylim=(-3, 105))
            if g.get("delta"):
                modes = ["none", "fillers_as_pairs", "uniform", "one", "pairs_as_fillers"]
                ml = ["learned", "fillers\nlike pairs", "uniform", "beta = 1", "pairs\nlike fillers"]
                vals = np.array([[100 * x["acc"][m] for m in modes] for x in g["delta"].values()])
                ax[1].bar(range(len(modes)), vals.mean(0), 0.6, color=DCOL["delta"])
                for j in range(len(modes)):
                    ax[1].scatter([j] * len(vals), vals[:, j], s=6, color="k", zorder=3)
                ax[1].set_xticks(range(len(modes)))
                ax[1].set_xticklabels(ml, fontsize=6)
                ax[1].set(ylabel="Test accuracy (%)", title="(b) Delta rule: write-gate interventions", ylim=(0, 105))
            save_fig(r, fig, "FD3_retraining_and_gates")
            made.append("FD3")
        if dc:
            fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
            arms = [a for a in dc["arms"] if a != "fp16"]
            em = [100 * dc["arms"][a]["exact_match"] for a in arms]
            ax[0].barh(range(len(arms)), em, color=["#1f77b4" if a == "k_channel" else "#d62728" if a == "k_token" else "0.55"
                                                    for a in arms])
            ax[0].set_yticks(range(len(arms)))
            ax[0].set_yticklabels([DARMS[a] for a in arms], fontsize=6.5)
            ax[0].invert_yaxis()
            ax[0].set(xlabel="Exact match (%)", title="(a) Interventions on token-wise keys", xlim=(0, 105))
            for a, c in [("k_channel", "#1f77b4"), ("k_token", "#d62728"), ("k_token_outlier8", "#2ca02c"),
                         ("k_token_center", "#ff7f0e"), ("k_token_sink8", "#8c564b")]:
                at = dc["arms"][a].get("attn_by_layer")
                if at:
                    ax[1].plot(range(len(at["kl"])), np.maximum(at["kl"], 1e-6), color=c, label=DARMS[a].replace("K token-wise, ", ""))
            ax[1].set_yscale("log")
            from matplotlib.ticker import MaxNLocator
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].set(xlabel="Layer", ylabel="Attention KL vs FP16 (nats)", title="(b) Attention distortion by layer")
            ax[1].legend(fontsize=6, loc="best")
            save_fig(r, fig, "FD4_tokenwise_key_mechanism")
            made.append("FD4")
    return ", ".join(made) or "no diagnostic figures"


FOLLOW_LAB = {"window_near_sparse_input": "window, near, 20.7% of queries in input",
              "window_mixed_far_removed": "window, mixed, far query tokens removed",
              "dilated_near_sparse_input": "dilated, near, 20.7% of queries in input",
              "dilated_mixed_far_removed": "dilated, mixed, far query tokens removed",
              "window_mixed_long": "window, mixed (main condition), 13,500 updates"}


def d_followup(r, plt):
    """D10 / FD5: follow-up retraining arms (what in mixed data blocks window/dilated learning)."""
    fdir = r.res / "diagnostics" / "followup"
    arms = {a: r.j("diagnostics", "followup", f"{a}.json") for a in FOLLOW_LAB} if fdir.exists() else {}
    arms = {a: x for a, x in arms.items() if x}
    if not arms:
        return "no follow-up arms"
    rows = []
    for a, x in arms.items():
        st, rn = x["input_stats"], x["runs"]
        rows.append([FOLLOW_LAB[a], x["updates"], f3(st["supervised_per_seq"], 2),
                     f3(st["other_queries_in_window_of_reachable"], 2), f3(st["far_query_tokens_per_seq"], 1),
                     " / ".join(f3(100 * v["acc_reachable_d_le_63"], 1) for v in rn.values()),
                     " / ".join(str(v["escape_update_acc50"] or "never") for v in rn.values())])
    write_table(r, "D10_mixed_data_followup", ["Arm", "Updates", "Supervised queries / seq",
                                               "Other queries in the window of a reachable query",
                                               "Far query tokens / seq", "Acc % on d <= 63 per seed", "Escape update"], rows,
                "Removed queries are replaced by random filler in the input and are unsupervised. Input statistics from "
                "the first 4,000 training sequences. Earlier arms (D7), same statistics: window/dilated mixed with "
                "reach-63 supervision 3.3 supervised / ~1.6 other queries / ~12.7 far tokens (never learned); window near "
                "thinned 3.3 / ~7.5 / 0 (learned in 2 of 3 seeds). 13,500-update arm: same recipe with the linear LR "
                "decay stretched over the longer budget.")
    with _pa3(plt):
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
        short = {"window_near_sparse_input": "W near\nsparse", "window_mixed_far_removed": "W mixed\nfar removed",
                 "dilated_near_sparse_input": "D near\nsparse", "dilated_mixed_far_removed": "D mixed\nfar removed"}
        prev = ((r.j("diagnostics", "part_b_diagnostic.json") or {}).get("retrain") or {}).get("arms", {})
        cols = [("W near\nthinned", prev.get("window_near_thinned"), "0.6"),
                ("W mixed\nreach-63", prev.get("window_mixed_reach63"), "0.6")]
        cols += [(short[a], arms[a], "#1f77b4" if a.startswith("window") else "#8c564b") for a in short if a in arms]
        cols = [c for c in cols if c[1]]
        for i, (lab, x, col) in enumerate(cols):
            ys = [100 * v["acc_reachable_d_le_63"] for v in x["runs"].values()]
            ax[0].scatter([i] * len(ys), ys, s=14, color=col, zorder=3)
        ax[0].set_xticks(range(len(cols)))
        ax[0].set_xticklabels([c[0] for c in cols], fontsize=5.5)
        ax[0].set(ylabel="Accuracy on d <= 63 (%)", ylim=(-3, 105), title="(a) Input edits (3 seeds; grey = D7)")
        if "window_mixed_long" in arms:
            for s, v in arms["window_mixed_long"]["runs"].items():
                c = np.array(v["loss_curve"])
                if len(c):
                    ax[1].plot(c[:, 0] / 1000, c[:, 1], label=f"seed {s}")
            ax[1].axvline(arms["window_mixed_long"]["updates"] / 3000, color="0.5", ls="--", lw=0.8)   # main budget
            ax[1].set(xlabel="Update (thousands)", ylabel="Training loss (all 16 queries)", title="(b) Window, mixed, 3x budget")
            ax[1].legend(fontsize=6)
        save_fig(r, fig, "FD5_mixed_data_followup")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    r = R(Path(a.results), load_config(a.config))
    plt = setup_mpl()
    failed = []
    for fn in (a_tables, a_bench, b_tables, b_figs, b_resources, c_tables, d_b_tables, d_c_tables, d_figs, d_followup):
        try:
            msg = fn(r, plt)
            print(f"{fn.__name__}: {msg or 'ok'}")
        except Exception:
            failed.append(fn.__name__)
            print(f"{fn.__name__}: FAILED\n{traceback.format_exc()}")
    print("failed:", failed or "none")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
