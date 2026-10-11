#!/usr/bin/env python3
"""Figures and tables for the COMP90083 Project 2 report from BehaviorSpace CSVs.

Usage (defaults: Assignment_2/results in, Assignment_2/figures out)
    python make_figures.py [--results DIR] [--out DIR]

Reads BehaviorSpace table CSVs from affordable_rental_allocation_v2.nlogox, using its
current variable names. Each experiment is found by the name on line 3 of its CSV,
not by file name; only the experiments below are loaded, and each figure is made
only if its experiments are present:

    pilot-v4, screening-check    -> fig2_calibration.pdf (P1 to P3 from the pilot,
                                    P4 screen off against on from screening-check)
    theory-test                  -> fig5_theory.pdf (search-theory ladder)
    stage1-supply, -window,      -> fig3_stage1_factors.pdf, fig4_stage1_priority.pdf
      -priority
    stage2-pooled                -> fig6_stage2.pdf (time series, boxplot, ring heat map,
                                    E by supply level), fig6e_budget.pdf if stage2-fixed-budget
    stage3-counter-mechanisms,   -> fig7_mechanisms.pdf (paired differences, 95% bootstrap)
      stage3-unit-decomposition
    sensitivity-*                -> tab_sensitivity.tex (S-bar per parameter per condition)
                                    and fig8_ranking.pdf for the most sensitive parameter

Conventions: one colour per condition everywhere; vector PDF; sentence-case
titles; every panel labelled with units and the number of runs.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- style ----
COND_ORDER = ["none", "concentrated", "demand-proportional", "dispersed"]
# legend names are the values of the model's strategy chooser
COND_LABEL = {"none": "none", "concentrated": "concentrated",
              "demand-proportional": "demand-proportional", "dispersed": "dispersed"}
COND_COLOR = {"none": "#7f7f7f", "concentrated": "#c0392b",
              "demand-proportional": "#2471a3", "dispersed": "#1e8449"}
KEY = "displaced-share"
STRESS = "mean-stress-duration"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 120, "savefig.bbox": "tight",
                     "figure.constrained_layout.use": True})


# ------------------------------------------------------------------ load ----
NEEDED = {"pilot-v4", "screening-check", "theory-test", "stage1-supply", "stage1-window", "stage1-priority",
          "stage2-pooled", "stage2-fixed-budget", "stage3-counter-mechanisms", "stage3-unit-decomposition"}


def experiment_name(path):
    with open(path, encoding="utf-8") as f:
        head = [next(f) for _ in range(3)]
    return head[2].strip().strip('"')


def load_bs(path):
    """Read a BehaviorSpace table CSV. Returns (experiment name, DataFrame)."""
    name = experiment_name(path)
    df = pd.read_csv(path, skiprows=6)
    df.columns = [c.strip("[]") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip('"')
    return name, df


def load_all(results_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        try:
            if not (experiment_name(p) in NEEDED or experiment_name(p).startswith("sensitivity")):
                continue
            name, df = load_bs(p)
        except Exception as e:  # noqa: BLE001
            print(f"skip {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        out.setdefault(name, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def final(df):
    """Rows at the last recorded step of each run."""
    if "step" not in df.columns:
        return df
    return df.loc[df.groupby("run number")["step"].idxmax()]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def paired_diff(df, metric, base_mask, cond_mask):
    """Within-seed differences cond - base, with 95 per cent bootstrap interval."""
    b = df[base_mask].set_index("seed")[metric]
    c = df[cond_mask].set_index("seed")[metric]
    common = b.index.intersection(c.index)
    d = (c.loc[common] - b.loc[common]).values
    lo, hi = boot_ci(d)
    return d.mean(), lo, hi, len(d)


def conds_in(df):
    return [c for c in COND_ORDER if c in set(df["strategy"])]


# --------------------------------------------------------------- figures ----
def fig_calibration(data, out):
    """Calibration panels: P1 to P3 from the final pilot (pilot-v4, called pilot-v26 before
    1 October); P4 from screening-check, screen off against on on the calibrated model."""
    if "pilot-v4" not in data:
        return
    raw = data["pilot-v4"]
    df = final(raw)
    early = raw[raw["step"].between(1, 4)].groupby("run number")["stress-prevalence"].mean()
    df = df.set_index("run number"); df["p1_early"] = early; df = df.reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # (a) P1: first-year stress and quarter-40 band, by move probability and hazard
    ax = axes[0, 0]
    g = df.groupby(["bg-target-move-prob", "termination-hazard-base"])[["p1_early", "stress-prevalence", "displaced-share"]].mean()
    labels = [f"{i[0]:g}\n{i[1]:g}" for i in g.index]
    x = np.arange(len(g))
    ax.bar(x - 0.27, g["p1_early"], 0.27, label="stress prevalence, quarters 1 to 4")
    ax.bar(x, g["stress-prevalence"], 0.27, label="stress prevalence, quarter 40")
    ax.bar(x + 0.27, g["displaced-share"], 0.27, label="displacements per household at quarter 40")
    ax.axhspan(0.85, 0.95, color="tab:blue", alpha=0.08); ax.axhspan(0.20, 0.60, color="tab:green", alpha=0.08)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_xlabel("bg-target-move-prob (top), termination-hazard-base (bottom)", fontsize=8); ax.set_ylabel("share of households")
    ax.set_title("(a) P1: stress prevalence and displacements per household (shaded targets)", fontsize=9); ax.set_ylim(0, 1.45); ax.legend(fontsize=6.5, loc="upper left", ncol=1)
    # (b) P2: forced share heat map, move probability x hazard
    ax = axes[0, 1]
    pv = df.pivot_table(index="termination-hazard-base", columns="bg-target-move-prob", values="forced-move-share", aggfunc="mean")
    im = ax.imshow(pv.values, aspect="auto", cmap="magma", origin="lower", vmin=0.2, vmax=0.45)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns); ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xlabel("bg-target-move-prob"); ax.set_ylabel("termination-hazard-base"); ax.set_title("(b) P2: forced-move share (target 0.20 to 0.35)")
    fig.colorbar(im, ax=ax)
    # (c) P3: Rent Increase by Tier against non-target-frac-mid
    ax = axes[1, 0]
    col = "non-target-frac-mid" if "non-target-frac-mid" in df.columns and df["non-target-frac-mid"].nunique() > 1 else "non-target-frac-high"
    g = df.groupby(col)[["rent-increase-high", "rent-increase-mid", "rent-increase-low"]].mean()
    for c, lab in zip(g.columns, ["high", "medium", "low"]):  # the pens of the model's plot
        ax.plot(g.index, g[c], marker="o", label=lab)
    ax.set_xlabel(col); ax.set_ylabel("rent / initial at quarter 40")
    ax.set_title("(c) P3: Rent Increase by Tier (target high > medium > low)"); ax.legend(title="Rent Increase by Tier")
    # (d) P4: screen off against on
    ax = axes[1, 1]
    src = data.get("screening-check")
    if src is not None:  # calibrated rows only: no new supply, competition on, landlord screening off
        src = src[(src["strategy"] == "none") & (src["non-target-competition?"].astype(str).str.lower() == "true")
                  & (src["landlord-screening?"].astype(str).str.lower() == "false")]
    if src is not None and src["max-rent-to-income"].nunique() > 1:
        d4 = final(src).groupby("max-rent-to-income")[["access-lost-displaced", "access-lost-not-displaced"]].mean().sort_index(ascending=False)
        x = np.arange(len(d4))
        ax.bar(x - 0.2, d4["access-lost-displaced"], 0.4, label="access-lost-displaced"); ax.bar(x + 0.2, d4["access-lost-not-displaced"], 0.4, label="access-lost-not-displaced")
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" + (" (screen off)" if v >= 1 else "") for v in d4.index]); ax.set_xlabel("max-rent-to-income")
        ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("access lost (access at setup minus current)"); ax.set_title("(d) P4: access-lost, ever displaced against the rest"); ax.legend()
    n = df["seed"].nunique()
    fig.suptitle(f"Calibration with strategy none: pilot-v4, mean over {n} seeds per cell", y=1.02)
    fig.savefig(os.path.join(out, "fig2_calibration.pdf")); plt.close(fig); print("fig2_calibration.pdf")


def fig_theory(data, out):
    """Search-theory ladder: E by strategy under random, cheapest and utility rules."""
    if "theory-test" not in data:
        return
    df = final(data["theory-test"]); n = df["seed"].nunique()
    rules = [r for r in ["random", "cheapest", "utility"] if r in set(df["search-rule"])]
    conds = conds_in(df)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, lab in zip(axes, [KEY, "stress-prevalence"], ["displacements per household at quarter 40", "stress prevalence at quarter 40"]):
        width = 0.8 / len(conds)
        for i, c in enumerate(conds):
            vals = [df[(df["search-rule"] == r) & (df["strategy"] == c)][metric] for r in rules]
            pos = np.arange(len(rules)) + (i - (len(conds) - 1) / 2) * width
            ax.bar(pos, [v.mean() for v in vals], width * 0.9, yerr=[v.std() for v in vals], color=COND_COLOR[c], alpha=0.7, label=COND_LABEL[c], capsize=2)
        ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules); ax.set_xlabel("search-rule")
        ax.set_ylabel(lab)
    axes[0].set_title(f"(a) displaced-share, mean and SD over {n} seeds"); axes[1].set_title("(b) stress-prevalence"); axes[0].legend()
    fig.savefig(os.path.join(out, "fig5_theory.pdf")); plt.close(fig); print("fig5_theory.pdf")


def _box_by_level(ax, df, level_col, metric, title, xlabel):
    conds = [c for c in conds_in(df) if c != "none"]
    levels = sorted(df[level_col].unique())
    width = 0.8 / max(len(conds), 1)
    for i, c in enumerate(conds):
        sub = df[df["strategy"] == c]
        data = [sub[sub[level_col] == l][metric].values for l in levels]
        pos = np.arange(len(levels)) + (i - (len(conds) - 1) / 2) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=COND_COLOR[c], alpha=0.6)
        for med in bp["medians"]:
            med.set(color="black")
        ax.plot([], [], color=COND_COLOR[c], lw=6, alpha=0.6, label=COND_LABEL[c])
    if "none" in set(df["strategy"]):
        ax.axhline(df[df["strategy"] == "none"][metric].median(), color=COND_COLOR["none"], ls="--", label="none, median")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
    ax.set_xlabel(xlabel); ax.set_ylabel("displacements per household at quarter 40"); ax.set_title(title); ax.legend()


def fig_stage1(data, out):
    have = [k for k in ("stage1-supply", "stage1-window") if k in data]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(5 * len(have), 3.4), squeeze=False)
        for ax, k in zip(axes[0], have):
            df = final(data[k]); n = df["seed"].nunique()
            if k == "stage1-supply":
                _box_by_level(ax, df, "govt-homes", KEY, f"(a) govt-homes, {n} seeds", "govt-homes (% of Households)")
            else:
                _box_by_level(ax, df, "govt-homes-build-time", KEY, f"(b) govt-homes-build-time, {n} seeds", "govt-homes-build-time (Ticks)")
        fig.savefig(os.path.join(out, "fig3_stage1_factors.pdf")); plt.close(fig); print("fig3_stage1_factors.pdf")
    if "stage1-priority" in data:
        df = final(data["stage1-priority"]); n = df["seed"].nunique()
        fig, ax = plt.subplots(figsize=(6, 3.4))
        _box_by_level(ax, df, "priority-rule", KEY, f"priority-rule per strategy, {n} seeds", "priority-rule")
        fig.savefig(os.path.join(out, "fig4_stage1_priority.pdf")); plt.close(fig); print("fig4_stage1_priority.pdf")


def fig_stage2(data, out, ref_supply=10):
    if "stage2-pooled" not in data:
        return
    all_df = data["stage2-pooled"]
    agents = all_df[all_df.get("mean-field?", "false").astype(str).str.lower() == "false"] if "mean-field?" in all_df else all_df
    ref = agents[agents["govt-homes"] == ref_supply] if "govt-homes" in agents else agents
    fin = final(ref); n = fin["seed"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # (a) time series with IQR band
    ax = axes[0, 0]
    for c in conds_in(ref):
        g = ref[ref["strategy"] == c].groupby("step")[KEY]
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.values, color=COND_COLOR[c], label=COND_LABEL[c])
        ax.fill_between(med.index, lo.values, hi.values, color=COND_COLOR[c], alpha=0.2)
    ax.set_xlabel("quarter"); ax.set_ylabel("displacements per household"); ax.set_title(f"(a) Displacement and stress: displacements per household, median and IQR, {n} seeds", fontsize=8); ax.legend()
    # (b) boxplot at quarter 40 with the mean-field? on marker
    ax = axes[0, 1]
    conds = conds_in(fin)
    bp = ax.boxplot([fin[fin["strategy"] == c][KEY].values for c in conds], patch_artist=True, showfliers=False)
    for b, c in zip(bp["boxes"], conds):
        b.set(facecolor=COND_COLOR[c], alpha=0.6)
    if "mean-field?" in all_df:
        mf = final(all_df[(all_df["mean-field?"].astype(str).str.lower() == "true") & (all_df["govt-homes"] == ref_supply)])
        for i, c in enumerate(conds):
            v = mf[mf["strategy"] == c][KEY]
            if len(v):
                ax.plot(i + 1, float(v.mean()), marker="D", color="black", ms=6, label="mean-field? on" if i == 0 else None)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=15)
    ax.set_ylabel("displacements per household at quarter 40"); ax.set_title("(b) displaced-share at quarter 40")
    # (c) stress-share-nb by neighbourhood at quarter 40
    ax = axes[1, 0]
    nb = [c for c in fin.columns if c.startswith("stress-share-nb")]
    if nb:
        nb = sorted(nb, key=lambda s: int(s.split()[-1]))
        mat = np.array([fin[fin["strategy"] == c][nb].mean().values for c in conds])
        im = ax.imshow(mat, aspect="auto", cmap="inferno", vmin=float(mat.min()), vmax=float(mat.max()))
        ax.set_yticks(range(len(conds))); ax.set_yticklabels([COND_LABEL[c] for c in conds])
        ax.set_xticks(range(0, 20, 2)); ax.set_xlabel("neighbourhood id (7 to 12 high tier)")
        ax.set_title("(c) stress-share-nb at quarter 40"); fig.colorbar(im, ax=ax, label="stress-share-nb")
        ax.axvspan(6.5, 12.5, color="white", alpha=0.15)
    # (d) E by supply level
    ax = axes[1, 1]
    if "govt-homes" in agents:
        f_all = final(agents)
        for c in conds_in(f_all):
            g = f_all[f_all["strategy"] == c].groupby("govt-homes")[KEY]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.errorbar(med.index.values, med.values, yerr=np.vstack([(med - q1).values, (q3 - med).values]),
                        marker="o", color=COND_COLOR[c], label=COND_LABEL[c], capsize=3)
        ax.set_xlabel("govt-homes (% of Households)"); ax.set_ylabel("displacements per household at quarter 40")
        ax.set_title("(d) displaced-share by govt-homes, median and IQR"); ax.legend()
    fig.savefig(os.path.join(out, "fig6_stage2.pdf")); plt.close(fig); print("fig6_stage2.pdf")
    # (e) fixed budget beside fixed supply
    if "stage2-fixed-budget" in data:
        fb = final(data["stage2-fixed-budget"]); fb = fb[fb["govt-homes"] == ref_supply] if "govt-homes" in fb else fb
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, (lab, d) in zip(axes, [("fixed-budget? off", fin), ("fixed-budget? on", fb)]):
            cs = conds_in(d)
            bp = ax.boxplot([d[d["strategy"] == c][KEY].values for c in cs], patch_artist=True, showfliers=False)
            for b, c in zip(bp["boxes"], cs):
                b.set(facecolor=COND_COLOR[c], alpha=0.6)
            units = d.groupby("strategy")["units-delivered-total"].mean() if "units-delivered-total" in d else None
            ax.set_xticks(range(1, len(cs) + 1))
            ax.set_xticklabels([COND_LABEL[c] + (f"\n{units[c]:.0f} units delivered" if units is not None else "") for c in cs], fontsize=7)
            ax.set_ylabel("displacements per household at quarter 40"); ax.set_title(f"{lab}, {d['seed'].nunique()} seeds")
        fig.savefig(os.path.join(out, "fig6e_budget.pdf")); plt.close(fig); print("fig6e_budget.pdf")


def fig_mechanisms(data, out):
    panels = []
    if "stage3-counter-mechanisms" in data:
        df = final(data["stage3-counter-mechanisms"])
        cases = [("induced-demand? off", (df["induced-demand?"].astype(str).str.lower() == "false") & (df["rent-feedback?"].astype(str).str.lower() == "true")),
                 ("rent-feedback? off", (df["induced-demand?"].astype(str).str.lower() == "true") & (df["rent-feedback?"].astype(str).str.lower() == "false")),
                 ("both off", (df["induced-demand?"].astype(str).str.lower() == "false") & (df["rent-feedback?"].astype(str).str.lower() == "false"))]
        base = (df["induced-demand?"].astype(str).str.lower() == "true") & (df["rent-feedback?"].astype(str).str.lower() == "true")
        panels.append(("(a) change in displaced-share", df, base, cases, KEY))
    if "stage3-unit-decomposition" in data:
        df = final(data["stage3-unit-decomposition"])
        df = df[df["affordable-rent-rule"] == "market-share"] if "affordable-rent-rule" in df else df
        sec = df["secure-tenure?"].astype(str).str.lower(); dis = df["rent-discount?"].astype(str).str.lower()
        base = (sec == "true") & (dis == "true")
        cases = [("secure-tenure? off", (sec == "false") & (dis == "true")), ("rent-discount? off", (sec == "true") & (dis == "false"))]
        panels.append(("(b) change in displaced-share", df, base, cases, KEY))
        panels.append(("(c) change in mean stress quarters", df, base, cases, STRESS))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.6), squeeze=False)
    for ax, (title, df, base, cases, metric) in zip(axes[0], panels):
        conds = [c for c in conds_in(df) if c != "none"]
        width = 0.8 / len(cases)
        for j, (lab, mask) in enumerate(cases):
            for i, c in enumerate(conds):
                m, lo, hi, n = paired_diff(df, metric, base & (df["strategy"] == c), mask & (df["strategy"] == c))
                x = i + (j - (len(cases) - 1) / 2) * width
                ax.bar(x, m, width * 0.9, color=COND_COLOR[c], alpha=0.4 + 0.2 * j, edgecolor="black")
                if not np.isnan(lo):
                    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=2)
            ax.plot([], [], color="black", alpha=0.4 + 0.2 * j, lw=6, label=lab)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=15)
        ax.set_ylabel("paired difference from all switches on, 95% bootstrap"); ax.set_title(title); ax.legend()
    fig.savefig(os.path.join(out, "fig7_mechanisms.pdf")); plt.close(fig); print("fig7_mechanisms.pdf")


# rows of the sensitivity table, in the report's order: the model's variable name, then the report's symbol
SENS_LABEL = {"alpha": r"alpha ($\alpha$)", "n-search-towns": r"n-search-towns ($k$)",
              "s-search-period": r"s-search-period ($s$)", "p-income-shock": r"p-income-shock ($p$)",
              "termination-hazard-base": r"termination-hazard-base ($h_0$)", "termination-hazard-coef": r"termination-hazard-coef ($\kappa$)",
              "stress-threshold": r"stress-threshold ($\theta$)",
              "non-target-frac-high": r"non-target-frac-high ($b_{\mathrm{high}}$)", "affordable-rent-ratio": r"affordable-rent-ratio ($\lambda$)",
              "income-penalty": r"income-penalty ($\rho$)", "n-households": r"n-households, with stock ($n$)",
              "govt-homes": r"govt-homes ($\sigma$)", "income-growth": r"income-growth ($g$)",
              "weight-rent": r"weight-rent ($w_r$)", "choice-noise": r"choice-noise ($\tau$)",
              "cost-high": r"cost-high, fixed budget ($c_{\mathrm{high}}$)"}


BOLD_AT = 0.20  # sensitivity values at or above this are bolded in the table


def table_sensitivity(data, out, shift=0.15):
    """S-bar per parameter per condition from experiments named sensitivity-<param>."""
    rows = {}
    for name, df in data.items():
        if not name.startswith("sensitivity"):
            continue
        df = final(df)
        # the swept parameter is named in the experiment; otherwise the numeric column with three values
        par = name[len("sensitivity-"):]
        if par not in df.columns or df[par].nunique() != 3:
            cand = [c for c in df.columns if c not in ("seed", "strategy", "run number", "step") and not c.startswith("stock-")
                    and df[c].nunique() == 3 and pd.api.types.is_numeric_dtype(df[c])]
            if not cand:
                continue
            par = cand[0]
        vals = sorted(df[par].unique()); ref = vals[1]
        for c in conds_in(df):
            sub = df[df["strategy"] == c]
            e0 = sub[sub[par] == ref].set_index("seed")[KEY]
            ebar = e0.mean()
            if not ebar:
                continue
            # divide by the actual relative change of the parameter (0.15 for
            # continuous parameters; 0.25 for k = 3/4/5, 0.5 for s = 1/2/3)
            up = (vals[2] - ref) / ref if ref else shift
            dn = (ref - vals[0]) / ref if ref else shift
            sp = ((sub[sub[par] == vals[2]].set_index("seed")[KEY] - e0) / (up * ebar)).mean()
            sm = ((sub[sub[par] == vals[0]].set_index("seed")[KEY] - e0) / (dn * ebar)).mean()
            rows.setdefault(par, {})[c] = (abs(sp) + abs(sm)) / 2
    if not rows:
        return
    tab = pd.DataFrame(rows).T.reindex(columns=[c for c in COND_ORDER if any(c in r for r in rows.values())])
    tab = tab.reindex([p for p in SENS_LABEL if p in tab.index] + [p for p in tab.index if p not in SENS_LABEL])
    # supply, its cost and the affordable rent do not exist in the baseline, so those cells are not applicable rather than zero
    for par in ("govt-homes", "cost-high", "affordable-rent-ratio"):
        if par in tab.index and "none" in tab.columns:
            tab.loc[par, "none"] = np.nan
    with open(os.path.join(out, "tab_sensitivity.tex"), "w") as f:
        f.write(f"% generated by make_figures.py; values of {BOLD_AT:.2f} or more in bold\n")
        f.write("\\begin{tabular}{l" + "c" * tab.shape[1] + "}\n\\toprule\nParameter & " + " & ".join(COND_LABEL[c] for c in tab.columns) + " \\\\\n\\midrule\n")
        cell = lambda v: "--" if pd.isna(v) else (f"\\textbf{{{v:.2f}}}" if round(v, 2) >= BOLD_AT else f"{v:.2f}")
        for par, r in tab.iterrows():
            f.write(SENS_LABEL.get(par, par) + " & " + " & ".join(cell(v) for v in r.values) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("tab_sensitivity.tex")
    # ranking stability for the most sensitive parameter
    top = tab.max(axis=1).idxmax()
    src = [n for n, d in data.items() if n.startswith("sensitivity") and top in d.columns and d[top].nunique() == 3]
    if src:
        df = final(data[src[0]]); vals = sorted(df[top].unique())
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
        for ax, v, lab in zip(axes, vals, ["Low (-15%)", "Reference", "High (+15%)"]):
            d = df[df[top] == v]; cs = conds_in(d)
            bp = ax.boxplot([d[d["strategy"] == c][KEY].values for c in cs], patch_artist=True, showfliers=False)
            for b, c in zip(bp["boxes"], cs):
                b.set(facecolor=COND_COLOR[c], alpha=0.6)
            ax.set_xticks(range(1, len(cs) + 1)); ax.set_xticklabels([COND_LABEL[c] for c in cs], rotation=15, fontsize=7)
            ax.set_title(f"{lab}: {top} = {v:g}")  # the model's variable name
        axes[0].set_ylabel("displacements per household at quarter 40")
        fig.savefig(os.path.join(out, "fig8_ranking.pdf")); plt.close(fig); print("fig8_ranking.pdf")


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Assignment_2
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--out", default=os.path.join(root, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load_all(a.results)
    print("experiments found:", ", ".join(sorted(data)) or "none")
    fig_calibration(data, a.out)
    fig_theory(data, a.out)
    fig_stage1(data, a.out)
    fig_stage2(data, a.out)
    fig_mechanisms(data, a.out)
    table_sensitivity(data, a.out)


if __name__ == "__main__":
    main()
