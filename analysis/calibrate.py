#!/usr/bin/env python3
"""Score calibration pilots against the four patterns and the saturation band,
and rank the parameter sets.

Usage (run from a scratch folder: calibration_scores.csv is written to the
current directory)  
    python calibrate.py --pilot ../results/pilot-v4.csv [--top 15]

The pilot is a BehaviorSpace table CSV from affordable_rental_allocation_v2.nlogox:
baseline only (strategy "none"), recorded every step, with the reporters in
REQUIRED. Runs are grouped by every name in FACTORS that varies in the file.

Criteria (Methods, Table 4 of the report), scored on each set's seed mean:
    P1  stress prevalence in the first year (steps 1-4) between 0.85 and 0.95,
        and at quarter 40 inside the band: displacement 0.20-0.60, stress 0.40-0.90
    P2  forced share of moves at quarter 40 between 0.20 and 0.35
    P3  rent increase at quarter 40 ordered high > medium > low
    P4  accessibility lost among ever-displaced > among the rest at quarter 40

Sets are ranked by passes, then by distance to target, weighted
P1 (3), P4 (2), P2 (1.5), P3 (1).
"""
import argparse
import numpy as np
import pandas as pd

FACTORS = ["income-max", "income-growth", "bg-target-move-prob", "max-rent-to-income", "alpha",
           "termination-hazard-base", "non-target-frac-high", "non-target-frac-mid", "non-target-frac-low",
           "non-target-competition?", "search-prob", "termination-hazard-coef", "rent-high"]

REQUIRED = ["run number", "seed", "step", "displaced-share", "stress-prevalence", "forced-move-share",
            "rent-increase-high", "rent-increase-mid", "rent-increase-low",
            "access-lost-displaced", "access-lost-not-displaced"]


def load(path):
    df = pd.read_csv(path, skiprows=6)
    df.columns = [c.strip("[]") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}. It needs a table-format run of the current model "
                         f"with the reporters in REQUIRED.")
    return df


def summarise(df):
    keys = [c for c in FACTORS if c in df.columns and df[c].nunique() > 1] or [c for c in FACTORS if c in df.columns][:1]
    early = df[df["step"].between(1, 4)].groupby(keys + ["seed"])["stress-prevalence"].mean().rename("p1_early")
    last = df.loc[df.groupby("run number")["step"].idxmax()].set_index(keys + ["seed"])
    per_seed = pd.concat([early, last[["displaced-share", "stress-prevalence", "forced-move-share",
                                       "rent-increase-high", "rent-increase-mid", "rent-increase-low",
                                       "access-lost-displaced", "access-lost-not-displaced"]]], axis=1)
    m = per_seed.groupby(keys).mean()
    m["n_seeds"] = per_seed.groupby(keys).size()
    return m


def score(m):
    s = pd.DataFrame(index=m.index)
    s["P1"] = m["p1_early"].between(0.85, 0.95) & m["displaced-share"].between(0.20, 0.60) & m["stress-prevalence"].between(0.40, 0.90)
    s["P2"] = m["forced-move-share"].between(0.20, 0.35)
    s["P3"] = (m["rent-increase-high"] > m["rent-increase-mid"]) & (m["rent-increase-mid"] > m["rent-increase-low"])
    s["P4"] = m["access-lost-displaced"] > m["access-lost-not-displaced"]
    s["passes"] = s[["P1", "P2", "P3", "P4"]].sum(axis=1)
    # distance to target, smaller is better: P1 early to 0.90, P2 to 0.28, P3 gradient margin, P4 margin
    d1 = (m["p1_early"] - 0.90).abs() + np.clip(0.20 - m["displaced-share"], 0, None) + np.clip(m["displaced-share"] - 0.60, 0, None) \
         + np.clip(0.40 - m["stress-prevalence"], 0, None) + np.clip(m["stress-prevalence"] - 0.90, 0, None)
    d2 = (m["forced-move-share"] - 0.28).abs()
    d3 = np.clip(m["rent-increase-mid"] - m["rent-increase-high"], 0, None) + np.clip(m["rent-increase-low"] - m["rent-increase-mid"], 0, None)
    d4 = np.clip(m["access-lost-not-displaced"] - m["access-lost-displaced"], 0, None)
    s["distance"] = 3 * d1 + 2 * d4 + 1.5 * d2 + d3
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="../../results/pilot-v4.csv")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    df = load(a.pilot)
    m = summarise(df)
    s = score(m)
    out = pd.concat([s, m[["p1_early", "displaced-share", "stress-prevalence", "forced-move-share",
                           "rent-increase-high", "rent-increase-mid", "rent-increase-low",
                           "access-lost-displaced", "access-lost-not-displaced"]].round(3)], axis=1)
    out = out.sort_values(["passes", "distance"], ascending=[False, True])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"{len(out)} parameter sets, {int(m['n_seeds'].iloc[0])} seeds each")
    print("pass counts:", out["passes"].value_counts().sort_index(ascending=False).to_dict())
    print("per-criterion pass rate:", {k: round(float(out[k].mean()), 2) for k in ["P1", "P2", "P3", "P4"]})
    print()
    print(out.head(a.top).to_string())
    best = out.index[0]
    best = best if isinstance(best, tuple) else (best,)  # one swept factor gives a plain index, not tuples
    print("\nBest set:", dict(zip(out.index.names, best)))
    out.to_csv("calibration_scores.csv")
    print("written calibration_scores.csv")


if __name__ == "__main__":
    main()
