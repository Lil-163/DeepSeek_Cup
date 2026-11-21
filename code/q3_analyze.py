# -*- coding: utf-8 -*-
"""q3_analyze.py — Q3 结果后处理：汇总、0.50 门槛、同 seed 配对比较。

用法：python q3_analyze.py --input <q3_trial_metrics.csv> --output <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["weighted_completion_rate", "weighted_on_time_rate", "weighted_tardiness_s",
           "cost", "observations", "elapsed_s", "observed_overrun_s", "reserved_idle_s",
           "censored_units", "invalid_actions"]


def paired_diff(df, m, base, alt):
    a = df[df.policy == alt].set_index(["case_id", "alpha", "beta", "seed"])[m]
    b = df[df.policy == base].set_index(["case_id", "alpha", "beta", "seed"])[m]
    d = (a - b).dropna()
    return {"metric": m, "base": base, "alt": alt, "n_pairs": int(len(d)),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std()),
            "ci95_lo": float(d.mean() - 1.96 * d.std() / np.sqrt(len(d))) if len(d) else None,
            "ci95_hi": float(d.mean() + 1.96 * d.std() / np.sqrt(len(d))) if len(d) else None,
            "win_rate_alt": float((d > 0).mean()) if len(d) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/q3_analysis"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    agg = df.groupby(["case_id", "policy", "alpha", "beta"], as_index=False).agg(
        n_trials=("seed", "size"),
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std": (m, "std") for m in METRICS})
    agg = agg.sort_values(["case_id", "alpha", "beta", "policy"]).reset_index(drop=True)
    agg.to_csv(args.output / "q3_validation_summary.csv", index=False, encoding="utf-8-sig")

    nominal = df[(df.alpha == 1.0) & (df.beta == 0.5)]
    gate = nominal.groupby(["case_id", "policy"])["weighted_on_time_rate"].agg(
        mean="mean", min="min", max="max", std="std").reset_index()
    gate["pass_0_50"] = gate["mean"] >= 0.50
    gate.to_csv(args.output / "q3_gate_0_50_nominal.csv", index=False, encoding="utf-8-sig")

    rows = []
    for case in sorted(df.case_id.unique()):
        sub = df[df.case_id == case]
        for m in ["weighted_on_time_rate", "weighted_completion_rate",
                  "weighted_tardiness_s", "cost", "observations"]:
            for base, alt in [("edd", "adaptive"), ("no_review", "adaptive"),
                              ("edd", "no_review")]:
                rows.append({"case_id": case, **paired_diff(sub, m, base, alt)})
    pair = pd.DataFrame(rows)
    pair.to_csv(args.output / "q3_paired_comparison.csv", index=False, encoding="utf-8-sig")

    print("=== 名义映射 0.50 门槛（adaptive/no_review/edd）===")
    print(gate.to_string(index=False))
    print("\n=== 名义映射每算例均值（加权准时率/完成率/成本/复核数）===")
    print(nominal.groupby(["case_id", "policy"])[
        ["weighted_on_time_rate", "weighted_completion_rate", "cost", "observations"]]
        .mean().round(4).to_string())
    print(f"\nWrote to {args.output}")


if __name__ == "__main__":
    main()

