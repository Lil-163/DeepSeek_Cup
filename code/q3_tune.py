# -*- coding: utf-8 -*-
"""q3_tune.py — 仅在验证集调 adaptive 策略参数（名义映射 α=1,β=0.5）。

两阶段搜索（控制计算量）：
  阶段1：效用/预留参数 w_cost×w_due×reserve_quantile_mix（18 组合，复核参数用默认）；
  阶段2：在阶段1最优基础上搜复核阈值 review_unc_min×review_value_min×review_vs_exec_factor（12 组合）。
可行条件：每个算例名义映射下平均加权准时率 ≥ 0.50；目标：最大化三算例均值、成本次优。
输出 q3_tune_results.csv、q3_tuned_params.json。
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd

from q3_scheduler import (DEFAULT_POLICY_PARAMS, fit_duration_models, run_trial)

STAGE1 = {
    "w_cost": [0.02, 0.035, 0.06],
    "w_due": [1.0, 1.4],
    "reserve_quantile_mix": [0.5, 0.75, 1.0],
}
STAGE2 = {
    "review_unc_min": [0.25, 0.40, 0.60],
    "review_value_min": [0.45, 0.70],
    "review_vs_exec_factor": [0.35, 0.55],
}


def evaluate(prefix, models, tasks, index, conditions, seeds, params):
    out = {"n_trials": 0, "on_time": [], "cost": [], "per_case": {}}
    for case in conditions["cases"]:
        case_df = tasks[tasks.case_id == case["case_id"]]
        per = []
        for seed in seeds:
            m, _ = run_trial(case_df, index, prefix, models, "validation", seed,
                             case["horizon_s"], case["cost_budget"], case["observation_limit"],
                             1.0, 0.5, "adaptive", params)
            per.append(m["weighted_on_time_rate"])
            out["on_time"].append(m["weighted_on_time_rate"])
            out["cost"].append(m["cost"])
            out["n_trials"] += 1
        out["per_case"][case["case_id"]] = sum(per) / len(per)
    out["mean_on_time"] = sum(out["on_time"]) / out["n_trials"]
    out["mean_cost"] = sum(out["cost"]) / out["n_trials"]
    out["min_case_on_time"] = min(out["per_case"].values())
    return out


def grid_search(stage: dict, base: dict, **ctx):
    keys = list(stage)
    results = []
    for i, combo in enumerate(itertools.product(*stage.values()), 1):
        params = {**base, **dict(zip(keys, combo))}
        r = evaluate(**ctx, params=params)
        results.append({"params": params, **{k: v for k, v in r.items()
                                             if k not in ("on_time", "cost", "per_case")}})
        print(f"  [{time.strftime('%H:%M:%S')}] combo {i}: mean={r['mean_on_time']:.4f} "
              f"min_case={r['min_case_on_time']:.4f} cost={r['mean_cost']:.2f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--task-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/q3_tune"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prefix = pd.read_csv(args.prefix)
    tasks = pd.read_csv(args.task_root / "tasks.csv").fillna({"predecessors": ""})
    index = pd.read_csv(args.task_root / "source_segment_index.csv")
    split = pd.read_csv(args.task_root / "record_split.csv").set_index("episode_uuid")["split"]
    index["split"] = index.episode_uuid.map(split)
    conditions = json.loads((args.task_root / "common_conditions.json").read_text(encoding="utf-8"))
    seeds = list(range(2026111700, 2026111730))
    models = fit_duration_models(prefix)
    ctx = dict(prefix=prefix, models=models, tasks=tasks, index=index,
               conditions=conditions, seeds=seeds)

    base = dict(DEFAULT_POLICY_PARAMS)
    r0 = evaluate(**ctx, params=base)
    print(f"[default] mean={r0['mean_on_time']:.4f} min_case={r0['min_case_on_time']:.4f} "
          f"cost={r0['mean_cost']:.2f}", flush=True)
    rows = [{"stage": "default", "params": json.dumps(base, ensure_ascii=False),
             "mean_on_time": r0["mean_on_time"], "min_case_on_time": r0["min_case_on_time"],
             "mean_cost": r0["mean_cost"]}]

    r1 = grid_search(STAGE1, base, **ctx)
    best1 = max(r1, key=lambda r: (r["min_case_on_time"] >= 0.50, r["mean_on_time"], -r["mean_cost"]))
    for r in r1:
        rows.append({"stage": "stage1", "params": json.dumps(r["params"], ensure_ascii=False),
                     "mean_on_time": r["mean_on_time"], "min_case_on_time": r["min_case_on_time"],
                     "mean_cost": r["mean_cost"]})
    print(f"[stage1 best] {json.dumps(best1['params'])} mean={best1['mean_on_time']:.4f} "
          f"min_case={best1['min_case_on_time']:.4f}", flush=True)

    r2 = grid_search(STAGE2, best1["params"], **ctx)
    for r in r2:
        rows.append({"stage": "stage2", "params": json.dumps(r["params"], ensure_ascii=False),
                     "mean_on_time": r["mean_on_time"], "min_case_on_time": r["min_case_on_time"],
                     "mean_cost": r["mean_cost"]})
    cands = [r for r in r2] + [best1]
    feasible = [r for r in cands if r["min_case_on_time"] >= 0.50]
    pool = feasible if feasible else cands
    best = max(pool, key=lambda r: (r["mean_on_time"], -r["mean_cost"]))
    pd.DataFrame(rows).to_csv(args.output / "q3_tune_results.csv", index=False,
                              encoding="utf-8-sig")
    (args.output / "q3_tuned_params.json").write_text(json.dumps({
        "chosen": best["params"], "mean_on_time": best["mean_on_time"],
        "min_case_on_time": best["min_case_on_time"], "mean_cost": best["mean_cost"],
        "feasible": bool(feasible), "stage1_grid": STAGE1, "stage2_grid": STAGE2,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # 调度器可直接读取的扁平参数（--params-json 期望顶层即为参数字典；无 BOM）
    (args.output / "q3_tuned_params_flat.json").write_text(
        json.dumps(best["params"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[flat params] {args.output / 'q3_tuned_params_flat.json'}")
    print(f"[chosen] feasible={bool(feasible)} mean={best['mean_on_time']:.4f} "
          f"min_case={best['min_case_on_time']:.4f} cost={best['mean_cost']:.2f}")
    print(f"[chosen params] {json.dumps(best['params'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()

