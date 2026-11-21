# -*- coding: utf-8 -*-
"""recompute_trial.py — 从 process log 独立重算单条试验的 10 项指标。

独立实现（不 import q3_scheduler，不复制其内部公式）：只读 process log 的动作行与 tasks.csv，
按 Problem_3_Requirements.pdf 的公开公式逐动作重放并计算指标，与 q3_trial_metrics.csv 对比。

题面公式（本脚本的唯一依据）：
- 区域坐标 Z0=(0,0) Z1=(1,0) Z2=(0,1) Z3=(1,1) Z4=(2,0)；D=欧氏距离；
- 执行：s=t+D+0.5；费用 2+0.1b+0.2D；p=finish_s−start_s；完成 C=s+p；设备 s+max(p,b) 释放；
- 跨 H 截尾：任务未完成、C=H；复核：成本 1、返回时刻=return_s；等待：now=until_s；
- weighted_tardiness_s = Σ w·max(C−due, 0)，**完成**任务 C=实际 finish，**未完成**任务 C=H；
- 分母：W=全部任务权重和；observed_overrun=Σ执行任务 max(min(p,H−s)−b,0)；
  reserved_idle=Σ完成任务 max(b−p,0)；censored=截尾任务数；invalid=违规数（0）。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ZONES = {"Z0": (0., 0.), "Z1": (1., 0.), "Z2": (0., 1.), "Z3": (1., 1.), "Z4": (2., 0.)}


def recompute(log_rows: pd.DataFrame, case_tasks: pd.DataFrame,
              seed: int, case_id: str, policy: str, alpha: float, beta: float) -> dict:
    jobs = {r.job_id: {"due": float(r.due_s), "weight": float(r.priority_weight),
                       "done": False, "attempted": False}
            for r in case_tasks.itertuples(index=False)}
    horizon = None
    now = 0.0
    cost = 0.0
    reviews = 0
    zone = "Z0"
    invalid = 0
    done_w = ontime_w = tardiness = overrun = idle = 0.0
    censored = 0
    n_actions = 0
    for row in log_rows.itertuples(index=False):
        n_actions += 1
        horizon = float(row.horizon_s) if hasattr(row, "horizon_s") else horizon
        a = row.action
        if a == "review":
            reviews += 1
            cost += 1.0
            now = max(now, float(row.return_s))
        elif a == "wait":
            now = float(row.until_s)
        elif a == "execute":
            j = jobs[row.job_id]
            j["attempted"] = True
            b = float(row.reservation_s)
            s = float(row.start_s)
            p = float(row.finish_s) - s
            d = math.dist(ZONES[row.from_zone], ZONES[row.to_zone])
            cost += 2 + 0.1 * b + 0.2 * d
            terminated = bool(row.terminated_at_H)
            overrun += max(min(p, horizon - s) - b, 0.0)
            if terminated:
                censored += 1
                now = horizon
                zone = row.to_zone
            else:
                j["done"] = True
                c = s + p
                done_w += j["weight"]
                if c <= j["due"]:
                    ontime_w += j["weight"]
                tardiness += j["weight"] * max(0.0, c - j["due"])
                idle += max(b - p, 0.0)
                now = min(s + max(p, b), horizon)
                zone = row.to_zone
        else:
            invalid += 1
            break
    # 未完成（含从未执行与截尾）：C=H
    for j in jobs.values():
        if not j["done"]:
            tardiness += j["weight"] * max(0.0, horizon - j["due"])
    W = sum(j["weight"] for j in jobs.values())
    return {
        "split": "validation", "case_id": case_id, "seed": int(seed),
        "alpha": float(alpha), "beta": float(beta), "policy": policy,
        "weighted_completion_rate": done_w / W if W else 0.0,
        "weighted_on_time_rate": ontime_w / W if W else 0.0,
        "weighted_tardiness_s": tardiness,
        "cost": cost, "observations": reviews, "elapsed_s": now,
        "observed_overrun_s": overrun, "reserved_idle_s": idle,
        "censored_units": censored, "invalid_actions": invalid,
        "valid_protocol": int(invalid == 0),
        "n_actions_seen": n_actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    args = ap.parse_args()

    metrics = pd.read_csv(args.metrics)
    target = metrics[(metrics.seed == args.seed) & (metrics.case_id == args.case)
                     & (metrics.policy == args.policy)
                     & (abs(metrics.alpha - args.alpha) < 1e-9)
                     & (abs(metrics.beta - args.beta) < 1e-9)]
    if target.empty:
        raise SystemExit(f"metrics 中找不到该试验: seed={args.seed} case={args.case} "
                         f"policy={args.policy} alpha={args.alpha} beta={args.beta}")
    target = target.iloc[0]

    log = pd.read_csv(args.log)
    trial_log = log[(log.seed == args.seed) & (log.case_id == args.case)
                    & (log.policy == args.policy)
                    & (abs(log.alpha - args.alpha) < 1e-9)
                    & (abs(log.beta - args.beta) < 1e-9)].copy()
    # 从日志中提取 horizon（execute/wait 行不携带，用 tasks 之外的日志字段兜底：由 metrics 行隐含）
    # 日志行合并了 metrics 字段，可能含 horizon_s；否则由 case 映射
    case_map = {"S1": 100.0, "S2": 100.0, "S3": 125.0}
    trial_log["horizon_s"] = case_map[args.case]

    tasks = pd.read_csv(args.tasks).fillna({"predecessors": ""})
    case_tasks = tasks[tasks.case_id == args.case]

    rec = recompute(trial_log, case_tasks, args.seed, args.case, args.policy, args.alpha, args.beta)

    fields = ["weighted_completion_rate", "weighted_on_time_rate", "weighted_tardiness_s",
              "cost", "observations", "elapsed_s", "observed_overrun_s", "reserved_idle_s",
              "censored_units", "invalid_actions", "valid_protocol"]
    print(f"trial: seed={args.seed} case={args.case} policy={args.policy} "
          f"alpha={args.alpha} beta={args.beta} (log actions={rec['n_actions_seen']})")
    mism = []
    for f in fields:
        a, b = float(rec[f]), float(target[f])
        ok = abs(a - b) < 1e-6 * max(1.0, abs(b))
        flag = "[OK]" if ok else f"[DIFF] recomputed={a:.6f} recorded={b:.6f}"
        if not ok:
            mism.append(f)
        print(f"  {f:28} {flag}")
    # 已完成但逾期的任务示例（用于验证延误口径）；review/wait 行的 terminated_at_H 为 NaN，先填 False
    ex = trial_log[(trial_log.action == "execute")
                   & (~trial_log.terminated_at_H.fillna(False).astype(bool))]
    if len(ex):
        late = ex[ex.finish_s > ex.job_id.map(case_tasks.set_index("job_id").due_s)]
        print(f"  completed_jobs={len(ex)} late_completed={len(late)}"
              + (f", example job={late.iloc[0].job_id} finish={late.iloc[0].finish_s:.2f}s"
                 if len(late) else ""))
    print(f"  conclusion: {'ALL MATCH' if not mism else 'MISMATCH FIELDS: ' + ', '.join(mism)}")
    return 0 if not mism else 1


if __name__ == "__main__":
    raise SystemExit(main())

