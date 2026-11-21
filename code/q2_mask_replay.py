# -*- coding: utf-8 -*-
"""q2_mask_replay.py — 阶段 E：Q2 观测缺失期间的预测与再观测（父 Agent 实现）。

固定实验条件（所有方法一致，看 test 前固定）：
  遮挡窗长 0.25/0.5/1.0/1.5/2.0 s；返回延迟 1 s；单窗最多 3 次请求；每次成本 1；
  固定间隔请求 0.5 s；名义 90% 区间；区间因子与请求容差仅在验证集校准。
方法：末值保持 / 常速度外推（零请求基线）、固定间隔复核、自适应复核（方差触发）。
严格因果回放：窗口内真实坐标只由评估器保存；仅当请求返回（延迟已过）才用记录中真实存在的
样本更新；隐藏参照时其派生特征一并不可见。窗口只从合格连续段（finite & 非坏帧 & 时间严格递增）抽取。
另附：单窗口手工追踪决策表（防泄漏证据）+ 自然低质量段统计。

用法：python q2_mask_replay.py --data-root <EN> --output <dir> [--smoke N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_state_quality import CVKalman, good_runs
from unified_data import attach_split, load_sequences, load_split

MASK_DURATIONS = (0.25, 0.5, 1.0, 1.5, 2.0)
RETURN_DELAY_S = 1.0
REQUEST_BUDGET = 3
REQUEST_COST = 1.0
FIXED_INTERVAL_S = 0.5
NOMINAL_COVERAGE = 0.90


def calibrate_on_validation(detail: pd.DataFrame, output: Path) -> tuple[float, float]:
    """只用验证集卡尔曼一步结果校准名义 90% 区间因子与请求容差。"""
    val = detail[(detail.split == "validation") & (detail.method == "cv_kalman")]
    ratio = val.error.to_numpy() / np.maximum(val.sigma.to_numpy(), 1e-8)
    factor = float(np.quantile(ratio, NOMINAL_COVERAGE))
    tolerance = float(np.quantile(val.error.to_numpy(), NOMINAL_COVERAGE))
    out = {"interval_factor_from_validation": factor,
           "request_tolerance_from_validation": tolerance,
           "nominal_coverage": NOMINAL_COVERAGE}
    (output / "q2_calibration.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return max(factor, 1e-8), max(tolerance, 1e-8)


def replay_window(position, timestamp, start, end, method, accel_var, meas_var,
                  interval_factor, tolerance, trace: bool = False):
    """因果回放一个遮挡窗 [start, end)。返回 (estimates, radii, usage, trace_rows)。"""
    p = np.asarray(position, float)
    t = np.asarray(timestamp, float)
    local_p, local_t = p[start - 2:end], t[start - 2:end]
    filt = CVKalman(local_p[0], local_p[1], local_t[1] - local_t[0], accel_var, meas_var)
    estimates, radii = [], []
    requests = wait_total = cost = 0.0
    pending = None
    last_request_time = float(local_t[1])
    window_end = float(local_t[-1])
    trace_rows = []

    for i in range(2, len(local_t)):
        dt = max(float(local_t[i] - local_t[i - 1]), 1e-8)
        estimate, cov = filt.predict(dt)
        # 已返回观测：本步若恰为该请求的返回索引（请求时刻+延迟已过），才允许使用该真实样本更新
        returned_obs_used = False
        pending_before = (pending["requested_at"], pending["index"]) if pending else None
        if pending is not None and i == pending["index"]:
            filt.update(local_p[i])
            estimate, cov = filt.state[:3].copy(), filt.cov[:3, :3].copy()
            returned_obs_used = True
            pending = None
        sigma = float(np.sqrt(np.trace(cov) / 3))
        radius = interval_factor * sigma
        requested_now = False
        new_request_expected_return = None
        if (method in ("adaptive_request", "fixed_interval")
                and requests < REQUEST_BUDGET and pending is None):
            if method == "adaptive_request":
                should = radius > tolerance
            else:
                should = local_t[i] - last_request_time >= FIXED_INTERVAL_S
            if should and local_t[i] + RETURN_DELAY_S <= window_end:
                return_time = local_t[i] + RETURN_DELAY_S
                resp = int(np.searchsorted(local_t, return_time, side="left"))
                if i < resp < len(local_t):
                    pending = {"index": resp, "requested_at": float(local_t[i])}
                    requests += 1
                    cost += REQUEST_COST
                    wait_total += float(local_t[resp] - local_t[i])
                    last_request_time = float(local_t[i])
                    requested_now = True
                    new_request_expected_return = resp
        estimates.append(estimate)
        radii.append(radius)
        if trace:
            trace_rows.append({
                "step": i, "t_s": float(local_t[i]),
                "returned_obs_requested_at": (pending_before[0] if pending_before else None),
                "returned_obs_index": (pending_before[1] if pending_before else None),
                "returned_obs_used_now": returned_obs_used,
                "new_request_issued": requested_now,
                "new_request_expected_return_index": new_request_expected_return,
            })
    return (np.asarray(estimates), np.asarray(radii),
            {"requests": requests, "observation_cost": cost, "waiting_s": wait_total},
            trace_rows)


def run_mask_replay(seqs, accel_var, meas_var, interval_factor, tolerance,
                    output: Path) -> tuple[pd.DataFrame, float]:
    rows = []
    trace_all = []
    trace_written = False
    started = time.perf_counter()
    for s in seqs:
        if s["split"] != "test":
            continue
        p = np.asarray(s["position"], float)
        t = np.asarray(s["time_s"], float)
        for run_id, run in enumerate(good_runs(s)):
            rp, rt = p[run], t[run]
            if len(run) < 12:
                continue
            for duration in MASK_DURATIONS:
                start = 2
                window_no = 0
                while start < len(run) - 1:
                    end = int(np.searchsorted(rt, rt[start] + duration, side="left"))
                    end = min(max(end, start + 2), len(run))
                    if end - start < 2:
                        break
                    truth = rp[start:end]
                    times = rt[start:end]
                    last = rp[start - 1]
                    dt_prev = max(float(rt[start - 1] - rt[start - 2]), 1e-8)
                    vel = (last - rp[start - 2]) / dt_prev
                    results = {
                        "last_value": (np.repeat(last[None, :], len(truth), axis=0), None, {}),
                        "constant_velocity": (np.asarray(
                            [last + vel * (x - rt[start - 1]) for x in times]), None, {}),
                    }
                    for method in ("adaptive_request", "fixed_interval"):
                        est, rad, usage, tr = replay_window(
                            rp, rt, start, end, method, accel_var, meas_var,
                            interval_factor, tolerance,
                            trace=(not trace_written and method == "adaptive_request"
                                   and duration == 1.5))
                        results[method] = (est, rad, usage)
                        if tr:
                            trace_all = [{"window_origin": f"{s['episode_uuid'][:8]}..{s['side']}",
                                          "duration_s": duration, **r} for r in tr]
                            trace_written = True
                    for method, (est, rad, usage) in results.items():
                        err = np.linalg.norm(truth - est, axis=1)
                        is_interval = method in ("adaptive_request", "fixed_interval")
                        rows.append({
                            "episode_uuid": s["episode_uuid"], "environment_id": s["environment_id"],
                            "dataset": s["dataset"], "side": s["side"],
                            "stage": str(s["stage"][run[start]]), "run": run_id,
                            "window": window_no, "duration_s": duration, "method": method,
                            "n_points": len(err), "mae": float(np.mean(err)),
                            "mse": float(np.mean(err ** 2)),
                            "p95_error": float(np.quantile(err, 0.95)),
                            "coverage_90": float(np.mean(err <= rad)) if is_interval else np.nan,
                            "mean_interval_radius": float(np.mean(rad)) if is_interval else np.nan,
                            "requests": usage.get("requests", 0.0),
                            "observation_cost": usage.get("observation_cost", 0.0),
                            "waiting_s": usage.get("waiting_s", 0.0),
                        })
                    window_no += 1
                    start = end
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "q2_test_replay.csv", index=False, encoding="utf-8-sig")
    if trace_all:
        pd.DataFrame(trace_all).to_csv(output / "q2_leakage_trace.csv",
                                       index=False, encoding="utf-8-sig")
    weighted = detail.assign(wmse=detail.mse * detail.n_points).groupby(
        ["duration_s", "method"], as_index=False).agg(
        points=("n_points", "sum"), wmse_sum=("wmse", "sum"))
    summary = detail.groupby(["duration_s", "method"], as_index=False).agg(
        records=("episode_uuid", "nunique"), windows=("window", "size"),
        points=("n_points", "sum"), mae=("mae", "mean"),
        p95_error=("p95_error", "mean"), coverage_90=("coverage_90", "mean"),
        mean_interval_radius=("mean_interval_radius", "mean"),
        requests=("requests", "mean"), observation_cost=("observation_cost", "mean"),
        waiting_s=("waiting_s", "mean"))
    summary["rmse"] = np.sqrt(weighted.wmse_sum / np.maximum(weighted.points, 1e-12))
    summary.to_csv(output / "q2_summary.csv", index=False, encoding="utf-8-sig")
    stage = detail.groupby(["duration_s", "stage", "method"], as_index=False).agg(
        windows=("window", "size"), mae=("mae", "mean"), requests=("requests", "mean"),
        waiting_s=("waiting_s", "mean"))
    stage.to_csv(output / "q2_by_stage.csv", index=False, encoding="utf-8-sig")
    return detail, time.perf_counter() - started


def natural_low_quality(seqs, output: Path) -> pd.DataFrame:
    events = []
    for s in seqs:
        t = np.asarray(s["time_s"], float)
        bad = np.asarray(s["bad"], bool) & np.asarray(s["quality_known"], bool)
        ids = np.flatnonzero(bad)
        if not len(ids):
            continue
        for run_no, ev in enumerate(np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)):
            st, en = int(ev[0]), int(ev[-1])
            nxt = next((i for i in range(en + 1, len(t))
                        if s["quality_known"][i] and not s["bad"][i]
                        and np.isfinite(s["position"][i]).all()), None)
            reasons = [x for x in np.asarray(s["bad_reason"], dtype=object)[ev] if str(x).strip()]
            text = "|".join(sorted(set(reasons))) if reasons else "unspecified"
            low = text.lower()
            group = "occlusion" if "occlusion" in low else ("near_edge" if "near edge" in low else "other")
            pos_dt = np.diff(t)
            pos_dt = pos_dt[pos_dt > 0]
            fi = float(np.median(pos_dt)) if len(pos_dt) else 0.0
            events.append({"dataset": s["dataset"], "episode_uuid": s["episode_uuid"],
                           "environment_id": s["environment_id"], "split": s["split"],
                           "side": s["side"], "event": run_no, "reason": text,
                           "reason_group": group, "stage": str(s["stage"][st]),
                           "n_frames": len(ev), "start_s": float(t[st]), "end_s": float(t[en]),
                           "flagged_span_s": float(max(t[en] - t[st] + fi, fi)),
                           "recovery_s": float(t[nxt] - t[en]) if nxt is not None else np.nan})
    detail = pd.DataFrame(events)
    detail.to_csv(output / "q2_low_quality_events.csv", index=False, encoding="utf-8-sig")
    if not detail.empty:
        detail.groupby(["split", "dataset", "reason_group"], as_index=False).agg(
            events=("event", "size"), mean_flagged_span_s=("flagged_span_s", "mean"),
            median_recovery_s=("recovery_s", "median")).to_csv(
            output / "q2_low_quality_summary.csv", index=False, encoding="utf-8-sig")
    return detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data_root)
    seqs = attach_split(load_sequences(args.data_root), split)
    if args.smoke > 0:
        kept = []
        for sp in ("train", "validation", "test"):
            part = [s for s in seqs if s["split"] == sp]
            kept.extend(part[: args.smoke])
        seqs = kept
        print(f"[smoke] {len(seqs)} 条序列", flush=True)

    # 一步回放（校准数据）复用 q1 脚本同一实现；区间因子与容差仅在验证集校准
    from q1_state_quality import estimate_motion_parameters, one_step_replay
    accel_var, meas_var, _params = estimate_motion_parameters(seqs)
    detail = one_step_replay(seqs, accel_var, meas_var, args.output)
    interval_factor, tolerance = calibrate_on_validation(detail, args.output)
    q2_detail, seconds = run_mask_replay(seqs, accel_var, meas_var, interval_factor, tolerance, args.output)
    events = natural_low_quality(seqs, args.output)

    report = {
        "interval_factor_from_validation": interval_factor,
        "request_tolerance_from_validation": tolerance,
        "nominal_coverage": NOMINAL_COVERAGE,
        "return_delay_s": RETURN_DELAY_S, "request_budget": REQUEST_BUDGET,
        "cost_per_request": REQUEST_COST, "mask_durations_s": list(MASK_DURATIONS),
        "q2_windows": int(q2_detail.drop_duplicates(
            ["episode_uuid", "side", "run", "duration_s", "window"]).shape[0]),
        "q2_replay_seconds": seconds,
        "low_quality_events": int(len(events)),
        "leakage_trace_written": (args.output / "q2_leakage_trace.csv").exists(),
    }
    (args.output / "q2_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

