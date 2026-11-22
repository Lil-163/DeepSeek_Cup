# -*- coding: utf-8 -*-
"""q3_scheduler.py — 阶段 F：Q3 不确定时长下的在线补货调度（父 Agent 实现）。

信息边界（严格）：策略只读任务公开字段（release/due/weight/zone/predecessors/family）+
当前已揭示前缀特征（0.2/0.6/1.0 s 的双手数值统计）；源区间 ID、结束时刻、真实时长由仿真器
持有，仅在执行后写入审计日志。真值绝不进入策略函数。

时长估计：训练环境 eligible 源区间的标注时长 d=end−start 为监督目标，log1p 目标下拟合
q50/q90 分位数梯度提升（特征=family + 前缀数值统计）；family 中位数为基线。Q90 额外用训练
环境留一交叉验证残差作保守加法校准，校准量不使用 validation/test。验证环境报告 MAE/Pinball/
覆盖率（q3_duration_validation.csv）。不默认对数正态：仿真用分位数+σ 近似。

仿真规程（逐条对照 Problem_3_Requirements.pdf）：
- 抽样：按 job_id 排序，random.Random(seed).choice 于 split/family 的 eligible 池有放回抽取，
  保持 source_segment_index.csv 行序；每 trial 新 RNG；同 seed 各策略同源分配。
- 复核：仅已释放且未执行任务；0.2→0.6→1.0；每次成本 1、占用设备 1 s；返回前不可用新前缀。
- 执行：b≥0.2；s=t+D+0.5；费用 2+0.1b+0.2D；C=s+p；设备 s+max(p,b) 释放；跨 H 截尾计未完成；
  预检查累计成本≤B 且 s+b≤H（不得用未揭示 p）；不可行指令不执行、计违规并终止试验。
- 指标：加权完成率/加权准时率/加权延误（未完成按 C=H）/成本/复核数/elapsed/超时/预留空闲/
  截尾/违规/valid_protocol；未完成任务保留在分母。

用法：
  python q3_scheduler.py --data-root <EN> --output <dir> [--split validation]
                         [--seeds 2026111700 2026111701 ...] [--policies edd no_review adaptive]
                         [--params-json <file>]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_data import attach_split, load_sequences, load_split

PREFIXES = (0.2, 0.6, 1.0)
MAPPINGS = ((1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0))
NUMERIC = ["prefix_s", "n_obs", "bad_fraction", "speed_mean", "speed_sd", "disp", "spread"]
ZONES = {"Z0": (0., 0.), "Z1": (1., 0.), "Z2": (0., 1.), "Z3": (1., 1.), "Z4": (2., 0.)}

# 策略参数（仅在验证集调；test 前冻结）
DEFAULT_POLICY_PARAMS = {
    "w_due": 1.2, "w_horizon": 0.25, "w_cost": 0.035, "w_dist": 0.02,
    "review_unc_min": 0.30, "review_value_min": 0.65,
    "review_first_scale": 0.6, "review_second_scale": 0.35,
    "review_vs_exec_factor": 0.45,
    "reserve_quantile_mix": 1.0,
    "sigma_min": 0.08,
}


# ---------- 前缀特征与时长模型 ----------

def _prefix_stats(seq: dict, start: float, end: float, q: float) -> dict:
    t = np.asarray(seq["time_s"], float)
    p = np.asarray(seq["position"], float)
    bad = np.asarray(seq["bad"], bool)
    mask = (t >= start) & (t < min(end, start + q))
    good = mask & np.isfinite(p).all(axis=1) & ~bad
    x, tt = p[good], t[good]
    if len(x) == 0:
        return {"n_obs": 0., "bad_fraction": 1., "speed_mean": 0.,
                "speed_sd": 0., "disp": 0., "spread": 0.}
    speed = np.array([], float)
    if len(x) > 1:
        dt = np.diff(tt)
        ok = dt > 0
        speed = np.linalg.norm(np.diff(x, axis=0)[ok], axis=1) / dt[ok]
    n_total = int(mask.sum())
    return {"n_obs": float(len(x)), "bad_fraction": float(1 - len(x) / max(1, n_total)),
            "speed_mean": float(np.mean(speed)) if len(speed) else 0.,
            "speed_sd": float(np.std(speed)) if len(speed) else 0.,
            "disp": float(np.linalg.norm(x[-1] - x[0])),
            "spread": float(np.mean(np.linalg.norm(x - np.median(x, axis=0), axis=1)))}


def build_prefix_table(data_root: Path) -> pd.DataFrame:
    task_root = data_root / "Restocking_Tasks_and_Constraints"
    index = pd.read_csv(task_root / "source_segment_index.csv")
    split = load_split(data_root).set_index("episode_uuid")["split"]
    index["split"] = index.episode_uuid.map(split)
    seqs = {}
    for s in load_sequences(data_root):
        seqs[(s["dataset"], s["episode_uuid"], s["side"])] = s
    rows = []
    for r in index.itertuples(index=False):
        if int(r.eligible) != 1 or r.split not in ("train", "validation", "test"):
            continue
        hands = []
        for side in ("left", "right"):
            seq = seqs.get((r.dataset, r.episode_uuid, side))
            if seq is not None:
                hands.append([_prefix_stats(seq, float(r.start_s), float(r.end_s), q)
                              for q in PREFIXES])
        for j, q in enumerate(PREFIXES):
            if hands:
                merged = {k: float(np.mean([h[j][k] for h in hands])) for k in hands[0][j]}
            else:
                merged = {k: 0.0 for k in ("n_obs", "bad_fraction", "speed_mean",
                                            "speed_sd", "disp", "spread")}
            rows.append({"unit_id": r.unit_id, "episode_uuid": r.episode_uuid,
                         "environment_id": r.environment_id, "split": r.split,
                         "family_id": r.family_id, "prefix_s": q,
                         "duration_s": float(r.duration_s), **merged})
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("前缀特征为空；请检查数据根与统一读取层")
    return out


def _fit_quantile_model(train: pd.DataFrame, quantile: float):
    """Fit one duration quantile model using only the supplied training rows."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("family", OneHotEncoder(handle_unknown="ignore"), ["family_id"])])
    model = make_pipeline(prep, GradientBoostingRegressor(
        loss="quantile", alpha=quantile, n_estimators=120, max_depth=2,
        min_samples_leaf=8, random_state=20261117))
    model.fit(train[NUMERIC + ["family_id"]], np.log1p(train.duration_s))
    return model


def fit_duration_models(features: pd.DataFrame) -> dict:
    """Fit train-only duration models and calibrate Q90 with grouped OOF residuals.

    Each training environment is held out while estimating its residual.  The
    90th percentile residual is then added to the full-train Q90 prediction.
    This guards against the observed Q90 undercoverage without learning from
    validation or test environments.
    """
    train = features[features.split == "train"].copy()
    if train.environment_id.nunique() < 2:
        raise RuntimeError("Q90 校准至少需要两个训练环境")
    family_median = train.groupby("family_id").duration_s.median().to_dict()
    q50 = _fit_quantile_model(train, .50)
    q90 = _fit_quantile_model(train, .90)

    oof_rows = []
    for environment_id in sorted(train.environment_id.unique()):
        fold_train = train[train.environment_id != environment_id]
        fold_holdout = train[train.environment_id == environment_id]
        fold_model = _fit_quantile_model(fold_train, .90)
        raw = np.maximum(0.05, np.expm1(fold_model.predict(
            fold_holdout[NUMERIC + ["family_id"]])))
        oof_rows.append(pd.DataFrame({
            "environment_id": environment_id,
            "duration_s": fold_holdout.duration_s.to_numpy(float),
            "q90_oof": raw,
        }))
    oof = pd.concat(oof_rows, ignore_index=True)
    residual = oof.duration_s.to_numpy(float) - oof.q90_oof.to_numpy(float)
    # One-sided additive conformal-style correction.  Never reduce the Q90
    # reservation because the scheduling objective treats underestimation as
    # capacity risk.
    margin = max(0.0, float(np.quantile(residual, .90)))
    oof_coverage_raw = float(np.mean(residual <= 0.0))
    oof_coverage_calibrated = float(np.mean(residual <= margin))
    calibration = {
        "method": "train_environment_leave_one_out_q90_residual",
        "n_train_environments": int(train.environment_id.nunique()),
        "n_oof_rows": int(len(oof)),
        "target_coverage": 0.90,
        "q90_additive_margin_s": margin,
        "oof_coverage_raw": oof_coverage_raw,
        "oof_coverage_calibrated": oof_coverage_calibrated,
    }
    return {"q50": q50, "q90": q90, "family_median": family_median,
            "q90_additive_margin_s": margin, "q90_calibration": calibration}


def validate_duration_models(features: pd.DataFrame, models: dict, output: Path) -> None:
    val = features[features.split == "validation"].copy()
    x = val[NUMERIC + ["family_id"]]
    q50 = np.maximum(0.05, np.expm1(models["q50"].predict(x)))
    q90_raw = np.maximum(q50, np.expm1(models["q90"].predict(x)))
    q90_calibrated = np.maximum(q50, q90_raw + models["q90_additive_margin_s"])
    base = val.family_id.map(models["family_median"]).to_numpy(float)
    y = val.duration_s.to_numpy(float)
    rows = []
    for name, pred, tau in (("family_median", base, None), ("q50", q50, 0.5),
                            ("q90_raw", q90_raw, 0.9),
                            ("q90_calibrated", q90_calibrated, 0.9)):
        row = {"model": name, "mae": float(np.mean(np.abs(y - pred))),
               "signed_bias": float(np.mean(y - pred))}
        if tau is not None:
            row["pinball"] = float(np.mean(np.where(y >= pred, (1 - tau) * (y - pred),
                                                    tau * (pred - y))))
            row["coverage"] = float(np.mean(y <= pred))
            row["nominal"] = tau
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "q3_duration_validation.csv", index=False,
                              encoding="utf-8-sig")
    (output / "q3_duration_calibration.json").write_text(
        json.dumps(models["q90_calibration"], ensure_ascii=False, indent=2),
        encoding="utf-8")


def predict_duration(models, obs, family, alpha, beta):
    x = pd.DataFrame([{**obs, "family_id": family}])
    q50 = max(.05, float(np.expm1(models["q50"].predict(x)[0])))
    q90_raw = max(q50, float(np.expm1(models["q90"].predict(x)[0])))
    q90 = max(q50, q90_raw + models["q90_additive_margin_s"])
    return alpha * q50 + beta, alpha * q90 + beta


# ---------- 抽样与策略 ----------

def sample_sources(index: pd.DataFrame, tasks: pd.DataFrame, split: str, seed: int) -> dict:
    rng = random.Random(int(seed))
    picked = {}
    pools = {f: index[(index.split == split) & (index.family_id == f) & (index.eligible == 1)]
             for f in tasks.family_id.unique()}
    for row in tasks.sort_values("job_id").itertuples(index=False):
        pool = pools[row.family_id]
        if pool.empty:
            raise RuntimeError(f"{split} 无 family {row.family_id} 的 eligible 源区间")
        picked[row.job_id] = rng.choice(pool.unit_id.tolist())
    return picked


def _slack_utility(job, now, zone, q50, q90, alpha, beta, weight, params):
    dist = math.dist(ZONES[zone], ZONES[job["zone_id"]])
    start = now + dist + .5
    mu = math.log(max(.05, q50 - beta) / max(alpha, 1e-8))
    sigma = max(params.get("sigma_min", .08),
                (math.log(max(q90 - beta, .051) / max(alpha, 1e-8)) - mu) / 1.28155)
    mean_d = alpha * math.exp(mu + sigma * sigma / 2) + beta
    sd_d = alpha * math.exp(mu + sigma * sigma / 2) * math.sqrt(math.exp(sigma * sigma) - 1)
    p_due = float(norm.cdf((job["due_s"] - start - mean_d) / max(.05, sd_d)))
    p_horizon = float(norm.cdf((job["horizon"] - start - mean_d) / max(.1, sd_d)))
    return (weight * (params.get("w_due", 1.2) * p_due + params.get("w_horizon", .25) * p_horizon)
            - params.get("w_cost", .035) * (2 + .1 * q90 + .2 * dist)
            - params.get("w_dist", .02) * dist)


def _obs_for(unit_id, prefix_s, pref):
    base = {k: 0.0 for k in NUMERIC if k != "prefix_s"}
    base["prefix_s"] = float(prefix_s)
    try:
        d = pref.loc[(unit_id, prefix_s)].to_dict()
    except KeyError:
        return base
    return {k: float(d.get(k, 0.0)) for k in NUMERIC} | {"prefix_s": float(prefix_s)}


def run_trial(case_tasks, source_index, prefix_features, models, split, seed,
              horizon, budget, review_limit, alpha, beta, policy, params=None):
    params = {**DEFAULT_POLICY_PARAMS, **(params or {})}
    sources = sample_sources(source_index, case_tasks, split, seed)
    pref = prefix_features.set_index(["unit_id", "prefix_s"])
    hidden, truth = {}, {}
    for job_id, unit in sources.items():
        hidden[job_id] = {q: _obs_for(unit, q, pref) for q in PREFIXES}
        truth[job_id] = (unit, float(source_index.loc[source_index.unit_id == unit,
                                                      "duration_s"].iloc[0]))
    jobs = {r.job_id: {"job_id": r.job_id, "family_id": r.family_id,
                       "release_s": float(r.release_s), "due_s": float(r.due_s),
                       "weight": float(r.priority_weight), "zone_id": r.zone_id,
                       "preds": [x for x in str(r.predecessors).split("|") if x and x != "nan"],
                       "level": .2, "done": False, "attempted": False,
                       "horizon": horizon} for r in case_tasks.itertuples(index=False)}
    now, cost, reviews, zone, actions, invalid = 0., 0., 0, "Z0", [], 0
    done_w = ontime_w = tardiness = overrun = idle = 0.
    censored = 0
    while now < horizon and len(actions) < 200:
        ready = [j for j in jobs.values()
                 if not j["done"] and not j["attempted"] and j["release_s"] <= now
                 and all(jobs[p]["done"] for p in j["preds"])]
        candidates = []
        for j in ready:
            obs = hidden[j["job_id"]][j["level"]]
            q50, q90 = predict_duration(models, obs, j["family_id"], alpha, beta)
            dist = math.dist(ZONES[zone], ZONES[j["zone_id"]])
            start = now + dist + .5
            reserve = max(.2, q50 + params["reserve_quantile_mix"] * (q90 - q50))
            fee = 2 + .1 * reserve + .2 * dist
            if start + reserve <= horizon and cost + fee <= budget:
                score = _slack_utility(j, now, zone, q50, q90, alpha, beta, j["weight"], params)
                if policy == "edd":
                    score = -j["due_s"] - .02 * dist
                elif policy == "no_review":
                    score += .001
                candidates.append((score, j, reserve, start, dist, fee))
        review = None
        reviewable = [j for j in jobs.values()
                      if not j["done"] and not j["attempted"] and j["release_s"] <= now]
        if (policy == "adaptive" and reviews < review_limit
                and cost + 1 <= budget and now + 1 < horizon):
            for j in reviewable:
                if j["level"] >= 1.0 or reviews >= review_limit:
                    continue
                q50, q90 = predict_duration(models, hidden[j["job_id"]][j["level"]],
                                            j["family_id"], alpha, beta)
                nxt = .6 if j["level"] < .6 else 1.0
                uncertainty = (q90 - q50) / max(.2, q50)
                scale = params["review_first_scale"] if nxt == .6 else params["review_second_scale"]
                value = j["weight"] * uncertainty * scale
                if uncertainty > params["review_unc_min"] and value > params["review_value_min"]:
                    cand = (value, j, nxt)
                    if review is None or cand[0] > review[0]:
                        review = cand
        # 将复核价值与当前最优可执行任务比较；ready 列表的首项不一定是
        # 得分最高的执行候选，直接取 candidates[0] 会让结果受 CSV 行序影响。
        best_exec_score = max((candidate[0] for candidate in candidates), default=None)
        if review is not None and (best_exec_score is None
                                   or review[0] > best_exec_score * params["review_vs_exec_factor"]):
            _, j, level = review
            reviews += 1
            cost += 1.
            t0 = now
            now += 1.
            j["level"] = level
            actions.append({"time_s": t0, "action": "review", "job_id": j["job_id"],
                            "prefix_s": level, "return_s": now, "cost": 1.,
                            "cumulative_cost": cost, "zone": zone})
            continue
        if candidates:
            _, j, reserve, start, dist, fee = max(candidates, key=lambda z: z[0])
            cost += fee
            j["attempted"] = True
            unit_id, true_d = truth[j["job_id"]]
            p = alpha * true_d + beta
            finish = start + p
            end = start + max(p, reserve)
            terminated = finish > horizon
            actions.append({"time_s": now, "action": "execute", "job_id": j["job_id"],
                            "from_zone": zone, "to_zone": j["zone_id"],
                            "reservation_s": reserve, "start_s": start, "finish_s": finish,
                            "terminated_at_H": terminated, "cost": fee,
                            "cumulative_cost": cost, "source_unit_id": unit_id})
            if not terminated:
                j["done"] = True
                done_w += j["weight"]
                if finish <= j["due_s"]:
                    ontime_w += j["weight"]
                # 题面口径：完成任务 C=实际 finish，延误 = w·max(C−due, 0)
                tardiness += j["weight"] * max(0., finish - j["due_s"])
                overrun += max(min(p, horizon - start) - reserve, 0.)
                idle += max(reserve - p, 0.)
            else:
                censored += 1
                overrun += max(min(p, horizon - start) - reserve, 0.)
            now = min(end, horizon)
            zone = j["zone_id"]
            continue
        future = [j["release_s"] for j in jobs.values()
                  if not j["attempted"] and j["release_s"] > now]
        if future:
            nxt = min(future)
            actions.append({"time_s": now, "action": "wait", "until_s": min(nxt, horizon),
                            "cost": 0., "cumulative_cost": cost, "zone": zone})
            now = min(nxt, horizon)
            continue
        break
    # 题面口径：未完成任务（含从未执行与跨 H 截尾）C=H，延误 = w·max(H−due, 0)
    for j in jobs.values():
        if not j["done"]:
            tardiness += j["weight"] * max(0., horizon - j["due_s"])
    W = sum(float(x.priority_weight) for x in case_tasks.itertuples(index=False))
    metrics = {"split": split, "case_id": str(case_tasks.case_id.iloc[0]),
               "seed": int(seed), "alpha": alpha, "beta": beta,
               "weighted_completion_rate": done_w / W,
               "weighted_on_time_rate": ontime_w / W,
               "weighted_tardiness_s": tardiness, "cost": cost,
               "observations": reviews, "elapsed_s": now,
               "observed_overrun_s": overrun, "reserved_idle_s": idle,
               "censored_units": censored, "invalid_actions": invalid,
               "valid_protocol": int(invalid == 0)}
    return metrics, actions


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(2026111700, 2026111730)))
    ap.add_argument("--policies", nargs="*", default=["edd", "no_review", "adaptive"])
    ap.add_argument("--params-json", type=Path, default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    task_root = args.data_root / "Restocking_Tasks_and_Constraints"
    tasks = pd.read_csv(task_root / "tasks.csv").fillna({"predecessors": ""})
    index = pd.read_csv(task_root / "source_segment_index.csv")
    index["split"] = index.episode_uuid.map(
        load_split(args.data_root).set_index("episode_uuid")["split"])
    prefix = build_prefix_table(args.data_root)
    prefix.to_csv(args.output / "prefix_features.csv", index=False, encoding="utf-8-sig")
    models = fit_duration_models(prefix)
    validate_duration_models(prefix, models, args.output)
    params = None
    if args.params_json:
        params = json.loads(args.params_json.read_text(encoding="utf-8"))
    conditions = json.loads((task_root / "common_conditions.json").read_text(encoding="utf-8"))
    all_metrics, all_logs = [], []
    for case in conditions["cases"]:
        case_df = tasks[tasks.case_id == case["case_id"]]
        for alpha, beta in MAPPINGS:
            for seed in args.seeds:
                for policy in args.policies:
                    # 调参只作用于风险感知族（adaptive/no_review）；EDD 为规范基线，
                    # 固定使用默认参数（纯 EDD 排序 + q90 预留）
                    p = params if policy != "edd" else None
                    m, log = run_trial(case_df, index, prefix, models, args.split, seed,
                                       case["horizon_s"], case["cost_budget"],
                                       case["observation_limit"], alpha, beta, policy, p)
                    m["policy"] = policy
                    all_metrics.append(m)
                    for row in log:
                        all_logs.append({**m, "policy": policy, **row})
    pd.DataFrame(all_metrics).to_csv(args.output / "q3_trial_metrics.csv",
                                     index=False, encoding="utf-8-sig")
    pd.DataFrame(all_logs).to_csv(args.output / "q3_process_log.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"Wrote {len(all_metrics)} trial rows to {args.output}")


if __name__ == "__main__":
    main()
