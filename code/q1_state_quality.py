# -*- coding: utf-8 -*-
"""q1_state_quality.py — 阶段 D：Q1 状态建模与观测可靠性（父 Agent 实现）。

状态：单手位置 p_t = 21 个 3D 关键点的逐坐标中位数；运动状态 s=[p,v]，六维常速度卡尔曼。
两只手独立建模；过程/观测尺度只用训练环境估计（环境级 bootstrap CI）；
末值保持与常速度外推为基线；误差按记录/环境/阶段汇总。
观测可靠性：逻辑回归（可解释基线），坏帧标签用原始质量标记（v3 逐手口径），
train 拟合、validation 定阈值、test 冻结评估（PR-AUC/AUROC/精确率/召回/平衡准确率/Brier/多数类基线）。

用法：python q1_state_quality.py --data-root <EN> --output <dir> [--smoke N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_data import attach_split, load_sequences, load_split

HORIZONS = (1, 2, 5, 10, 20)
HORIZON_STRIDE = 5
FEATURES = ["log_dt", "log_speed", "log_accel", "missing_joint_fraction",
            "log_spread", "log_spread90", "log_spread_change"]


# ---------- 卡尔曼（六维常速度） ----------

class CVKalman:
    def __init__(self, p0, p1, dt, accel_var, meas_var):
        v = (np.asarray(p1, float) - np.asarray(p0, float)) / max(float(dt), 1e-8)
        self.state = np.r_[p1, v].astype(float)
        self.cov = np.diag([meas_var] * 3 + [meas_var / max(dt * dt, 1e-8)] * 3)
        self.accel_var = max(float(accel_var), 1e-10)
        self.meas_var = max(float(meas_var), 1e-10)

    def _FQ(self, dt):
        I = np.eye(3); Z = np.zeros((3, 3))
        F = np.block([[I, dt * I], [Z, I]])
        G = np.vstack((0.5 * dt * dt * I, dt * I))
        Q = self.accel_var * (G @ G.T)
        H = np.hstack((I, Z))
        R = self.meas_var * I
        return F, Q, H, R

    def predict(self, dt):
        F, Q, _, _ = self._FQ(dt)
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        return self.state[:3].copy(), self.cov[:3, :3].copy()

    def update(self, obs):
        _, _, H, R = self._FQ(0.0)
        res = np.asarray(obs, float) - H @ self.state
        S = H @ self.cov @ H.T + R
        K = np.linalg.solve(S, H @ self.cov).T
        self.state = self.state + K @ res
        # Joseph form preserves a symmetric positive-semidefinite covariance
        # better under finite precision; this covariance drives Q2 intervals.
        A = np.eye(6) - K @ H
        self.cov = A @ self.cov @ A.T + K @ R @ K.T
        self.cov = 0.5 * (self.cov + self.cov.T)


def good_runs(seq: dict) -> list[np.ndarray]:
    p = np.asarray(seq["position"], float)
    t = np.asarray(seq["time_s"], float)
    ok = np.isfinite(p).all(axis=1) & ~np.asarray(seq["bad"], bool) & np.isfinite(t)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(t[idx]) <= 0)) + 1
    return [r for r in np.split(idx, breaks) if len(r) >= 3]


# ---------- 运动参数估计（仅训练环境） ----------

def estimate_motion_parameters(seqs: list[dict]) -> tuple[float, float, dict]:
    env_acc: dict[str, list[float]] = {}
    env_second_diff: dict[str, list[float]] = {}
    cadence: list[float] = []
    second_diff: list[float] = []
    for s in seqs:
        if s["split"] != "train":
            continue
        p = np.asarray(s["position"], float)
        t = np.asarray(s["time_s"], float)
        for run in good_runs(s):
            dt = np.diff(t[run])
            cadence.extend(dt[dt > 0].tolist())
            vel = np.diff(p[run], axis=0) / np.maximum(dt[:, None], 1e-9)
            if len(vel) > 1:
                mid = 0.5 * (dt[1:] + dt[:-1])
                acc = np.diff(vel, axis=0) / np.maximum(mid[:, None], 1e-9)
                env_acc.setdefault(s["environment_id"], []).extend(acc.ravel().tolist())
            cur_second_diff = np.diff(p[run], n=2, axis=0).ravel().tolist()
            second_diff.extend(cur_second_diff)
            env_second_diff.setdefault(s["environment_id"], []).extend(cur_second_diff)
    typical_dt = float(np.median(cadence)) if cadence else 1 / 30
    all_acc = np.asarray([x for v in env_acc.values() for x in v], float)
    accel_var = max(float(np.var(all_acc)), 1e-8) if len(all_acc) else 1e-4
    meas_var = max(float(np.var(second_diff) / 6), 1e-8) if second_diff else 1e-6
    rng = np.random.default_rng(2026)
    envs = sorted(set(env_acc) & set(env_second_diff))
    est_acc, est_meas = [], []
    if envs:
        for _ in range(300):
            sampled = rng.choice(envs, size=len(envs), replace=True)
            acc_vals = np.concatenate([np.asarray(env_acc[e]) for e in sampled])
            second_vals = np.concatenate([np.asarray(env_second_diff[e]) for e in sampled])
            est_acc.append(np.var(acc_vals))
            est_meas.append(np.var(second_vals) / 6)
    acc_ci = ([float(np.quantile(est_acc, 0.025)), float(np.quantile(est_acc, 0.975))]
              if est_acc else [None, None])
    meas_ci = ([float(np.quantile(est_meas, 0.025)), float(np.quantile(est_meas, 0.975))]
               if est_meas else [None, None])
    return accel_var, meas_var, {
        "median_frame_interval_s": typical_dt,
        "acceleration_residual_variance": accel_var,
        "acceleration_variance_env_bootstrap_95ci": acc_ci,
        "measurement_variance_scale": meas_var,
        "measurement_variance_env_bootstrap_95ci": meas_ci,
        "train_environment_count": len(envs),
    }


# ---------- 一步回放（末值 / 常速度 / 卡尔曼） ----------

def one_step_replay(seqs, accel_var, meas_var, output: Path) -> pd.DataFrame:
    rows = []
    for s in seqs:
        if s["split"] not in ("validation", "test"):
            continue
        p = np.asarray(s["position"], float)
        t = np.asarray(s["time_s"], float)
        for run_id, run in enumerate(good_runs(s)):
            filt = CVKalman(p[run[0]], p[run[1]], t[run[1]] - t[run[0]], accel_var, meas_var)
            for k in range(2, len(run)):
                i0, i1, i = run[k - 2], run[k - 1], run[k]
                dt = float(t[i] - t[i1])
                if dt <= 0:
                    continue
                pred, cov = filt.predict(dt)
                v_pred = filt.state[3:].copy()
                vel = (p[i1] - p[i0]) / max(t[i1] - t[i0], 1e-8)
                v_ref = (p[i] - p[i1]) / dt
                estimates = {"last_value": p[i1],
                             "constant_velocity": p[i1] + vel * dt,
                             "cv_kalman": pred}
                sigma = float(np.sqrt(np.trace(cov) / 3))
                for method, est in estimates.items():
                    rows.append({
                        "dataset": s["dataset"], "episode_uuid": s["episode_uuid"],
                        "environment_id": s["environment_id"], "split": s["split"],
                        "side": s["side"], "stage": s["stage"][i], "run": run_id,
                        "timestamp_s": t[i], "method": method,
                        "error": float(np.linalg.norm(p[i] - est)),
                        "velocity_error": (float(np.linalg.norm(v_pred - v_ref))
                                           if method == "cv_kalman" else np.nan),
                        "sigma": sigma if method == "cv_kalman" else np.nan,
                    })
                filt.update(p[i])
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "q1_state_replay.csv", index=False, encoding="utf-8-sig")
    if detail.empty:
        return detail
    summary = detail.groupby(["split", "dataset", "environment_id", "method"], as_index=False).agg(
        n_points=("error", "size"), mean_error=("error", "mean"),
        p95_error=("error", lambda x: float(np.quantile(x, 0.95))),
        mean_velocity_error=("velocity_error", "mean"), mean_sigma=("sigma", "mean"))
    summary.to_csv(output / "q1_state_metrics.csv", index=False, encoding="utf-8-sig")
    stage = detail.groupby(["split", "stage", "method"], as_index=False).agg(
        n_points=("error", "size"), mean_error=("error", "mean"),
        p95_error=("error", lambda x: float(np.quantile(x, 0.95))))
    stage.to_csv(output / "q1_state_by_stage.csv", index=False, encoding="utf-8-sig")
    return detail


def env_bootstrap_ci(detail: pd.DataFrame, output: Path) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for (sp, method), part in detail.groupby(["split", "method"]):
        envs = list(part.environment_id.unique())
        est = []
        for _ in range(400):
            sampled = rng.choice(envs, size=len(envs), replace=True)
            # 有放回：重复抽中的环境按其出现次数计入（concat 保留重复，不用 isin 去重）
            sub = pd.concat([part[part.environment_id == e] for e in sampled])
            est.append(float(sub.error.mean()))
        rows.append({"split": sp, "method": method, "n_environments": len(envs),
                     "mean_error": float(part.error.mean()),
                     "ci95_lo": float(np.quantile(est, 0.025)),
                     "ci95_hi": float(np.quantile(est, 0.975))})
    out = pd.DataFrame(rows)
    out.to_csv(output / "q1_env_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    return out


def horizon_sweep(seqs, accel_var, meas_var, output: Path) -> pd.DataFrame:
    rows = []
    for s in seqs:
        if s["split"] not in ("validation", "test"):
            continue
        p = np.asarray(s["position"], float)
        t = np.asarray(s["time_s"], float)
        for run in good_runs(s):
            if len(run) < 30:
                continue
            for k in range(2, len(run) - max(HORIZONS), HORIZON_STRIDE):
                i0, i1, ix = run[k - 2], run[k - 1], run[k]
                # 常速度基线：用最近两帧（当前帧 ix 与前一帧 i1）估速
                vel = (p[ix] - p[i1]) / max(t[ix] - t[i1], 1e-8)
                for h in HORIZONS:
                    j = k + h
                    if j >= len(run):
                        continue
                    # 卡尔曼：从 i1 起顺序推进（逐步 predict+update）至当前观测时刻 t[ix]，
                    # 得到 t[ix] 处的滤波状态后，再纯预测 h 步到目标时刻
                    filt = CVKalman(p[i0], p[i1], t[i1] - t[i0], accel_var, meas_var)
                    for step in range(k - 1, k):
                        nxt = run[step + 1]
                        filt.predict(float(t[nxt] - t[run[step]]))
                        filt.update(p[nxt])
                    dt_h = float(t[run[j]] - t[ix])
                    pred, _ = filt.predict(dt_h)
                    cv = p[ix] + vel * dt_h
                    for method, est in (("cv_kalman", pred), ("last_value", p[ix]), ("constant_velocity", cv)):
                        rows.append({"dataset": s["dataset"], "environment_id": s["environment_id"],
                                     "split": s["split"], "side": s["side"], "horizon": h,
                                     "method": method,
                                     "error": float(np.linalg.norm(p[run[j]] - est))})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    detail.to_csv(output / "q1_horizon_error.csv", index=False, encoding="utf-8-sig")
    summ = detail.groupby(["split", "horizon", "method"], as_index=False).agg(
        n_points=("error", "size"), mean_error=("error", "mean"),
        p95_error=("error", lambda x: float(np.quantile(x, 0.95))))
    summ.to_csv(output / "q1_horizon_error_summary.csv", index=False, encoding="utf-8-sig")
    return detail


# ---------- 观测可靠性模型 ----------

def frame_features(seq: dict) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(seq["position"], float)
    t = np.asarray(seq["time_s"], float)
    kp = np.asarray(seq["keypoints"], float)
    bad = np.asarray(seq["bad"], bool)
    n = len(t)
    dt = np.r_[np.nan, np.diff(t)]
    pos_dt = dt[1:][np.isfinite(dt[1:]) & (dt[1:] > 0)]
    typ = float(np.median(pos_dt)) if len(pos_dt) else 1 / 30
    dt[0] = typ
    dt = np.where((dt > 0) & np.isfinite(dt), dt, typ)
    valid = np.isfinite(p).all(axis=1)
    step = np.zeros(n)
    if n > 1:
        pair = valid[1:] & valid[:-1]
        dist = np.linalg.norm(np.diff(p, axis=0), axis=1)
        step[1:] = np.where(pair, dist / np.maximum(dt[1:], 1e-8), 0.0)
    acc = np.zeros(n)
    if n > 1:
        acc[1:] = np.abs(np.diff(step)) / np.maximum(dt[1:], 1e-8)
    joint_valid = np.isfinite(kp).all(axis=2)
    missing = 1.0 - joint_valid.mean(axis=1)
    center = np.nanmedian(kp, axis=1)
    radius = np.linalg.norm(kp - center[:, None, :], axis=2)
    spread = np.nanmedian(radius, axis=1)
    spread90 = np.nanquantile(radius, 0.90, axis=1)
    spread_change = np.r_[0.0, np.abs(np.diff(spread))]
    x = np.column_stack((np.log1p(dt), np.log1p(step), np.log1p(acc),
                         missing, np.log1p(spread), np.log1p(spread90),
                         np.log1p(spread_change)))
    x[~np.isfinite(x)] = 0.0
    return x, (bad | ~valid).astype(int)


def quality_model(seqs, output: Path) -> dict:
    rows = []
    for s in seqs:
        x, y = frame_features(s)
        for i in range(len(y)):
            rows.append({"dataset": s["dataset"], "episode_uuid": s["episode_uuid"],
                         "environment_id": s["environment_id"], "split": s["split"],
                         "side": s["side"], "stage": s["stage"][i], "frame": i,
                         "quality_known": bool(s["quality_known"][i]),
                         "finite": bool(np.isfinite(s["position"][i]).all()),
                         "bad": int(y[i]), **{f: float(x[i, j]) for j, f in enumerate(FEATURES)}})
    table = pd.DataFrame(rows)
    table.to_csv(output / "q1_observation_audit.csv", index=False, encoding="utf-8-sig")
    train = table[(table.split == "train") & table.quality_known & table.finite]
    if train.bad.nunique() < 2:
        raise ValueError("训练集质量标记不足两个类别")
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7))
    model.fit(train[FEATURES], train.bad)
    kf = table.quality_known & table.finite
    table["prob_bad"] = 1.0
    table.loc[kf, "prob_bad"] = model.predict_proba(table.loc[kf, FEATURES])[:, 1]

    val = table[(table.split == "validation") & table.quality_known & table.finite]
    prec, rec, ths = precision_recall_curve(val.bad, val.prob_bad)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    threshold = float(ths[int(np.argmax(f1))]) if len(ths) else 0.5

    rows2 = []
    for sp in ("validation", "test"):
        d = table[(table.split == sp) & table.quality_known]
        y, score = d.bad.to_numpy(int), d.prob_bad.to_numpy(float)
        pred = (score >= threshold).astype(int)
        row = {"split": sp, "threshold_from_validation": threshold, "n_frames": len(d),
               "bad_rate": float(y.mean()), "accuracy": float(accuracy_score(y, pred)),
               "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
               "bad_precision": float(precision_score(y, pred, zero_division=0)),
               "bad_recall": float(recall_score(y, pred, zero_division=0)),
               "bad_f1": float(f1_score(y, pred, zero_division=0)),
               "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
               "brier": float(brier_score_loss(y, score))}
        if len(np.unique(y)) == 2:
            row["auc"] = float(roc_auc_score(y, score))
            row["auprc"] = float(average_precision_score(y, score))
        rows2.append(row)
    metrics = pd.DataFrame(rows2)
    metrics.to_csv(output / "q1_quality_metrics.csv", index=False, encoding="utf-8-sig")
    groups = []
    for (sp, ds, env, stage, side), part in table[table.quality_known].groupby(
            ["split", "dataset", "environment_id", "stage", "side"], dropna=False):
        groups.append({"split": sp, "dataset": ds, "environment_id": env,
                       "stage": stage, "side": side, "n_frames": len(part),
                       "bad_rate": float(part.bad.mean()),
                       "complete_rate": float(part.finite.mean())})
    pd.DataFrame(groups).to_csv(output / "q1_factor_summary.csv", index=False, encoding="utf-8-sig")
    return metrics.iloc[-1].to_dict() if len(metrics) else {}, threshold


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

    accel_var, meas_var, params = estimate_motion_parameters(seqs)
    detail = one_step_replay(seqs, accel_var, meas_var, args.output)
    bootstrap = env_bootstrap_ci(detail, args.output) if not detail.empty else pd.DataFrame()
    horizon = horizon_sweep(seqs, accel_var, meas_var, args.output)
    q_test, threshold = quality_model(seqs, args.output)

    report = {
        "n_sequences": len(seqs),
        "motion_parameters_from_train": params,
        "q1_quality_test": q_test,
        "q1_env_bootstrap_ci": bootstrap.to_dict(orient="records") if len(bootstrap) else [],
        "q1_horizon_error_points": int(len(horizon)),
    }
    (args.output / "q1_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
