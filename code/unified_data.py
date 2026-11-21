# -*- coding: utf-8 -*-
"""unified_data.py — C 题统一读取层（阶段 C，父 Agent 设计）。

三类原始记录 → 规范化帧级序列（缓存 Parquet）→ 元数据表 / 区间表 / 审计问题清单。
口径与 schema 见 work/data_schema.md。只读题供数据，不合成样本。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KP_COLS = [f"kp_{j:02d}_{a}" for j in range(21) for a in "xyz"]

# ---------- 保留规则常量 ----------
MIN_INTERVAL_S = 1.2
BAD_RATIO_MAX = 0.10
NAN_MAX = 0.10
NOMINAL_DT = 1.0 / 30.0

DATASETS = ("EgoDemo", "EgoPro", "EgoStandard")


def default_roots() -> dict[str, Path]:
    # 缓存位置相对代码仓库推导，数据路径始终由命令行 --data-root 显式提供。
    return {"repo": Path(__file__).resolve().parents[1]}


# ---------- 官方划分 ----------

def load_split(data_root: Path) -> pd.DataFrame:
    p = data_root / "Restocking_Tasks_and_Constraints" / "record_split.csv"
    split = pd.read_csv(p, dtype=str)
    required = {"episode_uuid", "dataset", "environment_id", "split"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"record_split.csv 缺字段: {missing}")
    if split.episode_uuid.duplicated().any():
        raise ValueError("record_split.csv 存在重复 episode_uuid")
    if (split.groupby("environment_id").split.nunique() > 1).any():
        raise ValueError("同一环境出现在多个 split")
    return split


# ---------- 记录级解析 ----------

def _first(v):
    if v is None:
        return None
    if isinstance(v, (list, np.ndarray)):
        if len(v) == 0:
            return None
        return _first(v[0])
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _keypoints_21x3(cell) -> np.ndarray:
    """EgoDemo 单元格是 (21,) 对象数组（元素为 3 向量）；EgoPro/Standard 是 21×3 数值。
    统一为 (21,3) float64。"""
    a = np.asarray(cell, dtype=object)
    if a.shape == (21, 3):
        try:
            return a.astype(float)
        except (TypeError, ValueError):
            pass
    arr = np.stack([np.asarray(x, dtype=float).ravel() for x in a])
    if arr.shape != (21, 3):
        raise ValueError(f"expected 21x3 keypoints, got {arr.shape}")
    return arr.astype(float)


def _iter_egodemo(rec_dir: Path) -> Iterator[dict]:
    ep_files = sorted((rec_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"{rec_dir}: 缺 meta/episodes parquet")
    meta = pq.read_table(ep_files[0]).to_pandas()
    row = meta.iloc[0]
    episode = str(_first(row["episode_uuid"]))
    env = str(_first(row["environment_id"]))
    scene = str(_first(row.get("scene_id")) or "")
    operator = str(_first(row.get("operator_id")) or "")
    data_files = sorted((rec_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"{rec_dir}: 缺 data parquet")
    pdf = pq.read_table(data_files[0]).to_pandas()
    ts = pdf["timestamp"].to_numpy(dtype=float).ravel()
    fi = pdf["frame_index"].to_numpy()
    # 坏帧
    bad_map: dict[int, dict] = {}
    bfr = rec_dir / "bad_frame_ratio.json"
    if bfr.exists():
        for d in json.loads(bfr.read_text(encoding="utf-8")).get("hand_bad_frame_details", []):
            bad_map[int(d["frame"])] = d
    # 语义段
    segments = []
    ann = rec_dir / "annotation.json"
    if ann.exists():
        for st in json.loads(ann.read_text(encoding="utf-8")).get("subtask_annotations", []):
            s, e = float(st.get("start_time", 0.0)), float(st.get("end_time", 0.0))
            segments.append((s, e, str(st.get("description", "unlabeled"))))
    views = sorted(d.name for d in (rec_dir / "videos").iterdir()) if (rec_dir / "videos").exists() else []
    for side in ("left", "right"):
        col = f"observation.state.hand_{side}_world"
        cloud = np.stack([_keypoints_21x3(v) for v in pdf[col].to_numpy()]).astype(float)
        pos = np.nanmedian(cloud, axis=1)
        reasons = np.array([str(bad_map.get(int(k), {}).get(f"{side}_bad_reason") or "")
                            for k in fi], dtype=object)
        bad = reasons != ""
        stage = np.full(len(ts), "unlabeled", dtype=object)
        for s, e, label in segments:
            if np.isfinite(s) and np.isfinite(e) and e > s:
                stage[(ts >= s) & (ts < e)] = label if label else "unlabeled"
        yield {"dataset": "EgoDemo", "record_dir": rec_dir.name, "episode_uuid": episode,
               "environment_id": env, "scene_id": scene, "operator_id": operator,
               "side": side, "time_s": ts, "frame_index": fi.astype(np.int64),
               "position": pos, "keypoints": cloud, "bad": bad,
               "bad_reason": reasons, "quality_known": np.ones(len(ts), dtype=bool),
               "stage": stage, "views": views, "source": str(data_files[0])}


def _iter_mcap(rec_dir: Path, dataset: str) -> Iterator[dict]:
    meta_raw = pd.read_csv(rec_dir / "session_metadata.csv")
    fields = {str(k): v for k, v in zip(meta_raw["field"], meta_raw["value"])}
    episode = str(fields["task_info.episode_uuid"]).strip()
    env = str(fields.get("task_info.environment_id", "")).strip()
    scene = str(fields.get("task_info.scene_id", "")).strip()
    operator = str(fields.get("operator.operator_id", "")).strip()
    tables = rec_dir / "tables"
    flags = pd.read_csv(tables / "annotation_bad_frame_pose_hand.csv")
    bad_col = "is_bad" if "is_bad" in flags.columns else None
    reason_cols = [c for c in flags.columns if c.startswith("problem_type[")]
    flag_index = flags.set_index("_message_index") if "_message_index" in flags else None
    seg_csv = tables / "annotation_semantic_segments.csv"
    segments = []
    if seg_csv.exists():
        seg = pd.read_csv(seg_csv)
        s_col = "segment.start_time" if "segment.start_time" in seg else "start_time"
        e_col = "segment.end_time" if "segment.end_time" in seg else "end_time"
        d_col = "segment.subtask_description" if "segment.subtask_description" in seg else "description"
        if {s_col, e_col}.issubset(seg.columns):
            segments = [(float(a), float(b), str(c)) for a, b, c in
                        zip(seg[s_col], seg[e_col], seg.get(d_col, [""] * len(seg)))]
    views = sorted(f.stem for f in (rec_dir / "videos").glob("*.mp4"))
    for side in ("left", "right"):
        p = tables / f"pose_{side}_hand.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cloud = np.asarray([
            [[row[f"transforms[{j}].pos.{a}"] for a in "xyz"] for j in range(21)]
            for _, row in df.iterrows()], dtype=float)
        pos = np.nanmedian(cloud, axis=1)
        ts = df["_offset_from_first_log_s"].to_numpy(dtype=float)
        fi = df["_message_index"].to_numpy(dtype=np.int64)
        bad = np.zeros(len(df), dtype=bool)
        reasons = np.full(len(df), "", dtype=object)
        known = np.zeros(len(df), dtype=bool)
        if bad_col and flag_index is not None and "_message_index" in df.columns:
            matched = df["_message_index"].isin(flag_index.index).to_numpy()
            known = matched
            if bad_col in flag_index.columns:
                is_bad_frame = df["_message_index"].map(flag_index[bad_col]).fillna(False).astype(bool).to_numpy()
                rmap = {k: "|".join(str(row[c]) for c in reason_cols
                                     if pd.notna(row[c]) and str(row[c]).strip())
                        for k, row in flag_index.iterrows()}
                reason_all = np.array([rmap.get(k, "") for k in df["_message_index"]], dtype=object)
                # 标记是逐手的：reason 字符串编码 left/right（如 "right:occlusion"）。
                # 仅当该帧 reason 提及本侧（或完全不提及任何侧，防御性）时该侧记坏。
                side_bad = np.array([
                    (f"{side}:" in r) or (("left:" not in r) and ("right:" not in r))
                    for r in reason_all], dtype=bool)
                bad = is_bad_frame & side_bad
                reasons = reason_all
        stage = np.full(len(ts), "unlabeled", dtype=object)
        for s, e, label in segments:
            if np.isfinite(s) and np.isfinite(e) and e > s:
                stage[(ts >= s) & (ts < e)] = label if label else "unlabeled"
        yield {"dataset": dataset, "record_dir": rec_dir.name, "episode_uuid": episode,
               "environment_id": env, "scene_id": scene, "operator_id": operator,
               "side": side, "time_s": ts, "frame_index": fi,
               "position": pos, "keypoints": cloud, "bad": bad,
               "bad_reason": reasons, "quality_known": known,
               "stage": stage, "views": views, "source": str(p)}


def iter_sequences(data_root: Path, sample: int | None = None) -> Iterator[dict]:
    """遍历解压树全部序列；sample>0 时每个数据集只解析前 sample 条记录。"""
    raw = data_root / "Raw_Data"
    for ds in DATASETS:
        root = raw / ds / ds
        if not root.exists():
            raise FileNotFoundError(f"缺数据目录 {root}")
        recs = sorted(p for p in root.iterdir() if p.is_dir())
        if sample:
            recs = recs[:sample]
        for rec in recs:
            yield from (_iter_egodemo(rec) if ds == "EgoDemo" else _iter_mcap(rec, ds))


# ---------- 缓存 ----------

def cache_signature(data_root: Path) -> dict:
    sig = {}
    raw = data_root / "Raw_Data"
    for ds in DATASETS:
        z = raw / f"{ds}.zip"
        if z.exists():
            st = z.stat()
            sig[ds] = {"zip_size": int(st.st_size), "zip_mtime_ns": int(st.st_mtime_ns)}
        tree = raw / ds / ds
        if tree.exists():
            files = sorted(p for p in tree.rglob("*") if p.is_file())
            sig[f"{ds}_tree"] = {
                "n_files": len(files),
                "total_bytes": sum(p.stat().st_size for p in files),
                "mtime_ns_max": max(p.stat().st_mtime_ns for p in files),
            }
    return sig


def _flatten(seqs: list[dict]) -> pd.DataFrame:
    rows = []
    for s in seqs:
        n = len(s["time_s"])
        base = {"dataset": s["dataset"], "record_dir": s["record_dir"],
                "episode_uuid": s["episode_uuid"], "environment_id": s["environment_id"],
                "scene_id": s["scene_id"], "operator_id": s["operator_id"],
                "side": s["side"], "views": "|".join(s["views"]), "source": s["source"]}
        kp = np.asarray(s["keypoints"], float)
        for i in range(n):
            row = dict(base)
            row["time_s"] = float(s["time_s"][i])
            row["frame_index"] = int(s["frame_index"][i])
            row["px"], row["py"], row["pz"] = (float(s["position"][i, 0]),
                                               float(s["position"][i, 1]),
                                               float(s["position"][i, 2]))
            row["bad"] = bool(s["bad"][i])
            row["bad_reason"] = str(s["bad_reason"][i])
            row["quality_known"] = bool(s["quality_known"][i])
            row["stage"] = str(s["stage"][i])
            for j, col in enumerate(KP_COLS):
                row[col] = float(kp[i, j // 3, j % 3])
            rows.append(row)
    return pd.DataFrame(rows)


def _reconstruct(df: pd.DataFrame) -> list[dict]:
    seqs = []
    for (ds, ep, side), part in df.groupby(["dataset", "episode_uuid", "side"], sort=False):
        part = part.sort_values("frame_index", kind="stable")
        first = part.iloc[0]
        seqs.append({
            "dataset": str(ds), "record_dir": str(first["record_dir"]),
            "episode_uuid": str(ep), "environment_id": str(first["environment_id"]),
            "scene_id": str(first["scene_id"]), "operator_id": str(first["operator_id"]),
            "side": str(side), "time_s": part["time_s"].to_numpy(float),
            "frame_index": part["frame_index"].to_numpy(np.int64),
            "position": part[["px", "py", "pz"]].to_numpy(float),
            "keypoints": part[KP_COLS].to_numpy(float).reshape(len(part), 21, 3),
            "bad": part["bad"].to_numpy(bool),
            "bad_reason": part["bad_reason"].to_numpy(dtype=object),
            "quality_known": part["quality_known"].to_numpy(bool),
            "stage": part["stage"].to_numpy(dtype=object),
            "views": str(first["views"]).split("|"), "source": str(first["source"]),
        })
    return seqs


def load_sequences(data_root: Path, use_cache: bool = True) -> list[dict]:
    repo = default_roots()["repo"]
    cache_dir = repo / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pq_path = cache_dir / "sequences.parquet"
    sig_path = cache_dir / "signature.json"
    if use_cache and pq_path.exists() and sig_path.exists():
        try:
            if json.loads(sig_path.read_text(encoding="utf-8")) == cache_signature(data_root):
                return _reconstruct(pd.read_parquet(pq_path))
        except Exception:
            pass
    seqs = list(iter_sequences(data_root))
    try:
        _flatten(seqs).to_parquet(pq_path, index=False)
        sig_path.write_text(json.dumps(cache_signature(data_root), ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return seqs


# ---------- 表构建与审计 ----------

def attach_split(seqs: list[dict], split: pd.DataFrame) -> list[dict]:
    ids = split.set_index("episode_uuid")
    for s in seqs:
        u = s["episode_uuid"]
        if u not in ids.index:
            s["split"] = None
            continue
        m = ids.loc[u]
        s["split"] = str(m["split"])
    return seqs


def dt_stats(t: np.ndarray) -> dict:
    t = np.asarray(t, float)
    if len(t) < 2:
        return {"n": int(len(t)), "duplicates": 0, "non_monotonic": 0, "gaps": 0}
    dt = np.diff(t)
    return {"n": int(len(t)), "duplicates": int((dt == 0).sum()),
            "non_monotonic": int((dt < 0).sum()),
            "gaps": int((dt > 1.5 * NOMINAL_DT).sum())}


def build_meta_table(seqs: list[dict], split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in seqs:
        t = np.asarray(s["time_s"], float)
        p = np.asarray(s["position"], float)
        ds = dt_stats(t)
        nan_rate = float(np.isnan(p).mean())
        fps = float(1.0 / np.median(np.diff(t))) if len(t) >= 2 and np.median(np.diff(t)) > 0 else None
        rows.append({
            "dataset": s["dataset"], "record_dir": s["record_dir"],
            "episode_uuid": s["episode_uuid"], "environment_id": s["environment_id"],
            "scene_id": s["scene_id"], "operator_id": s["operator_id"],
            "split": s.get("split"), "side": s["side"],
            "n_frames": len(t), "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            "dup_timestamps": ds["duplicates"], "non_monotonic": ds["non_monotonic"],
            "dt_gaps": ds["gaps"], "fps_estimated": round(fps, 2) if fps else None,
            "nan_rate_position": round(nan_rate, 6),
            "bad_frames": int(np.asarray(s["bad"], bool).sum()),
            "bad_ratio": round(float(np.asarray(s["bad"], bool).mean()), 6),
            "quality_known_ratio": round(float(np.asarray(s["quality_known"], bool).mean()), 6),
            "n_views": len(s.get("views", [])), "views": "|".join(s.get("views", [])),
            "source": s.get("source"),
        })
    return pd.DataFrame(rows)


def build_interval_table(seqs: list[dict], split: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    index = pd.read_csv(data_root / "Restocking_Tasks_and_Constraints" / "source_segment_index.csv")
    ids = split.set_index("episode_uuid")["split"]
    index["split"] = index.episode_uuid.map(ids)
    # 帧级索引：按 (dataset, episode_uuid, side) 快速定位
    frames = {}
    for s in seqs:
        frames.setdefault((s["dataset"], s["episode_uuid"], s["side"]), []).append(s)
    rows = []
    for r in index.itertuples(index=False):
        # 使用官方 duration_s 与 eligible 列，避免浮点重算 end-start 引入 1.1999… 类误差
        dur = float(r.duration_s)
        eligible = int(r.eligible)
        # 逐手统计 + 帧级并集（质量标记为逐帧逐手；并集口径=任一手的帧被标记即该帧低质量）
        side_stats: dict[str, tuple[int, int, np.ndarray]] = {}  # side -> (n_frames, bad, bad_array)
        nan_vals = []
        for side in ("left", "right"):
            for s in frames.get((r.dataset, r.episode_uuid, side), []):
                t = s["time_s"]
                mask = (t >= float(r.start_s)) & (t < float(r.end_s))
                bad_arr = np.asarray(s["bad"], bool)[mask]
                n = int(mask.sum())
                side_stats[side] = (n, int(bad_arr.sum()), bad_arr)
                nan_vals.append(float(np.isnan(np.asarray(s["position"], float)[mask]).mean()))
        n_frames = side_stats["left"][0] if "left" in side_stats else (
            side_stats["right"][0] if "right" in side_stats else 0)
        bad_union = None
        for _, (_, _, arr) in side_stats.items():
            bad_union = arr.copy() if bad_union is None else (bad_union | arr)
        bad_any = int(bad_union.sum()) if bad_union is not None else 0
        nan_max = max(nan_vals, default=0.0)
        bad_ratio = bad_any / n_frames if n_frames else None
        if eligible == 0:
            cls = "excluded_too_short"
            why = str(r.exclusion_reason) if pd.notna(r.exclusion_reason) else f"official duration={dur}s"
        elif bad_ratio is None:
            cls, why = "excluded_unknown", "无帧数据"
        elif bad_ratio > BAD_RATIO_MAX or nan_max > NAN_MAX:
            cls, why = "analysis_only_low_quality", f"bad_ratio_any={bad_ratio}, nan={nan_max}"
        else:
            cls, why = "qualified_reference", "满足全部阈值"
        rows.append({"unit_id": r.unit_id, "dataset": r.dataset, "episode_uuid": r.episode_uuid,
                     "environment_id": r.environment_id, "split": r.split,
                     "family_id": r.family_id, "start_s": r.start_s, "end_s": r.end_s,
                     "duration_s": dur, "eligible": r.eligible, "description": r.description,
                     "n_frames": n_frames,
                     "bad_frames_left": side_stats.get("left", (0, 0, None))[1],
                     "bad_frames_right": side_stats.get("right", (0, 0, None))[1],
                     "bad_frames_any": bad_any,
                     "bad_ratio_left": (side_stats["left"][1] / side_stats["left"][0]
                                        if "left" in side_stats and side_stats["left"][0] else None),
                     "bad_ratio_right": (side_stats["right"][1] / side_stats["right"][0]
                                         if "right" in side_stats and side_stats["right"][0] else None),
                     "bad_ratio": bad_ratio,
                     "nan_rate_hand_max": nan_max, "retention_class": cls,
                     "retention_reason": why})
    return pd.DataFrame(rows)


def audit_checks(seqs: list[dict], split: pd.DataFrame, data_root: Path) -> list[dict]:
    issues: list[dict] = []
    seen: dict[str, str] = {}
    for s in seqs:
        u = s["episode_uuid"]
        key = f"{s['dataset']}/{s['record_dir']}/{s['side']}"
        if u and u in seen and seen[u] != f"{s['dataset']}/{s['record_dir']}":
            issues.append({"code": "duplicate_uuid", "severity": "critical",
                           "message": f"{key}: uuid 与 {seen[u]} 重复", "record": key})
        seen.setdefault(u, f"{s['dataset']}/{s['record_dir']}")
        if s.get("split") is None:
            issues.append({"code": "local_orphan", "severity": "critical",
                           "message": f"{key}: 不在官方划分中", "record": key})
        else:
            m = split.set_index("episode_uuid").loc[u]
            for field in ("dataset", "environment_id"):
                if str(s.get(field, "")).strip() and str(s[field]).strip() != str(m[field]).strip():
                    issues.append({"code": "meta_mismatch", "severity": "warning",
                                   "message": f"{key}: {field} 记录内 {s[field]} ≠ 官方 {m[field]}",
                                   "record": key})
        ds = dt_stats(np.asarray(s["time_s"], float))
        if ds["duplicates"] or ds["non_monotonic"] or ds["gaps"]:
            issues.append({"code": "time_anomaly", "severity": "warning",
                           "message": f"{key}: dup={ds['duplicates']} nonmono={ds['non_monotonic']} gaps={ds['gaps']}",
                           "record": key})
    local_uuids = {s["episode_uuid"] for s in seqs}
    for u in split.episode_uuid:
        if u not in local_uuids:
            issues.append({"code": "split_orphan", "severity": "critical",
                           "message": f"官方划分中的 {u} 无本地序列", "record": ""})
    return issues


if __name__ == "__main__":
    print(json.dumps({"repo_root": str(default_roots()["repo"])}, ensure_ascii=False, indent=2))

