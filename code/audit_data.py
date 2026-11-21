# -*- coding: utf-8 -*-
"""audit_data.py — 阶段 C 数据审计（由代码执行子 Agent B 运行）。

用法：
  python audit_data.py --data-root <EN目录> --output <目录> [--sample N] [--full] [--no-cache]
  --sample N ：小样本模式，每个数据集只解析前 N 条记录（不写缓存）
  --full     ：全量解析（69 条记录）并写 Parquet 缓存
输出：manifest.csv、record_audit.csv、interval_audit.csv、audit_issues.csv、audit_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_data import (attach_split, audit_checks, build_interval_table,
                          build_meta_table, cache_signature, iter_sequences,
                          load_sequences, load_split, MIN_INTERVAL_S)


def manifest(data_root: Path, output: Path) -> None:
    raw = data_root / "Raw_Data"
    rows = []
    for ds in ("EgoDemo", "EgoPro", "EgoStandard"):
        z = raw / f"{ds}.zip"
        if z.exists():
            rows.append({"kind": "zip", "dataset": ds, "name": z.name,
                         "bytes": z.stat().st_size})
        tree = raw / ds / ds
        if tree.exists():
            for p in sorted(tree.rglob("*")):
                if p.is_file():
                    rows.append({"kind": "file", "dataset": ds,
                                 "name": str(p.relative_to(tree)),
                                 "bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    split = load_split(args.data_root)
    manifest(args.data_root, args.output)

    if args.full:
        seqs = load_sequences(args.data_root, use_cache=not args.no_cache)
    else:
        seqs = list(iter_sequences(args.data_root, sample=args.sample or None))
    seqs = attach_split(seqs, split)

    meta = build_meta_table(seqs, split)
    meta.to_csv(args.output / "record_audit.csv", index=False, encoding="utf-8-sig")
    intervals = build_interval_table(seqs, split, args.data_root)
    intervals.to_csv(args.output / "interval_audit.csv", index=False, encoding="utf-8-sig")
    issues = audit_checks(seqs, split, args.data_root)
    pd.DataFrame(issues).to_csv(args.output / "audit_issues.csv", index=False, encoding="utf-8-sig")

    summary = {
        "mode": "full" if args.full else f"sample({args.sample or 'all'})",
        "n_sequences": int(len(seqs)),
        "n_records": int(meta.drop_duplicates("episode_uuid").shape[0]),
        "n_frames": int(meta.n_frames.sum()),
        "records_by_dataset": {ds: int((meta.dataset == ds).drop_duplicates().sum())
                               if False else int(meta[meta.dataset == ds].episode_uuid.nunique())
                               for ds in ("EgoDemo", "EgoPro", "EgoStandard")},
        "split_counts": meta.drop_duplicates("episode_uuid").split.value_counts().to_dict()
        if meta.split.notna().any() else {},
        "environments_by_split": split.groupby("split").environment_id.nunique().to_dict(),
        "n_intervals": int(len(intervals)),
        "n_intervals_ge_1_2s": int((intervals.duration_s >= MIN_INTERVAL_S).sum()),
        "retention_counts": intervals.retention_class.value_counts().to_dict(),
        "bad_frames_total": int(meta.bad_frames.sum()),
        "issues_by_code": pd.DataFrame(issues).code.value_counts().to_dict() if issues else {},
        "elapsed_s": round(time.time() - t0, 1),
        "signature": cache_signature(args.data_root),
    }
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

