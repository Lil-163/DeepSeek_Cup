# 统一数据层 schema（阶段 C 设计，父 Agent）

## 1. 输入源与口径

- 数据根 `EN` 目录：`Raw_Data/{EgoDemo,EgoPro,EgoStandard}`（解压树，主读取源）+ 三个 zip（校验对照）。
- 记录映射：EgoDemo 用 `meta/episodes/chunk-000/file-000.parquet` 的 `episode_uuid`；
  EgoPro/EgoStandard 用 `session_metadata.csv` 的 `task_info.episode_uuid`。
- 官方划分：`Restocking_Tasks_and_Constraints/record_split.csv`（69 行、24 环境、16/4/4），
  与记录内部 environment_id/scene_id/operator_id 交叉校验。

## 2. 规范表

### 2.1 元数据表 meta（每记录 1 行）
`dataset, record_dir, episode_uuid, environment_id, scene_id, operator_id, split,
 n_frames, duration_s, fps_estimated, n_views, views, bad_frames_total, bad_ratio_total,
 nan_rate_hand, n_segments, n_segments_ge_1_2s, source_path`

### 2.2 序列表（帧级，长表 → Parquet 缓存）
每 (dataset, episode_uuid, side) 一条序列：
`time_s f64 | frame_index i64 | px py pz f64 | kp_{00..20}_{x,y,z} 63×f64 |
 bad bool | bad_reason str | quality_known bool | stage str`
- 代表手点 = 21 个 3D 关键点的逐坐标中位数（px,py,pz）；关键点原值全部保留（kp_*）。
- bad：原始质量标记（EgoDemo bad_frame_ratio.json 按帧索引；EgoPro/Standard 按
  `_message_index` 对齐 annotation_bad_frame_pose_hand.csv 的 is_bad 与 problem_type）。
- quality_known：EgoDemo 恒 True；EgoPro/Standard = 是否在坏帧表中匹配到该消息索引。

### 2.3 区间表 interval（每语义区间 1 行）
以 `source_segment_index.csv` 为基准（349 行）：`unit_id, episode_uuid, dataset, split,
 family_id, start_s, end_s, duration_s, eligible, description`；
审计附加列（v4 修正版）：`n_frames, bad_frames_left, bad_frames_right, bad_frames_any,
 bad_ratio_left, bad_ratio_right, bad_ratio, nan_rate_hand_max, retention_class`
- 坏帧比例口径：**逐手报告**（left/right），保留判定用**帧级并集** `bad_ratio_any`
  （任一手的帧被质量标记即计该帧坏；不再只统计单侧）。
- retention_class：qualified_reference（duration≥1.2 且 bad_ratio_any≤0.10 且 nan≤0.10）
  / analysis_only_low_quality / excluded_too_short（官方 eligible=0）/ anomaly_flag。

## 3. 统一接口（code/unified_data.py）

```python
load_split(data_root) -> pd.DataFrame                    # 官方划分
iter_sequences(data_root, sample=None) -> Iterator[dict] # 解压树解析（可小样本）
load_sequences(data_root, use_cache=True) -> list[dict]  # + Parquet 缓存 data_cache/sequences.parquet
build_meta_table(seqs, split) -> pd.DataFrame
build_interval_table(seqs, split, index) -> pd.DataFrame
audit_checks(seqs, split, index) -> list[dict]           # 问题清单（code/severity/message）
cache_signature(data_root) -> dict                       # zip 大小+mtime、解压树文件大小摘要
```

## 4. 检查项（audit_checks）

1. 官方划分 ↔ 本地记录双向孤立（split_orphan / local_orphan）；
2. episode_uuid 跨目录/数据集重复；
3. 记录内元数据（env/scene/operator/dataset）与官方划分不一致；
4. 时间戳：重复、非单调、缺口（dt>1.5×名义 1/30s）、行数 vs 元数据统计、视频 to_timestamp 偏差；
5. 坐标：NaN 率（数值通道）、手部位置量级（米）、四元数模长≈1；
6. 质量标记：坏帧计数、原因分布、区间内坏帧率；
7. 环境级划分互斥：同 environment_id 只属一个 split。

## 5. 保留规则（与题面一致，写死为常量）

- min_interval_s = 1.2（349→343 口径）；bad_ratio_max = 0.10；nan_max = 0.10；
- 低质量区间仅用于低质量观测分析，**绝不作为无误差真值/参照**；
- 合格区间仅作记录参照（recorded reference），非物理真值；
- 环境级划分不可拆；坏帧从状态拟合样本剔除、保留给质量模型作正样本。

