# 结果文件索引（最终版本，含 2026-11-21 修正轮）

> 本机权威版本目录如下（旧版全部保留作对照）；本机全部输出的完整 64 位 SHA256 见 outputs/_files_manifest.txt。GitHub 仅提交汇总表，细粒度明细和该本机全量清单不上传。

> GitHub 提交包含下表的紧凑权威汇总文件和 Q3 试验指标；逐帧/逐窗口回放明细、Q3 process log、旧版重复结果及本地缓存仍在本机，不纳入本次提交。需要完整明细时按 README 复现命令生成。

| 阶段 | 本机权威目录 | GitHub 提交的关键汇总文件 | 说明 |
|---|---|---|---|
| C 审计 | outputs/audit_full_v4/ | audit_summary.json | v4：逐手+并集坏帧汇总；含记录/区间标识的明细留在本机 |
| D Q1 | outputs/q1_full_v2/ | q1_state_by_stage.csv / q1_horizon_error_summary.csv / q1_env_bootstrap_ci.csv / q1_quality_metrics.csv / q1_summary.json | v2：步长与误差汇总；逐记录/环境标识明细留在本机 |
| E Q2 | outputs/q2_full_v2/ | q2_summary.csv / q2_by_stage.csv / q2_calibration.json / q2_low_quality_summary.csv / q2_summary.json | v2 汇总；逐窗口 trace 与事件明细留在本机 |
| F Q3 调参 | outputs/q3_tune/ | q3_tune_results.csv / q3_tuned_params.json / q3_tuned_params_flat.json | 两阶段验证集搜索 |
| F Q3 tuned | outputs/q3_validation_tuned_v3/ + outputs/q3_analysis_tuned_v3/ | q3_trial_metrics.csv / q3_duration_validation.csv / q3_gate_0_50_nominal.csv / q3_paired_comparison.csv / q3_validation_summary.csv | **最终方案**：v3 延误修正（完成=finish/未完成=H）；逐动作 process log 与前缀特征表留在本机 |
| 独立重算 | code/recompute_trial.py | 按题面公式从本机 process log 重放单条试验 11 项指标 | 已对 v3 metrics 核验一致 |
| 缓存 | data_cache/sequences.parquet + signature.json | 不提交 | 本机帧级序列缓存；输入签名不变时复用 |

结论说明：work/q1_results_notes.md、work/q2_results_notes.md、work/q3_results_notes.md（均含修正轮说明）。
文献：docs/literature/literature_matrix.md。完整执行日志与全量文件清单仅保留在本机。

论文写作新增素材（2026-11-22）：`docs/local_validation_20261122.md`（三问复跑审查与限制）、`docs/figures/`（四张聚合指标图）、`code/plot_public_figures.py`（从已提交聚合 CSV 复现图片）。Q3 两阶段调参复核参数与冻结配置一致；独立 1,350 次验证复跑摘要与权威输出一致。
