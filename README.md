# DeepSeek_Cup — 2026 数维杯 C 题可复现代码包

> 本包包含代码、文档、汇总结果与复现说明，不包含原始数据。数据根目录需由运行者在本机自行指定；本仓库不保存个人目录路径。

> GitHub 中提交源码、复现说明和汇总结果表。包含记录/区间标识、原始文件清单或逐样本特征的细粒度导出、逐帧回放、Q2 全量窗口回放、Q3 逐动作 process log 与本地 Parquet 缓存保留在本机；按下文命令可重新生成。原始题目数据不提交。

## 1. 运行环境

- Python：Python 3.13；在 PyCharm 中选择已安装所需依赖的项目解释器
- 依赖版本（requirements.txt）：numpy 2.5.5 / pandas 3.1.1 / pyarrow 26.0.1 /
  scikit-learn 1.9.1 / scipy 1.18.1 / matplotlib 3.11.2
- 数据根（只读）：运行者本机的题目数据目录；运行脚本时通过 `--data-root` 指定

## 2. 代码结构

```
code/unified_data.py        统一读取层（v3：官方 eligible 口径 + 逐手坏帧标记）+ Parquet 缓存
code/audit_data.py          阶段 C 数据审计（--sample N / --full / --no-cache）
code/q1_state_quality.py    阶段 D Q1：CV-KF（Joseph 协方差更新）+ 观测可靠性逻辑回归
code/q2_mask_replay.py      阶段 E Q2：四方法因果回放 + 防泄漏追踪 + 环境分组汇总
code/q3_scheduler.py        阶段 F Q3：前缀分位数时长预测 + 协议仿真器 + 三策略（v3 延误修正）
code/q3_tune.py             Q3 验证集两阶段调参
code/q3_analyze.py          Q3 汇总/0.50 门槛/配对比较
code/recompute_trial.py     独立重算：按题面公式从 process log 重放单条试验 11 项指标
code/make_manifest.py       结果清单（完整 64 位 SHA256，排除清单自身）
work/*.md                   设计 schema、候选模型表、各问结论说明
docs/literature/            文献矩阵（12 篇，DOI 经注册库核实）
docs/local_validation_20261122.md 三问本地复跑审查、关键数值与论文限制
docs/figures/               三问论文用汇总图（仅使用聚合指标）
outputs/paper_ready_20261122/ 最新修正版三问结果、Q3试验级指标和聚合对比表
outputs/                    本地完整回放明细与过程日志；不上传原始/逐样本材料
data_cache/sequences.parquet 帧级序列缓存（输入签名不变时复用）
reports/run_log.md          本地执行日志（含本机数据标识，不上传）
```

## 3. 关键命令（复现顺序）

```powershell
$VENV = "python" # 或 PyCharm 当前项目解释器的完整路径
$DATA = "<本机题目数据根目录>"
$REPO = (Get-Location).Path # 先在项目根目录打开 Terminal

# 阶段 C：数据审计（全量，v4 逐手+并集口径；生成缓存）
& $VENV $REPO\code\audit_data.py --data-root $DATA --output $REPO\outputs\audit_full_v4 --full

# 阶段 D：Q1（v2：顺序推进卡尔曼 + 最近两帧常速度 + 保留重复环境 bootstrap）
& $VENV $REPO\code\q1_state_quality.py --data-root $DATA --output $REPO\outputs\q1_full_v2

# 阶段 E：Q2（v2：trace 字段细化）
& $VENV $REPO\code\q2_mask_replay.py --data-root $DATA --output $REPO\outputs\q2_full_v2

# 阶段 F：Q3 validation（默认参数）
& $VENV $REPO\code\q3_scheduler.py --data-root $DATA --output $REPO\outputs\q3_validation
& $VENV $REPO\code\q3_analyze.py --input $REPO\outputs\q3_validation\q3_trial_metrics.csv --output $REPO\outputs\q3_analysis_default

# 阶段 F：Q3 验证集调参 + tuned 重跑（tuned 参数只作用于 adaptive/no_review；EDD 为规范基线）
# q3_tune.py 会同时输出 q3_tuned_params.json（含诊断字段）与 q3_tuned_params_flat.json（顶层即参数字典，调度器直接读取）
& $VENV $REPO\code\q3_tune.py --prefix $REPO\outputs\q3_validation\prefix_features.csv --task-root $DATA\Restocking_Tasks_and_Constraints --output $REPO\outputs\q3_tune
& $VENV $REPO\code\q3_scheduler.py --data-root $DATA --output $REPO\outputs\q3_validation_tuned_v3 --params-json $REPO\outputs\q3_tune\q3_tuned_params_flat.json
& $VENV $REPO\code\q3_analyze.py --input $REPO\outputs\q3_validation_tuned_v3\q3_trial_metrics.csv --output $REPO\outputs\q3_analysis_tuned_v3

# 独立重算核验（单条试验，按题面公式从 process log 重放；输出为 ASCII，默认 GBK 控制台可直接运行）
& $VENV $REPO\code\recompute_trial.py --metrics $REPO\outputs\q3_validation_tuned_v3\q3_trial_metrics.csv --log $REPO\outputs\q3_validation_tuned_v3\q3_process_log.csv --tasks $DATA\Restocking_Tasks_and_Constraints\tasks.csv --seed 2026111700 --case S1 --policy adaptive --alpha 1.0 --beta 0.5

# 结果清单（完整 SHA256，排除清单自身）
& $VENV $REPO\code\make_manifest.py --repo $REPO
```

## 4. 纪律（与计划一致）

- 环境级划分 16/4/4；模型只在 train 拟合、validation 选模/调参、冻结后评 test；
- 无组委会 test seeds：Q3 停在 validation，不伪造 test；
- 低质量区间只作低质量分析，不作无误差真值；合格留出仅作记录参照；
- 手状态 ≠ 商品状态；区域坐标 ≠ 货架位置；场景结果 ≠ 机器人性能；
- Q3 同 seed 各策略同源配对；未完成按 C=H 计延误；违规试验保留。

## 5. 论文素材

三问复跑和代码审查结论见 `docs/local_validation_20261122.md`；最新结果表在 `outputs/paper_ready_20261122/`，图片在 `docs/figures/`，可用 `code/plot_public_figures.py` 重画。Q1 质量分类器弱区分力、Q3 分位数欠覆盖、EDD 未达门槛以及缺少独立官方测试 seeds 均需如实说明。

官方 Q3 要求提交试验级指标和逐动作 process log。两者现均已发布在 `outputs/paper_ready_20261122/`；process log 含源区间引用和逐动作资源变化，供队友核验协议。原始轨迹数据仍不随代码仓库发布。

当前论文数字以 `docs/local_validation_20261122.md` 和 `outputs/paper_ready_20261122/` 为准。`work/q3_results_notes.md` 是早期版本，含旧策略数字和逐任务示例；自动审核拒绝再次公开更新该文件，请勿引用其中的策略指标。

