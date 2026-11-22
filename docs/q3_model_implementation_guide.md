# Q3 模型、文献与代码实现指南

> 用途：供队友按当前 Q3 仓库代码复现模型、理解策略并撰写论文。本文描述的是 `code/q3_scheduler.py` 当前实际实现。理论背景或可扩展方法会单独标明，不会把它们说成已实现算法。

## 一、问题拆解与解题思路

Q3 不只是排列任务顺序。每项任务有释放时间、截止时间、优先权重、前置关系和货架区域；设备只有一台；真实执行时长不确定，只能在执行前看到 0.2、0.6、1.0 秒逐步揭示的前缀；复核需付费并等待 1 秒；此外还有总预算和计划时域约束。

当前代码分为两层：

1. **时长预测层**：根据当前已揭示的前缀统计量和任务 family，估计执行时长的条件分位数 Q50 与 Q90。
2. **在线调度层**：每到一个决策点，更新可执行任务集合，先检查硬约束，再比较任务得分；adaptive 策略可付费取得更长前缀并重新决策。

这是“预测—执行—获得新信息—重新决策”的在线启发式结构，便于逐动作审计并与基线做同源配对。它不是全局最优调度算法。

## 二、当前代码实际使用的模型

### 1. 时长特征与监督目标

代码从题目提供的合格源区间提取样本。在每个区间的 0.2、0.6、1.0 秒前缀上，对左右手分别计算统计量后取均值：

- `prefix_s`：当前揭示时长；
- `n_obs`、`bad_fraction`：有效观测数和坏帧比例；
- `speed_mean`、`speed_sd`：关键点代表位置的帧间速度均值和标准差；
- `disp`、`spread`：前缀首末位移和关键点位置离散度；
- `family_id`：任务类别，作为 one-hot 特征。

目标是区间标注时长 `duration_s = end_s - start_s`。拟合目标用 `log1p(duration_s)`，预测后以 `expm1` 还原为秒数。模型只用 `split=train` 环境拟合；validation 用于参数选择和指标报告；没有官方 test seeds 时不运行或声称测试结果。

### 2. Family 中位数基线

对每个 `family_id`，用训练环境时长中位数预测。该基线不使用前缀特征，可检查更复杂的模型是否真的增加预测价值。

### 3. Q50/Q90 分位数梯度提升

当前实现为两个独立的 `sklearn.ensemble.GradientBoostingRegressor`：

- `loss="quantile"`；
- `alpha=0.50` 和 `alpha=0.90`；
- `n_estimators=120`、`max_depth=2`、`min_samples_leaf=8`；
- 数值特征标准化，`family_id` one-hot；
- 固定 `random_state=20261117`。

分位数回归最小化 pinball/check loss。令误差 `u = y - predicted_q_tau(x)`，对应损失为：

`rho_tau(u) = u * (tau - I[u < 0])`

Q50 对应条件中位数；Q90 用来估计较保守的上尾时长。分位数能体现不对称低估/高估风险，但并不等于已经拟合出完整真实分布。

**实现引用**：分位数回归与 pinball loss 可引 Koenker & Hallock [1]；梯度提升框架可引 Friedman [2]；`loss="quantile"` 和 `alpha` 参数的实际软件实现应引用 scikit-learn 官方文档 [3]。

> **模型名称要写准确**：当前脚本用的是 scikit-learn Gradient Boosting Regressor，不是 XGBoost。文献矩阵中的 Chen & Guestrin (2016) 介绍 XGBoost，可作为备选模型背景，不是当前实现的直接算法引用。若论文写成 XGBoost，模型描述就与代码不符。

### 4. 训练环境留一 Q90 残差校准

原始 Q90 在验证环境覆盖率为 0.8000，低于名义 0.90。代码增加以下训练内校准：

1. 逐个留出一个训练环境，用其余训练环境拟合 Q90；
2. 对被留出的环境产生 OOF 预测，计算 residual = actual_duration - oof_q90；
3. 取 margin = max(0, residual 的 90% 分位数)；
4. 最终预测为 q90_cal = max(q50, q90_raw + margin)。

本次 16 个训练环境产生 738 条 OOF 行，margin 为 3.858971 秒；训练 OOF 覆盖率从 0.8157 提升到 0.8997。校准只使用训练环境。

Romano, Patterson & Candès 的 conformalized quantile regression (CQR) [4] 是分位数校准的重要背景，但**当前代码不是论文中的完整 CQR 算法**：这里采用逐环境留一 OOF、行级残差池化和单侧加法校准。因此不要宣称有限样本覆盖率保证或“严格 conformal”。验证集只有 21 条时，校准 Q90 覆盖率为 0.9524（20/21），同时 Q90 MAE 从 9.2395 上升到 12.2143 秒。应表述为“修正欠覆盖，但在当前小验证集偏保守”。

### 5. 预测不确定性、执行预留和任务效用

#### 5.1 分位数映射与不确定性近似

代码通过 Q50、Q90 反推出近似 log-normal 参数：

- mu = log(max(0.05, q50 - beta) / alpha)
- sigma = max(sigma_min, [log(max(q90 - beta, 0.051) / alpha) - mu] / 1.28155)

然后计算时长均值、标准差，并以正态 CDF 近似截止前完成概率 p_due 和时域内完成概率 p_horizon。场景映射为 p = alpha * duration + beta，预测分位数也按相同映射变换。

这是**计算策略得分用的分布近似**，并非经统计检验确认的真实时长分布。队友改动时要分别报告分位数覆盖率和策略效果，不能把正态 CDF 得到的概率说成精确真实概率。

#### 5.2 预留时长

当前预留公式：

`reserve = max(0.2, q50 + reserve_quantile_mix * (q90 - q50))`

调参后 `reserve_quantile_mix=0.5`，即在 Q50 与校准 Q90 之间线性插值。这是用验证集选出的风险—空闲折中启发式，不是由明确的低估成本/高估成本推导出的报童临界分位数。

Qin et al. [7] 的 newsvendor/critical-fractile 理论可作为“非对称过量/不足成本”的动机背景，但不能据此声称本代码求得了理论最优预留分位数。当前插值系数是验证集调参结果，没有从成本分布解析求解。

#### 5.3 执行得分

对当前可执行候选，代码近似计算 p_due 和 p_horizon，得分为：

`score = weight * (w_due * p_due + w_horizon * p_horizon) - w_cost * (2 + 0.1 * q90 + 0.2 * distance) - w_dist * distance`

注意：得分里的成本惩罚使用 q90 估计；硬约束检查实际费用时使用 reserve，即 fee = 2 + 0.1 * reserve + 0.2 * distance。选取得分最高且通过预算、时域、release、precedence 检查的任务。最新验证调参参数为 `w_due=1.0`、`w_cost=0.06`、`reserve_quantile_mix=0.5`；其余参数见 `outputs/paper_ready_20261122/q3_tuned_params_flat.json`。

#### 5.4 三种策略

- **EDD 基线**：实际排序分数为 `-due_s - 0.02 * distance`。距离项是额外移动惩罚，会改变纯 EDD 排序，因此应称为 EDD-like 基线，不能引用纯 EDD 的最优性结论。单机调度理论背景见 Pinedo [5]。
- **no_review**：使用风险得分调度，但不付费延长前缀。
- **adaptive**：当可复核任务的相对不确定性 `(q90-q50)/max(0.2,q50)` 和优先权加权的“价值代理”超过阈值，且复核分数高于当前最佳可执行候选分数乘 `review_vs_exec_factor` 时，支付 1 单位成本并等待 1 秒，更新到下一前缀后重新决策。

adaptive 的复核价值是启发式代理，不是由后验期望成本差算出的 EVII，也没有求解 POMDP。Howard 的信息价值理论可作为扩展方向，但不能称为当前复核策略的精确推导。Ouelhadj & Petrovic [6] 可作为动态/预测—反应式调度的背景。

## 三、仿真与代码边界

每个 trial 中，仿真器保留抽中的 `source_unit_id` 和真实时长；策略只接收已揭示前缀特征、公开任务字段和当前状态。执行后，仿真器才根据真实时长结算结果并写入过程日志。这一信息边界是防止数据泄漏的关键：

- 决策过程中不能读取 `duration_s`、未来前缀、源区间 ID 或验证结果；
- 同一 seed/case/mapping 下各策略复用相同源任务样本；
- 每个 trial 独立按官方 seed 初始化随机数，任务顺序固定；
- 复核返回前不能使用下一前缀；
- 截止 H 前只能用公开预留时长做预算和可行性预检，不能偷看真实未来时长；
- 费用、资源释放、完成/截尾和指标由模拟器按题面公式结算。

复现入口：

- `code/q3_scheduler.py`：前缀特征、训练、调度、过程日志；
- `code/q3_tune.py`：只用 validation seeds 调策略参数；
- `code/q3_analyze.py`：名义映射 0.50 门槛、五映射分组汇总、配对比较；
- `code/recompute_trial.py`：独立按过程日志重算单条试验；
- 当前权威结果：`outputs/paper_ready_20261122/`。

## 四、最新验证结果与论文表述边界

| 结果口径 | 场景 | adaptive | no-review | EDD |
|---|---|---:|---:|---:|
| 名义映射 alpha=1, beta=0.5，30 seeds | S1 | 0.7917 | 0.8271 | 0.3813 |
| 同上 | S2 | 0.6097 | 0.6236 | 0.2722 |
| 同上 | S3 | 0.5073 | 0.5250 | 0.2135 |
| 五种映射、30 seeds 合并均值 | S1 | 0.7925 | 0.8113 | 0.3750 |
| 同上 | S2 | 0.5872 | 0.6019 | 0.2628 |
| 同上 | S3 | 0.5004 | 0.5219 | 0.2135 |

no-review 在三种场景的加权准时率都高于 adaptive，且成本更低，所以当前结果不支持“adaptive 全面优于 no-review”。adaptive/no-review 在名义映射下都达到 0.50 均值门槛；五映射合并的 adaptive S3=0.5004，贴近门槛。EDD 未达门槛。校准诊断和局限详见 `docs/local_validation_20261122.md`。

这些是 validation 调参和验证结果；官方 Q3 test seeds 未提供，不能写成测试集表现。参数是在 validation 上选出，又在同一 validation 上报告策略效果，门槛表现可能有调参乐观偏差。官方 test seeds 到位后，应冻结模型、参数和代码，再做独立 test 评估。

## 五、代码实现注意事项

1. **只用题目文件内数据训练**；不要下载外部时长标签、公开数据或人为补标签。文献只能说明方法依据，不是训练数据来源。
2. **先按 environment 划分，再拟合**；同一环境的片段不能跨 train/validation/test。Q90 校准 OOF 也必须留环境。
3. **不要将调度器的隐变量暴露给策略**；真实 `duration_s` 只用于拟合模型和模拟器事后结算。
4. **保持分位数次序**；确保 `q90 >= q50`，校准量不允许为负。
5. **所有硬约束先检查再下单**：释放时间、前置任务、预算、`start + reserve <= H`、复核上限、动作上限。违规需留日志并计数，不能静默修复。
6. **完成/截尾口径不要改**：完成时 C=finish；未完成时 C=H；延误按 `weight * max(C-due,0)` 计。未完成任务仍在总权重分母内。
7. **不要只报均值**：同时给场景/映射分组、随机种子配对、成本、复核次数、协议违规、分位数 pinball 与覆盖率。
8. **保留限制和失败基线**：Q90 校准在小验证集偏保守、adaptive 未胜 no-review、EDD 未达门槛、无官方 test seeds 都要如实写。
9. **改代码后独立重算一条 trial**；核对日志字段、source 配对、指标口径和 `valid_protocol`。
10. **不要把背景文献当算法证明**：只有代码实现相应公式和假设后，才能称为 POMDP、EVII、newsvendor optimal 或严格 conformal 算法。

## 六、参考文献

[1] Koenker, R., & Hallock, K. F. (2001). Quantile Regression. *Journal of Economic Perspectives*, 15(4), 143–156. [DOI: 10.1257/jep.15.4.143](https://doi.org/10.1257/jep.15.4.143). 分位数回归与 pinball loss 背景。

[2] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232. [DOI: 10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451). 梯度提升框架依据。

[3] scikit-learn Developers. `GradientBoostingRegressor` API documentation. [Official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html). 当前代码 `loss="quantile"` 和 `alpha` 参数的软件实现说明。

[4] Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. *Advances in Neural Information Processing Systems 32*. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html). 仅作分位数校准背景；当前代码不是其完整方法。

[5] Pinedo, M. L. (2022). *Scheduling: Theory, Algorithms, and Systems* (6th ed.). Springer. [DOI: 10.1007/978-3-031-05921-6](https://doi.org/10.1007/978-3-031-05921-6). 单机调度理论背景；题目参考文献指定第 6 版时优先使用此版本。

[6] Ouelhadj, D., & Petrovic, S. (2009). A survey of dynamic scheduling in manufacturing systems. *Journal of Scheduling*, 12(4), 417–431. [DOI: 10.1007/s10951-008-0090-8](https://doi.org/10.1007/s10951-008-0090-8). 在线动态、预测—反应式调度背景。

[7] Qin, Y., Wang, R., Vakharia, A. J., Chen, Y., & Seref, M. M. H. (2011). The newsvendor problem: Review and directions for future research. *European Journal of Operational Research*, 213(2), 361–374. [DOI: 10.1016/j.ejor.2010.11.024](https://doi.org/10.1016/j.ejor.2010.11.024). 非对称过量/不足成本背景；当前插值预留并未实现报童临界分位数。

补充说明：现有 `docs/literature/literature_matrix.md` 另列了 Chen & Guestrin (2016) 的 XGBoost 论文（DOI: 10.1145/2939672.2939785）。该文可支持 XGBoost 备选模型的讨论，不应替代 [1]–[3] 作为当前 scikit-learn 模型的实现引用。
