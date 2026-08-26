# 实验设计书：ChronoRL：按任务难度自适应分配测试时推理预算以修复 LLM 金融时间理解

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea02_ChronoRL.md
> 依赖：Idea01_PiT-AuditBench（训练/评测数据）

## 0. 摘要

ChronoRL 把"测试时推理预算（reasoning effort）"作为金融时序任务上的**可学习决策**。核心观察来自 Chronology：推理模型 + 扩展推理能显著改善时间排序，但全量开启成本高，且过度推理反而伤害金融情感精度（Reasoning or Overthinking，2506.04574）。本设计训练一个轻量路由（router），输入"Chronology 复杂度特征"（序列长度、实体重叠、条件数、时间线数、术语密度、日期跨度），输出当前样本应使用的 effort 档（low/medium/high），目标是最小化"正确率损失 × 成本节约"的权衡。路由可用监督学习（收集三档 effort 的 (特征, 正确率, 成本) 数据后训练 XGBoost）或直接 RL（策略=选档），并在不同模型家族之间验证迁移性。产出包括 Pareto 曲线下面积（Accuracy-Cost AUC）、预算受限下的 exact-match，以及"路由策略 vs 固定 effort"的差距。

## 1. 研究背景与动机

### 1.1 问题定义

给定时序理解任务样本 s（来自 Idea 1 的 PiT-AuditBench），评测器 F 在 effort 档位 e∈{low, medium, high} 下给出正确率 a(s,e) 与成本 c(s,e)（token 数）。目标是学习路由策略 π: φ(s) → e，使 `E[ a(s,π(s)) ] − λ·E[ c(s,π(s)) ]` 最大化（λ 为成本惩罚，调参控制预算）。评测时：路由 → 选档 → 调 LLM → 规则计分 → 汇总。

### 1.2 相关工作不足

- **Chronology（AAAI Emerging Trends in AI · 论文 183）**：给出"推理预算越高排序越好"的现象，但未给出"该花多少推理"的决策方法，且其评测是固定 effort。
- **Reasoning or Overthinking（arXiv:2506.04574）**：证明金融情感任务上过度推理有害——说明"全高 effort"不是免费午餐，恰好是本设计的动机支点。
- **PiT-Inference（arXiv:2512.06607）**：用推理时 logits 调整去偏，但针对"遗忘"而非"预算分配"。
- 现有 effort 自适应研究多为通用推理（A/B 测试式），无面向金融时序复杂度特征的专门路由。

### 1.3 为什么是现在、为什么你的环境适合做

推理模型（DeepSeek V4 Pro、Kimi K2.6）按 token 计费且支持 effort 参数，使"预算分配"成为真实成本问题而非理论问题。环境优势：(1) 可直接复用 Idea 1 的 PiT-AuditBench 训练/评测管线；(2) 4×L40 用于本地小路由训练与批量评测并行，API 预算可控；(3) 可同时覆盖开源（Qwen2.5-7B 本地 + 推理）与闭源（DeepSeek V4 Pro/Kimi K2.6）模型，验证路由迁移性。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 不同复杂度样本的最优 effort 档显著不同 | 存在样本使 high-effort 比 low-effort 提升 >10 个点，同时存在样本提升 <2 个点（最优档分布非退化） |
| H2 | 复杂度特征可预测最优 effort 档 | 路由在 hold-out 样本上的 Accuracy-Cost AUC 显著优于固定 high/low 与随机基线（paired 检验，p<0.05） |
| H3 | 路由跨模型家族可迁移 | 在 DeepSeek V4 Pro 上学到的路由直接用于 Qwen2.5-7B 或 Kimi K2.6，性能退化 <15% |
| H4 | 路由决策能同时提升效率与正确率 | 在目标成本预算（如成本压缩 50%）下，路由的 exact-match ≥ 固定 high 的 95% |

第一验证实验：在 PiT-AuditBench 样本上采集三档 effort 数据（≥3k 样本 × 3 档），画"样本复杂度 → 最优档"的分布，检验 H1。

## 3. 总体方法设计

### 3.1 数据流水线

- 样本源：Idea 1 的 PiT-AuditBench（T1-T4 全量，含窗口内外分层）；
- **多档数据采集**（DeepSeek V4 Pro 为主，Kimi K2.6 做难样本补充）：对每个样本在 low/medium/high 三档 effort 各跑 1 次（temperature=0.2），记录正确性（规则计分）、输出 token 数、推理延迟；
- **复杂度特征 φ(s)**：排序长度 m、事件数、实体重叠率（同名实体跨事件比例）、条件数 c、时间线数 k、日期跨度（天）、术语密度（金融词表命中率）、事件类型混合度；
- 数据量：5,000 样本 × 3 档 = 15,000 条 (样本, 档, 正确率, 成本) 记录；其中 500 条双档重跑评估噪声；
- 防泄漏：特征与标签均只来自样本与评测器，不引用任何未来信息；样本按事件日期切分，窗口外样本不进训练。

### 3.2 方法设计

- **路由模型**：优先 XGBoost（可解释、快、无需 GPU）；同时实现小型 MLP（3 层 256 维）对比；RL 变体：策略 π_θ 输出三档概率，用 REINFORCE 优化 `E[a] − λ·E[c]`（λ∈{0.5,1,2} 网格），logits 加温度退火；
- **监督版目标**：先由数据求每个样本的"经验最优档" e*(s) = argmax_e [a(s,e) − λ̂·c(s,e)]（λ̂ 取 1），训练分类器；再对 λ 做多解，输出帕累托前沿；
- **推理时**：`e = π(φ(s))`；若 e=high 时仍有空（极难样本）可再做一致性重采样（pass@2）；
- **成本模型**：c 用实际 token 计数；跨模型迁移时对 token 做归一化（per-token 单价不同）；
- **前视偏差处理**：路由只依赖样本静态特征与历史（训练时已知）的评测结果，不依赖未来的正确率。

### 3.3 训练流程

- XGBoost：`max_depth=5, lr=0.1, n_estimators=300`，5-fold CV；
- MLP：Adam `lr=1e-3`，batch=256，early stop；
- RL 变体：4000 步、`lr=1e-4`、每步采样 64 个样本；<1 GPU·天；
- 全部在 4×L40 上运行，评测并行调用 API 多档生成。

### 3.4 回测与评测流程

- 评测集 = PiT-AuditBench 窗口外 hold-out（与训练无重叠）；
- 评估协议：对同一评测集跑"固定 low / 固定 medium / 固定 high / 随机 / 贪心（按预估难度）/ 路由"，输出各自 (exact-match, 总成本)；
- 预算受限比较：归一化到相同成本预算，报告 exact-match；以及帕累托前沿图（x=成本、y=正确率）；
- 显著性：5 种子 × 不同样本打乱，paired t-test 与 bootstrap CI；
- 无前视：路由推理时不引用评测样本的正确答案，也不引用未来数据。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 样本集 | PiT-AuditBench v1（Idea 1，~20k 事件装配出 8k+ 排序/日期样本） |
| 多档评测数据 | 自建 ChronoRL-Data：5,000 样本 × 3 档 = 15,000 条，含 (特征, 档, 正确, 成本, 延迟) |
| 划分 | 训练 60% / 验证 20% / 评测 20%（按事件日期分层，保证窗口外样本主要落在评测） |
| 许可 | 全部公开数据 + 自建评测数据随论文开源 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| 固定 low / medium / high | 策略基线 | 直接评测 | exact-match、成本 |
| 随机选档 | 策略基线 | 均匀随机 | 同上 |
| 贪心难度路由 | 启发式 | 按 m+c+k 简单阈值 | 同上 |
| 全高 effort | 上界参考 | 全样本 high | 正确率上界（成本最高） |
| 全低 effort | 下界参考 | 全样本 low | 成本下界 |
| Chronology 原模型设定 | 现象对照 | arXiv:2511.14214 复现 | 排序指标与原文对齐 |

**统一口径**：同一评测集、同一计分器（Idea 1 规则计分）、温度 0.2、各档 max_tokens 固定。

## 6. 实验矩阵

- **A 主实验**：路由（XGBoost/MLP/RL）vs 全部基线，主指标 Accuracy-Cost AUC；
- **B 特征消融**：去掉各特征组（长度/重叠/条件/时间线/日期跨度/术语密度）后 AUC 变化；
- **C 跨模型迁移**：DeepSeek V4 Pro 路由 → Qwen2.5-7B、Kimi K2.6 的直接迁移与微调迁移；
- **D λ 灵敏度**：λ∈{0.5,1,2,4} 下帕累托前沿；
- **E 成本结构消融**：按 API 单价加权的真实成本 vs 纯 token 数；
- **F 稳健性**：换打乱种子、换计分器（LLM judge vs 规则）、评测窗口 ±3 个月。

## 7. 评测协议

- **指标**：exact-match / Kendall τ（按 Idea 1 定义）；成本 = 实际输出 token 数 × 单价（归一化）；Accuracy-Cost AUC（帕累托曲线下面积，成本与正确率均 0-1 归一）；
- **统计**：5 种子 mean±std；路由 vs 基线 paired t-test（α=0.05）；bootstrap CI；
- **无前视**：训练特征与标签不含未来信息；评测样本窗口外优先；
- **报告模板**：每模型输出"正确率-成本"散点 + 前沿 + 每档使用比例。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 多档数据采集 | API（DeepSeek V4 Pro 主力 + Kimi K2.6 难样本） | 约 15,000 次推理，约 5,000 万 token |
| 本地路由训练 | 4×L40（MLP/RL 极小负载） | <1 GPU·天 |
| 评测与迁移 | 4×L40 + API 并行 | 1-2 GPU·天 |
| 存储 | 评测记录 JSONL | <50GB |
| 总计 | 4×L40 | 约 2-3 GPU·天 |

**API 成本**：DeepSeek V4 Pro 多档推理约 4,000 万 token；Kimi K2.6 约 800 万；DeepSeek V4 Flash 预筛约 500 万。估算 $300-700（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，7 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 接入 PiT-AuditBench + 复杂度特征提取 | 特征管线 + 预筛样本 |
| W2 | 多档数据采集（先 1k 样本 pilot） | ChronoRL-Data v0（H1 检验） |
| W3 | 全量采集 + 噪声评估 | ChronoRL-Data v1 |
| W4 | 路由训练（XGBoost/MLP/RL） | 路由模型 + 验证集指标 |
| W5 | 主实验 A/B | 帕累托图 + 消融表 |
| W6 | 跨模型迁移 + λ 灵敏度 | C/D 实验表 |
| W7 | 稳健性 + 论文初稿 | 提交稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 推理预算收益不稳定（H1 不成立） | 高 | 若三档差异小，转向"二档决策 + 校准重采样"；或把选题重心转为"证据 vs 成本的审计方法论" |
| 路由过拟合到单模型 | 中 | 迁移实验前置；若迁移差，定位为"单模型路由 + 少样本适配"并如实报告 |
| API 成本超预算 | 中 | Flash 预筛把高置信样本降档采集；对难样本重采样而非全档重复 |
| 过度推理伤害（对金融情感）与排序收益矛盾 | 中 | 报告中明确任务边界：本方法面向时序排序，情感任务作为鲁棒性对照 |
| 评测器噪声 | 低 | 规则计分为主 + judge 一致性 κ≥0.8 把关 |

## 11. 论文写作计划

- **目标会议**：ACL 2027 / NeurIPS 2026（推理效率方向）；以官方截稿公告为准。
- **差异化卖点**：(1) 首个把测试时推理预算作为金融时序任务可学习决策；(2) 复杂度特征来自 Chronology 任务族；(3) 跨模型迁移实验增强通用性。
- **图表清单**：图1 方法总览（路由→选档→评测）；图2 三档 effort 的复杂度-正确率-成本散点；图3 帕累托前沿；图4 迁移矩阵；表1 数据集统计；表2 主实验；表3 消融；表4 灵敏度。
- **相关工作覆盖**：Chronology（2511.14214）、Reasoning or Overthinking（2506.04574）、PiT-Inference（2512.06607）、Look-Ahead-Bench（2601.13770）、FinCast 等 TSFM 效率（2508.19609，作为下游动机）、Alpha Illusion（2605.16895）。

## 12. 参考文献

1. Wongchamcharoen, P. K., & Glasserman, P. *Do Large Language Models (LLMs) Understand Chronology? (Student Abstract)*. arXiv:2511.14214.
2. *Reasoning or Overthinking: Evaluating LLMs on Financial Sentiment Analysis*. arXiv:2506.04574.
3. *A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs*. arXiv:2512.06607.
4. *Look-Ahead-Bench: A Standardized Benchmark for Point-in-Time Look-Ahead Bias in LLMs*. arXiv:2601.13770.
5. *FinCast: A Financial Time Series Foundation Model*. arXiv:2508.19609.
6. *The Alpha Illusion*. arXiv:2605.16895.
7. *ChronoSense: Evaluating Temporal Understanding in LLMs*. arXiv:2501.03040.
8. *Structured yet Bounded Temporal Understanding in LLMs*. arXiv:2510.16685.
9. *A Survey of Large Language Models in Finance (FinLLMs)*. arXiv:2402.02315.
