# 实验设计书：Chrono-Aug：时间线一致的金融序列自监督数据增强

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea11_Chrono-Aug.md
> 关联收藏论文：AAAI Emerging Trends in AI · 论文 183（Chronology）；Machine Learning VII · 论文 47（Kronos）；叠加于 Idea 4（Kronos-CN 预训练）

## 0. 摘要

Chrono-Aug 在金融时序模型预训练阶段注入"事件时间线"自监督目标，让 TSFM/LLM 学到金融时间线先验（事件如何按时间组织、相对时距、周期/日历锚定），从源头缓解乱序与前视问题。现有 TSFM 预训练只有 next-token，未显式教模型时间线组织；Chronology 证明 LLM 全局时间线不一致。本设计新增三类自监督头：**(a) 时间线排序预测**（事件段打乱后的正确排序）、**(b) 相对时距回归**（"事件 A 比 B 早多少天"）、**(c) 周期/日历锚定分类**（财报季、除权日模式），与 next-token 联合训练、不改变推理接口。方法可叠加到任何 TSFM/金融 LLM 预训练（以 Idea 4 的 Kronos-CN 为主载体），下游用 Idea 1 的基准与 RankIC 评测。

## 1. 研究背景与动机

### 1.1 问题定义

给定预训练语料（K 线段 + 事件标记流），构造三类辅助任务：(a) 给定 m 个事件段（按时间打乱），预测正确排序；(b) 给定事件对 (A,B)，预测时距 d(A,B)（天）；(c) 给定事件上下文，预测日历/周期类别（财报季、除权日前后）。主目标为 next-token 预测（K 线）。联合训练后评估：(1) Idea 1 PiT-AuditBench 的时间排序/日期指标是否提升；(2) 下游 RankIC 是否保持或提升（不被辅助目标负迁移）。

### 1.2 相关工作不足

- **Chronology（AAAI Emerging Trends in AI · 论文 183）**：证明 LLM 全局时间线不一致，但只测不改，未给出预训练层面的补救；
- **Kronos（AAAI Machine Learning VII · 论文 47）/ 主流 TSFM**：预训练仅 next-token，无时间线组织目标；
- **Synthetic Preference Data for Temporal Understanding（arXiv:2510.03955）**：面向视频的合成偏好数据增强时间理解，未迁移到金融时序；
- **Timer-S1（arXiv:2603.04791）**：串行扩展时序模型，无事件时间线目标；
- **Point-in-Time 研究（2601.13770 等）**：评测前视但不解决"模型自身时序先验缺失"。

### 1.3 为什么是现在、为什么你的环境适合做

Chrono-Aug 叠加在 Idea 4 的 Kronos-CN 预训练上只增 10-20% 算力，直接复用同一语料与 tokenizer；Idea 1 提供现成的排序/日期评测；社区对"预训练注入时间线先验"几乎空白。环境优势：1-3B 预训练 4×L40 已预算（2-4 GPU·周），增量小、风险可控。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 时间线自监督目标可学习且不拖累 next-token | 辅助任务验证集指标收敛；next-token 困惑度与无增强版持平或更好 |
| H2 | 时间线先验提升时间理解评测 | 有 Chrono-Aug 的模型在 Idea 1 排序/日期指标上显著高于无增强版 |
| H3 | 时间线先验不损害（或提升）下游预测 | RankIC 保持 ≥ 基线，或提升（≥0 且显著） |
| H4 | 三种目标各自有贡献 | 去掉任一种目标的消融，其对应指标下降（排序/时距/周期各自绑定） |

第一验证实验：1B 模型在单行业语料上跑"有/无 Chrono-Aug"对照 3-5 天，测辅助目标收敛 + RankIC 是否不下降（H1/H3 pilot）。

## 3. 总体方法设计

### 3.1 数据流水线

- **语料**：复用 Idea 4 Kronos-CN 语料（A 股/港股/可转债 K 线 + 制度标记）；
- **事件标记流**：把事件（财报日、除权日、解禁日、宏观发布日）作为特殊 token 注入 K 线序列流，事件与 K 线共享时间轴（事件 token 在对应日期 K 线前插入）；
- **辅助任务样本构造**（规则 + 可选 LLM 增强）：
  - 排序：随机取 m∈{3,5,8} 个事件段（含相邻 K 线上下文），打乱生成；
  - 时距：随机事件对 (A,B)，标签 = B 日 − A 日（天）；
  - 周期：事件上下文 → 类别（财报季 1/4 月等、除权前后、宏观发布日类型）；
- **LLM 语义增强**（DeepSeek V4 Flash）：对事件段生成"带时间戳的事件描述 + 难排序变体"（同日多事件、跨市场事件），作为语义增强样本补充；
- **清洗与去泄漏**：辅助任务标签全部来自事件真实日期，无未来信息；训练/验证按时间切分（验证最后 3 个月）；
- **数量**：排序 ~50 万样本、时距 ~80 万对、周期 ~40 万样本（由语料自动装配，无需人工标注）。

### 3.2 方法设计

- **模型**：在 Kronos-CN 的 decoder 上叠三个辅助头（共享主干）：
  - 排序头：对 m 个事件段表示做 pairwise/binary 排序（对比损失或 listwise）；
  - 时距头：回归（MSE），对时距做对数变换 + 上界截断；
  - 周期头：分类（CE，K=8 类）；
- **联合损失**：`L = L_next + λ1·L_ord + λ2·L_dist + λ3·L_per`，λ 初值 (0.2, 0.1, 0.1)，可调度（前 10% 步预热）；
- **实现细节**：辅助头只在含事件 token 的样本上激活（mask），普通样本只有 L_next；
- **前视处理**：辅助任务标签用事件真实日期；排序/时距均不含评测期未来事件；tokenizer 与 Idea 4 一致；
- **推理接口不变**：推理时只走 next-token 主路径，辅助头仅在评测中评估。

### 3.3 训练流程

- 与 Idea 4 同一框架：1-3B，AdamW lr=1e-3，batch 0.5-1M token，warmup 2,000 步，cosine，4×L40 FSDP；
- 主实验 = Idea 4 预训练 + Chrono-Aug（增量 10-20% 步数）；对照 = Idea 4 无增强；
- 单行业 pilot 先行（银行），再全量。

### 3.4 回测与评测流程

- **评测 1（时间理解）**：Idea 1 PiT-AuditBench（T1 排序 exact-match/τ、T4 日期 MAE）；
- **评测 2（预测）**：Kronos-CN 标准评测（RankIC、vol MAE，窗口外 2025-07 至 2026-06）；
- **防泄漏**：辅助任务训练样本截止与预训练一致（窗口外事件不进训练）；评测与训练零重叠；
- 报告：有/无 Chrono-Aug 的配对对比 + 每辅助目标单独指标。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | Chrono-Aug-Tasks v1（自建，自动装配） |
| 来源 | 复用 Idea 4 语料 + 事件日历（财报/除权/解禁/宏观） |
| 许可 | 同 Idea 4（公开数据源） |
| 划分 | 训练（时间切分 90%）/ 验证（3 个月） |
| 规模 | 排序 50 万、时距 80 万、周期 40 万样本 |
| 预处理 | 事件 token 注入、样本装配、标签计算 |
| 质量门 | 辅助标签规则校验（事件日期交叉验证）；LLM 增强样本 5% 人工复核 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| Kronos-CN（无增强） | 主对照 | Idea 4 流程 | RankIC / 时间理解指标 |
| Kronos 原版 | TSFM | https://github.com/shiyu-coder/Kronos | 同上 |
| Chronos / TimeGPT | 通用 TSFM | https://github.com/amazon-science/chronos-forecasting / nixtla | 同上 |
| LLM 时间理解零样本 | 对照 | GPT-4.1/DeepSeek V4 Pro 直接回答 Idea 1 | 排序/日期指标 |
| Chrono-Aug（只加部分目标） | 消融 | 单目标版 | 各目标贡献 |
| Chrono-Aug（完整） | 本方法 | 本设计 | 全部指标 |

**统一口径**：同一语料、同一 tokenizer、同一评测期与计分器；预训练步数与 token 预算对齐（比较含辅助目标的相同算力下）。

## 6. 实验矩阵

- **A 主实验**：Kronos-CN±Chrono-Aug × 时间理解 + RankIC 全表；
- **B 目标消融**：单目标（排序/时距/周期）分别加入的效果（H4）；
- **C 权重灵敏度**：λ 网格（0.05-0.4）；
- **D 负迁移检测**：next-token 困惑度 + RankIC 变化（H3）；
- **E 事件类型影响**：仅财报 / 仅除权 / 全部事件的子集；
- **F 跨市场**：A 股 vs 港股 vs 可转债子集；
- **G 稳健性**：种子、排序长度 m、时距上界。

## 7. 评测协议

- **指标**：Idea 1 的排序 exact-match / Kendall τ / 日期 MAE；RankIC / ICIR；vol MAE；next-token 困惑度；
- **统计**：3 种子 mean±std；有/无增强配对 t 检验（α=0.05）；
- **无前视**：辅助任务标签不含窗口外事件；评测窗口外；预训练-评测日期零重叠。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 任务样本装配 | CPU | 1 周 |
| LLM 语义增强 | API（Flash） | 约 3,000 万 token |
| 1-3B 预训练（+Chrono-Aug） | 4×L40 FSDP | 2-5 GPU·周（含增量 10-20%） |
| 单行业 pilot | 4×L40 | 3-5 GPU·天 |
| 评测（时间理解 + 预测） | 4×L40 | 2-3 GPU·天 |
| 存储 | 语料 + 任务样本 + checkpoint | 2-3TB |
| 总计 | 4×L40 | 约 60-150 GPU·天（与 Idea 4 共用大部分算力，增量约 10-20%） |

**API 成本**：Flash 语义增强约 3,000 万 token；Pro 评测/审计约 500 万。估算 $150-450（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，10 周，与 Idea 4 并行排程）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 事件日历接入 + 任务样本装配 | 任务集 v1 |
| W2 | LLM 语义增强样本 | 增强任务集 v1 |
| W3 | 单行业 pilot（有/无增强） | pilot 表（H1/H3 初判） |
| W4 | 辅助头实现 + 联合损失 | 训练脚本 v1 |
| W5 | 1B 全量预训练（增强版） | 1B 增强 checkpoint |
| W6 | 3B 预训练（增强版） | 3B 增强 checkpoint |
| W7 | 主实验 A（时间理解 + RankIC） | A 表 |
| W8 | 目标消融 B/C | 消融表 |
| W9 | 负迁移检测 D + 事件类型 E | D/E 表 |
| W10 | 跨市场 F + 稳健性 + 论文初稿 | 提交稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 辅助目标干扰 next-token（负迁移） | 高 | λ 调度 + 预热；单行业 pilot 先行；若负迁移明显则降 λ 或只加排序头 |
| 时间理解指标提升不显著 | 中 | 增加事件密度语料；难样本生成（同日多事件）；以排序/时距 sub-metric 细分找增益 |
| 事件日历覆盖不全（可转债/新股） | 中 | 以财报/除权主干事件为准；缺失日历的事件段不构造辅助任务 |
| 与 Idea 4 的算力排期冲突 | 中 | 共享预训练跑批；先跑 1B 增强版，3B 视 1B 结果决定 |
| 收益归因困难（增强 vs 步数） | 中 | 对齐 token 预算与步数做公平对照（同算力比较） |

## 11. 论文写作计划

- **目标会议**：ICML 2027 / NeurIPS 2026（以官方截稿为准）。
- **差异化卖点**：(1) 首个在金融 TSFM 预训练注入时间线自监督（排序/时距/周期）的工作；(2) 与 next-token 联合且不改变推理接口；(3) 可迁移到任何 TSFM/金融 LLM。
- **图表清单**：图1 辅助任务示意图；图2 联合训练架构；图3 时间理解提升图；图4 负迁移检测（困惑度/RankIC）；表1 任务集统计；表2 主实验；表3 目标消融；表4 权重灵敏度。
- **相关工作覆盖**：Chronology（2511.14214）、Kronos（2508.02739）、Timer-S1（2603.04791）、Synthetic Preference Data（2510.03955）、Chronos（2403.07815）、Look-Ahead-Bench（2601.13770）、ChronoSense（2501.03040）、TLQA（2506.21783）。

## 12. 参考文献

1. Wongchamcharoen, P. K., & Glasserman, P. *Do Large Language Models (LLMs) Understand Chronology? (Student Abstract)*. arXiv:2511.14214.
2. Shi, Y., et al. *Kronos: A Foundation Model for the Language of Financial Markets*. arXiv:2508.02739.
3. *Timer-S1: Billion-Parameter Serial Scaling for Time Series*. arXiv:2603.04791.
4. *Synthetic Preference Data for Temporal Understanding*. arXiv:2510.03955.
5. Ansari, A., et al. *Chronos: Learning the Language of Time Series*. arXiv:2403.07815.
6. *Look-Ahead-Bench: A Standardized Benchmark for Point-in-Time Look-Ahead Bias in LLMs*. arXiv:2601.13770.
7. *ChronoSense: Evaluating Temporal Understanding in LLMs*. arXiv:2501.03040.
8. *TLQA: Time-Sensitive List QA Benchmark*. arXiv:2506.21783.
9. *FinCast: A Financial Time Series Foundation Model*. arXiv:2508.19609.
10. *The Alpha Illusion*. arXiv:2605.16895.
