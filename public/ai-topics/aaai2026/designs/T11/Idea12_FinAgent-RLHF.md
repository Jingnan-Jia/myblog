# 实验设计书：FinAgent-RLHF：融合回测信号与事实性奖励的金融决策智能体对齐

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea12_FinAgent-RLHF.md
> 关联收藏论文：AAAI Emerging Trends in AI · 论文 183（Chronology，前视红线）；Application Domains I · 论文 57（FinRpt，RL 奖励）；依赖 Idea 8（环境）与 Idea 9（事实性验证器）

## 0. 摘要

FinAgent-RLHF 用"可验证回测风险调整收益 + 输出事实性 + 行为一致性"组成的奖励函数，对 LLM 交易/投顾智能体做 RL 对齐，使智能体"既会赚钱（在可验证、防污染口径下）又讲真话"。现有 RL 对齐奖励多为文本质量/信号（FinRpt-Gen、FLAG-Trader），FinRL-DeepSeek 做风险敏感 RL 但未对齐事实性，Alpha Illusion 强调收益必须可验证。本设计定义奖励 `R = λ1·回测风险调整收益（严格成本/T+1/无前视） + λ2·事实性（Idea 9 验证器得分） − λ3·幻觉惩罚 + λ4·行为一致性（Idea 8 规则遵守）`，用 GRPO 在 MarketSim（Idea 8）+ 真实历史回测器双环境训练 7B 基座，输出对齐后智能体的"可验证归因"（钱从哪来、是否靠泄漏）。

## 1. 研究背景与动机

### 1.1 问题定义

给定 LLM 交易智能体 π_θ（输入市场状态 + 持仓，输出动作 + 理由），定义奖励：
`R(τ) = λ1·Sharpe_adj(τ) + λ2·Fact(τ) − λ3·Halluc(τ) + λ4·CS(τ)`
其中 Sharpe_adj 为防污染回测（成本后）的风险调整收益，Fact 为决策理由的事实性得分（Idea 9），Halluc 为幻觉惩罚，CS 为行为一致性（Idea 8 规则遵守）。目标：GRPO 优化 π_θ 使 E[R] 最大化，同时评测对齐前后在防污染评测上的表现与归因可解释性。

### 1.2 相关工作不足

- **FinRpt-Gen（AAAI Application Domains I · 论文 57）**：RL 奖励偏报告文本质量，未接可验证回测收益与事实性验证；
- **FLAG-Trader（arXiv:2502.11433）**：梯度 RL 融合信号，奖励偏信号级，未对齐事实性；
- **FinRL-DeepSeek（arXiv:2502.07393）**：风险敏感 RL，但无事实性/行为对齐；
- **Alpha Illusion（arXiv:2605.16895）/ Benchmarks Are Not Validation（arXiv:2607.28840）**：强调收益必须可验证、需五层验证——本设计把"可验证回测信号"作为奖励源并给出归因；
- **StockBench（arXiv:2510.02209）**：防污染评测框架，作为对齐后的评测集。

### 1.3 为什么是现在、为什么你的环境适合做

RL 对齐技术（GRPO）成熟且开源（TRL），Idea 8 的仿真环境与 Idea 9 的验证器可直接复用，形成"奖励=回测+事实性+行为"的完整闭环。环境优势：(1) 7B GRPO 在 4×L40 上 3-6 GPU·周可承担；(2) 环境与奖励计算与训练解耦（预生成 episode 缓存）；(3) 评审红线（前视、reward hacking）被主动处理成论文卖点。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 回测奖励驱动的对齐在防污染口径下提升风险调整收益 | 对齐后在 StockBench 风格防污染评测上 Sharpe/Calmar 显著高于 SFT 智能体（成本后） |
| H2 | 加入事实性奖励降低幻觉而不伤收益 | 事实性得分提升 ≥20%（相对），同时收益不显著下降 |
| H3 | 行为一致性奖励减少规则违反 | 对齐后规则违反率显著下降（Idea 8 协议） |
| H4 | 收益提升非来自前视泄漏（归因可验证） | 对齐后"泄漏归因比例"（通过前视审计的收益占比）不升高；泄漏对照无显著收益提升 |

第一验证实验：在单市场状态（牛市）环境上用 SFT 基座跑 200 步 GRPO pilot，验证奖励可学习 + 防污染回测管线（H0/H1 初判），3-5 天。

## 3. 总体方法设计

### 3.1 数据流水线

- **环境**：复用 Idea 8 MarketSim（可配置微观结构 + 市场状态剧本）+ 真实历史数据回测器（2020-2025 训练期，成本/T+1/涨跌停约束）；
- **决策理由事实性**：智能体输出的理由文本 → Idea 9 验证器打分（是否与点即时刻事实一致）；
- **行为一致性**：理由 + 动作 → Idea 8 规则集（风格指令约束）判分；
- **训练轨迹生成**（DeepSeek V4 Flash/Pro + Kimi K2.6）：生成多样市场状态下的决策轨迹与决策理由，作为 SFT 与 episode 的种子；pro 做奖励判分 judge；
- **清洗与去泄漏**：训练期 2020-2025，评测期 2025-07 至 2026-06 严格分离；episode 内动作只使用 ≤t 信息；理由文本引用的数字需在披露日之前可用；
- **数量**：SFT 训练 ~30k（状态, 动作, 理由）样本；RL episode 预生成 5,000 个 × 250 步；评测 StockBench 风格防污染集 + 自建 A 股评测期。

### 3.2 方法设计

- **奖励定义**（防 reward hacking）：
  - `Sharpe_adj`：成本后日收益的滚动 Sharpe（经 IR 缩放的 sigmoid 归一化，防极端值）；
  - `Fact`：验证器对理由中数字 claim 的"一致"比例（Idea 9，阈值保守）；
  - `Halluc`：矛盾/无法验证比例（惩罚项）；
  - `CS`：规则遵守率（Idea 8 规则集）；
  - `R = λ1·Sharpe_adj + λ2·Fact − λ3·Halluc + λ4·CS`，λ 初值 (0.5, 0.3, 0.2, 0.2)，多解消融；
- **GRPO**：每组问题（市场状态起点）采样 8 个 rollout，组内基线化优势，clip ε=0.2；KL 正则 β=0.04 锚定参考策略；
- **归因分析**：对齐后对策略收益做分解——按决策理由中的因子归因 + 前视泄漏对照（把未来信息随机注入理由的对照组，量化泄漏收益）；
- **前视处理**：episode 状态只含 ≤t 信息；回测与奖励均防污染（成本/滑点/T+1/涨跌停）；验证器只比对点即时刻证据。

### 3.3 训练流程

- 基座：Qwen2.5-7B（或 DeepSeek 蒸馏小模型）；阶段一 LoRA SFT（r=16, lr=2e-5, 2 epochs）；阶段二 GRPO（lr=1e-6，5,000 步，4×L40 FSDP）；
- 环境与奖励计算解耦：预生成 episode 缓存（状态序列 + 可打分理由），RL 只做策略更新，奖励查表；
- 多市场状态采样：训练时按剧本（牛/熊/高波动/流动性枯竭）比例 2:2:2:1 采样；
- 早停：验证集（hold-out 市场状态）奖励均值不再上升即停。

### 3.4 回测与评测流程

- **防污染评测**：StockBench 风格（真实市场信号、窗口外、成本后）+ 自建 A 股评测期（2025-07 至 2026-06）；回测协议同 Idea 4/8（T+1、成本、涨跌停、停牌）；
- **前视审计**：对评测期策略运行做泄漏检测（理由文本时间戳 + 动作时点交叉验证）；泄漏对照实验必需；
- **归因**：收益分解报告（决策理由因子的贡献 + 泄漏归因占比）；
- 报告：对齐前后对比（Sharpe/Calmar/事实性/一致性/泄漏占比）+ 3 种子 mean±std。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | FinAgent-RLHF-DS v1（SFT 样本 + RL episodes，自建） |
| 来源 | 行情数据（公开）+ 剧本（Kimi 生成）+ 决策轨迹（LLM 生成 + 规则基座合成） |
| 许可 | 行情按数据源许可；轨迹数据随论文开源 |
| 划分 | SFT 训练 ≤2024-12；RL 训练 2025H1（含脚本回测期 2020-2025）；评测 2025-07 至 2026-06 |
| 规模 | SFT 30k；episode 5,000×250 步；评测防污染集 1,000 样本 |
| 预处理 | 状态标准化、理由抽取、奖励字段缓存 |
| 质量门 | 理由-事实人工复核 200 条；episode 时点校验（无未来引用） |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| SFT 智能体（无 RL） | 基线 | 本设计阶段一 | Sharpe/事实性/一致性 |
| RL（无事实性奖励） | 消融 | λ2=λ3=0 | 收益（预期涨、事实性不涨） |
| RL（无行为一致性） | 消融 | λ4=0 | 收益 + 违规率 |
| FinMem / FinAgent 复现 | 参考智能体 | https://github.com/RobinWWang/FinMem 等 | 全指标对照 |
| FinRL-DeepSeek | 参考 | arXiv:2502.07393 | 风险敏感 RL 对照 |
| FinAgent-RLHF（完整） | 本方法 | 本设计 | 全部指标 |

**统一口径**：同一防污染评测器、同一成本/制度约束、同一事实性验证器、同一行为规则集。

## 6. 实验矩阵

- **A 主实验**：对齐前后全指标对比（收益/事实性/一致性/泄漏占比）；
- **B 奖励权重消融**：λ1-λ4 网格（H1-H3）；
- **C 训练步数**：1k/2k/5k 步曲线 + 过拟合检测（hold-out 状态）；
- **D 环境多样性**：不同市场状态比例的影响；
- **E 前视审计**：泄漏对照 vs 不泄漏（H4，必需）；
- **F 基座消融**：Qwen2.5-7B vs DeepSeek 蒸馏 vs Kimi 代理（无 RL 对照）；
- **G 稳健性**：种子、成本假设（20/50bp）、评测期滚动。

## 7. 评测协议

- **指标**：防污染 Sharpe/Calmar/最大回撤/换手（成本后）；事实性得分（Idea 9）；幻觉率；规则违反率（Idea 8）；泄漏归因占比；
- **统计**：3 种子 mean±std；对齐前后 paired t 检验（α=0.05）；多状态分解；
- **无前视**：评测窗口外 + 成本/制度约束 + 泄漏对照必需；reward 计算的验证器阈值固定。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 环境与 episode 预生成 | 4×L40 + API | 3-5 GPU·天 |
| SFT（7B LoRA） | 4×L40 | 7-14 GPU·天 |
| GRPO RL | 4×L40 | 21-42 GPU·天（3-6 GPU·周） |
| 防污染评测 + 前视审计 | 4×L40 + CPU | 3-5 GPU·天 |
| 存储 | episode 缓存 + checkpoint | <1TB |
| 总计 | 4×L40 | 约 35-65 GPU·天 |

**API 成本**：Flash/Pro/Kimi 轨迹生成 + 事实性 judge 约 8,000 万 token。估算 $700-1,800（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，10 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 环境复用 + 奖励管线（回测/验证器/规则） | 奖励计算闭环 v1 |
| W2 | SFT 样本构造 + 事实性人工复核 | SFT 集 v1 |
| W3 | SFT 训练 | SFT 模型 |
| W4 | GRPO pilot（单状态 200 步） | pilot 表（H0/H1 初判） |
| W5 | episode 预生成（5,000×250） | episode 缓存 |
| W6 | GRPO 全量训练 | RL 模型 |
| W7 | 主实验 A + 权重消融 B | 全表 |
| W8 | 前视审计 E + 归因 | E 表 + 归因报告 |
| W9 | 基座/环境多样性 F/D | F/D 表 |
| W10 | 稳健性 + 论文初稿 | 提交稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 回测奖励方差大、易过拟合特定市场状态 | 高 | 多状态采样 + hold-out 状态早停 + 滚动评测期 |
| reward hacking（刷回测收益而非真实能力） | 高 | 验证器交叉约束 + 泄漏对照 + 成本敏感；归因审计 |
| 事实性奖励拖累收益（H2 弱） | 中 | 权重网格；若冲突则主结论如实报告权衡曲线 |
| GRPO 训练不稳定 | 中 | KL 正则 + clip + 低 lr + 参考策略锚定；预生成缓存保证奖励确定性 |
| 与前视红线冲突被质疑 | 中 | 前视审计 + 泄漏对照为论文必需章节，主动回应 |

## 11. 论文写作计划

- **目标会议**：NeurIPS 2026 / ICML 2027（以官方截稿为准）。
- **差异化卖点**：(1) 首个把"可验证回测收益 + 事实性 + 行为一致性"三合一奖励的金融智能体 RL 对齐；(2) 归因可解释（钱从哪来、是否靠泄漏）；(3) 主动通过 Alpha Illusion 式审计。
- **图表清单**：图1 RL 对齐架构；图2 奖励函数示意；图3 训练曲线（收益/事实性/一致性）；图4 前视审计与归因；表1 数据集统计；表2 主实验；表3 奖励消融；表4 稳健性。
- **相关工作覆盖**：FinRpt-Gen（2511.07322）、FLAG-Trader（2502.11433）、FinRL-DeepSeek（2502.07393）、Alpha Illusion（2605.16895）、Benchmarks Are Not Validation（2607.28840）、StockBench（2510.02209）、Can LLMs Trade（2504.10789）、FinMem（2311.13743）、FinAgent（2402.18485）、Look-Ahead-Bench（2601.13770）。

## 12. 参考文献

1. Jin, S., et al. *FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation*. arXiv:2511.07322.
2. *FLAG-Trader: Fusion LLM-Agent with Gradient-based RL for Financial Trading*. arXiv:2502.11433.
3. *FinRL-DeepSeek: Risk-Sensitive Financial RL*. arXiv:2502.07393.
4. *The Alpha Illusion*. arXiv:2605.16895.
5. *Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications*. arXiv:2607.28840.
6. *StockBench: Pollution-Avoiding Evaluation of LLM Trading Agents on Real Market Signals*. arXiv:2510.02209.
7. *Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations*. arXiv:2504.10789.
8. Wang, R., et al. *FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory*. arXiv:2311.13743.
9. Zhang, W., et al. *FinAgent: A Multimodal Foundation Agent for Financial Trading*. arXiv:2402.18485.
10. *Look-Ahead-Bench: A Standardized Benchmark for Point-in-Time Look-Ahead Bias in LLMs*. arXiv:2601.13770.
11. *FINSABER: Twenty-Year Regime-Switching Backtesting of LLM Investment Strategies*. arXiv:2505.07078.
12. *A Survey of Large Language Model based Autonomous Agents for Financial Trading*. arXiv:2408.06361.
