# 实验设计书：MarketSim-Eval：市场仿真驱动的 LLM 投资智能体行为评测

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea08_MarketSim-Eval.md
> 关联收藏论文：AAAI Emerging Trends in AI · 论文 183（Chronology，时间墙审计）；AAAI Emerging Trends in AI · 论文 284（PortfolioPilot，算法平台对照）

## 0. 摘要

MarketSim-Eval 在可控多智能体市场仿真环境中系统评测 LLM 投资智能体的**行为一致性**与**制度稳健性**，回答"换市场环境后智能体是否还言行一致"。它扩展 2504.10789 的开源市场仿真框架，加入可配置微观结构（订单簿/手续费/涨跌停/流动性/T+1）与市场状态参数，对 FinMem、FinAgent、FLAG-Trader 等主流智能体施加行为压力测试。核心产出两类指标：(1) **行为一致性分数**（给定投资风格约束时规则遵守率）；(2) **制度迁移鲁棒性**（T+0→T+1、有涨跌停→无涨跌停下的表现变化）。回应 Alpha Illusion（2605.16895）与 Benchmarks Are Not Validation（2607.28840）的评审红线：把"回测数值"与"行为审计"结合，输出 deployment-readiness 评分。

## 1. 研究背景与动机

### 1.1 问题定义

给定 LLM 投资智能体 A 与市场环境参数 θ（微观结构 + 市场状态），在仿真中运行 T 步得到决策轨迹 τ_A = {行动_t, 持仓_t, 理由_t}。行为评测目标：(1) 一致性——A 在给定风格指令（如"价值投资"）下的实际行为与指令的相符度；(2) 制度适应性——θ 变化（T+0/T+1、涨跌停开关、手续费）时 A 的表现与行为变化；(3) 稳健性——重复运行/不同市场状态的方差。

### 1.2 相关工作不足

- **FinMem/FinCon/TradingAgents 等（2311.13743 / 2407.06567 / 2412.20138）**：报告高 Sharpe，但 Alpha Illusion 指出这些是架构研究而非部署证据，无行为审计；
- **Can Large Language Models Trade?（arXiv:2504.10789）**：提供异构 LLM 竞争的市场仿真框架，但未做系统行为评测（一致性/制度迁移）；
- **StockBench（arXiv:2510.02209）**：防污染真实市场评测，但无微观结构控制与行为压力测试；
- **Benchmarks Are Not Validation（arXiv:2607.28840）**：要求数据/检索/智能体行为/治理/实现五层验证——本设计实现其中的"行为层"。

### 1.3 为什么是现在、为什么你的环境适合做

社区对"LLM 智能体收益不可信"已有共识（Alpha Illusion 打假潮），但缺少可操作的行为评测标准；2504.10789 已开源仿真框架可扩展。环境优势：(1) 无需大模型训练，仿真 + API 推理为主，4×L40 可并行多智能体；(2) 行为审计判分走规则 + LLM judge，预算可控；(3) 可与 Idea 12（RL 对齐）共用环境与评测器。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 主流 LLM 交易智能体在行为压力测试下表现不一致 | 风格指令（价值/动量）遵守率在压力场景（高波动/流动性枯竭）显著下降（>15 个点） |
| H2 | 制度迁移显著改变智能体收益-风险 | T+0→T+1、涨跌停开关前后的 Sharpe/换手率显著变化，且部分智能体策略失效 |
| H3 | 行为一致性分数与"防污染收益"相关 | 高遵守率的智能体在窗口外真实回测的收益（成本后）与遵守率正相关 |
| H4 | 推理预算影响行为稳定性 | 高 reasoning effort 下智能体更遵守规则（或不，需实测）——与 Chronology 结论对照 |

第一验证实验：单智能体（DeepSeek V4 Flash 代理）在基础仿真环境跑 20 步，验证环境可运行 + 行为记录管线（H0 可行性），3-5 天。

## 3. 总体方法设计

### 3.1 数据流水线

- **仿真环境**：扩展 2504.10789 开源框架（若不可得则自建简化订单簿环境）；参数化：订单簿深度、手续费、涨跌停比例、T+1、流动性冲击、新闻事件注入；
- **市场状态剧本**（Kimi K2.6 生成）：牛市/熊市/高波动/流动性枯竭/事件冲击 5 类场景 × 各 3 个剧本；
- **智能体**：FinMem（2311.13743）、FinAgent（2402.18485）、FLAG-Trader（2502.11433）按官方仓库复现；DeepSeek V4 Flash/Pro 与 Kimi K2.6 直接代理；简单规则基线与 RL 代理作对照；
- **风格指令**：价值/动量/低波 3 类显式指令 + 无指令对照组；
- **行为记录**：每步记录 {行动, 持仓, 理由文本, 时点}；理由文本后处理成"规则遵守/违反"标注（规则 + LLM judge 双通道，不一致时人工仲裁）；
- **防泄漏**：仿真市场序列由历史真实数据重采样/合成生成，不直接使用评测期真实序列；智能体无未来信息访问权限（下一期价格仅在下期步骤才公开）。

### 3.2 方法设计

- **行为一致性分数**：`CS = (符合指令约束的步数 − 违规步数) / 总步数`，指令约束转成规则集（如"价值投资 → 换手率 < 阈值、市盈率筛选、持仓 ≥ 周期"）；
- **制度迁移鲁棒性**：同一智能体在 θ 网格（T+0/T+1 × 涨跌停 {0,10%,20%} × 手续费 {0,10,50bp}）下的 (收益, 回撤, 换手) 变化矩阵；用"策略失效比例"（某 θ 下负收益或异常换手）衡量；
- **deployment-readiness 评分**：加权 `DR = w1·CS + w2·制度稳健性 + w3·窗口外成本后收益(−1 惩罚泄漏) + w4·行为方差罚`，w=(0.3,0.3,0.3,0.1)；
- **前视审计**：智能体输出若引用了"t 之后的信息"（在理由文本中检测），计入违规；对照仿真中故意"泄露"信息组的分数差（呼应 Idea 1 LAS）；
- **评测**：规则 + LLM judge（DeepSeek V4 Pro）双通道，κ≥0.8。

### 3.3 训练流程

- 无模型训练；仿真与 API 推理并行（4×L40 分派不同智能体/场景）；每次运行固定种子，重跑 3 次取 mean±std。

### 3.4 回测与评测流程

- **仿真回测协议**：每场景运行 250 步（日频）；成本含手续费与滑点；T+1 与涨跌停按 θ 配置执行；
- **窗口外真实回测（H3 佐证）**：在 2025-07 至 2026-06 真实数据上对同一智能体跑防污染回测（成本/制度约束，见 Idea 4 协议），与行为分数相关分析；
- **防泄漏**：仿真序列与评测期错开；智能体运行时序严格 step 化，无未来回看；全程审计日志。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | MarketSim-Bench v1（仿真剧本 + 行为标注，自建） |
| 来源 | 历史行情重采样 + Kimi 生成剧本；智能体复现自公开仓库 |
| 许可 | 仿真环境沿用原框架许可；剧本与行为标注随论文开源 |
| 划分 | 场景内训练/验证（剧本调参）/评测（Hold-out 剧本 20%） |
| 规模 | 5 场景 × 3 剧本 = 15 环境配置；每环境 250 步 × 3 种子 |
| 预处理 | 剧本格式化、规则集人工编写、行为记录标准化 |
| 质量门 | 行为标注 200 条人工复核；κ≥0.8 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| FinMem | LLM 智能体 | https://github.com/RobinWWang/FinMem | CS + 制度迁移 + DR |
| FinAgent | LLM 智能体 | arXiv:2402.18485（官方仓库以论文页为准） | 同上 |
| FLAG-Trader | RL 智能体 | arXiv:2502.11433 | 同上 |
| 规则基线（价值/动量） | 确定性 | 自实现 | 行为上界参考 |
| 随机代理 | 噪声 | 随机决策 | 行为下界参考 |
| 无风格指令版 | 消融 | 同一 LLM 去掉指令 | 指令增益 |

**统一口径**：同一环境、同一规则集、同一 judge；每智能体在全部 15 配置 × 3 种子运行。

## 6. 实验矩阵

- **A 主实验**：全部智能体 × 风格指令的一致性分数矩阵；
- **B 制度迁移**：θ 网格下收益-风险变化矩阵 + 策略失效比例（H2）；
- **C 指令消融**：有/无显式风格指令、指令措辞变体；
- **D 推理预算**：Flash/Pro/Kimi 与 reasoning effort 档位的行为稳定性（H4）；
- **E 前视对照**：泄漏信息组 vs 不泄漏组的行为分数差；
- **F 窗口外真实回测**：与行为分数相关性（H3）；
- **G 稳健性**：种子、剧本变体、判分通道（规则 vs judge）。

## 7. 评测协议

- **指标**：行为一致性 CS、制度迁移矩阵（Sharpe/Calmar/换手变化）、策略失效比例、deployment-readiness DR、前视泄漏检出率；
- **统计**：3 种子 mean±std；制度间 paired t-test；相关性 Spearman ρ；
- **无前视**：仿真 step 化 + 无未来信息访问 + 泄漏对照为必需项。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 仿真环境构建 | CPU | 2-3 周 |
| 智能体复现 | CPU + 4×L40 推理 | 数天/智能体 |
| 批量仿真运行 | 4×L40 + API | 14-28 GPU·天（数 GPU·周） |
| 行为标注与 judge | API（Pro） | 约 5,000 万 token |
| 窗口外真实回测 | CPU | 3-5 天 |
| 存储 | 轨迹 + 标注 | <200GB |
| 总计 | 4×L40 | 约 20-35 GPU·天 |

**API 成本**：Flash/Pro/Kimi 多智能体推理 + judge 约 1 亿 token。估算 $1,000-2,500（按当期定价，重点控制：批量提示、轨迹采样降频）。

## 9. 里程碑与时间线（单人 + 4×L40，9 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 仿真环境构建 + H0 可行性验证 | 环境 v1 + 运行日志 |
| W2 | 剧本生成（Kimi）+ 规则集 | MarketSim-Bench 剧本 v1 |
| W3 | 智能体复现（FinMem/FinAgent/FLAG-Trader） | 复现智能体 + 冒烟测试 |
| W4 | 行为记录 + 标注通道（规则 + judge） | 标注管线 v1 |
| W5 | 主实验 A | 一致性矩阵 |
| W6 | 制度迁移 B | 迁移矩阵 + 失效比例 |
| W7 | 指令/推理预算消融 C/D | 消融表 |
| W8 | 前视对照 E + 真实回测 F | E/F 表（H3） |
| W9 | 稳健性 + 论文初稿 | 提交稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 仿真与真实市场差异大、结论外推受限 | 高 | 定位为"行为评测"而非"收益预测"；加窗口外真实回测佐证 |
| 智能体复现困难（官方代码不可得/版本旧） | 中 | 以论文方法摘要重实现；若不可行则用 DeepSeek/Kimi 代理 + 规则基线补齐矩阵 |
| LLM judge 判分偏置 | 中 | 规则优先 + judge 交叉 + 200 条人工仲裁 |
| 行为分数方差大 | 中 | 3 种子 + 剧本多样 + 分数聚合窗口 |
| API 成本超支 | 中 | 轨迹降频采样；judge 只在关键步骤启用 |

## 11. 论文写作计划

- **目标会议**：AAMAS 2027 / ICML 2027 / NeurIPS 2026（智能体评测，以官方截稿为准）。
- **差异化卖点**：(1) 可配置微观结构的"行为压力测试"；(2) 行为一致性 + 制度迁移双指标；(3) deployment-readiness 评分对齐五层验证框架。
- **图表清单**：图1 仿真环境架构；图2 一致性矩阵热图；图3 制度迁移桑基/矩阵图；图4 前视对照；表1 剧本统计；表2 主实验；表3 消融；表4 DR 评分。
- **相关工作覆盖**：FinMem（2311.13743）、FinAgent（2402.18485）、FinCon（2407.06567）、TradingAgents（2412.20138）、FLAG-Trader（2502.11433）、Can LLMs Trade（2504.10789）、Alpha Illusion（2605.16895）、Benchmarks Are Not Validation（2607.28840）、StockBench（2510.02209）、MASS（2505.10278）、AlphaAgents（2508.11152）、Macro Economists（2606.08283）、LLM Market Experiments（2409.08357）。

## 12. 参考文献

1. Wang, R., et al. *FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory*. arXiv:2311.13743.
2. Zhang, W., et al. *FinAgent: A Multimodal Foundation Agent for Financial Trading*. arXiv:2402.18485.
3. Xu, Y., et al. *FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement*. arXiv:2407.06567.
4. Li, Y., et al. *TradingAgents: Multi-Agents LLM Financial Trading Framework*. arXiv:2412.20138.
5. *FLAG-Trader: Fusion LLM-Agent with Gradient-based RL for Financial Trading*. arXiv:2502.11433.
6. *Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations*. arXiv:2504.10789.
7. *LLM Competitive Market Behavior Experiments*. arXiv:2409.08357.
8. *The Alpha Illusion*. arXiv:2605.16895.
9. *Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications*. arXiv:2607.28840.
10. *StockBench: Pollution-Avoiding Evaluation of LLM Trading Agents on Real Market Signals*. arXiv:2510.02209.
11. *MASS: Multi-Agent Simulation for Portfolio Construction*. arXiv:2505.10278.
12. *AlphaAgents: Multi-Agent Equity Portfolio*. arXiv:2508.11152.
13. *Macro Economists in the Machine: Multi-Agent Macro ETF Portfolio*. arXiv:2606.08283.
14. *A Survey of Large Language Model based Autonomous Agents for Financial Trading*. arXiv:2408.06361.
