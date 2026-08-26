# 实验设计书：QuantCoder：带防前视静态检查与回测审计的 NL→量化策略代码智能体平台

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea06_QuantCoder.md
> 关联收藏论文：AAAI Emerging Trends in AI · 论文 284（PortfolioPilot）

## 0. 摘要

QuantCoder 是一个"自然语言 → 可回测量化策略代码"的智能体平台，在 PortfolioPilot 的思路上补上三个缺失层：**防前视静态检查器**（AST/数据流分析检测未来数据引用、幸存者偏差）、**严谨回测审计模块**（交易成本/滑点/T+1/停牌涨跌停约束 + 与前视检查交叉验证）、以及**智能体修复闭环**（生成→静态检查→回测→失败反馈→LLM 修复，≤3 轮）。BacktestBench 已证明 LLM 生成回测代码错误率高，Alpha Illusion 指出 Sharpe 报告≠部署证据——本平台的目标是把"生成正确且无泄漏的策略代码"做成可系统保障、可审计复现的评测框架。评测基于 BacktestBench + 自建 QuantBugSuite（注入 10 类已知泄漏/逻辑 bug）。

## 1. 研究背景与动机

### 1.1 问题定义

给定自然语言策略描述 D（如"每季度末买入股息率最高的前 20% 股票，持有 3 个月"），系统需产出：(1) 满足 D 的可执行策略代码 C；(2) C 通过防前视静态检查（无未来数据引用、无幸存者偏差）；(3) 严谨回测报告（含成本与制度约束）；(4) 审计日志（可复现）。评测对象是"代码的正确性与无泄漏性"而非策略收益本身。

### 1.2 相关工作不足

- **PortfolioPilot（AAAI Emerging Trends in AI · 论文 284 · PortfolioPilot: An Agentic Platform for Financial Portfolio Management Algorithm Development and Evaluation）**：只做"安全校验（防注入）"，无逻辑正确性与防前视校验；经典算法库浅，未接因子模型/RL/TSFM；
- **BacktestBench（arXiv:2605.17937）**：证明 LLM 生成回测代码错误率高，但偏评测而非"修复闭环"平台；
- **Alpha Illusion（arXiv:2605.16895）**：指出收益报告不可作为部署证据——需要审计层；
- **Summoning the Oracle（arXiv:2605.24564）**：LLM 辅助去前视偏差，但针对回测流程本身而非生成代码的静态检查。

### 1.3 为什么是现在、为什么你的环境适合做

生成型策略平台缺少"防泄漏"系统保障是社区共识缺口；本设计纯工程 + API 驱动，无大模型训练，4×L40 仅用于并行推理候选策略，预算极友好；BacktestBench 提供现成评测集，自建 QuantBugSuite 补充 10 类 bug。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 防前视静态检查能检出大多数注入泄漏 bug | 在 QuantBugSuite 上泄漏检出率 >90%（按 bug 类） |
| H2 | LLM 修复闭环显著降低最终泄漏率 | 有修复闭环 vs 无修复（单次生成）的最终无泄漏率提升 ≥30 个点 |
| H3 | 生成代码意图与回测结果一致 | 代码意图（如"买高股息 top20%"）与回测持仓/换手统计的语义一致率 >85%（LLM judge 评分） |
| H4 | 检查器假阳性可控 | 对正确策略误报率 <10% |

第一验证实验：把 QuantBugSuite 的 10 类 bug 注入 50 个正确策略模板，测静态检查器检出率（H1），3 天出结果。

## 3. 总体方法设计

### 3.1 数据流水线

- **策略 DSL**：Python 子集（向量化算子库：`ref(close, k)`、`rank()`、`top_pct()`、`shift()`、`ewma()` 等），所有算子显式声明其使用的数据时点（bar t 内可用的字段白名单）；
- **数据源白名单**：日线 OHLCV、成交量、市值、基本面快照（含"披露日期"字段）、板块/指数成分（含"加入日期"）；**未来数据（未来收益、t+1 收盘）不在白名单**；
- **QuantBugSuite 构造**：10 类 bug——未来收益引用、t+1 收盘决策、lookahead 复权因子、幸存者股票池、shift 方向错误、条件中用错时点、四舍五入/单位错、跨周期聚合泄漏、用全量指数成分而非时点成分、滑点/成本缺失；每类 20 个样例 = 200 个 bug 注入策略 + 50 个正确策略；
- **人工标注**：每个策略配 ground-truth bug 类别与严重度（人工 3 人交叉核对）；
- **防泄漏**：评测策略不进入任何模型的训练/提示缓存；BugSuite 与评测回测窗口不重叠。

### 3.2 方法设计

- **静态检查器**：
  1. AST visitor：遍历策略代码，识别数据源调用（`ref`、`shift`、数据字段）；
  2. **污染传播（taint）分析**：把"未来信息源"（如 `future_return`、`close.shift(-1)`、复权因子、非时点成分）标记为 tainted，追踪其流向（变量、表达式、条件分支、交易信号）；
  3. 规则引擎：白名单数据源 + 时点字段校验 + 停牌/涨跌停约束检查；
  4. 输出：问题清单（类别、行号、说明、严重度）+ 修复建议；
- **回测审计模块**：事件驱动回测（成本、滑点、T+1、涨跌停不可成交、停牌跳过、成交量上限）；把回测中"实际成交时点 vs 信号时点"映射回代码行，与前视检查结果交叉验证（若回测出现未来时点成交而静态检查未报，则记 false-negative）；
- **智能体闭环**：DeepSeek V4 Flash 生成候选 → 静态检查 → 若报错则把错误类别喂回 LLM 要求修复（≤3 轮）→ 修复后重检 → 通过则回测 → 审计报告（含 LLM 修复轨迹）；
- **代码-意图对齐评分**：DeepSeek V4 Pro 作为 judge，比对"策略描述 D"与"生成代码的语义摘要 + 回测持仓统计"，输出一致率；
- **防前视**：所有评测在固定窗口（2020-2025 训练意图 / 2025-07 至 2026-06 回测验证）上跑；白名单与 taint 规则不因评测结果调整。

### 3.3 训练流程

- 无大模型训练；可选：用 BugSuite 的 (bug描述, 修复代码) 对做 SFT（QLoRA 7B，2-3 GPU·天）提升修复率，作为消融项；
- 静态分析器本地轻量运行（CPU）；4×L40 并行推理多候选策略。

### 3.4 回测与评测流程

- **回测协议**：T+1、双边成本 20-50bp、涨跌停不可成交、停牌跳过、成交上限 10% 日额；持仓按收盘价信号、次日可成交；
- **防泄漏**：股票池按信号时点存续成分；复权用无前视版本（见 Idea 4）；回测窗口外评测；
- **评测指标**：BugSuite 检出率/修复率；正确策略误报率；意图一致率；回测收益仅作辅助信息并标注"非部署证据"。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | QuantBugSuite v1（自建）+ 复用 BacktestBench |
| 来源 | 50 个正确策略模板（人工编写）× 10 类 bug 注入 |
| 许可 | 自建数据随论文开源（CC-BY-4.0 意向） |
| 划分 | 50 正确 + 200 含 bug；人工标注 bug 类别/严重度 |
| 预处理 | 策略统一 DSL 化、含 bug 版本与原版成对保存 |
| 质量门 | 3 人交叉核对；bug 注入后运行回测确认行为差异 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| 直接生成（无检查） | 平台基线 | DeepSeek Flash 单轮生成 | 无泄漏率、检出率 |
| 纯静态检查（无 LLM 修复） | 平台基线 | 本设计检查器只报不改 | 检出率 |
| PortfolioPilot 式安全校验 | 参考 | 仅注入防护 + 语法检查 | 检出率（预期低） |
| QuantCoder（完整闭环） | 本方法 | 本设计 | 全部指标 |
| BacktestBench 官方评测 | 对照 | arXiv:2605.17937（官方仓库为准） | 对齐其指标 |

**统一口径**：同一 BugSuite、同一回测引擎、同一评测窗口；温度 0.4（生成）。

## 6. 实验矩阵

- **A 主实验**：直接生成 vs 纯检查 vs 完整闭环（无泄漏率/检出率/修复率/意图一致率）；
- **B bug 类消融**：10 类 bug 的逐类检出/修复率表；
- **C 修复轮数消融**：0/1/2/3 轮的成本-收益曲线；
- **D 假阳性**：50 个正确策略的误报率；
- **E 基座模型消融**：DeepSeek V4 Flash vs Pro vs Kimi K2.6（生成与修复）；
- **F 可选 SFT**：BugSuite SFT 后修复率提升（消融）。

## 7. 评测协议

- **指标**：泄漏检出率（recall@bug）、修复率（最终无泄漏率）、假阳性率（precision）、意图一致率（Pro judge，人工复核 100 条）；成本 = 平均生成/修复 token 与轮数；
- **统计**：3 种子 mean±std；类别间显著性用 McNemar 检验；
- **无前视**：白名单与 taint 规则固定；回测窗口外；审计日志完整可复现。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| BugSuite 构造 | CPU + 人工 | 2-3 人·周 |
| 静态检查器实现 | CPU | 1-2 周 |
| 候选策略并行推理 | 4×L40 + API | 1-2 GPU·天 |
| 可选 SFT 修复器 | 4×L40 | 2-3 GPU·天 |
| 回测与审计 | CPU | 2-3 天 |
| 存储 | 代码/日志/回测结果 | <50GB |
| 总计 | 4×L40 | 约 3-5 GPU·天 |

**API 成本**：Flash 生成/修复约 3,000 万 token；Pro judge 约 1,000 万；Kimi K2.6 长文档约 500 万。估算 $150-450（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，8 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | DSL 与数据源白名单设计 | DSL 规范 v1 |
| W2 | 静态检查器 v1（AST + taint） | 检查器 + 单元测试 |
| W3 | BugSuite 构造 + 人工标注 | QuantBugSuite v1 |
| W4 | 回测审计模块 | 回测引擎 + 交叉验证 |
| W5 | 智能体闭环（生成→检查→修复） | 平台 v1 |
| W6 | 主实验 A/B | 全表 |
| W7 | 修复轮数/基座/假阳性 C/D/E | 消融表 |
| W8 | 论文初稿 + 开源 | 提交稿 + 仓库 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 静态分析对"隐性未来信息"（外部 API 取未来数据）检测不完整 | 高 | 白名单 + taint 规则明确覆盖受支持数据源；不支持的调用直接标"未知源"并拒绝回测 |
| 误报率高影响可用性 | 中 | 严重度分级（error/warning）；假阳性消融 + 规则迭代 |
| LLM 修复反复失败 | 中 | 3 轮上限 + 轮数消融；可选 SFT 修复器 |
| 与 PortfolioPilot 区分度不足 | 中 | 突出防前视静态检查 + 回测审计 + 修复闭环三件套 |
| BugSuite 人工标注偏差 | 低 | 3 人交叉核对 + 行为差异验证 |

## 11. 论文写作计划

- **目标会议**：ICML 2027（工具/Agent）/ AAAI 2027 Demo（以官方截稿为准）。
- **差异化卖点**：(1) 首个"防前视静态检查 + 回测审计 + 修复闭环"的量化策略生成平台；(2) 可审计日志与可复现报告；(3) 与 BacktestBench 对齐的量化评测。
- **图表清单**：图1 平台架构（生成→检查→回测→修复）；图2 taint 分析示例；图3 修复轮数成本曲线；表1 BugSuite 统计；表2 主实验；表3 逐类检出/修复；表4 基座消融。
- **相关工作覆盖**：PortfolioPilot（AAAI 论文 284）、BacktestBench（2605.17937）、Alpha Illusion（2605.16895）、Summoning the Oracle（2605.24564）、StockBench（2510.02209）、QuantEval（2601.08689）、FINESSE-Bench（2605.15482）、Benchmarks Are Not Validation（2607.28840）。

## 12. 参考文献

1. Xu Yang, J. C., Ma, H., Ma, Y. *PortfolioPilot: An Agentic Platform for Financial Portfolio Management Algorithm Development and Evaluation*. AAAI 2026；DOI: 10.1609/aaai.v40i48.42396.
2. *BacktestBench: Automated Strategy Backtest Evaluation*. arXiv:2605.17937.
3. *The Alpha Illusion*. arXiv:2605.16895.
4. *Summoning the Oracle to Slay It: Mitigating Look-Ahead Bias in Financial Backtesting with LLMs*. arXiv:2605.24564.
5. *StockBench: Pollution-Avoiding Evaluation of LLM Trading Agents on Real Market Signals*. arXiv:2510.02209.
6. *QuantEval: Evaluating Quant Strategy Code Generation*. arXiv:2601.08689.
7. *FINESSE-Bench: Evaluating Backtest Code Correctness*. arXiv:2605.15482.
8. *Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications*. arXiv:2607.28840.
9. *Market-Bench: A Benchmark for Financial Markets and Strategies*. arXiv:2512.12264.
