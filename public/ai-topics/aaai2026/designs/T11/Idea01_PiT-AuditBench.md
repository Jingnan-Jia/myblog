# 实验设计书：PiT-AuditBench：面向金融回测的 LLM 时序污染审计基准

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea01_PiT-AuditBench.md
> 关联收藏论文：AAAI Emerging Trends in AI · 论文 183（Chronology）；间接关联 Machine Learning VII · 论文 47（Kronos）、Application Domains I · 论文 57（FinRpt）

## 0. 摘要

PiT-AuditBench 是一个面向金融 LLM 的"点即时刻（Point-in-Time）污染 + 时间顺序理解"联合审计基准。它把 Chronology 的三类时间任务（乱序排序 / 条件排序 / 年代错误检测）落到具体金融事件实体（财报日、分红除权、限售解禁、宏观数据发布），并新增**绝对日期锚定**与**金融日历约束**两类任务。基准的核心资产是"训练窗口 overlap 已知"的事件集：按事件发布年度分层为"训练窗口内 / 窗口外"，从而把"记忆排序"与"时序推理"解耦，并输出标准化的 look-ahead risk score。该 score 可作为插件挂接到任意金融 LLM 管线，回答"这份回测结果有多少来自记忆污染"。与 Look-Ahead-Bench（2601.13770）等既有基准相比，本基准把"时间顺序理解能力"作为一等公民构造块，且全部事件附带可核验的权威来源与时间戳。

## 1. 研究背景与动机

### 1.1 问题定义

给定金融事件集合 E = {e_i}，每个事件 e_i = (entity_i, type_i, date_i, ts_utc_i, source_i)，其中 date_i 是事件发生/发布日（绝对日期），ts_utc_i 是权威来源披露的 UTC 时间戳。定义四类任务：

- **T1 乱序排序**：打乱 m 个（m∈{5,10,15}）同类或混合事件，要求按时间正序重排，测 exact-match 与 Kendall τ；
- **T2 条件排序**：先给过滤条件（如"只保留 2022 年后的财报事件"），再对过滤后事件排序；
- **T3 多时间线年代错误检测**：从两条及以上时间线的交叉处构造年代错误，要求指出"哪条时间线上的事件与全局时间线矛盾"；
- **T4 绝对日期锚定 + 金融日历约束**：问"该事件发生在哪一天？"（绝对日期 MAE）；以及"分红除权日与登记日顺序""财报发布日是否在 t 日之前""解禁日是否已过"等约束判定。

**look-ahead risk score** 定义为模型在"训练窗口内事件"上的超额表现（记忆红利）：
`LAS(m) = Acc_in(m) − Acc_out(m)`（同任务、同长度、同难度分层），其中 Acc_in/Acc_out 分别是模型在窗口内/窗口外事件集上的正确率。LAS>0 且显著时，说明模型在该任务上依赖记忆而非时序推理，其回测结果需要打污染折扣。

### 1.2 相关工作不足

- **Chronology（AAAI Emerging Trends in AI · 论文 183 · Do Large Language Models (LLMs) Understand Chronology? (Student Abstract)）**：证明 LLM 对"已知事件"也会乱序、条件排序困难、多时间线年代错误频发；但只用历史已知事件，未覆盖训练窗口外事件、未测绝对时间戳与金融日历锚点，且未给可操作去偏方法。
- **Look-Ahead-Bench（arXiv:2601.13770）**：做点即时刻污染规避评测，但以"问答/任务级"为主，未把"时间顺序理解"拆成可审计的构造块，也未对绝对时间戳打分。
- **StockBench（arXiv:2510.02209）**：强调用真实市场信号规避污染，但目标是被测模型的收益，而非时间理解本身。
- **Assessing Look-Ahead Bias（arXiv:2309.17322）**：实证 GPT 情感策略的 look-ahead 与 distraction 偏差并给出剥离方法，但只针对新闻情感单信号，未推广到事件排序/时间理解。
- **ExAnte（arXiv:2505.19533）**：事前推断基准关注"模型在评测日之前是否已见过信息"，与本文互补但侧重数据而非时间理解。

共同空白：**没有一个基准同时保证（a）事件训练窗口 overlap 已知、（b）时间理解能力可量化、（c）输出可直接挂接为污染审计分数**。

### 1.3 为什么是现在、为什么你的环境适合做

2025-2026 社区共识（Alpha Illusion 2605.16895、Benchmarks Are Not Validation 2607.28840）是把"防污染回测"从建议升级为评审红线；而 Chronology 进一步证明即便信息已知，LLM 时间线也会乱——这使"时序审计"成为基础设施级选题。本环境优势：(1) 数据全部来自公开权威源（SEC EDGAR、交易所公告、宏观发布日历），无需付费许可；(2) 基准构造是纯推理评测 + 数据工程，对算力极友好（4×L40 足够），适合作为多篇论文的公共基建；(3) 可用开源 cutoff 已知模型（Llama-3.1/Qwen2.5/DeepSeek-V3）做窗口分层代理，配合闭源 API 敏感性分析。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | LLM 在训练窗口内事件上的排序/日期表现显著高于窗口外事件（存在记忆红利） | 对多数基础模型，LAS>0 且在 95% 置信水平显著；窗口外 exact-match 明显低于窗口内 |
| H2 | 推理模型 + 测试时扩展推理主要提升的是"时序推理"（窗口外）而非"记忆"（窗口内） | 窗口外任务分数随 reasoning effort 上升的斜率大于窗口内；LAS 随 effort 上升而下降 |
| H3 | 绝对时间戳 / 金融日历约束是排序任务之外独立的薄弱点 | 模型在 T4 上的 MAE 与违反率显著高于 T1/T2 同难度样本；二者相关性低（ρ<0.5） |
| H4 | LAS 可作为金融回测污染的代理指标 | 在"用过去 K 线 + LLM 新闻打分"的回测中，评测期落在训练窗口内时的 RankIC 提升量与对应模型 LAS 正相关 |

第一验证实验：在 8 个模型（4 开源 cutoff 已知 + 4 API）上跑通 T1-T4，输出 LAS 与 Kendall τ 全表，检验 H1/H2。

## 3. 总体方法设计

### 3.1 数据流水线

**数据源（全部权威、公开）**：
1. SEC EDGAR（Filing Dates：8-K/10-K/10-Q 的 accepted date）；
2. 交易所公司行动公告（NYSE/NASDAQ/SSE/SZSE 官网 corporate actions：分红除权、配股、限售解禁、回购）；
3. 宏观发布日历（BLS 非农、CPI、美联储 FOMC、中国统计局 CPI/PMI/社融）；
4. 免费开源财经 API（tushare / akshare / yfinance 仅用于行情与公司行动元数据交叉校验，不含模型输出）。

**API 合成/抽取 prompt 思路**（DeepSeek V4 Flash）：
- **事件实体抽取**：给原文段落，要求输出 JSON 数组 `[{event_type, entity_name, ticker, exchange, event_date, announcement_ts_utc, source_url}]`；规则强调"event_date 仅取原文出现的信息、不确定字段填 null、禁止推断"。
- **难样本构造**（Kimi K2.6）：生成"发布日期与生效日期不同""同日多事件""跨市场同日事件"的金融事件描述用于 T3 难例。
- **答案判分**（DeepSeek V4 Pro，LLM-as-judge）：排序答案先按规则计分（exact-match / τ），再用 judge 对"部分正确但顺序局部颠倒"的答案复核，与规则计分交叉验证，judge 一致性用 κ≥0.8 把关。

**清洗与去泄漏规则**：
- 事件日 = 权威披露时间戳，不使用任何事后回溯数据（如复权因子、修正后的宏观值）；
- 去重：同 entity+date+type 只保留最早披露源；跨源冲突（如 SEC 与交易所日期不同）以最早披露为准并记录差异；
- 去污染：构造完成后，全部 20k 事件的 source URL 与日期保持可审计；标注集不接受任何 LLM 生成事件（防止基准本身被污染）；
- 窗口分层：按事件日期与各模型已知训练截止（Llama-3.1-8B≈2023-12、Qwen2.5-7B≈2024-06、DeepSeek-V3≈2024-06、GPT-4.1/DeepSeek V4/Kimi K2.6 闭源以"代理+敏感性分析"处理）切分窗口内外两层。

**数量**：约 20,000 条事件（T1≈8,000 / T2≈5,000 / T3≈4,000 / T4≈3,000），覆盖 2015-01 至 2026-06，其中窗口外（≥2024-06）约占 40%。每个样本含人工抽检 300 条（double-annotated，κ≥0.9）作为质量门。

### 3.2 方法设计

- **样本生成**：从事件池按任务模板装配（排序长度 m、条件复杂度 c、时间线数 k 可配置），配平难度阶梯（容易/中等/难），避免极端分布；
- **评测器**：纯规则（exact-match / Kendall τ-b / 绝对日期 MAE）+ LLM-as-judge 交叉验证；
- **LAS 计算**：按任务与难度分层后 `LAS = Acc_in − Acc_out`，再做 1000 次 bootstrap 求 CI；
- **打分器（可选蒸馏）**：LoRA 微调 7B 模型做"排序对/日期对"二元判定，输出 0-1 分数，作为 judge 的本地替代（省钱、可复现）；
- **防前视设计**：评测时给模型的任务描述只含事件名与类型，**不含日期**；T4 的"截至 t 日"问题中 t 从事件池随机抽取并保证 t 不晚于全部相关事件；所有 score 按"窗口内/窗口外"分层报告，禁止报告混合池单一数字。

### 3.3 训练流程

- 主体无大模型训练，仅推理评测；
- 可选 LoRA 打分器：7B 基座（Qwen2.5-7B），`r=16, α=32, dropout=0.05`，AdamW `lr=2e-5`，batch=8（grad-acc=8），cosine schedule，1 epoch，4×L40 FSDP，约 1-2 GPU·天；
- 评测并行：4 卡分 4 个模型并行推理；API 模型用并发池（并发 8-16），控价重试。

### 3.4 回测与评测流程

- **评测协议**：同一任务模板库分别灌入窗口内/外事件 → 分别计分 → 分层报告；
- **防泄漏**：评测样例绝不进入任何训练/蒸馏集；窗口外样例在"可公开验证到当日"的约束下构造（事件日 ≤ 数据冻结日 2026-06-30）；
- **回测挂接演示**（供论文论证"LAS 可用"）：在 A 股 CSI300 上跑"K 线特征 + LLM 新闻打分"策略，训练窗口（对标 LLM cutoff）内/外各一个回测期，展示 RankIC 差与 LAS 的正相关（H4）；
- 所有实验随机种子固定（seed∈{0,1,2}），报告 mean±std。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | PiT-AuditBench v1（自建，CC-BY-4.0 意向） |
| 来源 | SEC EDGAR、SSE/SZSE 官网公告、BLS/中国统计局宏观日历、tushare/akshare 元数据校验 |
| 许可 | 均为公开数据；加工后的标注（人工+LLM 双重）由本项目持有，随论文开源 |
| 划分 | 窗口内/窗口外（按模型 cutoff）；不设"训练/测试"划分（评测基准） |
| 事件数 | ~20k；实体去重后覆盖 ~1.2k 公司 + 60 类宏观发布 |
| 预处理 | UTF-8 统一、日期 ISO8601(UTC)、事件类型标准化到 6 类（财报/分红/解禁/并购/宏观/其他） |
| 质量门 | 人工抽检 300 条双标注 κ≥0.9；机器抽取事件 5% 随机复核 |

## 5. 基线复现

| 基线 | 类型 | 复现方式/官方来源 | 预期指标口径 |
|---|---|---|---|
| Chronology 原任务集 | 排序基准 | arXiv:2511.14214，按论文任务模板复现 | exact-match / τ 对齐原文 |
| Look-Ahead-Bench | 污染基准 | arXiv:2601.13770（官方仓库以论文页为准） | 报告其 pollution 指标作为对照 |
| ExAnte | 事前推断 | arXiv:2505.19533 | 报告其事前分数 |
| Llama-3.1-8B | 开源模型 | https://github.com/meta-llama/llama-models | 全任务指标 |
| Qwen2.5-7B | 开源模型 | https://github.com/QwenLM/Qwen2.5 | 全任务指标 |
| DeepSeek-V3 | 开源模型（cutoff 已知） | https://huggingface.co/deepseek-ai/DeepSeek-V3 | 全任务指标 |
| GPT-4.1 / DeepSeek V4 Pro / Kimi K2.6 | 闭源 API | 官方 API | 全任务指标（窗口分层用代理法） |

**统一口径**：所有模型用同一条系统提示与相同的任务模板，固定温度=0（排序类）或 temperature=0.2（judge），max_tokens 按任务上限；报告 3 次随机打乱的平均值。预期表：每模型 × 每任务 × 窗口层 = 平均 correct + ±std。

## 6. 实验矩阵

- **A 主实验**：8 模型 × 4 任务 × 2 窗口层，输出 LAS 与全部指标。
- **B 难度消融**：排序长度 m∈{5,10,15}、条件数 c∈{0,1,2}、时间线数 k∈{1,2,3} 下的分数梯度。
- **C 推理预算消融**：对推理模型按 low/medium/high 三档 effort，测窗口内外斜率差（验证 H2）。
- **D 绝对时间戳专项**：T4 的 MAE 与日历约束违反率；与 T1/T2 的相关性（验证 H3）。
- **E 污染挂接演示**：A 股新闻打分回测，窗口内外 RankIC 差 vs LAS 的相关性（验证 H4）。
- **F 稳健性**：换打乱种子、换 judge 模型、窗口划分 ±3 个月的敏感性。

## 7. 评测协议

- **指标定义**：exact-match（全序列正确比例）；Kendall τ-b（打结校正）；T4 绝对日期 MAE（天）；T3 年代错误检出率；LAS = Acc_in − Acc_out；
- **统计**：每个指标在 3 种子 × 3 次打乱 = 9 次测量上取 mean±std；窗口内外之差用 paired t-test（α=0.05）与 1000 次 bootstrap CI；
- **显著性**：LAS 显著为正需 bootstrap CI 下界 >0；模型间比较用配对检验；
- **无前视保证**：评测输入不含日期、不含事后修正数据；窗口外事件截止 2026-06-30；全程记录审计日志。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 数据收集与清洗 | CPU 为主 | 约 2-3 人·周（并行） |
| 开源模型评测 | 4×L40 并行 | 1-2 GPU·天 |
| 闭源 API 评测 | API | 约 3000 万 token（Flash 抽取 + Pro judge），成本估算见下 |
| 可选 LoRA 打分器 | 4×L40 | 1-2 GPU·天 |
| 存储 | 事件 JSON + 评测日志 | <100GB |
| 总计 | 4×L40 | 约 2-4 GPU·天 |

**API 用量**：DeepSeek V4 Flash 事件抽取与预筛约 2000 万 token；DeepSeek V4 Pro 判分约 800 万 token；Kimi K2.6 难样本约 200 万 token。成本以"输入/输出 token 数 × 当时官方单价"核算，预估值在 $150-400 区间（随价格波动，需按当期 API 定价更新）。

## 9. 里程碑与时间线（单人 + 4×L40，8 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 数据源接入，事件抽取 pipeline v1 | 抽取脚本 + 5k 事件 |
| W2 | 任务模板装配 + 人工抽检 | 全任务样例库 v1 |
| W3 | 窗口分层 + 开源模型评测 | LAS 初表（4 开源模型） |
| W4 | API 评测 + judge 交叉验证 | 全模型指标 + κ 报告 |
| W5 | 难度消融 + 推理预算消融 | B/C 实验表 |
| W6 | 污染挂接演示（A 股回测） | E 实验表（H4 检验） |
| W7 | 敏感性分析 + 文档 | F 实验 + 数据卡 |
| W8 | 论文初稿 + 基准开源页 | 提交稿 + 公开仓库 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 闭源 API 训练窗口未知，LAS 分层失效 | 高 | 用开源 cutoff 模型做主结论 + 对闭源做 3 种代理 cutoff 敏感性分析；或以"只报开源模型 LAS"收口 |
| 事件实体抽取噪声 | 中 | 权威源直读优先；LLM 抽取仅做辅助并用 5% 人工复核 |
| 任务难度失衡导致所有模型饱和/地板 | 中 | 预跑 pilot 调参（m、条件数、时间线数），保证 30-80% 正确率带 |
| 与既有基准（Look-Ahead-Bench）区分度不足 | 中 | 强调 T3/T4 与 LAS 挂接的唯一性；主动对比并报告增量 |
| 数据获取受限（交易所公告接口变更） | 低 | 以 EDGAR + 宏观日历为主干，交易所数据用 akshare 兜底 |

## 11. 论文写作计划

- **目标会议**：ACL 2027（预计 ARR 2027 年 1 月窗口）或 AAAI 2027（以官方截稿公告为准），benchmark 型论文；若周期合适可作为主论文的评测基建附赠。
- **差异化卖点**：(1) 首个把"记忆 vs 推理"解耦并输出可挂接 LAS 的金融时序审计基准；(2) 覆盖绝对时间戳与金融日历约束，补齐 Chronology 空白；(3) 全部事件可审计来源，规避基准自身污染。
- **图表清单**：图1 基准构造流程；图2 窗口内外性能散点（LAS 可视化）；图3 难度梯度曲线；图4 reasoning effort vs 窗口内外斜率；表1 数据集统计；表2 全模型×任务×窗口指标；表3 LAS 挂接回测 RankIC 相关性；表4 敏感性。
- **相关工作覆盖**：Chronology、Look-Ahead-Bench、StockBench、ExAnte、PiT-Inference（2512.06607）、ChronoSense（2501.03040）、TLQA（2506.21783）、Structured yet Bounded Temporal Understanding（2510.16685）、Assessing Look-Ahead Bias（2309.17322）、Alpha Illusion（2605.16895）、Benchmarks Are Not Validation（2607.28840）。

## 12. 参考文献

1. Wongchamcharoen, P. K., & Glasserman, P. *Do Large Language Models (LLMs) Understand Chronology? (Student Abstract)*. AAAI 2026；arXiv:2511.14214.
2. *Look-Ahead-Bench: A Standardized Benchmark for Point-in-Time Look-Ahead Bias in LLMs*. arXiv:2601.13770.
3. *StockBench: Pollution-Avoiding Evaluation of LLM Trading Agents on Real Market Signals*. arXiv:2510.02209.
4. *ExAnte: A Benchmark for Ex-Ante Inference in Financial LLMs*. arXiv:2505.19533.
5. *Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis*. arXiv:2309.17322.
6. *A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs*. arXiv:2512.06607.
7. *ChronoSense: Evaluating Temporal Understanding in LLMs*. arXiv:2501.03040.
8. *TLQA: Time-Sensitive List QA Benchmark*. arXiv:2506.21783.
9. *Structured yet Bounded Temporal Understanding in LLMs*. arXiv:2510.16685.
10. *The Alpha Illusion*. arXiv:2605.16895.
11. *Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications*. arXiv:2607.28840.
12. *FINSABER: Twenty-Year Regime-Switching Backtesting of LLM Investment Strategies*. arXiv:2505.07078.
13. *A Survey of Large Language Models in Finance (FinLLMs)*. arXiv:2402.02315.
