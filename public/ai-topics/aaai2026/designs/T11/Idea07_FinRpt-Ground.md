# 实验设计书：FinRpt-Ground：点即时刻事实锚定 + 数值一致性验证的股权研究报告生成

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea07_FinRpt-Ground.md
> 关联收藏论文：Application Domains I · 论文 57（FinRpt）；依赖 Idea09 的验证器能力

## 0. 摘要

FinRpt-Ground 把股权研究报告（Equity Research Report, ERR）生成从"写得好"升级为"说得对、说得可追溯、说得有预测力"。核心是三点即时刻约束：**(1) 点即时刻检索**——报告只能引用报告日之前已披露的财报/公告（时间墙），杜绝未来数字进入报告；**(2) 数值一致性验证器**——对报告每个财务数字做"claim→原始披露单元格"的来源追踪，检出幻觉并触发重写；**(3) 决策有效性指标**——报告的评级/目标价与后续实际收益的校准（rank-IC、命中率），补 FinRpt 11 指标的空白。方法为"多智能体分步生成 + 验证器 + RL 对齐"闭环：基于 FinRpt 数据集 SFT，再用验证器信号做 RL（奖励 = 事实性 + 数值一致性 + 决策有效性）。

## 1. 研究背景与动机

### 1.1 问题定义

输入：股票 s、报告日期 t_r、点即时刻检索的披露语料 D_pit（≤t_r 的财报/公告/研报）。输出：结构化研究报告 R（含评级、目标价、财务预测数值、行文）。评估维度：文本质量（对齐 FinRpt 11 指标）、事实锚定（每个数值可溯源到 D_pit 的单元格）、数值一致性（报告内数值与披露数值一致、报告间一致）、决策有效性（预测与后续实现吻合）。

### 1.2 相关工作不足

- **FinRpt（AAAI Application Domains I · 论文 57 · FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation）**：11 指标偏文本/结构相似度与事实核对，未接入"报告预测与后续股价一致"的经济有效性；数据切分未明确 point-in-time 处理；
- **Financial Statement Analysis with LLMs（arXiv:2407.17866）**：GPT-4 方向性盈利预测胜过人类分析师，但未与报告生成打通、未做点即时刻约束；
- **FAITH（arXiv:2508.05201）/ Deficiency of LLMs（arXiv:2311.15548）**：证明金融 LLM 幻觉严重、表格数值系统性失败——报告生成最怕编数字；
- **Point-in-Time Financial RAG（arXiv:2605.31201）**：展示点即时刻检索可行性，但未解决"生成后数值验证 + 重写"闭环；
- **Template-Based Financial Report Generation（arXiv:2504.14233）**：模板化检索生成，无决策有效性评估。

### 1.3 为什么是现在、为什么你的环境适合做

FinRpt 刚把 ERR 任务形式化（数据集+评测系统公开），但其"11 指标无经济有效性"与"切分非 point-in-time"是明确可补的评审关注点；点即时刻 RAG 与幻觉检测方法（2605.31201、2605.31064）已就绪；A 股研报公开数据可补中文空白。环境优势：(1) 7B SFT+RL 在 4×L40 上 3-6 GPU·周可承担；(2) 披露数据（财报/公告）点即时刻可构造；(3) 验证器（Idea 9）可与本 idea 共享模块。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 点即时刻检索显著降低未来信息泄漏 | 报告引用"报告日之后披露"数字的比例：点即时刻版 < 全量检索版（显著） |
| H2 | 数值验证器 + 重写闭环降低幻觉率 | 数值幻觉率（不可溯源数字占比）在闭环后下降 ≥50%（相对） |
| H3 | 决策有效性指标能区分报告质量 | 生成的报告评级/目标价的校准（rank-IC）显著优于随机；且与 11 文本指标低相关（说明补了盲区） |
| H4 | RL 奖励中"事实性+一致性"不损害"决策有效性" | RL 后目标价 IC 不降（或微升），同时幻觉率下降 |

第一验证实验：在 100 只股票的评测集上跑"全量检索 vs 点即时刻检索"对照，测未来泄漏比例（H1），1 周出结果。

## 3. 总体方法设计

### 3.1 数据流水线

- **披露语料**：美股（SEC EDGAR 10-K/10-Q/8-K，权威 filing date）；A 股（财报 + 公告，交易所披露时间）；每个披露项带"披露日期"；
- **点即时刻检索**：检索时过滤 `disclosure_date ≤ t_r`；索引按 (股票, 期间, 披露日期) 分桶，保证数值单元格可溯源；
- **报告数据集**：FinRpt 公开数据集（arXiv:2511.07322）+ 自建 A 股研报集（公开研报文本，2020-2025）；每份报告配 (股票, t_r, 评级, 目标价, 财务预测表) 结构化标注；
- **API prompt 思路**：
  - DeepSeek V4 Flash：从披露文本抽取"数值 claim"（数字 + 单位 + 期间 + 口径）→ JSON；
  - Kimi K2.6：生成中文研报长文草稿（按模板分段：行业/公司/财务/风险/评级）；
  - DeepSeek V4 Pro：数值-来源对齐判分与最终事实性 judge；
- **清洗与去泄漏**：报告 t_r 与披露日期强约束；训练/评测按 t_r 切分不重叠（训练 ≤2024-12，评测 2025-01 至 2026-06）；从语料剔除"含未来数据"的合成污染样本；
- **数量**：披露语料 ~100 万份（美股 + A 股）；报告训练集 FinRpt 规模 + 自建 A 股 ~5 万份；评测集人工标注 500 份（数值溯源 ground-truth）。

### 3.2 方法设计

- **多智能体管线**：数据抽取智能体（提取披露数值）→ 财务分析智能体（预测数值 + 逻辑）→ 行文智能体（生成段落）→ 合规智能体（免责声明/风险提示）→ **验证器**（见下）→ 重写循环（≤2 轮）；
- **数值验证器**：对报告每个数值 claim v，检索候选证据单元格（数值相近 + 期间/口径匹配），判定 {一致, 矛盾, 无法验证}；一致性判定用"数值相对误差 <1% + 单位/期间对齐"规则 + NLI 式对齐模型（可复用 Idea 9）；
- **决策有效性**：评级分档（1-5）与目标价收益率 vs 后续实现（h∈{30,90,180} 天），计算校准 IC 与命中率；
- **RL 奖励**：`R = λ1·事实性（可溯源率） + λ2·数值一致性（验证器得分） − λ3·幻觉惩罚（矛盾/无法验证率） + λ4·决策有效性（目标价 IC 的批内排名）`，λ=(0.4,0.3,0.2,0.1) 起步；
- **防前视**：t_r 为报告日；预测目标价只对标"报告日之后"的实现；检索、验证、评测全部点即时刻；
- **训练**：先在 FinRpt + 自建集 SFT（LoRA 7B），再用 GRPO 以 R 为奖励做 RL。

### 3.3 训练流程

- 7B（Qwen2.5-7B）LoRA SFT：r=16, α=32, lr=2e-5，batch 16（grad-acc 8），2 epochs，4×L40 FSDP，1-2 GPU·周；
- GRPO RL：采样 8 个 rollouts/问题，`lr=1e-6`，500-1000 步，3-5 GPU·周；奖励计算与训练解耦（预生成验证结果缓存）；
- 验证器独立训练（可选）：NLI 式对齐 7B LoRA 2-3 GPU·天。

### 3.4 回测与评测流程

- **评测集**：500 份人工标注报告（含数值溯源 ground-truth + 评级/目标价实现）；
- **防泄漏协议**：t_r 切分；检索只含披露日 ≤ t_r；实现窗口（t_r 之后）不与训练重叠；
- **评测项**：11 文本指标（对齐 FinRpt）、数值幻觉率/可溯源率、决策有效性（评级校准 IC、目标价 IC、命中率）、合规项（免责声明存在率）。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | FinRpt-Ground-DS v1（FinRpt 公开集 + 自建 A 股集） |
| 来源 | FinRpt 数据集（arXiv:2511.07322 声明公开）；SEC EDGAR；公开研报 |
| 许可 | FinRpt 数据按其许可；自建标注随论文开源 |
| 划分 | 训练 ≤2024-12；验证 2025H1；评测 2025-01 至 2026-06（500 份人工） |
| 规模 | 披露 ~100 万份；报告 ~5 万份训练；500 评测 |
| 预处理 | 点即时刻索引、数值 claim 结构化、报告-披露对齐、去重 |
| 质量门 | 评测标注双标注 κ≥0.85；数值溯源人工复核 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| FinRpt-Gen | 多智能体 SFT+RL | arXiv:2511.07322（官方代码以论文页为准） | 11 指标 + 决策有效性 |
| 直接 RAG 生成 | 朴素 | 检索 + 单轮生成 | 同上 |
| Template-Based 生成 | 模板 | arXiv:2504.14233 复现 | 同上 |
| 点即时刻 RAG（无验证器） | 消融 | 本设计去验证器 | 泄漏率 + 幻觉率 |
| FinRpt-Ground（完整） | 本方法 | 本设计 | 全部指标 |

**统一口径**：同一评测集、同一 11 指标实现、同一数值验证器（对基线同样打分）。

## 6. 实验矩阵

- **A 主实验**：FinRpt-Ground vs 基线（文本 + 事实 + 决策三块指标全表）；
- **B 检索消融**：全量 vs 点即时刻（H1）；
- **C 验证器消融**：有无验证器 + 重写轮数（0/1/2）（H2）；
- **D 决策有效性分析**：评级/目标价校准 IC、命中率、与 11 指标相关性（H3）；
- **E RL 奖励消融**：去掉各奖励项（H4）；
- **F 跨市场**：美股 vs A 股；中文 vs 英文报告；
- **G 稳健性**：t_r 切分 ±3 个月、种子、检索窗口大小。

## 7. 评测协议

- **指标**：11 文本指标（对齐 FinRpt 定义）；数值幻觉率 = 不可溯源 claim / 总 claim；可溯源率；评级校准（Spearman 评级 vs 后续收益）、目标价 IC（预测 vs 实现，日截面/事件法）；命中率（目标价方向正确率）；
- **统计**：3 种子 mean±std；paired t-test（α=0.05）；相关性用 Spearman ρ + bootstrap CI；
- **无前视**：t_r 切分 + 点即时刻检索 + 实现窗口后置；全部写进评测附录。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 披露语料 + 点即时刻索引 | CPU | 2 人·周 |
| 数值 claim 抽取 | API（Flash） | 约 5,000 万 token |
| SFT（7B LoRA） | 4×L40 | 7-14 GPU·天 |
| GRPO RL | 4×L40 | 21-42 GPU·天（3-5 GPU·周） |
| 验证器（可选） | 4×L40 | 2-3 GPU·天 |
| 评测与人工标注 | CPU + 人 | 500 份评测 |
| 存储 | 语料 + checkpoint | <1TB |
| 总计 | 4×L40 | 约 30-60 GPU·天 |

**API 成本**：Flash 抽取约 5,000 万 token；Kimi K2.6 长文草稿约 1,000 万；Pro judge 约 2,000 万。估算 $600-1,500（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，11 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 披露语料 + 点即时刻索引 | 索引 + 质量报告 |
| W2 | 数值 claim 抽取 + 人工复核 | 结构化 claim 库 |
| W3 | 评测集构建（500 份人工标注） | 评测集 v1 |
| W4 | 基线复现（FinRpt-Gen / RAG / 模板） | 基线表 |
| W5 | 检索消融 B（H1 pilot） | 泄漏率表 |
| W6 | 多智能体管线 + 验证器 v1 | 管线 v1 + 验证器 |
| W7 | 验证器消融 C | 幻觉率表 |
| W8 | SFT 训练 | SFT 模型 |
| W9 | GRPO RL | RL 模型 |
| W10 | RL 消融 + 跨市场 F | E/F 表 |
| W11 | 论文初稿 + 开源 | 提交稿 + 仓库 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 数值验证器召回/精度权衡（漏检比误报严重） | 高 | 引入人工标注幻觉测试集；阈值偏保守（宁可"无法验证"也不误判一致）；复用 Idea 9 模块 |
| 披露数据点即时刻可得性是工程重点 | 中 | SEC 权威日期 + A 股交易所披露时间；时间戳交叉验证 |
| RL 奖励方差大、reward hacking | 中 | 奖励各项固定权重 + 验证器交叉约束；RL 步数上限与早停 |
| 决策有效性提升不显著（H3/H4 弱） | 中 | 主结论收缩为"事实锚定 + 一致性 + 可溯源"，决策有效性作为辅助指标如实报告 |
| 中文研报合规口径复杂 | 中 | 合规智能体 + 免责模板；定位"辅助撰写"非"自动发布" |

## 11. 论文写作计划

- **目标会议**：ACL 2027 / ACL-FinNLP / EMNLP 2026（以官方截稿为准）。
- **差异化卖点**：(1) 首个"点即时刻 + 数值溯源 + 决策有效性"三位一体的 ERR 评测闭环；(2) 显式量化未来信息泄漏率；(3) 补 FinRpt 11 指标的经济有效性空白。
- **图表清单**：图1 管线总览（检索→多智能体→验证→重写→评测）；图2 数值溯源示例；图3 校准图（评级/目标价 vs 实现）；图4 泄漏率对照；表1 数据集统计；表2 主实验；表3 消融；表4 RL 奖励消融。
- **相关工作覆盖**：FinRpt（2511.07322）、Financial Statement Analysis with LLMs（2407.17866）、The Structure of Financial Equity Research Reports（2407.18327）、Template-Based（2504.14233）、FAITH（2508.05201）、Point-in-Time RAG（2605.31201）、Fighting Numerical Hallucinations（2605.31064）、Deficiency of LLMs（2311.15548）、FinReportBench（2608.04374）、FinDebate（2509.17395）。

## 12. 参考文献

1. Jin, S., Li, S., Zhang, S., Yan, R. *FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation*. arXiv:2511.07322.
2. *Financial Statement Analysis with Large Language Models*. arXiv:2407.17866.
3. *The Structure of Financial Equity Research Reports*. arXiv:2407.18327.
4. *Template-Based Financial Report Generation in Agentic and Decomposed Information Retrieval*. arXiv:2504.14233.
5. *FAITH: Evaluating Financial Table Hallucination*. arXiv:2508.05201.
6. *Point-in-Time Financial RAG with Frozen LLMs and Market-Feedback Adaptive Retrieval*. arXiv:2605.31201.
7. *Fighting Numerical Hallucinations via Data-centric Compilation for Online Financial QA*. arXiv:2605.31064.
8. *Deficiency of LLMs in Finance: An Empirical Examination of Hallucination*. arXiv:2311.15548.
9. *FinReportBench: Institution-Level Financial Report Generation Evaluation*. arXiv:2608.04374.
10. *FinDebate: Multi-Agent Financial Analysis*. arXiv:2509.17395.
11. *The Alpha Illusion*. arXiv:2605.16895.
12. *Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications*. arXiv:2607.28840.
