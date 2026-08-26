# 实验设计书：FinFactCheck：金融表格/数字幻觉检测与可溯源验证器

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea09_FinFactCheck.md
> 关联收藏论文：Application Domains I · 论文 57（FinRpt，验证器下游）；与 Idea 7 协同、独立可复用

## 0. 摘要

FinFactCheck 构建一个通用的金融"数值-来源"验证器模型与基准，可插拔到报告生成、QA、RAG 任何 LLM 管线，把幻觉检测从文本层下沉到**数字层**。现有工作 FAITH 止于检测分类、Fighting Numerical Hallucinations 偏工程编译技巧、通用 NLI 验证器不处理表格异构。本设计把验证建模为**数字级 claim-证据对齐**：输出"数字→表格单元格→披露来源"的三级溯源，并训练专用验证模型（数字抽取 → 对齐 → 矛盾判定）。发布 FinFactBench：SEC XBRL + A 股披露上合成的含错报告（DeepSeek 注入 10 类数字错误）+ 人工标注。验证器可作为 RL 奖励函数（与 Idea 7 协同），并输出可插拔 API。

## 1. 研究背景与动机

### 1.1 问题定义

输入：LLM 输出文本 O（含若干数值 claim c = (value, unit, period, context)）与点即时刻证据语料 E（结构化披露）。输出：对每个 claim 的判定 v∈{一致, 矛盾, 无法验证}，以及可溯源路径（claim → 表格单元格 → 披露来源）。目标指标：数字级精确率/召回率/F1、溯源准确率、幻觉检出率。

### 1.2 相关工作不足

- **FAITH（arXiv:2508.05201）**：表格幻觉评估，止于检测分类，未给"定位 + 溯源 + 修复"闭环；
- **Fighting Numerical Hallucinations（arXiv:2605.31064）**：数据为中心的编译技巧，非可插拔验证器；
- **Deficiency of LLMs in Finance（arXiv:2311.15548）**：指出现有 LLM 金融数字系统性失败，但未给验证方案；
- **InfoLoss 信息论去幻觉（arXiv:2512.03107）**：报告 92% 降幻觉，但面向通用文本非金融表格；
- **FAITH 之后缺"检测 + 定位 + 溯源 + 修复"端到端**——本设计补位。

### 1.3 为什么是现在、为什么你的环境适合做

SEC XBRL 是结构化、可机器对齐的黄金证据源；A 股披露也逐步结构化；验证器可复用 Idea 7 的 RL 奖励与评测闭环。环境优势：(1) 验证器 7B LoRA 1-2 GPU·天即可训练；(2) 数据合成靠 API 并行、成本可控；(3) 与 2605.31064、2508.05201 直接对比，评审关注点明确。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 数字级对齐优于文段级 NLI | 在 FinFactBench 上数字级 F1 高于文段级 NLI ≥10 个点 |
| H2 | 三级溯源（数字→单元格→来源）显著提升定位能力 | 溯源准确率 >85%；"矛盾"判定给出正确证据单元格比例高 |
| H3 | 验证器对未训练的错误类型泛化 | 对"未见错误类"（held-out 类）检出率 ≥ 训练类检出率 80% |
| H4 | 验证器可作 RL 奖励（与 Idea 7 协同） | 用本验证器做奖励的 FinRpt-Ground RL，其报告幻觉率显著低于无奖励版 |

第一验证实验：在 XBRL 证据上对 500 条含错 claim 跑验证器 v0（规则对齐版），测数字级 F1（H1 pilot），2-3 天。

## 3. 总体方法设计

### 3.1 数据流水线

- **证据语料**：SEC XBRL（10-K/10-Q 财务表格，结构化单元格）+ A 股披露（财报 PDF 转结构表格，或用公告结构化字段）；每证据带披露日期（点即时刻）；
- **含错报告合成**（DeepSeek V4 Flash，核心）：从正确报告注入 10 类数字错误——抄错（数字替换）、单位错（百万/十亿）、期间错（Q1→Q2）、口径错（营收→毛利）、四舍五入错、正负号错、来源张冠李戴（A 公司数字挂到 B）、百分比/绝对值混淆、加总不一致、币种错（人民币/美元）；
  - Prompt 思路：给正确 claim 与证据，要求"用指定错误类型改写 claim，保持文本通顺"，输出 {claim, true_value, error_type, evidence_cell}；
- **人工标注**：评测集 2,000 claim（含 1,000 含错 + 1,000 正确）双标注 κ≥0.85；500 条含三级溯源 ground-truth；
- **清洗与去泄漏**：训练合成与评测人工集不相交；证据按披露日期切分（评测期 2025 后证据不进训练）；错误注入的 LLM 版本与评测 LLM 版本解耦；
- **数量**：训练合成 ~50,000 claim（每类 5,000）；评测人工 2,000。

### 3.2 方法设计

- **验证器管线**：检索（claim 数字 + 上下文 → 候选证据，bge-m3/领域嵌入）→ 抽取（claim 的数字/单位/期间，规则 + 小模型）→ 对齐（数值相对误差、单位/期间归一化匹配，与候选单元格配对）→ 判定（规则阈值 + 判别模型融合）；
- **判别模型**：7B LoRA（输入 = claim + 候选证据 + 对齐特征，输出 {一致, 矛盾, 无法验证}）；训练损失 CE；可输出置信度；
- **三级溯源**：claim → 命中的单元格 ID → 所属表格/披露文件 + 披露日期；溯源准确率单独评测；
- **修复（可选）**：判定"矛盾"时给出正确值候选（对齐单元格值）；
- **防前视**：证据检索限制披露日 ≤ 输出日；评测与训练按日期切分。

### 3.3 训练流程

- 7B（Qwen2.5-7B）LoRA：r=16, α=32, lr=2e-5，batch 16（grad-acc 8），2 epochs，4×L40 FSDP，1-2 GPU·天；
- 检索模型：bge-m3 冻结 + 轻量 rerank 可选；对齐规则用阈值网格（±1%、单位归一化字典）离线调参（验证集）。

### 3.4 回测与评测流程

- **评测集**：FinFactBench 人工 2,000 claim + 真实 RAG 输出 1,000 claim（Idea 7 管线产出，含自然发生的幻觉）；
- **协议**：验证器输出逐 claim 判定；错误类型混淆矩阵；溯源链正确率（claim→单元格→来源，需人工核对 500 条）；
- **防泄漏**：评测 claim 及其证据来源日期晚于训练；验证器不接触评测人工标注；
- **插拔验证**：在 QA（FinBen 财务问答子集）与报告生成（Idea 7）两条管线上挂接验证器，测幻觉率变化。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | FinFactBench v1（自建，含合成 + 人工标注） |
| 来源 | SEC XBRL、A 股披露、DeepSeek 注入合成 |
| 许可 | XBRL 公开数据；合成与标注随论文开源（CC-BY-4.0 意向） |
| 划分 | 训练合成 50,000；验证 2,000；评测人工 2,000（含三级溯源 500） |
| 预处理 | claim 结构化、单位归一化、期间标准化、证据索引 |
| 质量门 | 评测双标注 κ≥0.85；错误注入后与原 claim 配对校验 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| 文段级 NLI 验证器 | 通用 | DeBERTa/Qwen NLI 微调 | F1（数字级） |
| GPT-4o / DeepSeek 零样本验证 | 通用 | 直接提示判 claim | F1 + 溯源 |
| FAITH 分类器 | 表格 | arXiv:2508.05201 复现 | 检出率 |
| InfoLoss 信息论方法 | 通用 | arXiv:2512.03107 | 降幻觉率 |
| FinFactCheck v0（纯规则对齐） | 消融 | 本设计无判别模型 | F1 |
| FinFactCheck（完整） | 本方法 | 本设计 | 全部指标 |

**统一口径**：同一评测集、同一 claim 划分、同一证据库；报告 F1 / 溯源准确率 / 混淆矩阵。

## 6. 实验矩阵

- **A 主实验**：FinFactCheck vs 全基线（数字级 P/R/F1 + 溯源）；
- **B 错误类型消融**：10 类错误逐类检出率；
- **C 泛化**：held-out 错误类（H3）；
- **D 证据噪声消融**：检索窗口大小、证据含噪声注入；
- **E 检索器消融**：bge-m3 vs 关键词 vs 领域嵌入（与 Idea 3 联动）；
- **F 插拔验证**：QA 与报告管线挂接后幻觉率变化（H4 与 Idea 7 协同）；
- **G 稳健性**：阈值灵敏度、种子、证据格式（XBRL vs PDF 文本）。

## 7. 评测协议

- **指标**：数字级精确率/召回率/F1（一致性判定混淆矩阵）；溯源准确率（三级链完整正确）；幻觉检出率（矛盾类召回）；插拔后管线的幻觉率降幅；
- **统计**：3 种子 mean±std；模型间 McNemar/paired 检验；
- **无前视**：证据披露日 ≤ 输出日；日期切分；评测标注人工来源。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 证据语料 + 索引 | CPU | 2 人·周 |
| 含错报告合成 | API（Flash 主力） | 约 6,000 万 token |
| 验证器训练 | 4×L40 | 4-8 GPU·天（1-2 天 × 2-3 版本） |
| 评测 + 插拔 | 4×L40 + API | 2-3 GPU·天 |
| 存储 | 证据 + 合成 + 标注 | <300GB |
| 总计 | 4×L40 | 约 6-12 GPU·天 |

**API 成本**：Flash 合成约 6,000 万 token；Pro 零样本基线 + judge 约 1,500 万；Kimi 自然化错误注入约 500 万。估算 $300-800（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，8 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | XBRL/A 股证据语料 + 索引 | 证据库 v1 |
| W2 | 错误注入合成（10 类）+ 配对校验 | 训练合成集 v1 |
| W3 | 人工标注评测集（2,000） | FinFactBench v1 |
| W4 | 规则对齐 v0 + H1 pilot | 规则验证器 + pilot 表 |
| W5 | 判别模型训练 | FinFactCheck 模型 |
| W6 | 主实验 A/B | 全表 |
| W7 | 泛化/噪声/检索消融 C/D/E | 消融表 |
| W8 | 插拔验证 F + 论文初稿 | 提交稿 + 开源 API |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 表格异构（XBRL/PDF/图片）下数字对齐召回低 | 高 | 先保证结构化 XBRL 基础集；PDF 用版面解析 + 低置信即"无法验证"；图片表作为扩展 |
| 漏检比误报更严重 | 中 | 判定阈值偏保守；"无法验证"优先级高于"一致" |
| 合成错误与真实错误分布差异 | 中 | 增加真实 RAG 输出评测子集；人工标注错误占比不低于 50% |
| 判别模型依赖对齐特征质量 | 中 | 对齐规则先离线调参；失败样本回注检索与抽取模块迭代 |
| 跨管线插拔收益不稳定 | 低 | 插拔验证作为协同实验，主结论以 FinFactBench 为准 |

## 11. 论文写作计划

- **目标会议**：ACL 2027 / EMNLP 2026（以官方截稿为准）。
- **差异化卖点**：(1) 数字级 claim-证据对齐 + 三级溯源（首个金融表格溯源验证器）；(2) 10 类错误合成 + 人工标注基准；(3) 可插拔 API 与 RL 奖励复用。
- **图表清单**：图1 验证器管线；图2 三级溯源示例；图3 错误类型混淆矩阵；图4 插拔效果；表1 数据集统计；表2 主实验；表3 消融；表4 泛化。
- **相关工作覆盖**：FAITH（2508.05201）、Fighting Numerical Hallucinations（2605.31064）、Deficiency of LLMs（2311.15548）、InfoLoss（2512.03107）、FinReportBench（2608.04374）、FinRpt（2511.07322）、FAITH 系表格幻觉（2508.05201）、BigFinanceBench（2606.03829）。

## 12. 参考文献

1. *FAITH: Financial Table Hallucination Evaluation Framework*. arXiv:2508.05201.
2. *Fighting Numerical Hallucinations via Data-centric Compilation for Online Financial QA*. arXiv:2605.31064.
3. *Deficiency of LLMs in Finance: An Empirical Examination of Hallucination*. arXiv:2311.15548.
4. *InfoLoss: Information-Theoretic Methods to Reduce Hallucination*. arXiv:2512.03107.
5. Jin, S., et al. *FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation*. arXiv:2511.07322.
6. *FinReportBench: Institution-Level Financial Report Generation Evaluation*. arXiv:2608.04374.
7. *BigFinanceBench: A Comprehensive Financial QA Benchmark*. arXiv:2606.03829.
8. *Point-in-Time Financial RAG with Frozen LLMs and Market-Feedback Adaptive Retrieval*. arXiv:2605.31201.
9. *FinEval: A Chinese Financial LLM Evaluation Benchmark*. arXiv:2308.09975.
10. *FinBen: A Holistic Financial Benchmark*. arXiv:2402.12659.
