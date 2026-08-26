# 实验设计书：NewsCast：防前视的"新闻事件 × K 线"多模态金融预测模型

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea05_NewsCast.md
> 关联收藏论文：Machine Learning VII · 论文 47（Kronos）；依赖 Idea04 的 Kronos-CN 权重

## 0. 摘要

NewsCast 把 LLM 抽取的**点即时刻（Point-in-Time）新闻事件表示**与 K 线 token 序列在共享时间轴上对齐后联合预测收益与波动率，核心是显式的"新闻发布时间墙"：交易信号只能使用 t 时刻之前已发布的新闻。现有工作如 Kronos 无文本模态；M2VN 用收费的 Time Machine GPT 做点即时刻 LLM 且未开源；SER 做事件级截面预测但无多模态时序融合；多数新闻-价格融合模型不设时间墙（实证见 2309.17322）。NewsCast 的差异化：(1) 显式时间墙设计 + "泄漏组对照实验"量化前视收益；(2) 结构化事件表示（谁-何时-何地-何事件）与 K 线 token 级表示交叉注意力融合；(3) 用开源/API LLM 做事件抽取，规避对专有点即时刻模型依赖，全程可复现。融合模型 1-3B 规模，K 线 encoder 复用 Kronos-CN 权重。

## 1. 研究背景与动机

### 1.1 问题定义

给定股票 s 在时刻 t 的可观测状态：K 线段 `X_s,t = {K_{t-L}, …, K_t}`（OHLCV）与新闻流 `N_s,t = {n_j : publish_time(n_j) ≤ t}`。目标：学习联合编码器 f 预测未来 h∈{1,5} 期的收益 r 与已实现波动率 σ。关键约束：**事件表示必须只用发布时间 ≤ t 的新闻**（时间墙），K 线特征只用 ≤t 收盘。评测含对照组（故意把 t 之后发布的新闻注入特征），量化前视收益。

### 1.2 相关工作不足

- **Kronos（AAAI Machine Learning VII · 论文 47）**：纯价格自回归，无文本模态，上限受弱有效市场假设约束；
- **M2VN（arXiv:2510.20699）**：融合新闻与波动率，但依赖收费 Time Machine GPT 做点即时刻 LLM，不公开、不可复现；
- **SER（arXiv:2512.19484）**：用 LLM 抽取结构化事件做截面收益预测，可解释但未做多模态时序融合、未设显式时间墙评测；
- **ChatTime（arXiv:2412.11376）**：数值-文本多模态 TSFM，但面向通用时序非金融事件、无制度约束；
- **Assessing Look-Ahead Bias（arXiv:2309.17322）**：实证 GPT 新闻情感策略的前视与 distraction 偏差，但只处理单一情感信号。

### 1.3 为什么是现在、为什么你的环境适合做

新闻数据公开可得（SEC 8-K 时间戳、交易所公告、Reuters 免费源 + A 股公告），Kronos-CN（Idea 4）提供可复用的 K 线权重；DeepSeek V4 Flash 可低成本大规模做点即时刻事件抽取；社区对"防前视多模态融合"有明确缺口。环境优势：(1) 事件抽取走 API 而非 GPU 训练；(2) 融合模型 1-3B 在 4×L40 上 2-4 GPU·天可训；(3) 时间墙协议可与 Idea 1 的审计基准呼应。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 事件级结构化表示优于朴素文本嵌入 | 结构化事件表示组 RankIC 高于句子嵌入组 ≥5%（相对） |
| H2 | 多模态融合优于纯 K 线/纯新闻 | NewsCast RankIC > Kronos-CN（纯 K 线）且 > 事件因子模型 |
| H3 | 时间墙有效且前视收益可量化 | 泄漏对照组的 RankIC 显著高于不泄漏组（量化污染）；不泄漏组仍保有显著增量 |
| H4 | 新闻时效窗口敏感 | 只取发布后 24h 内的新闻做特征，RankIC 高于全窗口 ≥ 3%（相对） |

第一验证实验：单市场（美股 SP500 子集）× 1 个月数据跑事件抽取 + 简单事件因子模型，验证事件抽取质量与基础信号（H1 pilot）。

## 3. 总体方法设计

### 3.1 数据流水线

- **新闻源**：SEC EDGAR 8-K/公告（权威时间戳，作为锚点）；Reuters 免费/API 快照；A 股公告（上交所/深交所 + akshare 聚合）；
- **事件抽取**（DeepSeek V4 Flash，API 核心）：
  - Prompt：给定新闻文本与发布时间，输出 `{event_type, entity, action, value, effect_direction, confidence}` JSON；强调"只使用文本内信息，禁止补充未来知识"；
  - 抽检：人工复核 200 条，与 Pro judge（LLM-as-judge）交叉验证抽取质量；
- **清洗与去泄漏**：每条新闻带 `publish_time_utc`；重发布/摘要去重（标题 MinHash）；时区统一 UTC；**训练期与评测期严格按新闻时间与行情时间对齐**；
- **数量**：美股 SP500 + A 股沪深300 相关新闻，2020-2026，约 50-100 万条新闻 → 事件化后 30-60 万事件；K 线对齐使用 Idea 4 的语料；
- **Kimi K2.6**：生成"新闻-事件难样例"与跨源冲突样本（同事件不同源时间戳），用于评测稳健性。

### 3.2 方法设计

- **K 线 encoder**：复用 Kronos-CN tokenizer + transformer encoder（冻结或低学习率适配），输出 token 级隐状态 H_K；
- **事件 encoder**：LLM 抽取的事件结构 → 事件嵌入（用固定版本的开源 LLM encoder，如 Qwen2.5-7B 的 layer 输出，**冻结**以保证点即时刻性——不随训练更新，避免未来知识注入）；或学习式事件类型 embedding + 数值字段 MLP；
- **对齐与融合**：事件按发布时间分配到 K 线时间桶 → 事件序列与 K 线序列做**交叉注意力**（事件 query 读取 K 线 key/value，或双向 cross-attention）；输出融合表示；
- **预测头**：收益回归（MSE）+ 方向分类（CE）+ 波动率回归（MSE），`L = w1·MSE_r + w2·CE_dir + w3·MSE_σ`（w=(1,0.5,0.5) 起步）；
- **时间墙实现**：特征构造阶段，事件特征矩阵仅含 `publish_time ≤ t` 的新闻；泄漏对照组在评测时额外注入 t+1 至 t+k 的新闻；
- **防前视**：事件 encoder 冻结在"训练日期之前发布"的版本；训练/评测按时间切分不重叠。

### 3.3 训练流程

- 1-3B 融合模型（encoder + cross-attention + head），AdamW lr=1e-4，warmup 500 步，batch 128 样本（每样本 64 K 线 + 8 事件），fp16，4×L40 FSDP；
- 分两段：先冻结 K 线 encoder 只训融合层 1 epoch，再全量（K 线 encoder 低 lr=1e-5）微调 2 epochs；
- 评测期 2025-07 至 2026-06，与训练零重叠；3 种子。

### 3.4 回测与评测流程

- **评测**：RankIC（价格）、vol MAE；分层组合回测（Top/Bottom decile，成本后）仅作辅助；
- **防前视协议**：事件时间戳交叉验证（SEC 锚点核对）；时间墙对照实验为必需项（H3）；回测含 T+1、成本、涨跌停约束（A 股）；
- 报告：不泄漏组 vs 泄漏组的 RankIC 差（= 前视收益上界）；所有指标 mean±std。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | NewsCast-Align v1（新闻-K线对齐，自建） |
| 来源 | SEC EDGAR、Reuters、交易所公告、akshare 聚合 |
| 许可 | 新闻源各自许可（公开摘要/API 快照）；项目发布事件化数据与对齐代码 |
| 划分 | 时间切分：训练 2018-2024、验证 2025H1、评测 2025-07 至 2026-06 |
| 规模 | 50-100 万条新闻 → 30-60 万事件；与 8,700 万根 K 线对齐 |
| 预处理 | UTC 统一、去重、事件结构化、发布时点对齐 K 线桶 |
| 质量门 | 200 条人工复核；SEC 时间戳锚点一致率 >99% |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| Kronos-CN（纯 K 线） | 时序 | Idea 4 权重 | RankIC / vol MAE |
| Kronos 原版 | 时序 | https://github.com/shiyu-coder/Kronos | 同上 |
| 事件因子模型 | 事件 | SER 风格（arXiv:2512.19484）截面因子 | 同上 |
| 朴素文本嵌入 + 回归 | 文本 | sentence embedding + 线性预测 | 同上 |
| M2VN 复现 | 多模态 | arXiv:2510.20699（受限复现，若不可得报其论文数字并注明） | 同上 |
| NewsCast（无时间墙对照） | 消融 | 本设计泄漏组 | 量化前视收益 |

**统一口径**：同一评测期、股票池、计分器；K 线输入统一 tokenizer（Kronos-CN）。

## 6. 实验矩阵

- **A 主实验**：NewsCast vs 全基线（RankIC / vol MAE / 分层组合）；
- **B 表示消融**：结构化事件 vs 朴素文本嵌入 vs 事件类型 one-hot（H1）；
- **C 对齐消融**：交叉注意力 vs 拼接 vs 事件-均值池化；新闻时效窗口（24h/7d/全量）（H4）；
- **D 时间墙对照**：不泄漏 vs 泄漏（H3，必需）；
- **E 跨市场**：美股 vs A 股的迁移与专项；
- **F 稳健性**：新闻密度阈值、抽取质量（含 10% 噪声注入）、种子、评测期滚动。

## 7. 评测协议

- **指标**：RankIC / ICIR（日截面 Spearman，日序列 t 检验）；vol MAE；分层组合年化收益与 Sharpe（成本后）；
- **时间墙指标**：`ΔIC = IC_leak − IC_clean`（前视收益上界），需报告并解释；
- **统计**：3 种子 mean±std；模型间 paired t-test（α=0.05）；
- **无前视**：见 3.2/3.4；事件 encoder 冻结版本说明写入方法节。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 新闻采集 + 事件抽取 | API（Flash 主力） | 约 3-6 亿 token |
| 融合模型训练 | 4×L40 FSDP | 8-16 GPU·天 |
| 评测与回测 | 4×L40 + CPU | 3-5 GPU·天 |
| 存储 | 新闻 + 事件 + 对齐特征 | <1TB |
| 总计 | 4×L40 | 约 12-22 GPU·天 |

**API 成本**：Flash 事件抽取 3-6 亿 token（这是本 idea 最大成本项）；Pro judge 约 2,000 万；Kimi K2.6 约 800 万。估算 $1,500-4,000（按当期定价，Flash 单价低，重点优化：批量提示、按实体去重、失败重试限流）。

## 9. 里程碑与时间线（单人 + 4×L40，10 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 新闻采集 + 时间戳锚定 | 新闻库 v1 + 质量报告 |
| W2 | 事件抽取 pipeline + 200 条复核 | 事件库 v0 |
| W3 | 事件因子 pilot（H1 信号检验） | pilot 报告 |
| W4 | 全量事件化 + K 线对齐 | NewsCast-Align v1 |
| W5 | 融合模型 v1 训练 | 初版模型 + 评测初表 |
| W6 | 表示/对齐消融 B/C | 消融表 |
| W7 | 时间墙对照 D（必需） | ΔIC 表 + 图 |
| W8 | 跨市场 E | E 表 |
| W9 | 稳健性 F + 回测审计 | F 表 + 回测报告 |
| W10 | 论文初稿 + 开源 | 提交稿 + 仓库 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 新闻时间戳噪声、历史覆盖不全 | 高 | 1-2 个权威源（SEC 8-K + Reuters）+ 人工抽检 200 条；时间戳用 SEC 锚点交叉验证 |
| 事件抽取质量不稳定 | 中 | 结构化 prompt + 字段置信度 + 抽检；抽取失败事件丢弃而非猜测 |
| API 成本超支（3-6 亿 token） | 中 | 按股票相关性过滤新闻；事件去重；Flash 批量长上下文合并 |
| 多模态融合无增益（H2 不成立） | 中 | 若 RankIC 无增量，转向"事件抽取质量评测 + 前视审计方法论"弱版本 |
| K 线 encoder 适配退化 | 低 | 冻结-再解冻两段式；监控纯 K 线基线不下降 |

## 11. 论文写作计划

- **目标会议**：ACL 2027 / EMNLP 2026 / IJCAI 2027（以官方截稿为准）。
- **差异化卖点**：(1) 显式时间墙 + 泄漏对照实验（前视收益可量化）；(2) 结构化事件与 K 线交叉注意力融合；(3) 全程可复现（开源 LLM 事件抽取，无专有点即时刻模型依赖）。
- **图表清单**：图1 时间墙与融合架构；图2 事件抽取 pipeline；图3 前视对照 ΔIC 图；图4 跨市场迁移；表1 数据集统计；表2 主实验；表3 消融；表4 回测审计。
- **相关工作覆盖**：Kronos（2508.02739）、M2VN（2510.20699）、SER（2512.19484）、ChatTime（2412.11376）、Assessing Look-Ahead Bias（2309.17322）、Instruct-FinGPT 情感（2306.12659）、Reasoning or Overthinking（2506.04574）。

## 12. 参考文献

1. Shi, Y., et al. *Kronos: A Foundation Model for the Language of Financial Markets*. arXiv:2508.02739.
2. *M2VN: Multimodal ... News and Volatility*. arXiv:2510.20699.
3. *Structured Event Representation and Stock Return Predictability (SER)*. arXiv:2512.19484.
4. *ChatTime: A Multimodal Time Series Foundation Model with Text-Numerical Fusion*. arXiv:2412.11376.
5. *Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis*. arXiv:2309.17322.
6. *Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning*. arXiv:2306.12659.
7. *Reasoning or Overthinking: Evaluating LLMs on Financial Sentiment Analysis*. arXiv:2506.04574.
8. *Enhancing Financial Sentiment Analysis via Retrieval Augmented LLMs*. arXiv:2310.04027.
9. *Doc2EDAG: End-to-End Chinese Financial Event Extraction*. arXiv:1904.07535.
10. *CausalStock: News-Driven Causal Discovery for Stock Movement*. arXiv:2411.06391.
11. *Ploutos: Interpretable Stock Movement Prediction with Financial LLM*. arXiv:2403.00782.
