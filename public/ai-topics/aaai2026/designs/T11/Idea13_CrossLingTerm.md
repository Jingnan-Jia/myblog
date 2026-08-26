# 实验设计书：CrossLingTerm：中英金融术语对齐与跨语言合规检索

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea13_CrossLingTerm.md
> 关联收藏论文：Application Domains II · 论文 24（TermGPT，TermBench 六语言宣称但评测未披露）

## 0. 摘要

CrossLingTerm 构建**中英金融监管术语对齐数据集 RegAlign-ZHEN** 与跨语言术语嵌入，解决"中文条款搜不到英文对照 / 术语翻译不一致"导致的合规与检索失败。现有 TermGPT 声称 TermBench 覆盖六语言但评测细节未披露，Dólares or Dollars 发现金融 LLM 跨语言能力不均衡，金融领域"条款级 + 术语级"对齐检索基准缺失。本设计：(1) 发布首个中英监管条款对齐数据集（条款级 + 术语级，含译法一致性标注）；(2) 跨语言对比微调：用对齐术语对做跨语言难负例对比训练，嵌入空间对齐后跨语言检索无需翻译；(3) 评测：跨语言监管检索（中文查询→英文条款）与术语一致性（同义不同译检出）。embed 模型（Qwen 多语言 / bge-m3）LoRA 微调 1-2 GPU·天即可完成，性价比极高。

## 1. 研究背景与动机

### 1.1 问题定义

给定中文监管条款语料 Z 与英文监管条款语料 E，构造术语对齐对 `(z_term, e_term)` 与条款级对齐对 `(z_clause, e_clause)`。任务：(1) 跨语言检索：中文查询 → 检索对应英文条款（或反向）；(2) 术语一致性：识别同一概念的不同译法（如同义词翻译不一致）；(3) 嵌入空间对齐：训练后跨语言检索无需机器翻译中间步骤。评测：跨语言 Recall@k / nDCG、术语对齐 F1、译法一致性检出率。

### 1.2 相关工作不足

- **TermGPT（AAAI Application Domains II · 论文 24）**：TermBench 宣称六语言但评测口径未披露、无跨语言对齐数据；
- **Dólares or Dollars（arXiv:2402.07405）**：发现金融 LLM 跨语言能力不均衡（数字/术语处理），但未做嵌入对齐与检索；
- **通用多语言嵌入（bge-m3、mE5）**：跨语言对齐有限，尤其金融监管术语（"适当性管理"与 "suitability"、"限售股" 与 "restricted shares" 等）不精确；
- **MT + 检索（翻译后检索）**：翻译误差传播，且无术语一致性保证。

### 1.3 为什么是现在、为什么你的环境适合做

中英监管文件主题可对齐（证监会↔SEC、交易所规则↔交易所规则），DeepSeek 可低成本抽取术语对 + 人工抽查；embed LoRA 微调 1-2 GPU·天极便宜。环境优势：(1) 纯数据工程 + 小模型微调，算力压力最小；(2) 与 Idea 3（FinAlign 术语嵌入）共用方法论与工具链；(3) 评测集人工可控（对齐子集）。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 跨语言对比训练显著提升跨语言检索 | 对齐后跨语言 Recall@10 高于 bge-m3/mE5 ≥10 个点 |
| H2 | 对齐训练后无需翻译即可跨语言检索 | 无 MT 路径的检索 ≥ 或 ≈ 有 MT 路径（翻译后检索） |
| H3 | 术语对齐 F1 与检索收益正相关 | 术语对齐 F1 高的子集，检索增益也高（Spearman ρ>0） |
| H4 | 译法一致性可被自动检出 | 同义不同译样本的检出 F1 >70%（人工标注 ground-truth） |

第一验证实验：在 500 条人工对齐术语对上跑 bge-m3 的跨语言检索 Recall@10 基线（H1 基线），3 天。

## 3. 总体方法设计

### 3.1 数据流水线

- **语料**：证监会/沪深交易所中文规则 + SEC/交易所英文规则；按主题章节对齐（如投资者保护↔Investor Protection）；
- **术语对抽取**（DeepSeek V4 Flash）：给定中英对应条款对，抽取术语对 `(z_term, e_term, 概念ID)`，输出概念聚类 + 译法变体；人工抽查 10%；
- **难负例**：跨语言难负例（中文"限售股"vs 英文"free float"等易混对）由 Flash 生成 + 人工确认；
- **Kimi K2.6**：生成中文合规长文片段作为检索查询池补充；
- **清洗与去泄漏**：条款版本时间戳；检索查询/条款按日期切分（评测期 2025 后）；对齐对去重（译法变体合并到概念 ID）；
- **数量**：条款级对齐 ~8k 对；术语对 ~30k（覆盖 ~5k 概念）；评测查询 2,000（含跨语言 1,000、术语一致性 500、逆查 500）；人工标注 600 对。

### 3.2 方法设计

- **模型**：bge-m3（多语言底座）LoRA；或 Qwen2.5-7B embedding 化（可选）；
- **跨语言对比损失**：对正对 (q, p⁺)（如中文查询→英文条款）与 batch 内难负例 p⁻ 做 InfoNCE：`L = −log[exp(sim(q,p⁺)/τ) / Σ_j exp(sim(q,p_j⁻)/τ)]`，τ=0.05；同时加入译法一致性约束（同一概念不同译法应近距）；
- **术语对齐 F1**：概念聚类 vs 人工概念标注的 F1；
- **译法一致性检出**：对"同一概念多译法"做聚类，检出不一致翻译对（聚类纯度 + 规则）；
- **防前视**：条款版本时间戳 + 查询/条款日期切分；评测查询不进入训练。

### 3.3 训练流程

- bge-m3 LoRA（r=16, α=32, lr=2e-5，batch 128，2 epochs），4×L40 FSDP，1-2 GPU·天；
- 难负例硬挖掘（在线 batch 内 + 离线 top-k）；温度 τ 网格 {0.03,0.05,0.1} 在验证集选。

### 3.4 回测与评测流程

- **检索评测**：跨语言 Recall@10 / nDCG@10（中文→英文、英文→中文双向）；对照 MT 路径（NLLB/GPT 翻译后检索）；
- **术语评测**：对齐 F1、译法一致性检出 F1（人工标注子集）；
- **防泄漏**：日期切分；评测查询由人工/固定模板构造，不来自训练；
- **插拔演示**：把对齐嵌入接入 Idea 3 的检索器与 Idea 10 的条款检索，测下游收益。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | RegAlign-ZHEN v1（自建，中英） |
| 来源 | 证监会/交易所中文规则、SEC/交易所英文规则 |
| 许可 | 监管文本公共领域；对齐标注随论文开源 |
| 划分 | 训练（条款/术语对）80%；评测人工 2,000 查询 + 600 标注对 |
| 规模 | 条款对齐 8k 对；术语对 30k（~5k 概念） |
| 预处理 | 条款块化 + 章节对齐 + 术语抽取 + 概念聚类 |
| 质量门 | 术语对 10% 人工抽查；评测 600 对双标注 κ≥0.85 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| bge-m3 | 通用多语言嵌入 | https://github.com/FlagOpen/FlagEmbedding | Recall@10 / nDCG |
| mE5 | 通用多语言嵌入 | 官方权重 | 同上 |
| MT + 检索 | 翻译路径 | NLLB/GPT 翻译 + 英文检索 | 同上 |
| TermGPT 复现嵌入 | 领域表示 | arXiv:2511.09854（单语）跨语言受限 | 同上 |
| CrossLingTerm（无难负例） | 消融 | 去难负例版 | 同上 |
| CrossLingTerm（完整） | 本方法 | 本设计 | 全部指标 |

**统一口径**：同一评测查询、同一归一化、同一指标实现；报告双向检索。

## 6. 实验矩阵

- **A 主实验**：CrossLingTerm vs 基线（双向 Recall@10/nDCG + 对齐 F1）；
- **B 负例消融**：难负例 vs batch 内负例 vs 无负例挖掘；
- **C 双语反向**：中→英 vs 英→中的非对称性；
- **D 术语专项**：对齐 F1、译法一致性检出（H3/H4）；
- **E 插拔演示**：接入 Idea 3/10 检索器的下游收益；
- **F 稳健性**：温度、seed、语料版本、查询噪声。

## 7. 评测协议

- **指标**：跨语言 Recall@10 / nDCG@10（双向）、术语对齐 F1（概念聚类 vs 人工）、译法一致性检出 F1、MT 对比增益；
- **统计**：3 种子 mean±std；模型间 paired 检验（α=0.05）；Spearman ρ（H3）带 bootstrap CI；
- **无前视**：条款版本时间戳 + 日期切分；评测查询人工/模板构造。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 条款对齐 + 术语抽取 | API（Flash 主力）+ 人工抽查 | 约 4,000 万 token |
| 评测集人工标注 | 人 | 600 对（1-2 周） |
| embed LoRA 微调 | 4×L40 | 4-8 GPU·天（1-2 天 × 2-3 版本） |
| 评测 | 4×L40 | 1-2 GPU·天 |
| 存储 | 语料 + 对齐 + 嵌入 | <200GB |
| 总计 | 4×L40 | 约 5-10 GPU·天 |

**API 成本**：Flash 术语对抽取约 3,000 万 token；Kimi 中文长文约 800 万；Pro 一致性 judge 约 500 万。估算 $200-500（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，7 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 中英条款收集 + 章节对齐 | 语料 v1 |
| W2 | 术语对抽取 + 人工抽查 | 术语对 v1（H1 基线） |
| W3 | 难负例生成 + 概念聚类 | 训练数据 v1 |
| W4 | 评测集构建（2,000 查询 + 600 标注） | RegAlign-ZHEN v1 |
| W5 | LoRA 微调 + 主实验 A | 全表 |
| W6 | 负例/双向/术语消融 B/C/D | 消融表 |
| W7 | 插拔演示 + 稳健性 + 论文初稿 | 提交稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 中英监管文件非逐条对应，对齐标注难 | 高 | 按主题/章节对齐而非逐条；人工审核 600 对子集作为 ground-truth |
| 术语对抽取噪声 | 中 | 10% 人工抽查 + 概念聚类合并；低置信术语对丢弃 |
| 跨语言检索增益不显著（H1 弱） | 中 | 增加难负例强度与术语约束损失；若仍弱则主结论收缩为"术语对齐数据 + 一致性评测" |
| 与 bge-m3 区分度不足 | 中 | 强调金融监管术语专项 + 译法一致性 + 对齐数据发布 |
| MT 路径作为上界 | 低 | MT 对比作为基线而非目标；报告无 MT 路径优势即可 |

## 11. 论文写作计划

- **目标会议**：ACL 2027 / EMNLP 2026（以官方截稿为准）。
- **差异化卖点**：(1) 首个中英金融监管术语对齐数据集（条款级 + 术语级 + 译法一致性）；(2) 跨语言难负例对比训练；(3) 无 MT 的跨语言检索验证。
- **图表清单**：图1 对齐数据构建流程；图2 跨语言对比训练；图3 双向检索对比；图4 术语一致性示例；表1 数据集统计；表2 主实验；表3 消融；表4 插拔效果。
- **相关工作覆盖**：TermGPT（2511.09854）、Dólares or Dollars（2402.07405）、bge-m3（FlagEmbedding）、mE5、CardioEmbed（2511.10930）、PatenTEB（2510.22264）、FINDAP（2501.04961）、TermBench（含于 2511.09854）。

## 12. 参考文献

1. Sun, Y., et al. *TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation in Legal and Financial Domains*. arXiv:2511.09854.
2. *Dólares or Dollars? Cross-Lingual Capability Disparity in Financial LLMs*. arXiv:2402.07405.
3. *CardioEmbed: Domain-Adapted Embeddings and Benchmark for Cardiology*. arXiv:2511.10930.
4. *PatenTEB: Patent Text Embedding Benchmark*. arXiv:2510.22264.
5. *Demystifying Domain-adaptive Post-training for Financial LLMs (FINDAP)*. arXiv:2501.04961.
6. *Integrating Contrastive Learning into a Multitask Transformer Model for Domain Adaptation*. arXiv:2310.04703.
7. *Domain Adaptation for Japanese Sentence Embeddings with Contrastive Learning based on Synthetic Sentence Generation*. arXiv:2503.09094.
8. *FiNER-ORD: Financial Named Entity Recognition Dataset*. arXiv:2302.11157.
9. *FinEval: A Chinese Financial LLM Evaluation Benchmark*. arXiv:2308.09975.
10. *FinBen: A Holistic Financial Benchmark*. arXiv:2402.12659.
