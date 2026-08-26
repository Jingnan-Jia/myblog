# 实验设计书：FinAlign：可证明术语表示收益的金融 RAG/合规对比微调框架

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea03_FinAlign.md
> 关联收藏论文：Application Domains II · 论文 24（TermGPT）

## 0. 摘要

FinAlign 以 TermGPT 的 isotropy 修复思想为基础，把"术语表示规整"改造成"可端到端验证的金融任务收益"。方法分三阶段：MLM 域继续预训练 → 术语级+句子级多层对比学习（TermGPT 式）→ 以 RAG 检索与合规分类为下游任务的联合精调。针对 TermGPT 依赖图拓扑负例、长尾监管条款退化的缺陷，FinAlign 用 DeepSeek 生成"仅一词之差/语义极易混淆"的监管条款难负例替代图拓扑。同时借鉴 CardioEmbed/PatenTEB 的"领域嵌入 + 下游任务联合基准"范式，发布 FinAlign-Bench（中英监管条款检索 + 合规分类）。核心贡献是给出"嵌入指标 → 任务指标"的可证明性链条：用 isotropy 分数与下游任务指标的 Spearman 相关，以及对照实验，论证表示收益确实转化为检索/合规收益。

## 1. 研究背景与动机

### 1.1 问题定义

给定金融领域语料（监管条款、公告、研报），目标是把 LLM/embedding 适配为"术语判别准确 + 检索召回高 + 合规分类可靠"的表示，并**量化**表示收益到任务收益的传导。三阶段：
1. 领域继续预训练（MLM）；
2. 多层对比学习（术语级 + 句子级），负例来源为"难负例挖掘（LLM 生成混淆对）+ 可选句子图拓扑"；
3. 下游联合精调（RAG 检索 + 合规分类），使表示收益在任务上可见。

### 1.2 相关工作不足

- **TermGPT（AAAI Application Domains II · 论文 24 · TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation in Legal and Financial Domains）**：只报告嵌入/判别指标，未证明 RAG 检索与合规分类的端到端收益；正负样本依赖图拓扑，稀疏长尾条款退化；未报 catastrophic forgetting。
- **CardioEmbed（arXiv:2511.10930）/ PatenTEB（arXiv:2510.22264）**：医学/专利领域已有"嵌入 + 下游"联合基准范式，金融缺失。
- **Isotropy 研究**（2109.04740、2106.01183、2411.17538）：给出 isotropy 度量的几何论证与修复手段，但均未落到金融合规任务。
- **FINDAP（arXiv:2501.04961）**：研究金融域持续预训练+指令微调配方，但未聚焦术语表示。

### 1.3 为什么是现在、为什么你的环境适合做

金融合规与 RAG 检索对"术语精确匹配"的需求强烈（条款引错一个词即失效）；同时 TermGPT 刚把"isotropy 修复"重新激活为热点但缺任务闭环。环境优势：(1) 监管条款公开可得（SEC 规则、沪深交易所规则、银保监文件），标注可控；(2) QLoRA 微调 7B（bge/Qwen）在 4×L40 上可负担（2-3 GPU·天/模型）；(3) DeepSeek/Kimi API 可大规模合成难负例与中文合规问答对，人工仅抽检。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | LLM 生成的混淆难负例优于图拓扑负例 | 对比实验中，难负例组的 Recall@10/F1 高于图拓扑组 ≥3 个点 |
| H2 | 三层对比训练带来任务收益（非仅嵌入指标） | FinAlign 在检索 Recall@10 与合规 F1 上显著优于原始模型与 TermGPT 复现（paired 检验 p<0.05） |
| H3 | 表示收益与任务收益可传导（可证明性） | isotropy 分数与下游任务指标的 Spearman ρ 在消融族内为正且显著 |
| H4 | 适配不破坏通用能力 | FinAlign 在通用嵌入基准上的退化 <5% 相对点 |

第一验证实验：在 FinAlign-Bench 的检索子集上跑 H1（难负例 vs 图拓扑），8 GPU·小时内出结果。

## 3. 总体方法设计

### 3.1 数据流水线

- **语料**：SEC 规则（Regulation S-K、Reg S-T 等）、沪深交易所规则与指引、银保监/证监会办法（中英双版，已公开）；
- **条款/术语标注**：条款编号、生效/修订日期、术语名词短语（规则级人工 seed + LLM 辅助扩展，人工抽检 10%）；
- **难负例生成**（DeepSeek V4 Flash，核心）：
  - Prompt 思路：给定目标条款 T 与金融术语 X，要求生成"语义相近、仅一词之差或高度易混淆"的负例条款 T'（如同主题不同适用对象、不同数值阈值、相反限定词），并给出混淆维度标签；
  - 数量：每条款 5-10 个难负例；总合成负例 ≥30k；
- **中文合规问答对**（Kimi K2.6）：按"问题-条款-答案"模板生成，覆盖条款援引、免责与风险提示；
- **清洗与去泄漏**：所有条款带发布/修订日期；问答对构造日期记录在案，评测期严格晚于构造期；负例不来自未来版本条款；去重（MinHash 近似去重）；
- **数量**：条款语料 ~60k 条款块；正样本对 ~40k；难负例 ~30k；合规 QA ~10k；评测集人工标注 1,000（双标注 κ≥0.85）。

### 3.2 方法设计

- **阶段一**：MLM 继续预训练（仅 decoder 层/嵌入层 LoRA），mask 比率 0.15，条款块 256-512 token；
- **阶段二**：多层对比学习。句子级 InfoNCE：
  `L_sent = −log[ exp(sim(q, q⁺)/τ) / Σ_j exp(sim(q, q_j⁻)/τ) ]`，τ=0.05；术语级按"术语 token 池化"做同样 InfoNCE，两损失加权 `L = α·L_sent + β·L_term`（α=β=0.5 起步）；负例由难负例挖掘器（LLM 混淆对）与可选句子图拓扑共同提供，采样时难负例占比 60%；
- **阶段三**：联合精调。检索损失 = batch 内难负例 InfoNCE；分类损失 = 合规二分类交叉熵（条款→合规/违规 + 匹配正确条款）；联合 `L = L_retr + γ·L_cls`（γ=0.5）；
- **超参数初值**：QLoRA（r=32, α=64, dropout=0.05）；AdamW lr=2e-5（阶段二/三）；cosine；batch 32（grad-acc 4）；epochs：阶段一 1、阶段二 2、阶段三 1；
- **前视/去泄漏**：条款版本号强制注入样本（`[条款 YYYYMMDD 生效]`），检索时用评测日当日的条款快照。

### 3.3 训练流程

- 7B 模型（Qwen2.5-7B / bge-m3 底座）QLoRA，4×L40 FSDP 分片，每阶段 2-3 GPU·天/模型；阶段间不重置 LoRA（连续训练）但保留各阶段 checkpoint 以做消融；
- 评估按模型 × 阶段 checkpoint 全量记录。

### 3.4 回测与评测流程

- **检索评测**：查询 = 场景描述或问题，gold = 条款；指标 Recall@10 / nDCG@10；查询库 3,000（含 1,000 人工）；
- **合规分类评测**：二分类 F1 + 条款命中率；1,000 人工样本；
- **防泄漏**：评测查询/条款按日期切分，训练阶段绝不接触评测期条款版本；检索结果人工复核 200 条；
- **可证明性分析**：在消融族（原始/TermGPT/FinAlign 各阶段）上计算 isotropy（如 IC 方向方差、各向异性比）与任务指标，报 Spearman ρ 及显著性。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | FinAlign-Bench（中英，CC-BY-4.0 意向） |
| 来源 | SEC EDGAR 规则文本、沪深交易所/证监会/银保监公开规则、公开公告 |
| 许可 | 监管文本属公共领域；自建标注归本项目 |
| 划分 | 检索 3,000 查询 / 合规 1,000 样本（评测）；训练语料 ~60k 条款块 |
| 预处理 | 条款块化、编号与生效日期结构化、MinHash 去重、UTF-8 |
| 人工质控 | 术语标注 10% 抽检；评测标注双标注 κ≥0.85 |

## 5. 基线复现

| 基线 | 类型 | 复现方式 | 预期指标 |
|---|---|---|---|
| 原始 Qwen2.5-7B | 无适配 | 官方权重，零样本/直接检索 | Recall@10 / nDCG |
| bge-m3（多语言） | 通用嵌入 | https://github.com/FlagOpen/FlagEmbedding | 同上 |
| TermGPT 复现 | 图拓扑对比 | arXiv:2511.09854 训练流程复现（无难负例） | 同上 + isotropy |
| TermGPT + 难负例 | 消融 | 本框架单阶段对照 | 同上 |
| FinAlign 三阶段 | 本方法 | 本设计 | 全指标 |

**统一口径**：同一检索/分类评测器、同一划分；嵌入模型用相同 max_seq=512；报告 mean±std（5 种子）。

## 6. 实验矩阵

- **A 主实验**：原始 vs bge-m3 vs TermGPT vs FinAlign（检索 + 合规 + isotropy 全表）；
- **B 负例消融**：难负例 vs 图拓扑 vs 两者混合（H1）；
- **C 阶段消融**：去掉阶段一/二/三的逐步退化（H2）；
- **D 可证明性**：isotropy ↔ 任务指标的 Spearman（H3）；
- **E 通用能力保持**：在通用嵌入基准（如 MTEB 中文子集）上测退化（H4）；
- **F 稳健性**：难负例数量 1k/5k/30k 曲线、τ/α/β 灵敏度、跨语言（中 vs 英子集）。

## 7. 评测协议

- **指标**：Recall@10、nDCG@10、合规二分类 F1 与条款命中率、isotropy（各向异性比与 IC 方向方差）、通用基准退化率；
- **统计**：5 种子 mean±std；模型间 paired t-test（α=0.05）；Spearman ρ 用 95% bootstrap CI；
- **无前视**：条款按版本快照、日期切分；评测查询日期 > 训练语料日期。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 数据合成与抽取 | API（Flash 难负例、Kimi QA、Pro judge） | 约 4,000 万 token |
| 三阶段 QLoRA 微调 | 4×L40 FSDP | 每模型 6-9 GPU·天（3 阶段），共 2-3 模型 |
| 评测 | 4×L40 + CPU | 1-2 GPU·天 |
| 存储 | 语料 + 嵌入向量 + checkpoint | <500GB |
| 总计 | 4×L40 | 约 14-30 GPU·天 |

**API 成本**：Flash 负例合成约 2,500 万 token；Kimi 中文 QA 约 800 万；Pro judge 约 500 万。估算 $250-600（按当期定价核算）。

## 9. 里程碑与时间线（单人 + 4×L40，9 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 语料收集 + 条款块化 | 语料库 v1 |
| W2 | 术语标注 + 难负例合成（1k pilot） | FinAlign-Bench 检索子集 v0（H1 pilot） |
| W3 | 全量难负例 + 中文 QA | FinAlign-Bench v1 |
| W4 | 阶段一 MLM + 阶段二对比训练 | checkpoint 阶段一/二 |
| W5 | 阶段三联合精调 | FinAlign 模型 |
| W6 | 主实验 A/B | 全表 |
| W7 | 阶段消融 + 可证明性 | C/D 表 |
| W8 | 通用能力 + 稳健性 | E/F 表 |
| W9 | 论文初稿 + 开源 | 提交稿 + 仓库 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 任务收益与嵌入收益解耦、归因不清 | 高 | 预注册对照族（原始/TermGPT/FinAlign）；用消融 + 可证明性 ρ 双保险 |
| 难负例合成质量不稳定 | 中 | 人工抽检负例 200 条；不合格主题迭代 prompt；增加规则化负例（数字/日期扰动）兜底 |
| 中文条款与英文条款结构差异 | 中 | 分语言子集评测与训练，避免跨语言负干扰 |
| 通用能力退化 | 中 | 保留通用基准评测；若退化 >5%，降阶段二/三学习率或提前终止 |
| 数据规模不足（长尾条款稀疏） | 中 | 用 LLM 重写扩展条款 + 概念聚类保证长尾覆盖 |

## 11. 论文写作计划

- **目标会议**：ACL 2027 / EMNLP 2027 / ACL-FinNLP（任务+数据+模型闭环型）。
- **差异化卖点**：(1) 金融首个"术语表示 → 检索/合规任务"端到端可证明框架；(2) LLM 难负例挖掘替代图拓扑；(3) 与 CardioEmbed/PatenTEB 对齐的联合基准范式。
- **图表清单**：图1 三阶段方法总览；图2 难负例示例与混淆维度分布；图3 isotropy vs 任务指标散点；表1 数据集统计；表2 主实验；表3 负例消融；表4 阶段消融 + 通用能力；表5 稳健性。
- **相关工作覆盖**：TermGPT（2511.09854）、CardioEmbed（2511.10930）、PatenTEB（2510.22264）、isotropy（2109.04740、2106.01183、2411.17538）、FINDAP（2501.04961）、FinEval（2308.09975）、FinBen（2402.12659）、FiNER-ORD（2302.11157）、SNFinLLM（2408.02302）。

## 12. 参考文献

1. Sun, Y., Zhu, M., Chen, F., et al. *TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation in Legal and Financial Domains*. arXiv:2511.09854.
2. *CardioEmbed: Domain-Adapted Embeddings and Benchmark for Cardiology*. arXiv:2511.10930.
3. *PatenTEB: Patent Text Embedding Benchmark*. arXiv:2510.22264.
4. *How Does Fine-tuning Affect the Geometry of Embedding Space*. arXiv:2109.04740.
5. *A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space*. arXiv:2106.01183.
6. *Isotropy Matters: Soft-ZCA Whitening of Embeddings*. arXiv:2411.17538.
7. *Demystifying Domain-adaptive Post-training for Financial LLMs (FINDAP)*. arXiv:2501.04961.
8. *FiNER-ORD: Financial Named Entity Recognition Dataset*. arXiv:2302.11157.
9. *FinEval: A Chinese Financial LLM Evaluation Benchmark*. arXiv:2308.09975.
10. *FinBen: A Holistic Financial Benchmark*. arXiv:2402.12659.
11. *SNFinLLM: Systematic and Nuanced Financial Domain Adaptation of Chinese LLMs*. arXiv:2408.02302.
12. *PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance*. arXiv:2306.05443.
