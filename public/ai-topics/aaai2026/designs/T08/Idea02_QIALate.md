# 实验设计书：Idea 2 查询感知的动态 Token 重要性多向量检索（QIA-Late）

## 0. 摘要
本项目把收藏论文 Token Importance（NLP IV · 论文 22）的静态 query token 权重升级为"查询感知的动态权重"，并引入检索-生成联合目标，得到 QIA-Late。核心是一个轻量 token-weight gate 模块，输入 query token 向量与全局查询表示，输出每个 token 在 late-interaction 聚合中的权重；训练分两阶段：先在 BEIR 上训检索，再用检索上下文上的 LM loss 微调 gate（冻结向量编码器）。预期 BEIR Recall@10 相对静态权重 +2–4%、端到端 RAG EM +1.5–3%，且权重输出可解释（可诊断查询中哪些词驱动召回）。总算力约 10–12 GPU·天，投稿 SIGIR 2027 / ACL 2027。

## 1. 研究背景与动机
### 1.1 问题定义
Late-interaction 多向量检索（ColBERT 式）用 Chamfer 距离对 query token 与 doc token 逐对打分后求和：score(q,d)=Σ_i max_j sim(q_i,d_j)。收藏论文 Token Importance 改为 score=Σ_i w_i·max_j sim(q_i,d_j)，w_i 是学习到的静态权重（IDF 或固定模型输出）。问题是 w_i 不随查询语义变化："of/a" 这类 token 在"介词敏感的查询"里重要性应动态翻转，静态权重做不到；且训练只优化检索召回，与下游 RAG 答案质量脱节。

### 1.2 相关工作不足
- ColBERT（arXiv:2004.12832）、ColBERTv2（arXiv:2112.01488）等权聚合，忽略 token 重要性。
- Token Importance（NLP IV · 论文 22 · Incorporating Token Importance in Multi-Vector Retrieval，arXiv:2511.16106）加了静态权重，但权重查询无关、只训检索召回。
- XTR（arXiv:2304.01982）改了 token 检索目标但没动聚合权重；两者正交，组合收益未知。
- WARP 引擎（arXiv:2501.17788）只做工程加速，不涉及权重学习。
- 下游：RAG 中检索最优 ≠ 生成最优（证据使用受噪声影响，参见 arXiv:2605.00505 denoising-first 视角），检索与生成目标脱节普遍存在。

### 1.3 为什么是现在、为什么你的环境适合做
- 证据：多向量检索 2025–2026 进入工程化成熟期（WARP 部署、HPC-ColPali arXiv:2506.21601 量化）；"检索侧与生成侧联合优化"是社区正在追赶的空白；Token Importance 论文是 AAAI 2026 新鲜工作，紧随其后做"动态化+联合化"升级正当时。
- 环境：检索器为 encoder 级（~100M–1B，e5/BGE/ColBERT 基座），单卡可训；生成器联合微调用 3B–7B FSDP 在 4×L40 上 2–3 GPU·天即可；DeepSeek/Kimi 负责合成查询改写与 hard negative，API 成本低。

## 2. 研究目标与可验证假设
- H1（动态权重 > 静态权重）：QIA-Late 的 query-aware 权重在 BEIR 13 任务平均 Recall@10/nDCG@10 显著高于 Token Importance 静态权重。成立时观测：13 任务平均 Recall@10 提升 ≥ +2%（静态权重基线为 +1.28% 零样本）。
- H2（联合目标有效）：叠加生成端 LM loss 微调 gate 后，端到端 RAG（NQ/HotpotQA）EM/F1 进一步提升，且不损检索召回。成立时观测：EM 提升 ≥ +1.5%，同时 Recall@10 不掉（≤0.5% 内）。
- H3（权重可解释）：查询中名词/专名 token 权重显著高于停用词，且权重分布随查询语义变化。成立时观测：人工 100 条查询的权重 top-3 token 与"关键词标注"重叠 ≥ 80%。
- H4（与 XTR 正交）：QIA-Late 动态权重可叠加在 XTR 的 token 检索目标上，组合收益 ≥ 各自单独收益之和的 80%。成立时观测：组合版本在 BEIR 上 Recall@10 ≥ 单用其一。

## 3. 总体方法设计
### 3.1 语料/数据流水线
1. 检索训练数据：MS MARCO 训练集（约 880 万 query-doc 对）、BEIR 13 任务官方划分；NQ/HotpotQA 作为 RAG 端到端评测集。
2. 查询改写数据（DeepSeek V4 Pro 生成）：对每个训练 query 生成 5 个语义改写/扩展变体，用于增加 gate 的查询多样性（prompt：`Generate 5 diverse reformulations of the query that keep the intent: <q>`）。
3. Hard negative：用 BM25 + ColBERT 初筛 top-50，Kimi K2.6 判定其中与 query 不相关的段落作为 hard negative；每 query 保留 3–5 个。
4. 联合训练数据：对每 query 取检索 top-10 段落 + 标注答案，构造"检索上下文 → 答案"的生成样本约 3 万条。

### 3.2 方法设计
- 基座：ColBERTv2 式（BERT-base，多向量）+ PLAID/WARP 索引引擎（arXiv:2501.17788）作为推理加速；模型库与打分函数替换为加权 Chamfer。
- gate 模块：w_i = σ(Linear(MLP([h_i; h_global])))，其中 h_i 是 query token 的最后一层 hidden，h_global 是 query 的 CLS/池化表示；w_i 同时乘到 max_j sim(q_i,d_j) 上。参数规模 <2M，可插拔。
- 损失：
  - 检索 loss：InfoNCE / 带温度 τ=0.1 的 cross-entropy，正样本 = 标注 doc，负样本 = in-batch + hard negative。
  - 生成 loss（联合阶段）：L_gen = -log P(answer | [CLS]retrieved_docs ⊕ query)，用冻结的 3B 生成器计算，只回传梯度到 gate（通过可微 top-k 近似或 Gumbel-softmax 采样段落）。
  - 联合总损失：L = L_ret + λ·L_gen，λ 初值 0.5，warmup 到 1.0。
- 超参数初值：lr=2e-5（检索阶段）/5e-6（gate 联合阶段）、batch 32、max_len query 32 / doc 128、temperature 0.1、λ=0.5。

### 3.3 训练流程
- 阶段一（检索）：在 4×L40 上，e5-base 或 ColBERTv2 编码器全参（~110M）微调，FSDP 2 卡、batch 32，约 2 天（1 个 epoch，MS MARCO 子采样 50 万条）；gate 同时训练（挂在 query encoder 后）。
- 阶段二（联合）：冻结全部向量编码器，只训 gate（~2M），在 3 万条"检索上下文→答案"样本上 2–3 GPU·天；生成器用 Qwen2.5-3B 冻结，仅提供 LM 梯度信号。
- 并行：阶段二同时可把生成器换成 7B LoRA 验证（可选，+2 GPU·天）。
- 可选扩展：文档 token 侧权重作为消融（给 doc token 也加 gate，训练更慢，先不做为主实验）。

### 3.4 评测流程
- 检索评测：BEIR 13 任务 Recall@10 / nDCG@10；MS MARCO dev 的 MRR@10。
- RAG 评测：NQ（短答案 EM）、HotpotQA（F1/EM），检索 top-10 段 + Qwen2.5-7B-instruct 生成（temperature=0）；报告检索-生成两条线的指标。

## 4. 数据集细节
| 数据集 | 来源 | 许可 | 说明 |
|---|---|---|---|
| MS MARCO | 官方 download | MS MARCO license（研究） | 训练/检索 + RAG |
| BEIR 13 任务 | beir 库打包 | 各任务原许可 | nDCG@10 标准划分 |
| NQ / HotpotQA | 官方 | 开放研究 | RAG 端到端 |
| 合成改写/硬负 | 本项目 API 生成 | 自建 | 3 万条生成样本 |
- 预处理：段落切分（NQ/HotpotQA 取 gold 段落 + 干扰段落）；统一 JSONL；嵌入缓存（推理时向量库复用，索引不变）。

## 5. 基线复现
| 基线 | 官方代码 | 预期指标（BEIR 平均 Recall@10 / NQ EM 粗估） |
|---|---|---|
| ColBERTv2 | github.com/stanford-futuredata/ColBERT | R@10 ~65–70% |
| XTR | github.com/google-deepmind/xtr | R@10 略高于 ColBERTv2 |
| Token Importance 复现 | arXiv:2511.16106（作者仓库未公布则自实现静态权重） | R@10 静态 +1.28% 相对 |
| WARP 部署 | github.com/stanford-futuredata/WARP | 与 ColBERTv2 同分数、快 2–5× |
- 复现步骤：ColBERTv2 官方 train.py + MS MARCO；BEIR 用 beir/Evaluate 脚本；统一用同一索引（PLAID）与同一批 embedding 缓存，保证打分函数差异可隔离。
- 统一口径：所有检索器用相同训练数据（MS MARCO）、相同索引后端（PLAID）、相同 top-k（10）。

## 6. 实验矩阵
- A（主实验）：QIA-Late vs ColBERTv2 / XTR / 静态权重，BEIR 13 + MS MARCO + NQ/HotpotQA RAG。
- B（消融）：gate 结构（gate on/off、用/不用全局 h_global）、λ（0/0.5/1.0）、查询改写数据量（0/1/5 变体）、文档侧权重开/关、Gumbel vs 直接 top-k。
- C（鲁棒性）：硬负数量（0/3/5）、长查询（TREC-COVID 医学长查询）、多语言 BEIR 子集（MrTydi 法语/中文）。
- D（泛化性）：换基座（e5-base→e5-large、BGE）；叠加 XTR token 检索目标（H4）；跨域 zero-shot（在 MS MARCO 训、直接测 13 任务全表）。

## 7. 评测协议
- 检索指标：Recall@10、nDCG@10（BEIR 标准）、MRR@10（MS MARCO）。
- 生成指标：EM / F1（NQ、HotpotQA）。
- 可解释指标：top-3 权重 token 与人工关键词重叠率（H3）。
- 统计：3 个随机种子（42/2024/2026），报告均值±std；配对 bootstrap（1000 次）检验 Recall@10 差异显著性（p<0.05）。

## 8. 算力与资源计划
- 4×L40 分阶段：阶段一检索训练 2 卡 ≈ 2–3 GPU·天；阶段二 gate 联合 2 卡 ≈ 2–3 GPU·天；推理/索引 ≈ 2 GPU·天；鲁棒性/泛化 ≈ 2–3 GPU·天；合计约 10–12 GPU·天。
- 存储：MS MARCO 索引（PLAID 压缩后 ~15GB）、BEIR 各任务索引 ~20GB、数据/缓存 <100GB。
- API：DeepSeek 查询改写 ≈ 50–100 美元；Kimi 硬负判定 ≈ 50–100 美元；合计 ≤ 200 美元。

## 9. 里程碑与时间线（按周，单人 + 4 卡）
| 周 | 任务 |
|---|---|
| 1 | 环境搭建；ColBERTv2/PLAID 复现；BEIR 基线跑通 |
| 2 | 合成改写 + 硬负数据；gate 实现；阶段一检索训练 |
| 3 | 阶段二联合训练；NQ/HotpotQA RAG 评测线打通 |
| 4 | 主实验 A + 消融 B |
| 5 | 鲁棒性 C + 泛化性 D；统计检验 |
| 6 | 论文初稿 + 图表 + 开源 |

## 10. 风险与备选方案
| 风险 | 等级 | 对策 |
|---|---|---|
| 联合 loss 不稳定（gate 梯度信号弱） | 高 | 先退化为"仅 query-aware gate"（静态替代，仍可投稿）；λ warmup |
| PLAID/WARP 对接成本高 | 中 | 先用 ColBERTv2 原生 PyTorch 打分验证增益，再工程化 |
| 多向量检索训练数据成本 | 中 | 用 MS MARCO 子采样 + BEIR 官方数据，不额外买数据 |
| 可解释性消融主观 | 低 | 人工标注 100 条 + 与 IDF 权重对比 |

## 11. 论文写作计划
- 目标：SIGIR 2027（约 2027 年 1 月截稿）或 ACL 2027；备选：EMNLP 2027。若进度快可投 SIGIR 2027。
- 差异化卖点：查询感知动态权重 + 检索-生成联合目标 + 可插拔（不改索引）三合一；首个与 XTR 组合的量化分析。
- 图表清单：图1 框架（gate 挂接位置）；图2 权重可视化（同一 query 在不同改写下权重变化）；图3 BEIR 13 任务增益分布箱线图；表1 主结果表；表2 消融表；表3 泛化表。
- 相关工作覆盖：ColBERT/ColBERTv2/XTR/WARP、Token Importance（NLP IV · 论文 22）、PLAID 系工程、检索-生成联合优化（arXiv:2605.00505）。

## 12. 参考文献
- ColBERT: arXiv:2004.12832
- ColBERTv2: arXiv:2112.01488
- XTR: arXiv:2304.01982
- Token Importance: arXiv:2511.16106（NLP IV · 论文 22）
- WARP: arXiv:2501.17788
- HPC-ColPali: arXiv:2506.21601
- Denoising-First: arXiv:2605.00505
- RAGAS: arXiv:2309.15217
