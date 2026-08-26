# 实验设计书：Idea 10 · ABAT —— 主动黑盒调优

> Active Black-box Adaptation with GP Surrogate and LLM-as-Teacher Active Learning
> 目标会议：EMNLP 2026（中优先级）
> 硬件：4 × NVIDIA L40（192GB）｜API：DeepSeek V4 Pro/Flash、Kimi K2.6

---

## 0. 摘要

ABAT 把收藏论文（NLP V·51，arXiv:2511.10210）的 GP 代理黑盒调优升级为**主动学习**：用 GP 对预测的不确定性（方差）驱动 API 查询，在极少调用下完成 LLM 黑盒适配；DeepSeek V4 对选中样本生成 rationales 作为 proxy 的额外监督（不止 logit）；冷启动用一致性/熵在无 API 下预筛易样本，把 API 预算集中在难样本。以 0.3–1B proxy 微调 + GP 代理，覆盖 GLUE/CLUE 分类、情感分析、意图识别、MMLU 多选题，预计 5–10 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

黑盒 LLM 适配：只能通过 API 查询（拿 logits 近似/输出）把能力转移到本地 proxy 小模型，查询成本昂贵。NLP V·51（arXiv:2511.10210）用 GP 代理拟合"LogitMap 对"，把 API 查询频率降至 1.38%，但：
1. **静态子集**：固定选样本查询，未利用 GP 不确定性做主动选择；
2. **无 teacher 多轮蒸馏**：只拿 logits，不利用 rationale 监督；
3. **冷启动**：初始核心集选择无先验。

### 1.2 相关工作不足（收藏论文用「(Session·论文N·英文标题)」格式；外部文献保留真实 arXiv ID/DOI）

- **Advanced Black-Box Tuning**（NLP V·51，arXiv:2511.10210）：GP 代理 + LogitMap 对，静态子集、无主动查询；
- **Gaussian Process 代理调优类**：多用于 prompt 优化，不做 proxy 蒸馏；
- **少样本 ICL**：无训练；
- **soft prompt 全查询黑盒调优**：每轮全量查询太贵。

空白：**GP 不确定性主动查询 + LLM-as-teacher 多轮蒸馏 + 冷启动预筛**的黑盒适配。

### 1.3 为什么是现在、为什么你的环境适合做

- NLP V·51 证明了 GP 代理路线有效，主动学习是明确下一步；
- 0.3–1B proxy + GP 训练只需 1 卡，算力需求极低（5–10 GPU·天）；
- DeepSeek V4/Pro 是现成的黑盒 teacher，Kimi K2.6 作对照 teacher，API 成本透明。

## 2. 研究目标与可验证假设（2-4 条，每条给出"成立时的可观测结果"）

- **H1（主动查询有效）**：GP 不确定性驱动的主动选择优于随机/固定子集。
  - 可观测结果：固定 API 调用预算下，主动选择的 acc 高于随机选择 ≥3pt、高于固定子集（NLP V·51 式）≥2pt。
- **H2（rationale 监督增益）**：DeepSeek V4 的 rationale 监督（不止 logit）提升 proxy。
  - 可观测结果：加 rationale 后 acc 高于仅 logit 蒸馏 ≥1.5pt（同查询预算）。
- **H3（冷启动预筛）**：一致性/熵预筛把预算集中在难样本。
  - 可观测结果：预筛后选中的样本难度分布更偏难（与真实查询收益相关）；同预算 acc 更高。
- **H4（预算-acc 曲线）**：固定 API 调用数下 acc 单调上升且优于基线。
  - 可观测结果：预算-acc 曲线在 500/1k/2k/5k/10k 调用点上均优于 5 条基线。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

**数据**：GLUE（SST-2、CoLA、QQP 子集）、CLUE（中文 TNEWS、CMNLI 子集）、情感分析（Amazon 子集）、意图识别（SNIPS）、MMLU 多选题子集。训练集各取 2k，评测取官方 dev/test。

**冷启动预筛（无 API）**：
- 对未标注池跑 proxy 多采样（N=4），统计答案一致性/预测熵；
- 一致且熵低 → 简单样本，标记"低优先"；
- 不一致或熵高 → 候选难样本池。

### 3.2 模型/算法设计（模块拆解、关键公式、超参数初值）

**proxy 模型**：0.3–1B（Qwen2.5-0.5B/1.5B），分类头（GLUE/CLUE）或多选生成（MMLU）。

**GP 代理**：
- 输入：样本特征（embedding + proxy logit 统计）；
- 拟合 LogitMap 对（样本 → teacher logits/标签），输出均值与方差；
- 高斯核 RBF，feature 降维（PCA 到 64 维，缓解高维退化）。

**主动查询流程**：
1. 初始化：冷启动难样本池取 k=100 条查询 teacher（DeepSeek V4 Pro），获得 logits 近似（输出分布采样 K=8）或直接标签 + rationale；
2. 训练/更新 proxy（用已有查询数据微调）；
3. GP 拟合（更新 LogitMap）；
4. 选择：`score = GP_uncertainty(x) = variance(x)`，取 top-k 高不确定性样本；
5. 查询 teacher：logits + rationale；
6. 更新 GP 与 proxy，重复直到预算耗尽（API 调用上限 2k–10k）；
7. 输出：proxy 模型 + 可选推理时黑盒偏移。

**多轮蒸馏**：每轮用最新 proxy 的伪标签 + teacher 的 rationale 共同微调（teacher-as-teacher，proxy 为 student）。

**超参数初值表**：k=100（每轮新增），预算上限 10k，PCA 64 维，RBF kernel lengthscale=1.0，proxy lr=2e-4，N=4（冷启动采样），K=8（分布近似）。

### 3.3 训练流程（优化器/学习率/批次/调度/FSDP 或 QLoRA 并行方案）

- proxy 微调：1 卡，AdamW，lr=2e-4，bs=32，3 epoch（每轮），全参（0.5B）或 LoRA（1B）；
- GP：scikit-learn（GaussianProcessRegressor），每轮重拟合；
- 并行：1 卡训练 + 其余空闲（本方案算力低，可与他人共享集群时间片）；
- 无 FSDP 需求；bf16。

### 3.4 推理与评测流程

- 评测：固定 API 预算下各数据集 acc；预算-acc 曲线；
- 指标：acc、GP 不确定性校准（ECE）、主动选择 vs 随机/固定子集、rationale 增益、proxy 规模影响、API 调用数。

## 4. 数据集细节（来源/许可/划分/预处理）

| 数据集 | 来源 | 许可 | 划分 | 预处理 |
|---|---|---|---|---|
| GLUE（SST-2/CoLA/QQP） | HF glue | Apache-2.0 | train 2k/val 官方 | 分类 |
| CLUE（TNEWS/CMNLI） | HF CLUE | 开放 | train 2k/val 官方 | 中文分类 |
| Amazon 情感 | HF | 开放 | 2k | 二分类 |
| SNIPS 意图 | HF | 开放 | 2k | 分类 |
| MMLU 子集（arXiv:2009.03300） | HF cais/mmlu | CC-BY-NC-4.0 | 2k 子集 | 多选题 |

预处理：统一 prompt；分类映射到标签；teacher 输出分布采样 K=8 缓存。

## 5. 基线复现（基线列表+官方代码地址；复现步骤与预期指标表；统一评测口径）

| 基线 | 官方实现 | 复现要点 |
|---|---|---|
| 无 API 离线微调 | HF | 只用本地数据，无 teacher |
| 固定子集 GP 代理（NLP V·51，arXiv:2511.10210） | 论文重写 | 静态子集 LogitMap |
| 每轮全查询黑盒调优（soft prompt） | 论文重写 | 每轮全量查询 |
| 少样本 ICL | API 直调 | 无 proxy 训练 |
| **ABAT** | 自研 | 主动 + rationale + 冷启动 |

**预期指标表（预算 2k 调用，多基准平均 acc）**：

| 方法 | 平均 acc | API 调用（×全量） |
|---|---|---|
| 无 API 离线微调 | 55.0 | 0 |
| 固定子集 GP（NLP V·51） | 82.0 | 1.4% |
| 每轮全查询 | 86.0 | 100% |
| 少样本 ICL | 70.0 | 8% |
| **ABAT** | **87.0** | **1.1%** |

> 预估值；口径：多基准平均 acc（SST-2/TNEWS/MMLU 子集等）。

## 6. 实验矩阵（A/B/C…：主实验、消融、鲁棒性、泛化性）

- **A（主实验）**：ABAT vs 4 条基线，预算 {500,1k,2k,5k,10k}；
- **B（消融）**：
  - B1 随机选择（去掉 GP 主动）；
  - B2 去掉 rationale（仅 logit）；
  - B3 去掉冷启动预筛；
  - B4 PCA 维度 {16,64,256}；
  - B5 每轮 k ∈ {50,100,200}；
  - B6 proxy 规模 {0.3B,0.5B,1B}；
- **C（鲁棒性）**：API 预算随机波动；teacher 降级（只给标签不给分布）；seed×3；标注噪声；
- **D（泛化性）**：中文 CLUE 迁移；生成任务（指令跟随子集，可选）；跨 teacher（Kimi K2.6）。

## 7. 评测协议（指标定义、均值±方差、显著性检验、随机种子）

- 指标：acc、预算-acc 曲线、GP 不确定性校准（ECE）、API 调用次数/成本、rationale 增益、proxy 规模影响；
- 主实验 3 seed；均值±方差；配对 bootstrap p<0.05；
- 随机种子 {42,7,2026}；teacher 采样 seed 固定。

## 8. 算力与资源计划（4×L40 分阶段 GPU·天；存储；API 用量与成本估算）

| 阶段 | 内容 | GPU·天 |
|---|---|---|
| P1 数据准备 + 冷启动预筛 | 1 卡 1 天 | 1 |
| P2 proxy 微调 + GP 拟合（迭代） | 1 卡 4 天 | 4 |
| P3 主实验 A + 基线 | 2 天 | 2 |
| P4 消融 B + 鲁棒 C + 泛化 D | 3 天 | 3 |
| **合计** | | **10（预算 5–10）** |

- 存储：proxy 模型 + 数据 ≈ 20GB；
- API：预算上限 10k 调用，实际主实验 ~2k×多轮，DeepSeek Pro 为主 ~$20–60；Kimi K2.6 对照 2k。

## 9. 里程碑与时间线（按周，单人+4 卡）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 数据 + 冷启动预筛 + GP 实现 | 管线 ready |
| W2 | 主动查询循环 + rationale 注入 | 端到端 |
| W3 | 主实验 A + 基线 | 预算-acc 曲线 |
| W4 | 消融 B + 鲁棒 C | 消融表 |
| W5 | 泛化 D + 论文初稿 | 投稿稿 |

## 10. 风险与备选方案（表）

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| GP 在高维 logit 空间退化 | 中 | 高 | PCA 降维（B4）；低秩近似；退化为不确定性启发式（熵） |
| API 预算内收益不显著 | 中 | 中 | 与固定子集严格对比；扩大预算扫描到 10k |
| 文本生成 LogitMap 构造难 | 中 | 中 | 先从分类/多选起步（本方案定位），生成任务留作 D 可选 |
| teacher 只给标签不给分布 | 中 | 中 | 分布近似用 K 采样；标签直接监督 proxy |
| 主动选择退化为随机 | 低 | 中 | GP 校准监控（ECE）；必要时用 ensemble 不确定性 |

## 11. 论文写作计划（目标会议/截稿日期、差异化卖点、图表清单、相关工作覆盖）

- 目标：EMNLP 2026（中优先级）；差异化卖点：GP 不确定性主动查询 + LLM-as-teacher 多轮蒸馏 + 冷启动预筛的三合一黑盒适配，比 NLP V·51 少 20% API 调用或同预算高 1pt+；
- 图表清单：Fig1 方法图（冷启动→GP 主动→teacher 查询→proxy 更新）；Fig2 预算-acc 曲线；Fig3 GP 不确定性校准（ECE）；Fig4 主动 vs 随机选择；Fig5 rationale 增益；Fig6 跨 teacher 对比；Tab1 主实验；Tab2 消融；
- 相关工作：Advanced Black-Box Tuning（arXiv:2511.10210）、Gaussian Process（scikit-learn 基线，不列文献）、少样本 ICL（以论文重写实现）、MMLU（arXiv:2009.03300）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）

- Advanced Black-Box Tuning of LLMs with Limited API Calls arXiv:2511.10210
- MMLU arXiv:2009.03300
- 收藏论文：Black-box GP 调优（Natural Language Processing V·论文 51，AAAI 2026，arXiv:2511.10210）
- 注：GP 为经典统计方法，实现使用 scikit-learn 标准库，不列第三方 arXiv。
