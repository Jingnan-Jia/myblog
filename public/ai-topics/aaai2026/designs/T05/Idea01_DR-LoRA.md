# 实验设计书：Idea 1 · DR-LoRA —— 难度自适应动态秩 LoRA 与遗忘补偿

> Difficulty-Aware Dynamic-Rank LoRA with Catastrophic-Forgetting Compensation
> 目标会议：NeurIPS 2026（高优先级）
> 硬件：4 × NVIDIA L40（192GB）｜API：DeepSeek V4 Flash/Pro、Kimi K2.6

---

## 0. 摘要

DR-LoRA 把 PEFT 的"秩分配、更新强度、知识保留"三个维度放进同一个可微框架内协同优化：按 token/样本难度动态分配 LoRA 秩分量，同时以冻结原模型的 reverse-KL 蒸馏显式补偿灾难性遗忘，训练后按门控统计对低激活秩分量做减法合并。与 GateRA（仅强度）、AdaLoRA（静态秩）、LoKI（仅遗忘）相比，DR-LoRA 首次在同一基座上同时给出动态秩 + 遗忘补偿，并以"通用能力保持率"为主衡量指标。方案以 LLaMA-3-8B/DeepSeek-R1-Distill-7B 为基座，全程 QLoRA + FSDP 双卡可跑，预计 10–20 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

给定基座模型 f_θ（如 LLaMA-3-8B），指令微调数据集 D={x_i, y_i}。PEFT 目标是在保持通用能力的前提下，以少量可训练参数把模型适应到目标任务。现有 PEFT 有三个独立维度，尚未统一：

- **强度（modulation）**：GateRA 用 token 级门控调制更新强度（Natural Language Processing III·论文 94·GateRA: Token-aware Modulation for Parameter-Efficient Fine-tuning），但 rank 固定；
- **秩（rank）**：AdaLoRA 用 SVD 显著性做静态秩分配（arXiv:2303.10512），无输入自适应；
- **遗忘（forgetting）**：LoKI 用机制先验防遗忘（Natural Language Processing IV·论文 107·LoKI: Low-Damage Knowledge Implanting of Large Language Models），但不感知难度。

形式化目标：min_{ΔW} L_task(x,y;W0+ΔW) + λ·L_gen(θ0; x) ，其中 ΔW 由"难度自适应秩掩码 × 强度门控"参数化，L_gen 为对冻结 teacher（原模型）的 reverse-KL 蒸馏项。

### 1.2 相关工作不足（收藏论文用「(Session·论文N·英文标题)」格式；外部文献保留真实 arXiv ID/DOI）

| 工作 | 维度 | 不足 |
|---|---|---|
| GateRA（NLP III·94） | 强度 | rank 固定；无遗忘保护；门控端到端开销未量化 |
| AdaLoRA（arXiv:2303.10512） | 秩 | 静态、输入无关 |
| LoKI（NLP IV·107） | 遗忘 | 依赖"前向 MLP 知识书写"假设；不感知难度 |
| LoRA Learns Less and Forgets Less（arXiv:2405.09673） | 分析 | 指出 LoRA 数学/推理弱、遗忘轻，但未给联合框架 |
| SFP（CV IV·48） | 减法 | 减法范式仅验证视觉；无自适应机制 |
| InTRO（NLP VI·81） | 难度信号 | 修正因子用于推理自我反馈，未接入 PEFT 秩分配 |

空白：**动态秩 × 强度门控 × 遗忘补偿**三维协同在已有文献中不存在。

### 1.3 为什么是现在、为什么你的环境适合做

- GateRA（arXiv:2511.17582）、AdaLoRA、LoKI 等近 1–2 年密集出现，说明"维度融合"是 PEFT 社区共识方向，审稿人容易认可空白点；
- 4×L40 足够跑 8B 级 QLoRA 双卡并行 + FSDP；8B 级模型难度信号（per-token 信息差异）可在单卡前向内完成，无需 70B 级算力；
- DeepSeek V4 API 可低成本为训练样本生成多步推理链与难度标注（1–10），作为难度估计器的监督与评测 judge。

## 2. 研究目标与可验证假设（2-4 条，每条给出"成立时的可观测结果"）

- **H1（动态秩提升）**：在同一参数量预算下，难度自适应秩分配优于固定秩 LoRA 与 AdaLoRA 静态秩。
  - 可观测结果：MMLU/GSM8K/MATH 平均 acc 高于同等参数预算的 LoRA(r=16) 与 AdaLoRA ≥ 1.0pt；且难样本子集（按难度标签 top-20%）上收益更大（≥2pt）。
- **H2（遗忘补偿）**：reverse-KL 蒸馏项在任务适应中保留通用能力，显著缓解灾难性遗忘。
  - 可观测结果：微调后 zero-shot MMLU/常识基准（HellaSwag 等）保持率（后/前 acc）高于无蒸馏的 DR-LoRA 变体 ≥ 5pt；遗忘率指标（类 LoRA Learns Less 测法）下降。
- **H3（减法合并）**：训练后按门控统计剪除低激活秩分量并合并，性能不降、参数量与推理延迟下降。
  - 可观测结果：合并后平均 acc 下降 ≤0.3pt，可训练参数减少 ≥30%，推理延迟降低（长序列 batch 场景 ≥10%）。
- **H4（开销可量化）**：token 级门控的端到端开销小于其收益。
  - 可观测结果：在 4k/8k 上下文 batch 推理中，含门控的推理吞吐不低于无门控基线的 90%，否则在论文中报告开销曲线。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

**训练数据**：混合指令集，取 CommonsenseQA（约 9.7k）、GSM8K 训练集（约 7.5k）、MATH 训练集子集（取 10k 难度分级样本）、MMLU 训练划分 5k 条，总量 ~32k 条，打散后按 90/10 分训练/验证。

**API 合成（DeepSeek V4 Pro）**：
- 对每道题用 prompt 生成多步 CoT（"请逐步推理并给出最终答案，输出以 FInal Answer: 结尾"）；
- 同一 prompt 用不同温度（t=0.3/0.7）采样 2 条，供难度估计器使用；
- 难度标注 prompt：`请对以下问题在 1–10 上打分（1 极简单，10 极难，考虑推理步数与知识要求），只输出一个数字。\n\n题目：{problem}`，DeepSeek V4 Flash 批量执行（约 32k 次调用，Flash 低价档）。
- 过滤规则：答案经规则化（去掉 CoT 只保留 FInal Answer 后字符串匹配）通过才保留；难度标注重复调用 2 次取均值，标准差>2 的样本丢弃（标注不稳定）。

**数量**：SFT 混合 32k 条 + 每样本 2 条多温度采样 = 有效难度对 64k；难度标签 32k。

### 3.2 模型/算法设计（模块拆解、关键公式、超参数初值）

**基座**：LLaMA-3-8B（主）；DeepSeek-R1-Distill-7B（泛化复现用）。

**LoRA 分支扩展为 R 个秩分量**（R=8，秩取 4/8/16/32/4/8/16/32 混合，总容量约等于 r=120）：

前向：`h = W0 x + Σ_r g_r(x)·(B_r A_r x)`，A_r∈ℝ^{r×d}, B_r∈ℝ^{d×r}。

**门控路由器** `g(x) ∈ [0,1]^R`：
- 取每层输入 token 隐状态 h_t，过一个共享 MLP（hidden 256，ReLU）到 R 维，再经 entmax（Γ=1.5）得到软权重；
- **难度信号** d_t：参考 InTRO（NLP VI·81）的"生成策略 vs 答案条件策略"信息差异：d_t = KL(p_θ(·|x, y_<t) ∥ p_θ(·|x, y))，其中 p_θ(·|x,y) 为 teacher-forcing 答案条件分布。实现上对每个 token 需一次额外前向（teacher-forcing 分支），初始化后与小模型共训（共享 q 层梯度）；
- 门控最终值：`g_r(x) = sigmoid(α_r · (w_r^⊤ h_t + β_r · d_t))`，α 初始化 1，β 初始化 0.1；用熵正则把门控推向近二值（同 GateRA）。

**训练损失**：

`L = CE(x,y) + λ1·Σ_t d_t·CE(x_t,y_t) + λ2·KL_rev(p_θ∥p_θ0) + λ3·H(g) + λ4·‖B‖_F`

- λ1=0.5, λ2=0.1（前 30% 步 warm-up 到 0.1，防 early 干扰）, λ3=0.01, λ4=1e-4；
- KL_rev 用冻结原模型在**同一 batch 前缀**上的分布做 reverse-KL（先验兼容 GKD 思想，arXiv:2306.13649），只对通用 token（非任务专属位置）加权；
- 秩稀疏正则 λ3·H(g) 鼓励近二值门控。

**训练后减法合并**（SFP 思想，CV IV·48）：
- 统计每个秩分量在验证集上的平均门控激活 a_r；
- 剪除 a_r < τ（τ 默认 0.05·max）的分量；
- 剩余分量：`ΔW = Σ_r ḡ_r B_r A_r` 直接与 W0 相加合并（保留门控均值缩放 ḡ_r），实现"零额外延迟"推理。

**超参数初值表**：R=8，lora_alpha=16，dropout=0.05，τ=0.05，entmax Γ=1.5，λ=(0.5,0.1,0.01,1e-4)，难度估计器 MLP hidden=256。

### 3.3 训练流程（优化器/学习率/批次/调度/FSDP 或 QLoRA 并行方案）

- 优化器 AdamW（β=(0.9,0.999)，wd=0.02），lr=2e-4（LoRA）配余弦调度 + 200 步 warm-up；门控路由器 lr=5e-4；
- batch：global bs=64，seq_len 2048，grad_accum=16 × per-gpu 1 × 4 卡；
- 步数：~12k 步（约 1 epoch 混合数据）；每 500 步评估验证 acc；
- 并行：QLoRA 4-bit NF4（arXiv:2305.14314）量化基座，双卡 FSDP（shard 门控与 LoRA 参数），另 2 卡跑难度估计器预热与数据难度特征缓存（避免重复前向）；
- 混合精度 bf16；gradient checkpointing 开。

### 3.4 推理与评测流程

- 训练后：合并版模型（ΔW 已并入）用于评测，保证零门控开销；
- 评测调用统一 prompt 模板与温度（见 §7）；生成 max_new_tokens=1024，GSM8K/MATH 用 8 样本多数投票报告；
- 通用能力保持率：微调前后同一评估脚本跑 MMLU（5-shot）、HellaSwag（10-shot）、PIQA、ARC-e，输出"后/前"比值。

## 4. 数据集细节（来源/许可/划分/预处理）

| 数据集 | 来源 | 许可 | 划分 | 预处理 |
|---|---|---|---|---|
| MMLU（arXiv:2009.03300） | huggingface cais/mmlu | CC-BY-NC-4.0 | train 取 5k 做 SFT，test 全量 5-shot 评测 | 标准化 choice prompt；MCQ 模板 |
| GSM8K（arXiv:2110.14168） | huggingface openai/gsm8k | MIT | sft 7.5k；test 1.3k 评测 | 答案后处理（去掉 CoT） |
| MATH（arXiv:2103.03874） | huggingface lighteval/MATH | MIT | sft 10k（按难度 1–5 分层抽样）；test 5k | 同 GSM8K |
| CommonsenseQA | huggingface tau/commonsense_qa | CC-BY | 9.7k 全部 SFT | MCQ 模板 |
| HellaSwag/PIQA/ARC-e | 公开 HF 数据集 | 各许可 | 仅评测 | 无 |
| 合成 CoT/难度标签 | DeepSeek V4 API 生成 | 自建，随论文开源 | 32k 条 | 见 §3.1 过滤规则 |

预处理通用：统一 chat 模板（`<|user|>`/`<|assistant|>`），tokenize 时 pack 到 2048；答案与问题同 prompt 拼接。

## 5. 基线复现（基线列表+官方代码地址；复现步骤与预期指标表；统一评测口径）

| 基线 | 官方实现 | 复现要点 |
|---|---|---|
| LoRA（arXiv:2106.09685） | PEFT (hf) | r=16, alpha=32, lr=2e-4，同数据同步数 |
| DoRA（arXiv:2402.09353） | hf PEFT 支持 | 同 LoRA 超参，加 weight decomposition |
| AdaLoRA（arXiv:2303.10512） | hf PEFT | 默认初始化，r=16 |
| GateRA（arXiv:2511.17582） | 官方 repo（若不可得按论文重写） | token 门控 + 熵正则，r=16 |
| HiRA | 论文重写（rank 扩展初始化） | r=16 |
| 全参微调（full FT） | 自写 | 4 卡 FSDP，lr=1e-5，仅 3k 步（控制成本） |

**预期指标表（统一 8B 基座、32k SFT、同评测协议）**：

| 方法 | MMLU(5-shot) | GSM8K(8-vote) | MATH(8-vote) | 通用保持率(MMLU 后/前) | 可训练参数 |
|---|---|---|---|---|---|
| 基座 LLaMA-3-8B | 68.4 | 45.8 | 17.0 | 1.0 | – |
| LoRA r16 | 67.2 | 52.1 | 19.5 | 0.98 | 16M |
| DoRA | 67.6 | 52.8 | 19.9 | 0.98 | 16M |
| AdaLoRA | 67.9 | 53.0 | 20.1 | 0.98 | ~13M |
| GateRA | 68.0 | 53.2 | 20.3 | 0.97 | 16M |
| 全参 FT | 65.5 | 53.5 | 20.8 | 0.90 | 8B |
| **DR-LoRA** | **68.5** | **54.5** | **21.5** | **0.99** | **~14M(合并后)** |

> 上表数字为预估值，用于验收阈值；以真实复现为准。口径：GSM8K/MATH 均做"答案规则提取+数值匹配"；MMLU 官方 choices 模板。

## 6. 实验矩阵（A/B/C…：主实验、消融、鲁棒性、泛化性）

- **A（主实验）**：DR-LoRA 完整版 vs 六条基线，全指标；
- **B（消融）**：
  - B1 去掉难度加权 CE（λ1=0）；
  - B2 去掉 reverse-KL（λ2=0）；
  - B3 去掉秩稀疏正则（λ3=0，门控连续化）；
  - B4 固定秩（用单一 r=32 替代 8 分量，仅强度门控=GateRA 式）；
  - B5 静态难度（用 API 标签替代 per-token 在线难度）；
  - B6 不剪枝 vs 剪枝+合并；
- **C（鲁棒性）**：随机种子 ×3；训练子集采样 ×2（10k/32k）；难度标签扰动（±1 标注）；门控初始化 α 不同取值；
- **D（泛化性）**：基座换成 DeepSeek-R1-Distill-7B；下游换成代码（HumanEval，取 2k 条 MBPP 子集 SFT）；上下文 4k→8k 门控开销曲线。

## 7. 评测协议（指标定义、均值±方差、显著性检验、随机种子）

- 指标：acc（一致形式）、通用保持率（后/前）、遗忘率 = 1−(通用后/前)、平均有效秩（Σ ḡ_r·r/Σḡ_r）、门控稀疏度（激活占比）、延迟（ms/token，batch=32 长序列）、可训练参数量；
- 均值±方差：主实验 3 个种子，报告 mean±std；
- 显著性：配对 bootstrap（1000 次重采样），p<0.05 视为显著；GSM8K/MATH 报告 8 样本投票 acc；
- 随机种子：数据打散、初始化、采样均固定 seed ∈ {42, 7, 2026}。

## 8. 算力与资源计划（4×L40 分阶段 GPU·天；存储；API 用量与成本估算）

| 阶段 | 内容 | GPU·天 |
|---|---|---|
| P1 数据合成+难度标注 | DeepSeek API（不占 GPU，GPU 用于难度估计器预热 0.5 天） | 0.5 |
| P2 主实验（A） | 7 方法 × 12k 步 × 4 卡 | 8 |
| P3 消融（B） | 6 组 × 8k 步 | 4 |
| P4 鲁棒/泛化（C/D） | 种子与基座扩展 | 4 |
| **合计** | | **16.5（预算 10–20）** |

- 存储：8B 模型 ×3（基座/teacher/checkpoint）≈ 50GB；数据 ≈ 5GB；合计 <100GB，SSD 即可；
- API 用量：难度标注 32k×(2 次 flash) + CoT 合成 64k 次（Pro，t=0.3/0.7）+ judge 约 10k；估算 DeepSeek 调用 ~110k 次，Flash 为主，成本 ~$25–60；Kimi K2.6 交叉标注 5k 次作对照。

## 9. 里程碑与时间线（按周，单人+4 卡）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 环境搭建、数据下载预处理、难度标注 API 脚本 | 数据 JSONL + 标注 CSV |
| W2 | 难度估计器预热、LoRA 8 分量 + 门控实现 | 可训练代码 + smoke test |
| W3 | reverse-KL 蒸馏项与剪枝合并实现 | 消融开关齐全 |
| W4 | 主实验 A 全量跑 + 基线复现 | 指标表初版 |
| W5 | 消融 B + 鲁棒 C | 消融表 |
| W6 | 泛化 D + 开销曲线 + 论文图表 | 论文初稿 |
| W7 | 复跑、写论文、内部评审 | 投稿稿 |

## 10. 风险与备选方案（表）

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 门控路由器长序列开销放大 | 中 | 高 | 3.4 中量化；备选：门控只在 decode 首 token 计算一次（降频），论文报告开销-收益曲线 |
| 秩分量并行显存 +15% 溢出 | 低 | 中 | 降 R=6；QLoRA 双卡 FSDP 已留余量 |
| 难度估计器受基模型采样质量影响 | 低 | 中 | 用 DeepSeek V4 生成第二参考采样增强信号；难度改为粗粒度（3 档） |
| reverse-KL 干扰任务学习 | 中 | 中 | λ2 warm-up；蒸馏只作用通用 token 子集 |
| 剪枝后掉点 | 低 | 低 | 阈值 τ 网格搜索 0.01–0.15；或保留 top-3 分量不剪 |

## 11. 论文写作计划（目标会议/截稿日期、差异化卖点、图表清单、相关工作覆盖）

- 目标：NeurIPS 2026（高优先级，投稿截止约 2026-05，以官网为准）；
- 差异化卖点："秩×强度×遗忘"三维联合自适应 + 减法合并，直接回应 GateRA（强度）、AdaLoRA（秩）、LoKI（遗忘）三篇的各自空白；
- 图表清单：Fig1 方法图（门控+秩分量+蒸馏+剪枝）；Fig2 难度-门控可视化（难/易 token 的门控激活）；Fig3 通用保持率曲线；Fig4 开销-收益曲线；Tab1 主实验；Tab2 消融；Tab3 泛化；
- 相关工作覆盖：PEFT（LoRA/DoRA/AdaLoRA/rsLoRA/LoRA+/PiSSA/MoRA，arXiv:2106.09685/2402.09353/2303.10512/2312.03732/2402.12354/2404.02948/2405.12130）；遗忘分析（arXiv:2405.09673）；收藏论文（NLP III·94, NLP IV·107, CV IV·48, NLP VI·81）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）

- LoRA arXiv:2106.09685；QLoRA arXiv:2305.14314
- DoRA arXiv:2402.09353；AdaLoRA arXiv:2303.10512；rsLoRA arXiv:2312.03732；LoRA+ arXiv:2402.12354；PiSSA arXiv:2404.02948；MoRA arXiv:2405.12130
- LoRA Learns Less and Forgets Less arXiv:2405.09673
- GKD (On-Policy Distillation) arXiv:2306.13649
- GateRA arXiv:2511.17582；LoKI arXiv:2505.22120；InTRO arXiv:2511.09865
- MMLU arXiv:2009.03300；GSM8K arXiv:2110.14168；MATH arXiv:2103.03874；HumanEval arXiv:2107.03374
- Self-Consistency arXiv:2203.11171；BBH arXiv:2210.09261
- 收藏论文：SFP（Computer Vision IV·论文 48，AAAI 2026，无 arXiv）
