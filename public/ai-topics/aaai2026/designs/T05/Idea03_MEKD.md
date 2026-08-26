# 实验设计书：Idea 3 · MEKD —— 微专家均衡 MoE→MoE 蒸馏

> Micro-Expert Balanced MoE-to-MoE Distillation with Subspace Alignment
> 目标会议：NeurIPS 2026（高优先级）
> 硬件：4 × NVIDIA L40（192GB）｜API：DeepSeek V4 Pro/Flash、Kimi K2.6

---

## 0. 摘要

MEKD 把 MoE→MoE 蒸馏推进到**跨矩阵微专家**层面：对 teacher/student 的 MoE 层做联合 SVD，在共享子空间内逐微专家对齐隐层激活；同时继承 B-Distill 的蒙特卡洛覆盖采样与熵感知路由蒸馏，解决专家覆盖不足与路由不平衡；蒸馏后再接 CAMERA 式训练-free 冗余分析（微专家贡献剪枝/量化），形成"蒸馏→压缩"闭环。以 Mixtral-8×7B（INT4）为 teacher、OLMoE-1B（8 专家）为 student，预计 30–40 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

MoE→MoE 蒸馏是新增设置：teacher T（大 MoE）把能力迁移给 student S（小 MoE，同样有路由结构）。存在两类特有不平衡：
1. **专家覆盖不足**：传统蒸馏只利用 teacher 路由器激活的少数专家，未激活专家学不到；
2. **路由不平衡**：student 的路由分布偏离 teacher，无法学到正确的专家分发策略。

此外，B-Distill 在**整专家粒度**蒸馏，忽略了 CAMERA 揭示的**跨矩阵微专家**冗余结构（微专家 = 跨 W_up/W_gate/W_down 的矩阵块单元）。本方案在三个粒度上联合：微专家（隐层对齐）、整专家（logit/路由）、全局（平衡正则）。

### 1.2 相关工作不足（收藏论文用「(Session·论文N·英文标题)」格式；外部文献保留真实 arXiv ID/DOI）

- **B-Distill**（ML V·63）：解决覆盖与路由不平衡，但仅整专家粒度；缺隐层子空间对齐；指标单一（Rouge-L）；
- **Sub-MoE**（ML IV·86，arXiv:2506.23266）：子空间专家合并（共享 U/合并 V），training-free，无蒸馏恢复闭环；
- **CAMERA**（ML IX·68）：微专家冗余分析 + 剪枝/量化，训练-free，未与蒸馏结合；
- **MiniLLM**（arXiv:2306.08543）：dense→dense 反向 KL 蒸馏，无 MoE 设置；
- **MoE→Dense 类 KD**：架构不匹配，student 吸收差。

空白：**微专家粒度 × 子空间对齐 × 路由平衡**三者联合的 MoE→MoE 蒸馏 + 蒸馏后压缩闭环，尚未存在。

### 1.3 为什么是现在、为什么你的环境适合做

- Mixtral-8×7B（INT4，~24GB）与 OLMoE-1B（~1B 总参，激活 ~0.4B，arXiv:2409.02060）在 4×L40 上可同驻分时调度，是社区可得的开放 MoE 对；
- CAMERA 证明微专家冗余分析 <5 分钟/A100 即可完成，训练-free 分析在本环境 4 卡上轻松复现；
- DeepSeek V4 API 可补足 student 容量不足时的高质量数据（长 CoT、指令与多步推理）。

## 2. 研究目标与可验证假设（2-4 条，每条给出"成立时的可观测结果"）

- **H1（微专家对齐增益）**：子空间内微专家级隐层对齐优于整专家级对齐蒸馏。
  - 可观测结果：在相同数据与预算下，下游平均 acc（MMLU/GSM8K/HumanEval/BBH）比 B-Distill 式整专家蒸馏高 ≥2pt；微专家对齐度（对齐后激活 cosine）更高。
- **H2（路由平衡）**：熵感知路由蒸馏 + 蒙特卡洛覆盖显著改善 student 路由分布。
  - 可观测结果：student 路由与 teacher 的路由分布 KL 下降；负载均衡度（max load / 平均 load）更接近 1；死专家率 <5%。
- **H3（覆盖均匀）**：蒙特卡洛扰动使每个 teacher 专家都被采样训练。
  - 可观测结果：teacher 专家采样覆盖度 = 1.0（全部专家出现在蒸馏样本中），对比朴素蒸馏覆盖度 <0.7。
- **H4（蒸馏→压缩闭环）**：蒸馏后 CAMERA 式剪枝/量化恢复优于直接对 teacher 剪枝。
  - 可观测结果：对蒸馏后的 student 做 30% 微专家剪枝，性能下降 ≤2pt（teacher 直接剪枝同比例下降 >5pt）。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

**数据来源**：UltraChat（arXiv:2305.14233 对应论文，对话）、OpenWebMath、Flan（arXiv:2109.01652，多任务）、MATH 合成 CoT（用 DeepSeek V4 Pro 生成 step-by-step）。总量目标 80k 条，比例 4:3:2:1。

**API 合成**：
- MATH 题目的 CoT 由 DeepSeek V4 Pro 生成（temperature 0.4，max 1024 token），过滤"答案与给定答案不一致"；
- 当 student 容量不足以承载全量数据时，用 API 数据优先（Kimi K2.6 也生成一批做多样性对照）。

**过滤规则**：长度 <10 或 >2048 token；重复文本（SimHash 去重）；teacher 输出的 logit 覆盖检查（对输入运行一次 teacher，logit 分布近均匀的样本弃）。

**数量**：训练 80k；验证 2k；评测基准见 §4。

### 3.2 模型/算法设计（模块拆解、关键公式、超参数初值）

**模型对**：
- teacher：Mixtral-8×7B，INT4（GPTQ 量化，使用 HF bitsandbytes，权重 ~24GB），驻留 1–2 卡；
- student：OLMoE-1B（8 专家，top-1 路由，激活 ~0.4B，arXiv:2409.02060），2–3 卡 FSDP。

**联合 SVD 对齐**：对 teacher/student 的每个 FFN 层的三个投影（W_up, W_gate, W_down）拼接后联合 SVD：把 teacher 权重矩阵与 student 权重矩阵各自左乘/右乘投影到共享 U 子空间（子空间秩 r=64），在 U 空间中计算微专家激活对齐损失。实现上：
- 对齐损失只作用于**路由激活的专家**对应的微专家块（减内存）；
- 子空间秩 r 在 {32,64,128} 消融（对应 CAMERA 微专家维度）。

**蒸馏损失**：

`L = λ1·L_logit + λ2·L_hidden + λ3·L_route + λ4·L_balance + λ5·L_CE(真标签)`

- `L_logit = KL(softmax(z_s/τ) ∥ softmax(z_t/τ))`，τ=2；
- `L_hidden = Σ_l cos(a_s^l, a_t^l)`（微专家激活，在共享 U 子空间内）；
- `L_route = KL(p_route_s ∥ p_route_t)`（熵感知：对 student 路由加熵正则，遏制负载坍缩）；
- `L_balance = (mean load / max load)` 梯度惩罚 + 覆盖均匀性正则（B-Distill 的蒙特卡洛：训练时以概率 ε 对 teacher 路由器输出加扰动，ε=0.1）；
- `L_CE` 真标签交叉熵（保留领域知识）。

**λ 初值**：λ1=1.0，λ2=0.5，λ3=0.5，λ4=0.1，λ5=0.2；前 10% 步只训 logit（避免隐层噪声）。

**压缩闭环**（蒸馏后，对 student 做）：
- 训练-free 微专家贡献分析：对每个微专家（矩阵块）计算其在一批校准数据上的输出方差贡献（CAMERA 思想），标记低贡献微专家；
- 低贡献微专家（贡献 < 阈值 p20）→ 并入共享 V 或降精度到 INT2（其余保持 INT8/FP16）；
- 量化用 GPTQ/AWQ 风格（AWQ arXiv:2306.00978），校准集取验证数据 512 条。

**超参数初值表**：τ=2，ε=0.1，子空间秩 r=64，压缩阈值 p20，λ=(1,0.5,0.5,0.1,0.2)，seq_len=2048，bs=32。

### 3.3 训练流程（优化器/学习率/批次/调度/FSDP 或 QLoRA 并行方案）

- 优化器 AdamW，student 全参微调 lr=5e-5（decay 0.1，warmup 500 步，cosine），student 路由/专家无关层 lr=3e-5；
- batch：global bs=32（2 卡 × grad_accum 8 × per-gpu 2），seq_len 2048；teacher 前向 batch 与 student 错峰（teacher 前向 1 批 → student 训练 4 步，避免显存竞争）；
- 步数：~40k 步（80k 数据 × 1 epoch）；每 2k 步评估；
- 并行：teacher INT4 驻 1 卡（或 2 卡 TensorParallel），student FSDP 2 卡；bf16 + checkpoint；
- 压缩阶段：训练-free 分析 1 天，微专家剪枝/量化后再微调恢复 2 天。

### 3.4 推理与评测流程

- 推理：student 合并/量化后跑 7 基准；对比 teacher（全精度 + INT4）作为上界与下界；
- 评测指标：MMLU（5-shot）、GSM8K（8-vote）、HumanEval（pass@1）、BBH（3-shot CoT）、激活参数/总参数比、推理吞吐（token/s）；
- 路由指标：专家覆盖度、负载均衡熵、max load、死专家率（<0.01% 激活）。

## 4. 数据集细节（来源/许可/划分/预处理）

| 数据集 | 来源 | 许可 | 划分 | 预处理 |
|---|---|---|---|---|
| UltraChat | HF stingning/ultrachat | 自定义（研究） | 40k 子集 | 对话转指令对 |
| OpenWebMath | HF open-web-math | ODC-By | 24k 子集 | 清洗，取短文 |
| Flan（arXiv:2109.01652） | HF flan-v2 | 开放 | 16k 子集 | 指令模板 |
| MATH（arXiv:2103.03874） | HF lighteval/MATH | MIT | 8k 子集 + API CoT | CoT 由 DeepSeek V4 生成 |
| 评测：MMLU/GSM8K/HumanEval/BBH | 标准 HF | 各许可 | 仅评测 | 标准 prompt |

预处理：统一 tokenizer（OLMoE 的 tokenizer）；pack 到 2048；路由标签（teacher 的专家激活）在数据管线中提前缓存（离线生成，避免训练时重复 teacher 前向）。

## 5. 基线复现（基线列表+官方代码地址；复现步骤与预期指标表；统一评测口径）

| 基线 | 官方实现 | 复现要点 |
|---|---|---|
| MoE→Dense KD（MiniLLM，arXiv:2306.08543） | 论文重写 | 用 Mixtral 蒸馏 OLMoE（当作 dense 处理，去掉路由损失） |
| MoE→MoE（B-Distill，ML V·63） | 论文重写 | 整专家 logit + 路由 + 蒙特卡洛覆盖 |
| 专家合并（Sub-MoE，arXiv:2506.23266） | 论文重写 | 对 Mixtral 直接子空间合并 |
| 专家剪枝（CAMERA，ML IX·68） | 论文重写 | 微专家剪枝，训练-free |
| QLoRA 直接微调小 MoE | HF PEFT | 4-bit 量化 student 微调，无蒸馏 |

**预期指标表**：

| 方法 | 平均 acc | 激活参数比 | 专家覆盖度 | 路由平衡熵 |
|---|---|---|---|---|
| Mixtral teacher (全精度) | 70.0 | – | – | – |
| Mixtral teacher (INT4) | 68.5 | – | – | – |
| OLMoE-1B 基座（未训练） | 35.0 | 0.4B | 0.3 | 低 |
| MoE→Dense KD | 45.0 | 0.4B | – | – |
| B-Distill | 52.0 | 0.4B | 0.8 | 中 |
| Sub-MoE（无训练） | 40.0 | – | – | – |
| CAMERA 剪枝 | 38.0 | 0.3B | – | – |
| QLoRA 直接微调 | 47.0 | 0.4B | – | – |
| **MEKD（蒸馏）** | **55.0** | 0.4B | **1.0** | 高 |
| **MEKD（蒸馏+压缩）** | **53.0** | **0.28B** | 1.0 | 高 |

> 预估值用于验收；统一口径：MMLU/GSM8K/HumanEval/BBH 平均 acc。

## 6. 实验矩阵（A/B/C…：主实验、消融、鲁棒性、泛化性）

- **A（主实验）**：MEKD 蒸馏 vs 6 条基线；MEKD 蒸馏+压缩闭环 vs 直接压缩；
- **B（消融）**：
  - B1 去掉微专家隐层对齐（仅整专家 logit+路由）＝退化到 B-Distill+；
  - B2 去掉蒙特卡洛覆盖（ε=0）；
  - B3 去掉熵感知路由平衡；
  - B4 子空间秩 r∈{32,64,128}；
  - B5 压缩阈值 p∈{p10,p20,p30}；
  - B6 数据规模 {40k,80k,120k}；
- **C（鲁棒性）**：teacher 换 DeepSeek-MoE-16B（可用 API 或 HF 模型）或 Qwen-MoE；teacher 量化位宽 {FP16,INT8,INT4}；随机种子×2；
- **D（泛化性）**：student 换 Qwen2.5-1.5B（dense）验证"微专家对齐"在 dense student 上的泛化（对比 MoE student）；跨域评测（新增 MBPP/MMLU-Pro）。

## 7. 评测协议（指标定义、均值±方差、显著性检验、随机种子）

- 指标：下游 acc（4 基准平均）、激活参数/总参比、专家覆盖度（teacher 专家被采样比例）、路由负载均衡（熵、max load）、微专家对齐度（cosine）、推理吞吐（token/s）、压缩率；
- 主实验 2 种子（算力受限）但补 bootstrap；均值±方差报告；
- 显著性：配对 bootstrap（1000 次），p<0.05；
- 随机种子 {42,7}；生成温度：评测统一 greedy（GSM8K 8-vote 用 t=0.7）。

## 8. 算力与资源计划（4×L40 分阶段 GPU·天；存储；API 用量与成本估算）

| 阶段 | 内容 | GPU·天 |
|---|---|---|
| P1 数据合成 + teacher 缓存（logit/路由标签） | teacher INT4 1–2 卡 × 3 天 | 5 |
| P2 主实验蒸馏 A | student 2–3 卡 FSDP，40k 步 | 12 |
| P3 消融 B | 6 组 × 20k 步 | 10 |
| P4 压缩闭环 + 恢复微调 | 1 天分析 + 2 天微调 | 3 |
| P5 鲁棒/泛化 C/D | 4 天 | 5 |
| **合计** | | **35（预算 30–40）** |

- 存储：Mixtral INT4 24GB + OLMoE 2GB + teacher 缓存数据（logit/路由）约 40GB + checkpoint 30GB ≈ 100GB；
- API：MATH CoT 8k + UltraChat/OpenWebMath 清洗后补全 + judge 评测 2k ≈ 12k 次，DeepSeek Pro 为主 ~$15–30；Kimi K2.6 对照 4k 次。

## 9. 里程碑与时间线（按周，单人+4 卡）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | teacher INT4 部署 + 数据管线 + 缓存脚本 | teacher 缓存 ready |
| W2 | 微专家对齐 + 路由损失实现 | student 训练通过 |
| W3 | 蒙特卡洛覆盖 + 熵平衡实现 | 消融开关齐全 |
| W4 | 主实验 A + 基线复现 | 指标表 |
| W5 | 消融 B | 消融表 |
| W6 | 压缩闭环 + 鲁棒 C/D | 压缩曲线 + 泛化表 |
| W7 | 论文初稿 + 复跑 | 投稿稿 |

## 10. 风险与备选方案（表）

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 微专家对齐对 SVD 数值敏感 | 中 | 高 | 子空间秩网格（B4）；对齐 loss 加 warm-up；改用 Frobenius 投影替代硬 SVD |
| 2B student 与 47B teacher 容量鸿沟 | 中 | 高 | 加 KL 下界约束（θ 约束）；数据缩放实验（B6）；先用 80k 数据 |
| teacher/student 同驻 4 卡显存竞争 | 高 | 中 | 分阶段调度（teacher 缓存离线，训练期不驻留）；错峰前向 |
| CAMERA 式分析在 student 上不适用 | 低 | 中 | 退化为整专家剪枝 + 恢复微调 |
| API 合成数据质量波动 | 低 | 中 | 过滤规则 + 5% judge 抽检 |

## 11. 论文写作计划（目标会议/截稿日期、差异化卖点、图表清单、相关工作覆盖）

- 目标：NeurIPS 2026（高优先级）；差异化卖点：MoE→MoE 蒸馏的"微专家粒度 × 子空间对齐 × 路由平衡"三合一 + 蒸馏→压缩闭环，与收藏论文（B-Distill/Sub-MoE/CAMERA）形成最清晰互补；
- 图表清单：Fig1 方法图（微专家对齐 + 蒙特卡洛 + 熵平衡 + 压缩闭环）；Fig2 专家覆盖度曲线；Fig3 负载均衡 vs 训练步；Fig4 压缩-质量曲线；Tab1 主实验；Tab2 消融；Tab3 鲁棒/泛化；
- 相关工作覆盖：MiniLLM（arXiv:2306.08543）、Sub-MoE（arXiv:2506.23266）、CAMERA（ML IX·68）、B-Distill（ML V·63）、MoEfication（arXiv:2110.01786）、DeepSeekMoE（arXiv:2401.06066）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）

- MiniLLM arXiv:2306.08543；Sub-MoE arXiv:2506.23266；OLMoE arXiv:2409.02060
- DeepSeekMoE arXiv:2401.06066；Mixtral of Experts arXiv:2401.04088
- AWQ arXiv:2306.00978
- UltraChat（Enhancing Chat LMs, arXiv:2305.14233）；Flan（Finetuned LMs Are Zero-Shot Learners, arXiv:2109.01652）
- MMLU arXiv:2009.03300；GSM8K arXiv:2110.14168；MATH arXiv:2103.03874；HumanEval arXiv:2107.03374；BBH arXiv:2210.09261
- 收藏论文：B-Distill（Machine Learning V·论文 63，AAAI 2026，无 arXiv）、CAMERA（Machine Learning IX·论文 68，AAAI 2026，无 arXiv）
