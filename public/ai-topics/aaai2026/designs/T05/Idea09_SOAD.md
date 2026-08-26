# 实验设计书：Idea 9 · SOAD —— 状态自适应在线蒸馏

> State-Aware On-policy Adaptive Distillation with Token-Difficulty Curriculum
> 目标会议：ACL 2027（中高优先级）
> 硬件：4 × NVIDIA L40（192GB）｜API：DeepSeek V4 Pro/Flash、Kimi K2.6

---

## 0. 摘要

SOAD 在 GKD 式 on-policy 蒸馏（arXiv:2306.13649）中，按**学生自身学习状态**动态调节每个 token 的蒸馏强度与训练课程：以学生 loss 稳定性/预测熵为在线信号，难 token 低温纠错、易 token 高温平滑（比 AdaKD 离线温度更实时）；课程进度由学生自身学习曲线驱动而非固定 epoch；teacher 为 DeepSeek V4 API（黑盒兼容，logit 不可见时用输出分布采样），支持多 rationale 注入（GrayKD 思想）。以 1B–3B student（Qwen2.5-1.5B/3B）蒸馏，预计 20–30 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

从大 teacher（DeepSeek V4，黑盒 API）蒸馏到小 student（1–3B）时，经典方法存在三组缺陷：
1. **GKD**（arXiv:2306.13649）on-policy 但温度/权重静态；
2. **AdaKD**（NLP V·50）token 温度离线、未用学生在线状态；
3. **MiniLLM**（arXiv:2306.08543）反向 KL 无课程调度；
4. **GrayKD**（NLP III·26）黑盒 rationale 注入但 teacher mode 固定、多 rationale 成本高。

目标：把"学生状态感知调制 + 难度课程 + on-policy 黑盒兼容"三者统一进一个在线蒸馏框架。

### 1.2 相关工作不足（收藏论文用「(Session·论文N·英文标题)」格式；外部文献保留真实 arXiv ID/DOI）

- **GKD**（arXiv:2306.13649）：on-policy，静态温度/权重；
- **MiniLLM**（arXiv:2306.08543）：反向 KL，无课程；
- **AdaKD**（NLP V·50）：token 自适应聚焦 + 逆难度温度，离线调度、单指标；
- **GrayKD**（NLP III·26）：黑盒多 rationale 注入，teacher mode 固定、开销大；
- **STaR**（arXiv:2203.14465）：自我引导推理增强，非在线蒸馏；
- **Distilling System 2 into System 1**（arXiv:2407.06023）：长 CoT→短 CoT 蒸馏，无学生状态调制。

空白：**在线学生状态调制 × 学习曲线驱动课程 × on-policy 黑盒 teacher** 的联合蒸馏。

### 1.3 为什么是现在、为什么你的环境适合做

- GKD 之后 on-policy 蒸馏是主流方向，AdaKD 的 token 温度提供直接对比；
- 3B student 2 卡 FSDP + API teacher 并行，20–30 GPU·天可完成；
- DeepSeek V4 提供 logit 不可见时的输出分布采样（黑盒兼容），Kimi K2.6 可作独立 teacher 交叉对比。

## 2. 研究目标与可验证假设（2-4 条，每条给出"成立时的可观测结果"）

- **H1（状态调制增益）**：在线学生状态调制优于静态温度/离线温度。
  - 可观测结果：下游 acc（MMLU 子集/GSM8K/BBH）比固定温度与 AdaKD 离线温度高 ≥1.5pt；调制前后的 token 难度与 loss 更对齐（相关性更高）。
- **H2（课程调度增益）**：学习曲线驱动的课程优于固定 epoch 课程。
  - 可观测结果：课程版在相同训练步数下 acc 更高（或达到相同 acc 步数更少）；收敛曲线更平滑。
- **H3（on-policy 有效）**：学生自采样蒸馏优于离线 teacher 数据。
  - 可观测结果：on-policy 版 acc 高于 off-policy（同数据预算）≥1pt；分布对齐 KL 更低。
- **H4（黑盒兼容）**：只用 API 输出分布（无 logit）时性能下降可控。
  - 可观测结果：输出分布采样（temperature 采样）的 acc 比白盒 logit 蒸馏下降 ≤1pt；多 rationale 注入进一步回收。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

**训练数据**：Alpaca（Self-Instruct，arXiv:2212.10560）、UltraChat（arXiv:2305.14233 对应论文）、MathInstruct（对应 MAmmoTH，arXiv:2309.05653，含 CoT）、CodeAlpaca。总量 ~40k 条，比例 4:3:2:1。

**API 多 rationale（DeepSeek V4 Pro）**：
- 对每样本生成 2–3 条 rationale（不同温度/提示角度：`请用另一种思路解答`）；
- rationale 作为 teacher 输出注入（GrayKD 思想，NLP III·26）；
- 过滤：最终答案一致才保留；长短 rationales 各 1 条。

**难度分桶（课程用）**：用 IFD 式难度（arXiv:2308.12032）或 DeepSeek V4 难度标签将样本分 3 桶（易/中/难）。

**数量**：训练 40k；验证 2k；课程桶 3 档。

### 3.2 模型/算法设计（模块拆解、关键公式、超参数初值）

**teacher**：DeepSeek V4（API），黑盒——logit 不可见时，用输出分布采样：对同一输入采样 K 次（K=8）近似分布，或直接作为目标序列；
**student**：Qwen2.5-1.5B / Qwen2.5-3B（对比规模）。

**蒸馏目标**：

`L = CE(正确答案) + λ(t)·KL_rev(π_student ∥ π_teacher) + L_curriculum`

- `λ(t) = λ0·(1 − loss_stab(t))`：λ 随学生 loss 稳定性上升（loss 波动大→λ 小，先学任务；稳定→λ 大，加强蒸馏）；
- token 级调制：`τ(tok) = τ0 · (1 + Δ(s_tok))`，s_tok 为学生对 token 的预测熵/置信度；
  - 难 token（学生置信低）：低温 τ<τ0 → 强制对齐 teacher（纠错）；
  - 易 token（置信高）：高温 τ>τ0 → 平滑（泛化）；
  - 与 AdaKD 逆难度温度相反：AdaKD 难 token 低温纠错、易 token 高温泛化，本方案把温度改为**在线学生状态驱动**而非离线标签驱动；
- `L_curriculum`：课程调度——按学生能力（验证集 acc 或 loss）达标进入下一难度桶；能力函数 `cap = 1/val_loss`；
- 学生自采样 on-policy：student 采样响应 → teacher 对响应打分（黑盒 judge 或输出分布）→ 蒸馏。

**超参数初值表**：λ0=1.0，τ0=2.0，Δ∈[−0.5, 0.5]，K=8（分布近似），EMA 温度平滑（EMA α=0.9），课程达标阈值 acc>60%，梯度裁剪 1.0。

### 3.3 训练流程（优化器/学习率/批次/调度/FSDP 或 QLoRA 并行方案）

- 优化器 AdamW，student 全参微调 lr=5e-5，warmup 500，cosine；蒸馏分支 lr=3e-5；
- batch：global bs=32（2 卡 FSDP），seq_len 2048；teacher API 并行采样（异步预缓存）；
- 步数：~30k 步（40k 数据）；每 2k 步评估；
- on-policy 采样：每 500 步用当前 student 采样一批（K=8），送入 teacher 打分；
- 多 rationale：teacher mode（cross-attention 注入）+ student mode（softmax 知识）两阶段（GrayKD 式），teacher mode 只用于前 10% 步（省 API）；
- bf16 + gradient checkpointing；EMA 温度状态存 buffer。

### 3.4 推理与评测流程

- 评测：MMLU 子集（取 4 个子领域 2k 条）、GSM8K、BBH（arXiv:2210.09261）全量；
- 指标：下游 acc、分布对齐 KL（student∥teacher）、温度调制收益（vs 固定温度）、课程收益、on/off-policy 对比、teacher 调用次数成本。

## 4. 数据集细节（来源/许可/划分/预处理）

| 数据集 | 来源 | 许可 | 划分 | 预处理 |
|---|---|---|---|---|
| Alpaca（Self-Instruct，arXiv:2212.10560） | HF tatsu-lab | CC-BY | 16k 子集 | 对话 |
| UltraChat（arXiv:2305.14233） | HF | 自定义 | 12k 子集 | 对话 |
| MathInstruct（MAmmoTH，arXiv:2309.05653） | HF | 开放 | 8k 子集 | CoT |
| CodeAlpaca | HF | 开放 | 4k 子集 | 代码 |
| 评测：MMLU/GSM8K/BBH | 标准 | 各许可 | 评测 | 标准 prompt |

预处理：统一 chat 模板；难度桶标注；rationale 拼接；答案规则化。

## 5. 基线复现（基线列表+官方代码地址；复现步骤与预期指标表；统一评测口径）

| 基线 | 官方实现 | 复现要点 |
|---|---|---|
| SFT-only | HF | 无蒸馏 |
| logit-KD（Hinton，arXiv:1503.02531） | 论文重写 | 白盒 logit KL |
| GKD（arXiv:2306.13649） | 论文重写 | on-policy 静态温度 |
| MiniLLM（arXiv:2306.08543） | 论文重写 | 反向 KL |
| AdaKD（NLP V·50） | 论文重写 | 离线 token 难度温度 |
| GrayKD（NLP III·26） | 论文重写 | 多 rationale 注入 |

**预期指标表（Qwen2.5-3B student，多基准平均 acc）**：

| 方法 | 平均 acc | 与 teacher 分布 KL↓ | teacher 调用（×样本） |
|---|---|---|---|
| SFT-only | 50.0 | 0.8 | – |
| logit-KD | 54.0 | 0.5 | 1 |
| GKD | 55.0 | 0.45 | 2 |
| MiniLLM | 55.5 | 0.4 | 2 |
| AdaKD | 56.0 | 0.38 | 1 |
| GrayKD | 56.5 | 0.35 | 3 |
| **SOAD** | **58.0** | **0.30** | **2.2** |

> 预估值；口径：MMLU 子集+GSM8K+BBH 平均。

## 6. 实验矩阵（A/B/C…：主实验、消融、鲁棒性、泛化性）

- **A（主实验）**：SOAD vs 6 条基线，1.5B/3B 两个 student；
- **B（消融）**：
  - B1 去掉状态调制（固定 λ、τ）；
  - B2 去掉课程调度（固定顺序）；
  - B3 off-policy（不用学生采样）；
  - B4 白盒 logit vs 黑盒采样；
  - B5 多 rationale 注入开关；
  - B6 λ/τ 超参网格；
- **C（鲁棒性）**：学生规模 {1B,1.5B,3B}；数据量 {20k,40k,80k}；seed×3；API teacher 延迟模拟（异步/同步）；
- **D（泛化性）**：跨域（代码/数学/对话）；teacher 换 Kimi K2.6（交叉蒸馏对比）；长 CoT 蒸馏（Distilling System 2 into System 1 式，arXiv:2407.06023）。

## 7. 评测协议（指标定义、均值±方差、显著性检验、随机种子）

- 指标：下游 acc、分布对齐 KL、温度调制收益、课程收益、on/off-policy 对比、teacher 调用成本（次数与 $）、loss 稳定性；
- 主实验 3 seed；均值±方差；配对 bootstrap p<0.05；
- 随机种子 {42,7,2026}；生成 greedy。

## 8. 算力与资源计划（4×L40 分阶段 GPU·天；存储；API 用量与成本估算）

| 阶段 | 内容 | GPU·天 |
|---|---|---|
| P1 数据准备 + rationale + 难度桶 | API 3 天（GPU 0.5） | 0.5 |
| P2 主实验 A（2 student × 30k 步） | 2 卡 × 8 天 | 16 |
| P3 消融 B | 6 组 × 2 天 | 6 |
| P4 鲁棒 C + 泛化 D | 6 天 | 6 |
| **合计** | | **28.5（预算 20–30）** |

- 存储：teacher 缓存 + student ×2 + 数据 ≈ 40GB；
- API：multi-rationale 40k + on-policy 采样 8k×K8 + judge ≈ 110k 次（大量 Flash 采样，Pro 生成 rationale），~$40–100；Kimi K2.6 交叉 5k。

## 9. 里程碑与时间线（按周，单人+4 卡）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 数据 + rationale + 课程桶 | 数据 ready |
| W2 | 状态调制 + 课程实现 | 可训练 |
| W3 | on-policy 采样管线 | 端到端 |
| W4 | 主实验 A + 基线 | 指标表 |
| W5 | 消融 B | 消融表 |
| W6 | 鲁棒 C + 泛化 D + 论文 | 初稿 |

## 10. 风险与备选方案（表）

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| on-policy 采样分布偏移导致 loss 不稳定 | 中 | 高 | EMA 温度、梯度裁剪；采样前对 student 做 KL 温度上限；退化为固定批次采样 |
| API teacher 延迟影响训练吞吐 | 高 | 中 | 预缓存高价值 teacher 响应；异步队列；离线预生成 teacher 数据（off-policy 兜底） |
| 黑盒分布近似（K=8 采样）有偏 | 中 | 中 | K 增大到 16；用 teacher 采样分数加权；多 rationale |
| 课程调度过慢/过快 | 中 | 中 | 课程阈值自适应（能力函数）；达标即进下一桶，失败回滚 |
| 1B student 容量不足以对齐 | 中 | 中 | 3B 为主实验，1B 作缩放消融 |

## 11. 论文写作计划（目标会议/截稿日期、差异化卖点、图表清单、相关工作覆盖）

- 目标：ACL 2027（中高优先级）；差异化卖点：首个把"在线学生状态调制 + 学习曲线驱动课程 + on-policy 黑盒兼容"统一进蒸馏框架，直接补 GKD 静态温度、AdaKD 离线温度的空白；
- 图表清单：Fig1 方法图（状态调制 + 课程 + on-policy）；Fig2 token 温度随训练变化（可视化）；Fig3 学习曲线（课程 vs 固定）；Fig4 分布对齐 KL；Fig5 跨 teacher 对比；Fig6 调用成本；Tab1 主实验；Tab2 消融；
- 相关工作：GKD（arXiv:2306.13649）、MiniLLM（arXiv:2306.08543）、logit-KD（arXiv:1503.02531）、STaR（arXiv:2203.14465）、System2→System1（arXiv:2407.06023）、IFD（arXiv:2308.12032）、收藏论文（NLP V·50、NLP III·26）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）

- GKD arXiv:2306.13649；MiniLLM arXiv:2306.08543；logit-KD（Hinton）arXiv:1503.02531
- STaR arXiv:2203.14465；Distilling System 2 into System 1 arXiv:2407.06023
- Self-Instruct arXiv:2212.10560；UltraChat arXiv:2305.14233；MAmmoTH arXiv:2309.05653；IFD arXiv:2308.12032
- MMLU arXiv:2009.03300；GSM8K arXiv:2110.14168；BBH arXiv:2210.09261
- 收藏论文：AdaKD（Natural Language Processing V·论文 50，AAAI 2026，无 arXiv）、GrayKD（Natural Language Processing III·论文 26，AAAI 2026，无 arXiv）
