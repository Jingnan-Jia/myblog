# 实验设计书：Idea 7 · TBC —— 通用思考预算控制器

> Training a Universal Thinking-Budget Controller via Distilled Difficulty Labels and Length-Aware RL
> 目标会议：NeurIPS 2026（高优先级）
> 硬件：4 × NVIDIA L40（192GB）｜API：DeepSeek V4 Pro/Flash、Kimi K2.6

---

## 0. 摘要

TBC 从强模型 DeepSeek V4 蒸馏**连续难度标签**（0–10 分 + 最短可行推理参考），训练一个预算可连续调节的思考控制器：SFT 注入预算 token 与难度提示，再以 GRPO 式长度感知 RL（奖励 = 答案正确 + 按难度缩放的减长惩罚）+ 对基座的 reverse-KL 保留原推理能力。推理时用户/路由层输入任意预算 b∈[0,1]，模型按难度自适应分配 CoT 深度，无需离散档位。以 DeepSeek-R1-Distill-7B 为基座，预计 40–60 GPU·天（本主题最贵之二）。

## 1. 研究背景与动机

### 1.1 问题定义

CoT 推理"统一长度"是 overthinking 的根源（简单题也付全量成本，见 Overthinking 分析 arXiv:2604.10739）。已有方案用离散预算：SABER 预设四档（NoThink/FastThink/CoreThink/DeepThink）并逐模型 profile（NLP VI·54）；MARS 三阶段自制数据（NLP VI·8）；TIV 两阶段 RL（NLP I·22）。共同缺陷：
1. **离散、需人工 profile**，迁移成本高；
2. **难度信号粗糙**（用基座 token 用量画像，未利用强模型难度先验）；
3. **预算接口不连续**，无法按成本实时调节。

目标：一个"连续难度标签 × 连续预算接口"的通用思考预算控制器，一次训练、多粒度可用。

### 1.2 相关工作不足（收藏论文用「(Session·论文N·英文标题)」格式；外部文献保留真实 arXiv ID/DOI）

- **SABER**（NLP VI·54，DOI:10.1609/aaai.v40i41.40799）：离散四档 + 逐模型 profile，难度用 token 用量画像；
- **MARS**（NLP VI·8）：三阶段（CoT Masking→自适应难度指令→GRPO），自制 MART 数据、跨模型迁移差；
- **TIV**（NLP I·22）：向量注入思维 + 两阶段 RL，训练复杂、仅 3 基准；
- **PI**（NLP V·67，arXiv:2508.02511）：测试时提示干预，依赖人工先验，无训练；
- **TBALR**（arXiv:2412.18547）：token 预算感知推理，偏推理侧；
- **Scaling LLM Test-Time Compute**（arXiv:2408.03314）：预算扩展理论，无"难度先验 + 连续预算"控制器。

空白：**连续难度标签蒸馏 + 连续预算 RL 控制器**（替代离散档位与逐模型 profile）。

### 1.3 为什么是现在、为什么你的环境适合做

- R1 系 + o1 后 overthinking 是顶会热点；DeepSeek-V3/V4 与 R1（arXiv:2412.19437 / 2501.12948）证明"可验证奖励 RL 激励 CoT"路线有效；
- 7B 基座 + QLoRA（2 卡）+ GRPO（4 卡）在 4×L40 上可跑，属本主题算力上限但可行；
- DeepSeek V4 API 天然是"强模型难度先验"来源（难度标签 + 长短 CoT 对），一次合成可复用，成本可控。

## 2. 研究目标与可验证假设（2-4 条，每条给出"成立时的可观测结果"）

- **H1（连续预算可控）**：控制器能按任意预算 b∈[0,1] 调节输出长度，且 acc-长度帕累托优于离散档位。
  - 可观测结果：在 20%–80% 长度预算下，acc 高于 SABER 同长度四档插值 ≥1.5pt；长度-acc 曲线单调可控（预算 0.5 时长度≈基座 50%）。
- **H2（难度标签有效）**：DeepSeek V4 蒸馏的连续难度标签优于 token 用量画像。
  - 可观测结果：难度标签与 LLM-judge 一致性（Spearman ≥0.8）高于 token 用量画像；用难度标签训练的控制器在"难题保持 acc、易题大幅缩短"上更优。
- **H3（保留原推理能力）**：reverse-KL 保底使控制器在宽松预算下不劣于基座。
  - 可观测结果：预算 b=1.0 时 acc ≈ 基座（差 ≤1pt）；no-think 模式（b≈0）下 acc 保持 ≥基座零样本的 90%。
- **H4（跨域迁移）**：控制器在未见基准（AIME/ARC/DROP/LiveBench）上仍按预算调节。
  - 可观测结果：跨域长度-acc 帕累托曲线接近训练域（MATH/GSM8K），降幅 ≤10%。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

**数据生成（DeepSeek V4 Pro 为主）**：
- 训练：MATH + GSM8K 全量（MATH 7.5k 训练 + GSM8K 7.5k）；
- 每题三路输出：
  1. **短 CoT**：prompt `请用最简短的步骤解决，尽量少 token`（t=0.3）；
  2. **长 CoT**：`请详细逐步推理`（t=0.6）；
  3. **难度标签**：`请对问题难度打分 0–10（考虑推理步数与知识），只输出数字`（Flash，重复 2 次取均值）；
- **最短可行推理参考**：judge prompt 让 DeepSeek V4 判断短 CoT 是否"信息充分且正确"，输出 [可行/不可行]；
- 过滤：三路输出最终答案一致的样本才保留（答案不一致说明采样不稳定）。

**数量**：MATH 7.5k + GSM8K 7.5k = 15k 题，每题长短 CoT 各 1 条 + 难度标签；短 CoT 可行率目标 >70%，不可行样本重生成 1 次。

### 3.2 模型/算法设计（模块拆解、关键公式、超参数初值）

**基座**：DeepSeek-R1-Distill-7B（QLoRA 4-bit）。

**训练数据格式**：系统提示注入预算 token：
`你在{b}预算下解题（b=0 不思考直接答，b=1 充分思考）。难度估计：{d}。`

**SFT 阶段**：
- 用长短 CoT 对 + 预算 token 构造 SFT 样本；
- 损失：`L_SFT = CE(x, y; prompt(b,d))`，b 从 {0,0.25,0.5,0.75,1} 采样，难度 d 用标签；
- 学习模型：按难度调节长度的策略 π_θ(y|x,b,d)。

**RL 阶段（GRPO）**：
- 采样：对每组样本采 G=8 个响应；
- 奖励：`R = r_acc + λ(b,d)·ℓ(len)`：
  - r_acc = 1（答案正确）/ 0（错误）；
  - ℓ(len) = 按难度缩放的长度惩罚：`ℓ = (b·L_max − len)/L_max`（len<目标长度得正、超长得负）；
  - 权重：`λ(b,d) = α·(1−d/10)·(1−b)`（易题、紧预算 → 惩罚更大）；α=0.5；
- 损失：GRPO 式相对奖励（参照 DeepSeekMath/GRPO，arXiv:2402.03300）+ reverse-KL：
  `L_RL = −E[log π_θ(y|x)·A(y)] + β·KL(π_θ ∥ π_θ0)`，β=0.05，π_θ0 为基座冻结；
- reverse-KL 保底：防止 RL 后推理能力坍塌。

**推理接口**：
- 输入预算 b，采样时按难度动态调节：max_length = b·L_max(d)，并配合系统提示；
- no-think（b=0）：直接生成答案（默认带少量推理的 FastThink 档）；
- 支持连续 b∈[0,1]。

**超参数初值表**：GRPO G=8，α=0.5，β=0.05，b 采样 {0,0.25,0.5,0.75,1}，QLoRA r=16，lr=1e-5（RL）/2e-4（SFT），L_max=1024。

### 3.3 训练流程（优化器/学习率/批次/调度/FSDP 或 QLoRA 并行方案）

- 数据合成：DeepSeek V4 API 4 天（可并行，GPU 空闲）；
- SFT：QLoRA 2 卡，bs=32，3 epoch，~10k 步，2 天；
- RL（GRPO）：4 卡并行 FSDP，bs=128（16 组 × 8 采样），~5k 步，5 天；
- 策略更新：AdamW，lr=1e-5（SFT 后继续），cosine，warmup 200；
- 每 500 步评估长度-acc 帕累托；早停（帕累托退化即停）；
- bf16 + gradient checkpointing；FSDP shard QLoRA 参数。

### 3.4 推理与评测流程

- 评测基准：AIME（30 题）、ARC（arXiv:1803.05457）、DROP（arXiv:1903.00161）、LiveBench-Reasoning、GSM8K/MATH 测试集；
- 指标：固定长度预算下 acc（长度-acc 帕累托曲线）、no-think 退化、跨域泛化、RL 稳定性（奖励曲线）、长度压缩率。

## 4. 数据集细节（来源/许可/划分/预处理）

| 数据集 | 来源 | 许可 | 划分 | 预处理 |
|---|---|---|---|---|
| MATH（arXiv:2103.03874） | HF lighteval/MATH | MIT | 训练 7.5k/评测 5k | 长短 CoT 由 API 生成 |
| GSM8K（arXiv:2110.14168） | HF openai/gsm8k | MIT | 训练 7.5k/评测 1.3k | 同上 |
| AIME 2024 | 公开 | 开放 | 评测 30 | 标准 |
| ARC（arXiv:1803.05457） | HF | 开放 | 评测 | 多选 |
| DROP（arXiv:1903.00161） | HF | 开放 | 评测 | 阅读理解 |
| LiveBench-Reasoning | 公开 | 开放 | 评测 | 推理子集 |

预处理：统一 chat 模板 + 预算 token；长 CoT tokenize 截断 2048；答案规则化提取。

## 5. 基线复现（基线列表+官方代码地址；复现步骤与预期指标表；统一评测口径）

| 基线 | 官方实现 | 复现要点 |
|---|---|---|
| DeepSeek-R1-Distill-7B（基座） | HF | 无预算控制 |
| SABER（NLP VI·54，DOI:10.1609/aaai.v40i41.40799） | 论文重写 | 画像 + 长度奖励 + 四档 |
| MARS（NLP VI·8） | 论文重写 | 三阶段（自制数据） |
| TIV（NLP I·22） | 论文重写 | 向量注入 + 两阶段 RL |
| TBALR（arXiv:2412.18547） | 论文重写 | token 预算感知 |

**预期指标表（长度 40% 预算，MATH/GSM8K 平均 acc）**：

| 方法 | 长度压缩率 | acc（40% 预算） | acc（100% 预算） |
|---|---|---|---|
| R1-Distill-7B 基座 | 0% | – | 60.0 |
| SABER | 55% | 52.0 | 60.5 |
| MARS | 60% | 50.5 | 59.5 |
| TIV | 65% | 51.0 | 60.0 |
| TBALR | 50% | 53.0 | 60.0 |
| **TBC（b=0.4）** | **60%** | **55.0** | **60.5** |

> 预估值；口径：MATH+GSM8K 平均，同评测协议。

## 6. 实验矩阵（A/B/C…：主实验、消融、鲁棒性、泛化性）

- **A（主实验）**：TBC vs 5 条基线，长度-acc 帕累托；
- **B（消融）**：
  - B1 去掉 reverse-KL（β=0）→ 只 RL；
  - B2 去掉难度缩放（λ 用固定值）；
  - B3 难度标签用 token 画像替代（对比蒸馏标签）；
  - B4 预算 token 去掉（只 SFT 长短 CoT，无 RL 接口）；
  - B5 GRPO vs PPO 简化；
  - B6 基座换 Llama-3.1-8B；
- **C（鲁棒性）**：b 连续扫描 {0,0.2,0.4,0.6,0.8,1}；难度分布偏斜；seed×3；RL 训练稳定性（奖励方差）；
- **D（泛化性）**：跨域（AIME/ARC/DROP/LiveBench）；no-think 退化；与 Idea 6（RKA 的 KV 释放）组合验证全链路。

## 7. 评测协议（指标定义、均值±方差、显著性检验、随机种子）

- 指标：长度-acc 帕累托曲线（横轴平均生成长度，纵轴 acc）、长度压缩率、no-think 退化、跨域 acc、难度标签一致性（Spearman）、RL 奖励曲线、RL 稳定性（终态奖励 std）；
- 主实验 3 seed；均值±方差；配对 bootstrap p<0.05；
- 随机种子 {42,7,2026}；GRPO 采样 seed 固定。

## 8. 算力与资源计划（4×L40 分阶段 GPU·天；存储；API 用量与成本估算）

| 阶段 | 内容 | GPU·天 |
|---|---|---|
| P1 数据合成（长短 CoT + 难度标签） | API 4 天（GPU 0.5 预检） | 0.5 |
| P2 SFT | 2 卡 × 2 天 | 4 |
| P3 RL（GRPO） | 4 卡 × 5 天 | 20 |
| P4 主实验 A + 基线 | 3 天 | 3 |
| P5 消融 B + 鲁棒 C | 8 天 | 8 |
| P6 泛化 D + 组合 | 5 天 | 5 |
| **合计** | | **40.5（预算 40–60）** |

- 存储：R1-Distill-7B 16GB + QLoRA checkpoint ×3 + CoT 数据 15k×2 ~8GB ≈ 60GB；
- API：长短 CoT 30k 次 + 难度 30k 次（Flash）+ judge 15k ≈ 75k 次，Pro/Flash 混合 ~$40–90；Kimi K2.6 交叉 3k 次。

## 9. 里程碑与时间线（按周，单人+4 卡）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1–W2 | 数据合成 + 难度标签 | 数据 ready |
| W3 | SFT + 预算 token 实现 | SFT checkpoint |
| W4–W5 | RL（GRPO）+ reverse-KL | RL checkpoint |
| W6 | 主实验 A + 基线 | 帕累托曲线 |
| W7–W8 | 消融 B + 鲁棒 C | 消融表 |
| W9 | 泛化 D + 论文初稿 | 投稿稿 |

## 10. 风险与备选方案（表）

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 长度奖励与正确率奖励冲突导致退化 | 中 | 高 | 按难度缩放 λ + reverse-KL 保底；B1/B2 消融验证；GRPO 早停 |
| RL 4 卡 FSDP 不稳定 | 中 | 中 | 降 bs、加梯度裁剪；用 QLoRA + offload；必要时降 G=4 |
| 难度标签与真实难度相关性不足 | 中 | 中 | 与 LLM-judge/Human 对齐检验（Spearman）；合成数据重采样 |
| 跨域迁移差 | 中 | 中 | 训练数据加入 ARC/DROP 子集；D 泛化实验量化 |
| 算力超预算 | 高 | 高 | SFT-only 版本先投稿（仍含连续预算接口 + 难度标签贡献）；RL 缩到 3k 步 |

## 11. 论文写作计划（目标会议/截稿日期、差异化卖点、图表清单、相关工作覆盖）

- 目标：NeurIPS 2026（高优先级）；差异化卖点：首个"连续难度标签蒸馏 + 连续预算接口"的通用思考预算控制器，无需离散档位与逐模型 profile，长度-acc 帕累托超越 SABER/MARS/TIV；
- 图表清单：Fig1 方法图（难度标签 + SFT + GRPO + 连续预算）；Fig2 长度-acc 帕累托曲线；Fig3 难度-长度关系（控制器行为）；Fig4 no-think 退化；Fig5 跨域泛化；Fig6 RL 训练曲线；Tab1 主实验；Tab2 消融；
- 相关工作：DeepSeek-R1（arXiv:2501.12948）、GRPO/DeepSeekMath（arXiv:2402.03300）、SABER（NLP VI·54）、MARS（NLP VI·8）、TIV（NLP I·22）、TBALR（arXiv:2412.18547）、Scaling Test-Time Compute（arXiv:2408.03314）、Overthinking（arXiv:2604.10739）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）

- DeepSeek-R1 arXiv:2501.12948；DeepSeek-V3 arXiv:2412.19437；GRPO/DeepSeekMath arXiv:2402.03300
- TBALR arXiv:2412.18547；Scaling LLM Test-Time Compute arXiv:2408.03314；Overthinking arXiv:2604.10739
- MATH arXiv:2103.03874；GSM8K arXiv:2110.14168；ARC arXiv:1803.05457；DROP arXiv:1903.00161
- 收藏论文：SABER（Natural Language Processing VI·论文 54，DOI:10.1609/aaai.v40i41.40799）、MARS（Natural Language Processing VI·论文 8，AAAI 2026，无 arXiv）、TIV（Natural Language Processing I·论文 22，AAAI 2026，无 arXiv）、PI（Natural Language Processing V·论文 67，arXiv:2508.02511）
