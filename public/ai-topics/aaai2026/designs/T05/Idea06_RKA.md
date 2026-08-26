# 实验设计书：Idea 6 · RKA —— 推理-缓存耦合的 KV 释放管理

> Reasoning-KV Coupled Memory Management with Early-Exit-Aware KV Release
> 目标会议：ACL 2027（中优先级）
> 硬件：4 × NVIDIA L40（192GB）｜API：DeepSeek V4 Pro/Flash、Kimi K2.6

---

## 0. 摘要

RKA 让 CoT 预算控制与 KV cache 释放联动：训练一个小判断器（0.3B）识别"已完成/冗余推理块"，判定后对该块 KV 激进 merge/evict（复用 KeepKV 补偿），同时保住结论性推理步，与 SABER 式预算层组合实现"预算紧→更激进释放"。以 DeepSeek V4 生成长 CoT 并 LLM-as-judge 标注必要/冗余步骤作为训练信号，DeepSeek-R1-Distill-7B 推理评测，首次以"吞吐 × 内存 × acc"三方权衡为主指标统一建模推理预算与 KV 内存。预计 8–12 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

长 CoT 推理（R1 系，如 DeepSeek-R1，arXiv:2501.12948）带来两个耦合瓶颈：
1. **token 预算**：模型"overthink"，简单题也输出长 CoT（见 Overthinking 分析，arXiv:2604.10739）；
2. **KV 内存峰值**：CoT 每步都产生 KV，长 CoT 推理时 KV 缓存占满显存。

现有 CoT 压缩（TIV/SABER/MARS/PI）只减少生成 token 数，不解决 KV 峰值；KV 压缩（KeepKV/H2O）不感知推理阶段，早期验证性推理步仍占满 KV。两者在长 CoT 推理下是真实瓶颈，但从未统一建模。

### 1.2 相关工作不足（收藏论文用「(Session·论文N·英文标题)」格式；外部文献保留真实 arXiv ID/DOI）

- **TIV**（NLP I·22）：向量注入思维压缩 token 推理，两阶段 RL，不解决 KV；
- **SABER**（NLP VI·54，DOI:10.1609/aaai.v40i41.40799）：预算画像 + 长度感知奖励 + 四档推理，减 token 不减 KV 峰值；
- **MARS**（NLP VI·8）：三阶段多模态 CoT 压缩，无 KV 耦合；
- **PI**（NLP V·67，arXiv:2508.02511）：测试时提示干预，无训练；
- **KeepKV**（NLP IV·67，arXiv:2504.09936）：合并式 KV 压缩，不感知推理阶段；
- **H2O**（arXiv:2306.14048）：启发式 KV 逐出，不感知推理阶段；
- **InTRO**（NLP VI·81，arXiv:2511.09865）：token 级修正因子，用于推理质量，未与 KV 管理结合。

空白：**推理阶段感知的 KV 释放 × CoT 预算控制的耦合**（预算-内存统一建模）。

### 1.3 为什么是现在、为什么你的环境适合做

- R1 系长 CoT 推理的 KV 瓶颈是当下热点（Overthinking arXiv:2604.10739、TBALR arXiv:2412.18547 均近 1–2 年）；
- 判断器 0.3B + 7B 推理在 4×L40 上绰绰有余，算力需求低（8–12 GPU·天）；
- DeepSeek V4 生成长 CoT + LLM-as-judge 标注，Kimi K2.6 交叉验证，API 管线成熟。

## 2. 研究目标与可验证假设（2-4 条，每条给出"成立时的可观测结果"）

- **H1（判断器有效）**：判断器能准确识别冗余推理块。
  - 可观测结果：判断器（P(redundant)）与 LLM-as-judge 标注的一致性（AUC ≥0.8）；冗余块判定的人类评审准确率 ≥80%。
- **H2（KV 释放有效）**：块级 KV 释放显著降峰值内存，不丢分。
  - 可观测结果：MATH/GSM8K/AIME 固定 acc 下峰值内存下降 ≥30%、吞吐提升 ≥40%（相对全缓存），acc 下降 ≤1pt。
- **H3（预算耦合增益）**：与 SABER 预算层组合时，预算紧→更激进释放，三方权衡更优。
  - 可观测结果：组合版在固定 acc 下吞吐×内存综合指标优于"各自独立"（SABER 单独 + KeepKV 单独）。
- **H4（块级 vs token 级）**：块级释放优于逐 token 释放。
  - 可观测结果：块级释放的 acc-吞吐权衡曲线严格在逐 token 之上。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

**数据生成（DeepSeek V4 Pro 为主）**：
- 长 CoT 样本：MATH/AIME 2024/GSM8K 题目，用 DeepSeek V4 Pro 生成 step-by-step CoT（temperature 0.6，max 4000 token），每样本 2 次采样；
- **LLM-as-judge 标注**：对 CoT 按"推理块"（每 1 步≈1 块，块边界 = 换行/步骤分隔符）标注必要/冗余，prompt：
  `以下推理过程按步骤划分。对每一步，判断其对得出最终答案是否必要，输出 [必要/冗余] 与一句理由。`
  DeepSeek V4 Flash 批量；Kimi K2.6 对 10% 子集做交叉验证（一致性 κ≥0.7 才保留）；
- 过滤：最终答案正确的样本才保留（避免把错误推理当"必要"）。

**数量**：MATH 3k + AIME 0.5k + GSM8K 3k ≈ 6.5k 样本，每样本 2 次采样 = 13k 长 CoT；标注 6.5k（每次采样标注）。

### 3.2 模型/算法设计（模块拆解、关键公式、超参数初值）

**推理块边界检测判断器**（0.3B，如 Qwen2.5-0.5B 或自训小模型）：
- 输入：每步的隐状态（取 last-token hidden）、预测分布熵、token 分布统计、位置；
- 结构：小 Transformer + 线性头，输出 P(redundant)∈[0,1]；
- 训练损失：BCE(P, y_judge)，y_judge 为 LLM-as-judge 标注。

**块级 KV 释放**：
- 推理时按块流式生成；每完成一个块，判断器打分；
- 若 P(redundant)>τ（τ 默认 0.6），将该块 KV 与前一必要块 merge（KeepKV 式零扰动合并，NLP IV·67）或 8× 稀疏存储（保留 top-1/8 位置）；
- 结论性推理步（块含 FInal Answer 或 P<1-τ）始终保留全精度 KV；
- 实现：复用 KeepKV 的 merge 补偿逻辑，避免注意力分布突变。

**与 SABER 预算层组合**：
- 预算层（NoThink/FastThink/CoreThink/DeepThink，SABER 思想）给出每样本 max_token 预算；
- KV 释放阈值 τ 按预算自适应：预算紧（FastThink）→ τ 降 0.15（更激进释放）；预算松 → τ 升 0.15；
- 统一损失：`L = L_acc(判断器) + λ·L_kv(内存惩罚)`。

**超参数初值表**：τ=0.6，稀疏度 8×，λ=0.5，预算自适应 τ±0.15，判断器 hidden 512，epoch 3。

### 3.3 训练流程（优化器/学习率/批次/调度/FSDP 或 QLoRA 并行方案）

- 判断器：单卡训练，AdamW，lr=1e-4，bs=64，seq_len 512（只取每步统计），~20k 步，2–3 天；
- 数据生成：DeepSeek V4 API 为主（3 天，可并行），GPU 用于 R1-Distill-7B 推理验证；
- 推理评测：R1-Distill-7B 在 4 卡上并行（或 2 卡 + 判断器 1 卡）；
- 与 SABER 组合：先用论文复现的 SABER 预算层（FastThink 档）叠加；
- bf16；判断器无 FSDP 需求。

### 3.4 推理与评测流程

- 推理：R1-Distill-7B + 判断器联合流式生成；每块打分 → 释放/保留 KV；
- 评测基准：MATH、GSM8K、AIME 2024、LiveBench-Reasoning、DROP（arXiv:1903.00161）；
- 指标：固定 acc 下的吞吐提升、峰值内存下降、τ 敏感性、判断器一致性、长 CoT（>2k token）压测。

## 4. 数据集细节（来源/许可/划分/预处理）

| 数据集 | 来源 | 许可 | 划分 | 预处理 |
|---|---|---|---|---|
| MATH（arXiv:2103.03874） | HF lighteval/MATH | MIT | 训练 3k/评测 5k | 题目取测试集做推理 |
| AIME 2024 | 公开 | 开放 | 评测 30 | 标准格式 |
| GSM8K（arXiv:2110.14168） | HF openai/gsm8k | MIT | 训练 3k/评测 1.3k | 标准 |
| LiveBench-Reasoning | LiveBench 公开 | 开放 | 评测 | 推理子集 |
| DROP（arXiv:1903.00161） | HF | 开放 | 评测 | 阅读理解 |
| 长 CoT 训练集 | DeepSeek V4 API | 自建 | 6.5k 样本 | 按块切分 + 标注 |

预处理：块切分（按 `\n` 分隔步骤）；标注格式 JSON。

## 5. 基线复现（基线列表+官方代码地址；复现步骤与预期指标表；统一评测口径）

| 基线 | 官方实现 | 复现要点 |
|---|---|---|
| 全缓存 | – | 无 KV 释放 |
| KeepKV（NLP IV·67，arXiv:2504.09936） | 论文重写 | Electoral Votes + 零扰动合并 |
| H2O（arXiv:2306.14048） | HF | 启发式重击逐出 |
| SABER（NLP VI·54，DOI:10.1609/aaai.v40i41.40799） | 论文重写 | 预算画像 + FastThink 档 |
| TIV（NLP I·22）+ KeepKV | 论文重写 | 向量注入 + KV 合并（组合） |

**预期指标表（R1-Distill-7B，MATH 固定 acc 下）**：

| 方法 | acc | 峰值内存（相对全缓存） | 吞吐提升 |
|---|---|---|---|
| 全缓存 | 62.0 | 1.0× | 1.0× |
| KeepKV | 61.2 | 0.55× | 1.8× |
| H2O | 60.5 | 0.6× | 1.7× |
| SABER(FastThink) | 61.0 | 0.9× | 1.5× |
| TIV+KeepKV | 61.5 | 0.55× | 1.8× |
| **RKA** | **61.8** | **0.40×** | **2.2×** |
| **RKA+SABER** | **61.5** | **0.32×** | **2.5×** |

> 预估值；口径：MATH 固定 acc 下的内存与吞吐倍数。

## 6. 实验矩阵（A/B/C…：主实验、消融、鲁棒性、泛化性）

- **A（主实验）**：RKA vs 5 条基线，5 个基准；
- **B（消融）**：
  - B1 释放策略 merge vs 8× 稀疏 vs evict；
  - B2 阈值 τ ∈ {0.4,0.5,0.6,0.7}；
  - B3 判断器特征消融（去掉熵/位置/隐状态）；
  - B4 块边界 vs 逐 token 释放；
  - B5 预算自适应（开/关）；
  - B6 判断器训练数据量 {1k,3k,6.5k}；
- **C（鲁棒性）**：长 CoT 压测（>2k token）；不同难度分布；seed×3；判断器对 OOD 题目的错误释放影响；
- **D（泛化性）**：多模态长上下文（可选，接 Qwen2.5-VL）；跨模型（Llama-3.1-8B 推理）；与 Idea 7（TBC 预算控制器）组合验证全链路。

## 7. 评测协议（指标定义、均值±方差、显著性检验、随机种子）

- 指标：acc、峰值内存（GB/相对）、吞吐（token/s 或 sample/min）、判断器一致性（AUC/κ）、块释放比例、τ 敏感性曲线、长 CoT 压测；
- 固定 acc 下比较内存与吞吐（即"在同一 acc 水平"，可用多个 τ 插值对齐）；
- 主实验 3 seed；均值±方差；配对 bootstrap p<0.05；
- 随机种子 {42,7,2026}；生成 greedy + 少量投票。

## 8. 算力与资源计划（4×L40 分阶段 GPU·天；存储；API 用量与成本估算）

| 阶段 | 内容 | GPU·天 |
|---|---|---|
| P1 数据生成 + 标注 | API 为主，GPU 用于预判 0.5 | 0.5 |
| P2 判断器训练 | 1 卡 × 3 天 | 3 |
| P3 主实验 A | R1-Distill-7B + 判断器评测 | 3 |
| P4 消融 B + 鲁棒 C | 2 天 | 2 |
| P5 泛化 D（跨模型/长上下文） | 2 天 | 2 |
| **合计** | | **10.5（预算 8–12）** |

- 存储：R1-Distill-7B 16GB + 判断器 0.3B + CoT 数据 ~10GB ≈ 30GB；
- API：长 CoT 生成 13k 次（Pro）+ 标注 6.5k×2（Flash）+ Kimi 交叉 1.3k ≈ 23k 次，~$30–60（Pro 为主）。

## 9. 里程碑与时间线（按周，单人+4 卡）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | CoT 生成 + 标注管线 | 数据 + 标注 |
| W2 | 判断器训练 + 块检测 | checkpoint |
| W3 | 块级 KV 释放 + 稀疏存储实现 | 端到端可跑 |
| W4 | 主实验 A + 基线 | 指标表 |
| W5 | 消融 B + 鲁棒 C | 消融表 |
| W6 | 泛化 D + 论文 | 初稿 |

## 10. 风险与备选方案（表）

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 判断器对 OOD 问题错误释放导致丢分 | 中 | 高 | 保守阈值（默认 τ=0.6）；可回滚缓存（释放前存一份轻量备份）；若判断器失败退回全缓存 |
| 块边界检测在多模态/长上下文泛化差 | 中 | 中 | 先用文本数学验证；多模态作为可选扩展 |
| LLM-as-judge 标注噪声 | 中 | 中 | Kimi 交叉验证 κ≥0.7；标注一致性差的样本丢弃 |
| merge 补偿（KeepKV）数值不稳定 | 低 | 中 | 复用论文原实现；稀疏存储作为替代后端 |
| 与 SABER 组合的耦合收益不显著 | 中 | 中 | 放宽到"独立各自收益 + 简单拼接"，论文报告分解收益 |

## 11. 论文写作计划（目标会议/截稿日期、差异化卖点、图表清单、相关工作覆盖）

- 目标：ACL 2027（中优先级）；差异化卖点：首个把"推理阶段感知的 KV 释放"与"CoT 预算控制"统一建模的工作，提出"吞吐×内存×acc"三方权衡指标，补齐 TIV/SABER/MARS 与 KeepKV/H2O 之间的链路；
- 图表清单：Fig1 方法图（判断器 + 块级释放 + 预算耦合）；Fig2 acc-吞吐-内存三角图；Fig3 τ 敏感性曲线；Fig4 长 CoT 压测；Fig5 判断器一致性可视化；Tab1 主实验；Tab2 消融；
- 相关工作：SABER（NLP VI·54）、TIV（NLP I·22）、MARS（NLP VI·8）、PI（NLP V·67）、KeepKV（arXiv:2504.09936）、H2O（arXiv:2306.14048）、InTRO（arXiv:2511.09865）、DeepSeek-R1（arXiv:2501.12948）、TBALR（arXiv:2412.18547）、Overthinking（arXiv:2604.10739）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）

- DeepSeek-R1 arXiv:2501.12948；TBALR arXiv:2412.18547；Overthinking arXiv:2604.10739
- KeepKV arXiv:2504.09936；H2O arXiv:2306.14048；InTRO arXiv:2511.09865
- MATH arXiv:2103.03874；GSM8K arXiv:2110.14168；DROP arXiv:1903.00161
- 收藏论文：SABER（Natural Language Processing VI·论文 54，AAAI 2026，DOI:10.1609/aaai.v40i41.40799）、TIV（Natural Language Processing I·论文 22，AAAI 2026，无 arXiv）、MARS（Natural Language Processing VI·论文 8，AAAI 2026，无 arXiv）、PI（Natural Language Processing V·论文 67，AAAI 2026，arXiv:2508.02511）
