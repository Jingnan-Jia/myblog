# 实验设计书：MME-SCI 驱动的闭环科学推理课程式强化学习

## 0. 摘要

本实验设计利用 MME-SCI（arXiv:2508.13938）的细粒度知识点标注构建「薄弱点诊断→自动出题→课程式 GRPO→再诊断」闭环：先按知识点建模型薄弱点图谱，用 DeepSeek V4 Pro 生成同知识点变体题（换数字/换题干/换语言，验证器过滤），按"先薄弱后一般"课程化 GRPO（奖励=答案匹配 + LLM judge）训练 7B MLLM，每轮评测更新图谱，并验证多语言迁移。目标是 MME-SCI 各学科/语言 ACC 系统提升（薄弱知识点 +10 点以上）、MathVista 保持、可归因到知识点。4×L40 约 9 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

MLLM 科学推理闭环训练：给定一个带细粒度知识点标注的多语言科学基准（MME-SCI），目标是系统性地提升模型在薄弱知识点上的表现，且不伤害其他能力。形式化：设知识点集 $K=\{k_1..k_M\}$，每知识点有题目集与当前模型准确率 $a_k$，定义课程 $\pi(K)$（薄弱优先），每轮用生成器 $G$ 产出同知识点变体题训练，用评测 $E$ 更新 $a_k$，直到收敛。

### 1.2 相关工作不足

- 收藏论文 `MME-SCI`（Computer Vision VIII·15·MME-SCI: A Comprehensive and Challenging Science Benchmark for Multimodal Large Language Models）：1019 题、多语言、3 评测模式、细粒度知识点标注（o4-mini 在"磁场"类 33 题中仅对 5 题）。但**只评测不闭环**，知识点归因未被用于针对性训练。
- `MathVista`（arXiv:2310.02255，ICLR 2024）与 `MMMU`（arXiv:2311.16502，CVPR 2024 Oral）数据量不足以做自举训练，且无知识点路由。
- 现有自举 RL 选题随机，无知识点先验；GRPO/DPO 类方法未利用"知识点薄弱点图谱"。

空白：**知识点薄弱点图谱 + 自动出题变体 + 课程式 RL 的评测-训练闭环**（评测论文的下半场）。

### 1.3 为什么是现在、为什么你的环境适合做

MME-SCI 2026 AAAI 刚中，其 1019 题规模小、正好适合做闭环研究（题目可控、可诊断、可防泄漏）；"评测→诊断→自适应训练→再评测"正是 2025–2026 MLLM 工作最缺的一环。本环境：7B 全参 FSDP 4×L40 可行；DeepSeek V4 Pro 出题/答案/解析、V4 Flash 批量 judge、Kimi K2.6 多语言校验都便宜；小规模 RL 3 轮课程约 6–10 GPU·天。

## 2. 研究目标与可验证假设

- **H1**：薄弱点路由优于随机选题。*可观测结果*：同算力下，薄弱点优先训练的 MME-SCI 总体 ACC 显著高于随机选题 RL（+3 点以上），且薄弱知识点提升 ≥10 点。
- **H2**：课程式（先薄弱后一般）优于均匀混合。*可观测结果*：课程式收敛更快、最终 ACC 更高（+1–2 点），MathVista 保持更稳。
- **H3**：自动出题 + 验证器可防简单化与泄漏。*可观测结果*：变体题合格率（抽检）≥80%，评测集与训练变体无重复（同题字面重合 <1%）。
- **H4**：多语言迁移。*可观测结果*：用中文/英文变体训练后，其他语言（法/西/日）的对应知识点 ACC 同步提升（≥5 点）。

## 3. 总体方法设计

### 3.1 数据流水线

- **初始评测**：用 MME-SCI 官方评测（Image-only 模式为主）跑一次全量，得每知识点 $a_k$，建薄弱点图谱 $K_{weak}=\{k: a_k<\theta\}$（$\theta=0.6$）。
- **自动出题（DeepSeek V4 Pro）**：
```
You are a science exam author. Given the knowledge point "{kp}", the original problem (text + answer), generate 10 variations:
- change numbers/units where possible; rephrase the stem; rewrite distractors; (a) same image or (b) no-image variant; keep the SAME knowledge point and answer format.
Also provide a step-by-step solution. Language: {lang}.
Original: {problem}
```
每知识点 ×10 变体；3 语言（中/英/法）各一批。
- **验证器过滤（V4 Flash + Kimi K2.6）**：(a) 答案与解析自洽校验；(b) 知识点不变校验；(c) 与 MME-SCI 原题去重（字面/语义相似度 <0.85）；(d) Kimi 检查多语言等价性（翻译一致性）。
- **数量**：MME-SCI 约 300 个知识点 × 10 变体 × 3 语言 ≈ **9k 训练变体**/轮；每轮重新生成（防过拟合）。

### 3.2 模型/算法设计

- **骨干**：7B MLLM（LLaVA-1.5-7B，arXiv:2310.03744；或 Qwen2-VL-7B 备选，arXiv:2409.12191）。
- **课程式 GRPO**：组内采样 $G$ 个响应（G=8），收益：
  $r=r_{exact}(答案匹配)+\beta\,r_{judge}(\text{V4 Flash 部分分})+\gamma\,r_{format}(格式/步骤完整)$，
  组归一化优势 $\hat{A}_i=(r_i-\text{mean}(r))/\text{std}(r)$，策略梯度 + KL(π‖π_ref) 正则（系数 0.01）。答案匹配先做字符串归一化（数字/单位/多选字母）。
- **知识点路由先验**：训练 batch 按 $w_k=1/(a_k+\epsilon)$ 加权抽样（薄弱优先）；课程安排：前 40% 轮次只采样 $K_{weak}$，后 60% 混入一般知识点。
- **防泄漏**：禁止同题复现；评测用官方固定划分；变体与官方题语义去重。

### 3.3 训练流程

- 全参 FSDP on 4×L40（或 QLoRA 降本），GRPO：batch=128（4 卡 × 32），gen 8 → 有效 1024；lr=5e-7（策略）/3e-6（value，若有）；每轮课程 = 2k 步。
- 3 轮课程循环：评测 → 更新图谱 → 生成新变体 → 训练。
- SFT 预对齐：先用一轮变体 SFT（1 epoch）稳定格式，再进 GRPO。

### 3.4 推理与评测流程

- 每轮评测：MME-SCI 三模式（Image-only/Text-only/Multi-modal）× 5 语言 × 每学科；旁路评测 MathVista/MMMU（防过拟合）。
- 输出：每知识点 ACC 增量表、学科×语言热图、课程收敛曲线。

## 4. 数据集细节

| 数据集 | 用途 | 说明 | 许可 |
|---|---|---|---|
| MME-SCI（arXiv:2508.13938） | 诊断 + 最终评测 | 官方 1019 题、知识点/语言标注 | 随论文 |
| **SciGen-Boot（本工作）** | 训练变体 | 每轮 9k 变体（3 语言 × 300 知识点 × 10） | 随论文开源（去重后） |
| MathVista（arXiv:2310.02255） | 防过拟合评测 | 官方 | 学术 |
| MMMU（arXiv:2311.16502） | 防过拟合评测 | 官方 val | 学术 |

## 5. 基线复现

| 基线 | 实现 | 复现要点 |
|---|---|---|
| 原模型（LLaVA-1.5-7B） | 官方权重 | 直接评测 |
| 均匀 SFT-only | 本流水线 | 变体全量均匀 SFT |
| 均匀 DPO | 本流水线 | 正负对（对/错答案）DPO |
| 无课程 RL（均匀 GRPO） | 本流水线 | 去掉薄弱点路由/课程 |
| 有课程 RL（Ours） | §3 全量 | 完整闭环 |

**预期指标表**（MME-SCI Image-only 总 ACC 与薄弱知识点提升，数值以复现为准）：

| 方法 | MME-SCI ACC↑ | 薄弱知识点 ACC↑ | MathVista↑/↓ | MMMU↑/↓ |
|---|---|---|---|---|
| 原模型 | 基准 | 基准 | 基准 | 基准 |
| 均匀 SFT-only | +小 | +小 | 持平/微降 | 持平/微降 |
| 均匀 DPO | +小 | +小 | 微降 | 微降 |
| 无课程 RL | +中 | +中 | 微降 | 微降 |
| **Ours（课程式）** | **+显著** | **+≥10 点** | **保持** | **保持** |

统一口径：同一 MME-SCI 官方脚本、同一温度（评测 T=0）、同一 judge 模板。

## 6. 实验矩阵

- **A. 主实验**：Ours 全闭环 vs 全部基线 × 3 模式 × 语言。
- **B. 消融**：B1 薄弱点路由 vs 随机；B2 课程 vs 均匀混合；B3 课程轮数（1/2/3）；B4 变体语言（单语/三语）；B5 奖励成分（exact/judge/format 各剔除）；B6 模型骨干（LLaVA vs Qwen2-VL）。
- **C. 鲁棒性**：C1 每题难度分层（易/中/难）增量；C2 语言迁移；C3 变体数量（5/10/15）。
- **D. 泛化性**：D1 MathVista/MMMU 保持；D2 跨学科迁移（数→物→化→生）；D3 下游科学 QA 微调。

## 7. 评测协议

- **指标**：MME-SCI 各学科/语言/模式 ACC；薄弱知识点 ACC；MathVista 官方 ACC；MMMU val ACC。
- **均值±方差**：3 个 RL 种子（123/2024/7），报 mean±std。
- **显著性**：主对比配对 Bootstrap（n=1000）；多语言一致性 Kappa。
- **防泄漏审计**：评测集与训练变体相似度审计报告（最小/均值）。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 初始评测 + 薄弱点图谱 + 出题（API） | 0.5 |
| P2 | SFT 预对齐（1 epoch） | 1 |
| P3 | GRPO ×3 轮（每轮 2k 步） | 6–9 |
| P4 | 每轮评测 + 最终全量评测（3 模式 × 5 语言 + 旁路） | 1.5 |
| **合计** | | **≈9–12** |

存储：MME-SCI + 变体 + 检查点 ≈ 60GB。API：DeepSeek V4 Pro 出题 ≈ $40；V4 Flash judge ≈ $20；Kimi K2.6 多语言校验 ≈ $12。总计 ≤ **$80**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | MME-SCI 获取 + 基线评测（原模型）+ 图谱建立 |
| W2 | 出题流水线 + 验证器 + 去重 + 抽检合格率 |
| W3 | SFT 预对齐 + GRPO 实现 + 轮1 |
| W4 | 轮2/轮3 + 图谱更新 + B1/B2 消融 |
| W5 | 全消融 + 语言迁移 + 防泄漏审计 + 论文初稿（NeurIPS 2026 截稿前 4 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 自举题目简单化 | 高 | 高 | 变体换题面 + 验证器 + 难度分层；与官方题去重 |
| 评测泄漏 | 中 | 高 | 同源约束（禁止同题）+ 相似度审计 + MathVista/MMMU 保持门禁 |
| GRPO 训练不稳定 | 中 | 中 | 小 lr + KL 系数调优 + 若发散退回 DPO |
| 多语言迁移弱 | 中 | 中 | 三语联合训练 + 语言间一致性正则 |
| MME-SCI 规模小波动大 | 中 | 低 | 报告分知识点聚合 + 置信区间 |

## 11. 论文写作计划

- **目标**：NeurIPS 2026（高）。
- **差异化卖点**：(1) 首个"评测基准→薄弱点诊断→自动出题→课程式 RL→再评测"闭环；(2) 知识点路由先验 + 多语言迁移证据；(3) 防泄漏审计协议（可复现）。
- **图表清单**：图1 闭环框架图；图2 薄弱点图谱（知识点×ACC 热图）；图3 课程收敛曲线；图4 语言迁移矩阵；图5 失败案例分析；表1 主实验；表2 消融；表3 防泄漏审计；表4 跨学科/语言。
- **相关工作覆盖**：科学基准（MME-SCI CV VIII·15、MathVista arXiv:2310.02255、MMMU arXiv:2311.16502）、RL 对齐（GRPO 原论文按规范引用为 DeepSeekMath 或有名的 RL 工作——本设计引用时仅列已验证项）、幻觉/评测（POPE arXiv:2305.10355、Res-Bench NLP II·77）。

## 12. 参考文献

1. MME-SCI（Computer Vision VIII·15）· MME-SCI: A Comprehensive and Challenging Science Benchmark for Multimodal Large Language Models（收藏论文，arXiv:2508.13938）
2. Lu et al. MathVista: Evaluating Mathematical Reasoning in Foundation Models with Visual Contexts. arXiv:2310.02255（ICLR 2024）.
3. Yue et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. arXiv:2311.16502（CVPR 2024 Oral）.
4. Li et al. Improved Baselines with Visual Instruction Tuning（LLaVA-1.5）. arXiv:2310.03744.
5. Wang et al. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191.
6. Res-Bench（Natural Language Processing II·77）· Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input（收藏论文）
