# 实验设计书：视频原生自监督 RLVR：状态转移验证奖励（Video-Native Self-Supervised RLVR with State-Transition Rewards）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：中高 ｜ 目标会议：CVPR 2027

## 0. 摘要

视频 LLM 的时间推理训练长期缺乏"绝对可验证"的奖励：TEMPLE（Computer Vision V·论文53）用扰动对比 DPO 只能教模型区分 clean/corrupted 输入，VideoSSR（arXiv:2511.06281）虽提出三预文本自监督 RL 却未覆盖"状态转移"与长程因果结构。本工作提出**状态转移验证（State-Transition Verification）**预文本任务族：判断两帧是否构成合法动作转移、检测倒放/乱序、多尺度时间 Jigsaw——所有奖励由任务构造本身给出 GT，**零人工标注、零外部模型依赖**。用 GRPO 训练 7B 视频 LLM 在这些符号可验证任务上学习时间结构，再零样本迁移评测下游 17 基准子集。预期贡献：(1) 首个面向"先 A 后 B 才 C"状态转移/长程因果的自监督时间奖励；(2) 与 VideoSSR/Spatial-SSRL 预文本可叠加的模块化配方；(3) 全套预文本任务构造代码与负样本增强协议。

## 1. 研究背景与动机

### 1.1 问题定义

视频 LLM 需要内隐的时间因果结构理解（状态转移合法性、事件顺序、插值预测）。问题在于：这些能力难以用答案监督（成本高），又无法用纯偏好信号（DPO 只给相对比较）精确刻画。自监督可验证奖励（self-supervised verifiable rewards）指：任务由程序化构造生成，其正确标签 = 构造本身，可零成本获得符号级验证信号，用于 RL 训练。

### 1.2 相关工作不足

- **扰动-DPO 流派**：TEMPLE（Computer Vision V·论文53）的 DPO 偏好对（clip dropping/shuffling/reversal）只提供相对偏好，无绝对正确性；且偏好信号来自"描述"单一任务。
- **自监督 RL 流派**：VideoSSR（arXiv:2511.06281）提出 Anomaly Grounding / Object Counting / Temporal Jigsaw 三预文本，但 (a) 不专门覆盖"状态转移合法性"（动作是否合法衔接）；(b) 不强调长程依赖；Spatial-SSRL（arXiv:2510.27606）只做空间；Conan（arXiv:2510.20470）是证据接地监督 RL 而非自监督。
- **空白**：时间维"状态转移合法性 + 长程依赖"的自监督可验证奖励缺失。

### 1.3 为什么是现在、为什么你的环境

- **时机**：RLVR + 自监督验证奖励（VideoSSR/Spatial-SSRL）被证实在 17 基准平均 +5%，是 2026 最前沿信号；补上"状态转移"维度即可形成组合卖点。
- **环境契合**：预文本任务构造为纯离线 CPU/单卡工作；7B LoRA GRPO 12-15 GPU·天在 4×L40 内可行。

## 2. 研究目标与可验证假设

- **H1（预文本自身可学）**：三个预文本任务准确率在训练后显著高于随机，且随训练稳定收敛。
  *成立时的可观测结果*：预文本任务准确率 ≥90%（状态转移合法性、乱序检测、Jigsaw 拼序）。
- **H2（下游正迁移）**：预文本 RLVR 显著提升下游时间推理基准。
  *成立时的可观测结果*：TempCompass MC、Vinoground(Text)、VideoMME 时间子集相对基座提升 ≥2 点。
- **H3（状态转移增益）**：状态转移预文本对"长程因果/状态变化"类问题的迁移增益大于 VideoSSR 已有预文本。
  *成立时的可观测结果*：在视频顺序/因果类基准子集上 State-Transition 预文本 > Temporal Jigsaw。
- **H4（可叠加性）**：与 VideoSSR 三预文本叠加产生增益。
  *成立时的可观测结果*：叠加版 ≥ 各单独训练之和的一半。

## 3. 总体方法设计

### 3.1 数据流水线

1. **语料**：Kinetics-400/700、VGGSound、ActivityNet（公开视频采样），离线切 16-32s 片段，每片段抽 32 帧。
2. **预文本构造（纯程序化，CPU 并行）**：
   - **T1 状态转移合法性**：对相邻片段对 (A→B)，构造"合法转移"（真实时序）与"非法转移"（随机组合/倒放 B），标签=构造本身。合法/非法 1:1。
   - **T2 倒放/乱序检测**：输入帧序列，随机做全局倒放或局部乱序（shuffle 块大小 2/4/8），模型判断是否倒放/乱序。
   - **T3 多尺度时间 Jigsaw**：把片段切成 4-16 块打乱，模型输出正确顺序（相对秩）。
3. **负样本增强（DeepSeek V4 Flash 辅助）**：对"边界模糊"样本（两帧差异极小）做难例标记；生成补充伪负样本。
4. **数量**：T1 40K 对、T2 40K、T3 30K，合计 ~110K 样本（离线构造，零标注成本）。

### 3.2 模型/算法设计

- **基座**：Qwen2.5-VL-7B（LoRA rank 64）。
- **输出格式**：T1 输出 `{legal: true/false, reason}`；T2 输出 `{order: correct/reversed}`；T3 输出 `{perm: [4,1,3,2,...]}`。
- **奖励（全部符号验证）**：
  - T1: r = 1[分类正确] + 0.2·边界置信校准（ECE 惩罚可选）
  - T2: r = 1[检测正确]
  - T3: r = Kendall-τ(预测排列, GT 排列) ∈ [-1,1]
- **GRPO**：每组 8 rollout；lr 1e-6；clip ε=0.2；KL 0.05；三预文本轮替训练（每 epoch 切换任务），4 epoch。
- **与 Pre-SFT DPO 组合**：可选通道 B——先 TEMPLE 式 Pre-SFT DPO（25K 对）再自监督 RL，形成"无监督自检 + 偏好对齐"双通道。

### 3.3 训练流程
- 数据构造：离线（CPU 多进程 + 单卡抽帧），2-3 天并行。
- 训练：7B LoRA GRPO，三预文本 × 4 epoch，2 卡训练 + 2 卡 rollout，10-14 GPU·天。

### 3.4 推理与评测流程
- 下游评测零样本：TempCompass MC、Vinoground(Text)、VideoMME 时间子集、LVBench、MLVU；temperature=0。
- 预文本自评：同分布测试集准确率。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| Kinetics-400/700 | 预文本构造 | 公开（CC-BY）|
| VGGSound | 预文本构造 | 公开（research）|
| ActivityNet | 预文本构造 | 公开（CC-BY）|
| TempCompass（arXiv:2403.00476）| 下游评测 | 公开 |
| Vinoground（arXiv:2410.02763）| 下游评测 | 公开 |
| VideoMME（arXiv:2405.21075）| 下游评测 | 公开 |
| MLVU（arXiv:2406.04264）/ LVBench（arXiv:2406.08035）| 下游评测 | 公开 |

### 4.2 划分与数量
- 预文本：110K（T1 40K/T2 40K/T3 30K），train 95%、test 5%。
- 下游：官方测试集。

### 4.3 预处理与格式
- 16-32s 片段、32 帧、224×224；标签内嵌（构造器直接写 JSONL）；
- JSONL：`{path, frames, task_type, label, extra}`。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| Qwen2.5-VL-7B | 官方 | github.com/QwenLM/Qwen2.5-VL |
| TEMPLE | Computer Vision V·论文53 | 论文开源 |
| VideoSSR | arXiv:2511.06281 | github.com/tinyvideo/VideoSSR（论文开源）|
| Spatial-SSRL | arXiv:2510.27606 | 论文开源则复现 |

### 5.2 复现步骤与预期指标表
统一 32 帧、temperature=0。预期主表：

| 方法 | TempCompass MC | Vinoground(Text) | VideoMME-Time | 预文本平均准确率 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 基准 | 基准 | 基准 | — |
| TEMPLE | +2-3 | +3.2 | +3.6(感知)/+2.8(推理) | — |
| VideoSSR | 17 基准平均+5%（官方）| 提升 | 提升 | — |
| **Ours（状态转移三预文本）** | ≥基线+2 | ≥基线+2 | ≥基线+2 | ≥90 |

### 5.3 统一评测口径
所有方法同 prompt 模板、同帧采样；预文本任务使用同一套构造器与解析器。

## 6. 实验矩阵

- **A（主实验）**：三预文本联合 GRPO。
- **B1（预文本消融）**：仅 T1 / 仅 T2 / 仅 T3 / 两两组合 / 三者联合。
- **B2（状态转移 vs 已有预文本）**：仅状态转移 vs VideoSSR 三预文本复现。
- **B3（叠加性）**：Ours + VideoSSR 预文本叠加。
- **B4（双通道）**：Pre-SFT DPO + 自监督 RL vs 纯自监督 RL。
- **C（鲁棒性）**：片段长度 {16,32,64}、噪声帧、不同帧率采样。
- **D（泛化性）**：跨基座（Qwen2-VL、VideoLLaMA3）；下游零样本子集。

## 7. 评测协议

- 指标：各基准官方 accuracy；预文本自身准确率/Kendall-τ；下游迁移 Δ。
- 3 种子 mean±std；bootstrap p<0.05；报告逐任务表。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：数据构造 2（离线 CPU/单卡）+ GRPO 12 + 评测 2 = **≈14 GPU·天**（预算 12-15）。
- 存储：视频缓存 500GB、JSONL 60GB。
- API：DeepSeek V4 Flash 数据清洗/难例筛选 ≈ 300 万 token；Kimi K2.6 抽查输出合理性 ≈ 80 万 token；成本 ≈ **$100-250**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | 视频采集 + 预文本构造器实现（T1/T2/T3）+ 数据生成 |
| 2 | 数据质量检查（构造正确率 100% 验证）；基座复现 |
| 3 | GRPO 训练 v0（单预文本 T1 先验证可收敛）|
| 4 | 三预文本联合训练 + 自评 |
| 5 | 下游迁移评测 + 消融 B1-B4 |
| 6 | 与 VideoSSR 叠加实验 + 鲁棒性 |
| 7 | 论文初稿 |
| 8 | 投稿 CVPR 2027（deadline ~2026-11）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 预文本到下游正迁移不确定 | 中高 | 高 | 先复现 VideoSSR 对照确认增益方向；若弱转"对齐冷启动 + DPO"混合 |
| 状态转移预文本判别易被外观捷径破解 | 中 | 中高 | 负样本增强（外观相似但时序非法）；均衡类别；难例挖掘 |
| 与已有预文本叠加增益不足 | 中 | 中 | 任务轮替/课程调度调整；或只推状态转移单点 |
| GRPO 在排列输出上不稳定 | 中 | 中 | T3 用 Kendall-τ 连续奖励 + 序列解码约束 |

## 11. 论文写作计划

- **目标会议/截稿**：CVPR 2027。
- **差异化卖点**：首个状态转移自监督可验证奖励；三预文本程序化构造协议（零标注）；与 VideoSSR/Spatial-SSRL 的模块化叠加配方。
- **图表清单**：Fig.1 三预文本任务示意；Fig.2 框架；Fig.3 预文本训练曲线；Fig.4 迁移案例；Tab.1 下游主表；Tab.2 预文本消融；Tab.3 叠加性。
- **相关工作覆盖**：偏好对齐（TEMPLE/TPO）、自监督 RL（VideoSSR/Spatial-SSRL/Conan）、RLVR（Video-R1）、诊断基准（TempCompass/Vinoground）。

## 12. 参考文献（真实核验）

- Computer Vision V·论文53·TEMPLE（arXiv:2503.16929）
- VideoSSR: arXiv:2511.06281
- Spatial-SSRL: arXiv:2510.27606
- Conan: arXiv:2510.20470
- Video-R1: arXiv:2503.21776
- TPO: arXiv:2501.13919
- TempCompass: arXiv:2403.00476
- Vinoground: arXiv:2410.02763
- VideoMME: arXiv:2405.21075
- MLVU: arXiv:2406.04264
- LVBench: arXiv:2406.08035
