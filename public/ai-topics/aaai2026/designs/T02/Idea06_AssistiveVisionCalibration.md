# 实验设计书：视障辅助视频理解的"缺失-画质"判别与空间推理校准（Assistive Vision: Missing-Info vs. Poor-Quality Discrimination with Calibrated Spatial Reasoning）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：中高 ｜ 目标会议：CVPR 2027（Assistive Vision workshop/主会）或 AAAI 2027

## 0. 摘要

VisAssist（Computer Vision III·论文7）揭示视障用户拍摄视频中三类系统性缺陷：第一人称空间推理弱、无法区分"信息缺失"与"拍摄质量差"（导致危险幻觉）、非拉丁文本脆弱——且目前无任何方法层工作。本工作提出面向辅助视觉的三类决策框架：模型必须输出"回答 / 报告信息缺失 / 报告画质受限"三者之一，并对回答做不确定性校准；通过合成第一人称空间推理与退化场景数据（运动模糊/遮挡/裁剪/抖动增强）进行 RL 校准，设计**安全拒绝奖励**——对"证据不足却自信作答"重罚。评测含 VisAssist accuracy、三类决策 F1 与期望校准误差（ECE）。预期贡献：(1) 首个面向视障辅助的"缺失-画质"判别与安全校准训练方法；(2) 可复现的退化增强与拒绝样本合成协议；(3) 校准-安全-准确率的三角权衡分析。

## 1. 研究背景与动机

### 1.1 问题定义

辅助系统对视频问答的错误成本不对称：模型在信息不足时"自信胡编"比"报告缺失"危害大得多。任务要求模型输出 (a) 三类决策（answer / missing-info / poor-quality），(b) 若有回答给出置信度，(c) 第一人称空间推理正确（左右/上下/相对位置）。安全校准指：模型的不确定性估计应与其真实错误率一致（低 ECE），且对证据不足的题目宁可拒绝也不臆造。

### 1.2 相关工作不足

- **基准无方法**：VisAssist（Computer Vision III·论文7）只揭示缺陷、无任何方法；同类第一人称基准 EgoCross（Computer Vision V·论文77）只有微调/RL 试点。
- **安全校准缺失**：通用视频 QA 基准（VideoMME/MLVU）均无"拒绝/不确定性"评估；通用 RLVR 奖励只奖励正确、不惩罚"证据不足却自信"。
- **合成退化数据**：低画质视频 QA 训练数据在公共领域稀缺，需自建。
- **空白**：面向视障辅助的"缺失-画质"判别 + 空间推理 + 安全校准的联合训练目标。

### 1.3 为什么是现在、为什么你的环境

- **时机**：Assistive Vision 是 2026 年增长中的研究热区（多模态辅助系统、可穿戴）；VisAssist 数据已公开，方法空白明确。
- **环境契合**：SFT + GRPO ≈ 8-10 GPU·天在 4×L40 内可行；合成退化数据用 DeepSeek 离线生成 + 图像处理增强，成本低。

## 2. 研究目标与可验证假设

- **H1（三类决策可学）**：合成退化增强 + SFT 让模型学会区分 missing-info 与 poor-quality。
  *成立时的可观测结果*：VisAssist 三类决策 F1 显著高于基座（默认只 answer）。
- **H2（安全拒绝奖励有效）**：安全拒绝奖励显著降低危险幻觉（低置信度作答被压制）。
  *成立时的可观测结果*：拒绝率↑、危险幻觉率↓、ECE 显著下降（≤0.1）。
- **H3（空间推理提升）**：第一人称空间推理合成数据提升 VisAssist 空间推理题准确率。
  *成立时的可观测结果*：空间题子集 accuracy 提升 ≥5 点。
- **H4（校准-覆盖权衡可控）**：通过校准奖励权重可平滑调节拒绝率与准确率（coverage-accuracy 曲线单调）。
  *成立时的可观测结果*：coverage-accuracy 曲线单调，存在可操作的工作点。

## 3. 总体方法设计

### 3.1 数据流水线

1. **语料**：VisAssist（13,413 视频，许可核对）+ Ego4D 子集 + Kinetics 子集。
2. **退化增强（图像处理 + DeepSeek 标签）**：对干净视频应用运动模糊/高斯模糊/遮挡（黑框、随机遮挡物）/裁剪/抖动/低分辨率/亮度异常，生成"画质受限"样本；对视频内容本身不完整（物体出画、被遮挡、缺失）生成"信息缺失"样本。每样本带类别标签。
3. **合成第一人称空间推理数据（DeepSeek V4 Pro）**：从 Ego4D 生成空间相对位置问答（"手机在杯子的哪边"），以物体框（Grounding-DINO/标注）为 GT 锚点，过滤不一致。
4. **拒绝样本合成（DeepSeek V4 Pro + Kimi K2.6）**：对信息缺失样本生成"应拒绝"标签；对可答样本生成"应回答"标签；双 LLM 一致才保留。
5. **数量**：退化增强 15K、信息缺失 8K、空间推理 10K、拒绝微调 5K，合计 **~38K**。

### 3.2 模型/算法设计

- **基座**：Qwen2.5-VL-7B（LoRA rank 64）。
- **三类决策输出**：`{decision: answer|missing|poor_quality, confidence: 0-1, answer_text?, evidence_notes?}`。
- **安全拒绝奖励（GRPO）**：
  `r = r_correct · I[decision=answer] + λ1·I[decision=missing 且确实缺失] + λ2·I[decision=poor 且确实低画质] − 2.0·I[决策 answer 但置信度>0.8 且答案错误] − 0.3·I[决策 answer 但置信度<0.6]`
  λ1=0.8、λ2=0.8；置信度由模型自报 + 输出 logprob 归一化融合。
- **校准目标**：附加 ECE 正则项（对温度/置信度偏移做温度缩放校准）。
- **SFT**：标准 NLL + 三类决策交叉熵 + 空间推理 NLL。
- **超参初值**：lr 1e-5（SFT）/1e-6（GRPO）；batch 16；8 rollout/组；3 epoch。

### 3.3 训练流程
- Stage A SFT 4-5 GPU·天；Stage B GRPO（含安全奖励）4-5 GPU·天；2 卡训练+2 卡推理/合成并行。

### 3.4 推理与评测流程
- 推理 temperature=0（决策+回答）；confidence 独立评估温度缩放。
- 评测：VisAssist accuracy + 三类 F1 + ECE + 危险幻觉率 + coverage-accuracy 曲线。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| VisAssist（DOI: 10.1609/aaai.v40i6.42410）| 训练/评测 | 公开（核对许可）|
| Ego4D 子集 | 空间推理合成源 | 公开（research license）|
| Kinetics 子集 | 退化增强源 | 公开（CC-BY）|
| EgoSchema（NeurIPS 2023）| 评测（泛化）| 公开 |
| NExT-QA 子集 | 评测（泛化）| 公开 |

### 4.2 划分与数量
- 训练 38K；VisAssist 官方评测集；EgoSchema/NExT-QA 测试。

### 4.3 预处理与格式
- 32 帧、224×224；退化参数记录（类型/强度）；
- JSONL：`{path, decision_gt, confidence_gt, answer, spatial_gt, degradation_info}`。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| Qwen2.5-VL-7B | 官方 | github.com/QwenLM/Qwen2.5-VL |
| VideoLLaMA3 | 官方 | github.com/DAMO-NLP-SG/VideoLLaMA3 |
| Gemini-2.5-Pro | API 参考 | — |

### 5.2 复现步骤与预期指标表
统一 32 帧。预期主表：

| 方法 | VisAssist Acc | 三类 F1 | ECE | 危险幻觉率 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 官方值（低）| 无 | 高 | 高 |
| Gemini-2.5-Pro | 官方值 | 无 | — | 高 |
| **Ours-7B** | ≥基线+3 | ≥0.6 | ≤0.10 | 相对基线减半 |

### 5.3 统一评测口径
所有方法同 prompt 模板、同决策解析器；对不做三类决策的基线，将其输出映射为"answer-only"并单独报告其 ECE。

## 6. 实验矩阵

- **A（主实验）**：完整管线（退化增强 SFT + 安全拒绝 GRPO + 温度校准）。
- **B1（决策消融）**：仅 answer（基线）/ +missing / +poor / 三类全开。
- **B2（奖励消融）**：无安全拒绝 / 安全惩罚系数 {0,1,2,4}。
- **B3（数据消融）**：退化增强比例 {0,30%,60%,100%}；无空间合成数据。
- **B4（校准消融）**：温度缩放 / ECE 正则 / 无校准。
- **C（鲁棒性）**：退化强度分级、遮挡类型、跨分辨率。
- **D（泛化性）**：EgoSchema、NExT-QA 子集；跨基座（VideoLLaMA3）。

## 7. 评测协议

- 指标：VisAssist accuracy、三类决策 F1、ECE（期望校准误差）、危险幻觉率、coverage-accuracy 曲线下面积、空间子集 accuracy。
- 3 种子 mean±std；bootstrap p<0.05；安全评审人工抽查 100 条（高危拒绝样例）。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：SFT 5 + GRPO 4 + 评测 2 = **≈10 GPU·天**（预算 9-11）。
- 存储：模型 + 视频缓存 300GB。
- API：DeepSeek V4 Pro 合成退化/缺失标签 + 空间 QA ≈ 500 万 token；Kimi K2.6 安全性一致性验证 ≈ 200 万 token；成本 ≈ **$200-450**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | VisAssist/Ego4D 获取；退化增强管线实现 |
| 2 | 合成标签 + 拒绝样本生成 + 双 LLM 过滤 |
| 3 | 空间推理合成数据 + GT 锚点校验 |
| 4 | SFT 训练 + 三类决策评测 v0 |
| 5 | GRPO 安全奖励 + ECE 调参 |
| 6 | 消融 + coverage-accuracy 分析 |
| 7 | 安全边界人工评审 + 论文初稿 |
| 8 | 投稿 CVPR 2027 或 AAAI 2027（若 deadline 允许）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 数据集规模有限、安全要求高 | 中高 | 中高 | 人工评审安全边界；先做"拒绝→补拍"闭环 demo；保守 λ |
| 合成退化与真实分布 gap | 中 | 中 | 退化参数域随机化 + 真实 VisAssist 样本混合 |
| 校准奖励权重敏感 | 中 | 中 | coverage-accuracy 曲线多点报告；权重扫描 |
| 拒绝率过高拖累 recall | 中 | 中 | λ 平衡 + 报告中报告操作点而非单点 |

## 11. 论文写作计划

- **目标会议/截稿**：CVPR 2027（Assistive Vision）或 AAAI 2027。
- **差异化卖点**：首类面向视障辅助的"缺失-画质"判别方法；安全拒绝奖励；校准-覆盖-准确率三角分析。
- **图表清单**：Fig.1 三类决策示例（危险幻觉案例）；Fig.2 框架；Fig.3 校准曲线（可靠性图）；Fig.4 coverage-accuracy 曲线；Tab.1 主表；Tab.2 消融；Tab.3 泛化。
- **相关工作覆盖**：第一人称基准（VisAssist/EgoCross/EgoSchema）、不确定性校准、辅助视觉、RLVR 安全。

## 12. 参考文献（真实核验）

- Computer Vision III·论文7·VisAssist（DOI: 10.1609/aaai.v40i6.42410）
- Computer Vision V·论文77·EgoCross（arXiv:2508.10729）
- EgoSchema（NeurIPS 2023）
- VideoMME: arXiv:2405.21075
- MLVU: arXiv:2406.04264
