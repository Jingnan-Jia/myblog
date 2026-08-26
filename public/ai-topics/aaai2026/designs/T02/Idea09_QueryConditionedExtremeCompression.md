# 实验设计书：查询条件的极端视频 token 压缩 + 位置偏置校正（Query-Conditioned Extreme Token Compression for Long Video）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：中高 ｜ 目标会议：NeurIPS 2026 / ECCV 2027

## 0. 摘要

长视频 LLM 推理的 token 预算瓶颈突出：XComp（arXiv:2604.14149）做到每帧 1 token 但压缩无查询条件、仅用 2.5% SFT 数据；FlexSelect（arXiv:2506.00993）/AdaptToken（arXiv:2603.28696）训练无关但信息丢失；APVR（Computer Vision III·论文3）K 固定。现有方法普遍忽略 LLM 长上下文的头尾位置偏置。本工作提出 **查询条件极端压缩 + 位置偏置校正**：用 query embedding 门控每帧压缩（区分"信息锚"与"故事线锚"），配合局部注意力分块 + 位置重标注消除头尾偏置，仅用 <5% 数据量做压缩 SFT，即插即用于 Qwen2.5-VL 系。训练信号 = 帧重建自监督 + 少量 LVQA；推理时 ≤8K 视觉 token 预算。预期贡献：(1) 首个查询条件门控的"每帧 ≤1 token"压缩器；(2) 位置偏置诊断与校正方法；(3) 压缩比-准确率全程曲线与 8K 预算下 SoTA 级效果。

## 1. 研究背景与动机

### 1.1 问题定义

视频 LLM 推理时，视觉 token 数随帧数线性增长，长视频场景显存/算力爆炸。目标：在 ≤8K 视觉 token 预算下最大化下游 QA 准确率。两个子问题：(a) 信息锚（对作答关键的内容）需保真，故事线锚（冗余帧）可极限压缩；(b) LLM 长上下文中头/尾位置有注意力偏置，会高估首末帧。

### 1.2 相关工作不足

- **XComp（arXiv:2604.14149，NeurIPS 2025）**：每帧 1 token 极端压缩 + 局部注意力消除位置偏置，但压缩无查询条件（query-agnostic）、只用 2.5% SFT 数据，上限受限。
- **训练无关压缩**：FlexSelect（arXiv:2506.00993）、AdaptToken（arXiv:2603.28696）、APVR（Computer Vision III·论文3）无学习或固定 K；信息丢失不可控。
- **其他压缩范式**：VideoChat-Flash（arXiv:2501.00574）分层压缩无查询门控；Video-XL-Pro（arXiv:2503.18478）重构式但非查询条件；Tempo（arXiv:2604.08120）小 VLM 压缩器无位置校正。
- **空白**：查询条件门控 + 头尾偏置校正 + 极端压缩比的统一配方。

### 1.3 为什么是现在、为什么你的环境

- **时机**：XComp 刚证明"每帧 1 token"可行，查询条件化是明显增量；8K 预算在 4×L40 上可测。
- **环境契合**：7B LoRA 压缩 SFT 4-6 GPU·天，评测 2 GPU·天，总计 <10 GPU·天。

## 2. 研究目标与可验证假设

- **H1（查询条件增益）**：查询条件门控压缩优于 query-agnostic 压缩。
  *成立时的可观测结果*：同 8K 预算下 LVBench/VideoMME accuracy 提升 ≥1.5 点。
- **H2（位置偏置校正有效）**：局部注意力分块 + 位置重标注显著消除首/中/尾区间命中率不平衡。
  *成立时的可观测结果*：首/中/尾区间命中率方差显著下降（原方差/校正后方差 ≥2）。
- **H3（数据效率）**：<5% 数据量的压缩 SFT 足以逼近全量 SFT。
  *成立时的可观测结果*：5% vs 全量 SFT 的 accuracy 差 ≤1 点。
- **H4（即插即用）**：压缩器可迁移到 Qwen2.5-VL 系其他规格。
  *成立时的可观测结果*：7B 训练压缩器在 Qwen2-VL-7B 上零样本微调后仍有增益。

## 3. 总体方法设计

### 3.1 数据流水线

1. **语料**：LVBench（arXiv:2406.08035）、VideoMME、MLVU、LongVideoBench、StreamingBench（arXiv:2411.03628）训练拆分（许可核对）。
2. **压缩训练数据（DeepSeek V4 Flash）**：对抽帧视频生成"帧→关键描述"，用于重建监督与故事线锚分类；另生成少量 LVQA（问答对）作为任务监督（<5% 数据量）。
3. **信息锚/故事线锚标注**：用 query-aware 打分（CLIP 文本-帧 + 帧间相似度）启发式初标，DeepSeek 校验；用作门控训练弱监督。
4. **数量**：压缩自监督 30K 视频片段；LVQA 5K。

### 3.2 模型/算法设计

- **基座**：Qwen2.5-VL-7B（LoRA rank 64）或 VideoChat-Flash 初始化压缩器。
- **查询条件 token 合成**：每帧视觉特征 v_f 与 query 嵌入 q 拼接 → 门控网络输出：
  - 信息锚帧：保留多 token（top-r% 帧全 token 兜底）；
  - 故事线锚帧：合成 ≤1 token（`t_f = Proj([v_f, g(q)])`，g 为门控线性层）。
- **局部注意力分块 + 位置重标注**：帧内 token 局部注意力（每帧内全连接、帧间稀疏）；位置编码重标注为"帧序 + 帧内序"，消除全局长序列头尾偏置。
- **损失**：`L = L_recon(帧重建自监督) + λ·L_anchor(锚分类 BCE) + γ·L_LVQA(少量问答 NLL)`，λ=0.3、γ=1.0。
- **超参初值**：lr 1e-5；batch 16；3 epoch；视觉 token 预算 ≤8K（推理时动态）。

### 3.3 训练流程
- 7B LoRA 压缩 SFT 4-6 GPU·天（2 卡）；评测 2 GPU·天（2 卡并行）。

### 3.4 推理与评测流程
- 推理：query 嵌入 → 门控合成 token → 局部注意力解码；temperature=0。
- 评测：8K token 预算下 LVBench/VideoMME/MLVU/LongVideoBench/StreamingBench accuracy；首/中/尾区间命中率诊断。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| LVBench（arXiv:2406.08035）| 训练/评测 | 公开 |
| VideoMME（arXiv:2405.21075）| 训练/评测 | 公开 |
| MLVU（arXiv:2406.04264）| 训练/评测 | 公开 |
| LongVideoBench（arXiv:2407.15754）| 评测 | 公开 |
| StreamingBench（arXiv:2411.03628）| 评测 | 公开 |

### 4.2 划分与数量
- 压缩自监督 30K、LVQA 5K、评测官方测试集。

### 4.3 预处理与格式
- 帧 1fps、224×224；JSONL：`{path, frames, q_expansion, anchor_labels, lvqa}`。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| XComp | arXiv:2604.14149 | 官方开源则复现 |
| FlexSelect | arXiv:2506.00993 | 官方开源则复现 |
| AdaptToken | arXiv:2603.28696 | 官方开源则复现 |
| VideoChat-Flash | arXiv:2501.00574 | 官方开源 |
| Tempo | arXiv:2604.08120 | 官方开源则复现 |

### 5.2 复现步骤与预期指标表
统一 8K token 预算。预期主表（accuracy）：

| 方法 | LVBench | VideoMME | MLVU | LongVideoBench | StreamingBench |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B（全 token 基准）| 基准 | 基准 | 基准 | 基准 | 基准 |
| XComp | 官方值 | 官方值 | — | — | — |
| **Ours-7B（8K）** | ≥XComp | ≥XComp | ≥XComp×0.95 | ≥基准×0.95 | ≥基准×0.95 |

### 5.3 统一评测口径
所有方法同 prompt、同帧采样；统一以"视觉 token 数 ≤8K"计量；报告压缩比-准确率曲线。

## 6. 实验矩阵

- **A（主实验）**：完整（查询门控 + 局部注意力 + 位置重标注 + 三损失）。
- **B1（门控消融）**：无查询条件（query-agnostic）/ 门控 / 门控+信息锚兜底。
- **B2（位置偏置消融）**：全局注意力 / 局部分块 / 分块+位置重标注。
- **B3（损失消融）**：仅 L_recon / +L_anchor / +L_LVQA。
- **B4（数据量消融）**：压缩 SFT 数据量 {1%,2.5%,5%,10%}。
- **C（鲁棒性）**：预算 {4K,8K,16K}、视频长度分级、帧率变化。
- **D（泛化性）**：跨基座（Qwen2-VL-7B、VideoChat-Flash 初始化）即插即用。

## 7. 评测协议

- 指标：各基准 accuracy、压缩比-准确率曲线、首/中/尾区间命中率（位置偏置诊断）、token 数、延迟。
- 3 种子 mean±std；bootstrap p<0.05。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：压缩 SFT 5 + 评测 2 = **<10 GPU·天**。
- 存储：模型 + 视频缓存 300GB。
- API：DeepSeek V4 Flash 帧描述/LVQA 生成 ≈ 800 万 token；Kimi K2.6 压缩后可读性验证 ≈ 150 万 token；成本 ≈ **$200-450**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | 基线 XComp/FlexSelect 复现（若开源）+ 数据准备 |
| 2 | 门控 + 局部注意力 + 位置重标注实现 |
| 3 | 压缩自监督数据生成 + SFT v0 |
| 4 | LVQA 注入 + 数据量消融 |
| 5 | 位置偏置诊断 + 鲁棒性/泛化 |
| 6 | 压缩比-准确率曲线 + 论文初稿 |
| 7 | 投稿 NeurIPS 2026 或 ECCV 2027（视 deadline）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 极端压缩信息丢失 | 中高 | 高 | 重建损失 + 查询门控 + 关键帧全 token 兜底 |
| 位置偏置校正破坏语义连贯 | 中 | 中 | 位置重标注 + 局部注意力分块协同调参；消融验证 |
| 查询条件依赖 query 质量 | 中 | 中 | query 展开（DeepSeek）增强；无查询时退化为 agnostic 模式 |
| 与现有评测口径不可比 | 中 | 中 | 统一 token 预算计量 + 官方脚本 |

## 11. 论文写作计划

- **目标会议/截稿**：NeurIPS 2026 或 ECCV 2027。
- **差异化卖点**：首个查询条件门控极端压缩器；首/中/尾位置偏置诊断与校正；<5% 数据量 SFT 的即插即用。
- **图表清单**：Fig.1 框架（门控+局部注意力+位置重标注）；Fig.2 位置偏置诊断热图（校正前后）；Fig.3 压缩比-准确率曲线；Fig.4 压缩 token 可视化；Tab.1 主表；Tab.2 消融；Tab.3 泛化。
- **相关工作覆盖**：token 压缩综述（arXiv:2507.20198）、极端压缩（XComp）、可学习压缩（VideoChat-Flash/Video-XL-Pro/Tempo）、训练无关（FlexSelect/AdaptToken/APVR）。

## 12. 参考文献（真实核验）

- XComp: arXiv:2604.14149
- FlexSelect: arXiv:2506.00993
- AdaptToken: arXiv:2603.28696
- VideoChat-Flash: arXiv:2501.00574
- Video-XL-Pro: arXiv:2503.18478
- Tempo: arXiv:2604.08120
- A Survey of Token Compression: arXiv:2507.20198
- Computer Vision III·论文3·APVR（arXiv:2506.04953）
- LVBench: arXiv:2406.08035
- VideoMME: arXiv:2405.21075
- MLVU: arXiv:2406.04264
- LongVideoBench: arXiv:2407.15754
- StreamingBench: arXiv:2411.03628
