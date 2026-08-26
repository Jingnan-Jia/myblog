# 实验设计书：意图感知难负样本三元组与重排的组合图像检索

## 0. 摘要

本实验设计为组合图像检索（CIR）引入「意图条件难负样本三元组挖掘 + 检索后重排」：用 DeepSeek V4 Pro 为 CIRR/CIRCO/FashionIQ 三元组生成细粒度意图标签（换属性/增删对象/风格迁移/场景变化）与"为什么像又为什么不是"的 distractor 理由，训练双塔（CLIP-L）时用意图条件对比损失 + 意图分类辅助头，推理时双塔召回 top-k 后用轻量 cross-encoder 对 top-50 重排（控延迟）。目标是 R@1/5/10 相对 SEARLE/LinCIR/iSEARLE/good4cir/De-MINDS 显著提升，同时报告延迟-精度权衡。4×L40 约 6 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

零样本组合检索（ZS-CIR）：给定参考图 $I_{ref}$ 与修改文本 $t$，检索目标图 $I_{tgt}$。核心难点在意图理解：$t$ 往往简短模糊（"换红色"），且存在大量"只差一个属性/对象"的难负样本（distractor）。我们定义意图 $u\in U=\{换属性, 增删对象, 风格迁移, 场景变化\}$，把检索训练改为意图条件三元组 $(I_{ref},t,I_{tgt},I_{neg},u)$，并加检索后重排（对 top-k 内用 cross-encoder 精排）。

### 1.2 相关工作不足

- 收藏论文 `De-MINDS`（Computer Vision VIII·93·Manipulation Intention Understanding for Zero-Shot Composed Image Retrieval）：蒸馏操纵意图，但意图类型固定、无意图条件难负样本、无重排、未见意图泛化。
- `SEARLE`（arXiv:2303.15247，CVPR 2024）文本反演；`LinCIR`（arXiv:2312.01998，CVPR 2024）仅文本高效训练；`iSEARLE`（arXiv:2405.02951）；`good4cir`（arXiv:2503.17871）合成描述；`MCoT-RE`（arXiv:2507.12819）免训练 CoT 重排——均未把"意图结构化 + 难负样本 + 重排"组合到训练端。

空白：**意图条件难负样本挖掘（"为什么像又为什么不是"）在训练端的系统建模 + 与重排的组合**。

### 1.3 为什么是现在、为什么你的环境适合做

De-MINDS 2026 AAAI 刚中，"意图"成为 ZS-CIR 关键词；good4cir/MCoT-RE 说明合成数据与重排是当下热点。本环境：双塔训练 4×L40 或单卡即可；重排模型轻量；DeepSeek V4 Pro 合成意图标签与 distractor 理由便宜可控。

## 2. 研究目标与可验证假设

- **H1**：意图条件难负样本提升判别力。*可观测结果*：加 intent-aware triplet 后，CIRR/CIRCO/FashionIQ 的 R@1 提升 ≥2 点，且"只差一个属性"子集提升更明显。
- **H2**：意图分类辅助监督带来可解释意图预测。*可观测结果*：意图分类 ACC ≥75%，且错误检索案例可按意图归因。
- **H3**：cross-encoder 重排（top-50）显著提精度且延迟可控。*可观测结果*：R@1 +3–5 点；每查询延迟 ≤20ms（vs 全量重排不可行）。
- **H4**：细粒度意图标签优于固定意图集。*可观测结果*：4 类意图 vs 固定 2 类，未见意图子集提升且训练更稳。

## 3. 总体方法设计

### 3.1 数据流水线

- **输入**：CIRR（官方数据，含参考/目标图与修改文本）、CIRCO（官方数据）、FashionIQ（arXiv:1905.12794 官方三元组）。
- **意图标签生成（DeepSeek V4 Pro）**：
```
Given ref image caption "{c_ref}", target caption "{c_tgt}", modification text "{t}", classify the manipulation into {attribute_change, object_add_remove, style_transfer, scene_change, other} and give a 1-sentence rationale.
```
- **难负样本挖掘（DeepSeek V4 Pro）**：对每个三元组生成 1 个 distractor：
```
Find a third image (from the same dataset's candidate pool by CLIP top-30) that looks similar to target but differs by ONE attribute specified by the modification text. Output its id and the reason "why it looks like the target yet is NOT the target".
```
理由文本随样本进入训练（作为文本增强的一种）。
- **过滤**：意图标签与理由用 DeepSeek V4 Flash 去噪（与图一致性校验）；distractor 需与 target 的 CLIP 相似度在 [0.7, 0.95]（太近太远都弃）。
- **数量**：CIRR train 17k 三元组 + FashionIQ 39k + CIRCO（少量）+ 合成 distractor 各 1 → 训练样本 ≈ **75k 意图三元组**；理由文本总词数 <64。

### 3.2 模型/算法设计

- **双塔**：CLIP-ViT-L/14 视觉 + 文本编码器。文本侧组合 $t_{composed}=[t_{cls};\,f_{text}(t)]$ 与图像组合 $f_{img}(I_{ref})$ 相加或交叉（继承 LinCIR 的轻量组合，仅训文本侧线性投影 + LoRA）。
- **意图条件对比损失**：
  $\mathcal{L}_{contr}=\sum_i -\log\frac{\exp(\cos(q_i, v_{tgt,i})/\tau)}{\sum_{j\in\{tgt,neg\}}\exp(\cos(q_i,v_j)/\tau)}$，
  其中 $q_i=\text{combine}(v_{ref,i}, t_i)$；负样本中至少含 1 个意图条件 distractor $v_{neg,i}$（"为什么像又为什么不是"）。
- **意图分类辅助头**：在组合 query 上 $p(u|q)$，$\mathcal{L}_{intent}=CE(p,u)$，$\lambda=0.3$ 初值。
- **理由辅助（可选）**：轻量生成头在检索前生成 distractor 排除理由，融合进 query 文本（消融项）。
- **重排器**：轻量 cross-encoder（DeBERTa-base 或 CLIP text-image cross，~300M 参数），输入 $(q, \text{top-50 候选图像})$，输出相似度重排。训练：用训练集检索出的 top-50 对做排序学习（listwise/pointwise，label=是否目标）。
- **延迟控制**：双塔召回 top-50 → cross-encoder 精排 top-50 → 输出；全流程 vLLM/ONNX 量化（fp16）。

### 3.3 训练流程

- 阶段1 双塔：batch=256（跨 4 卡），lr=3e-4（组合层）/1e-4（LoRA），AdamW，10k 步；负样本 batch 内 hard negative + 意图条件 distractor。
- 阶段2 重排器：加载阶段1 检索 top-50 缓存训练，2k 步，lr=1e-5。
- 冻结 CLIP 骨干（仅 LoRA 全量训练侧），降低过拟合与成本。

### 3.4 推理与评测流程

- 检索：$I_{ref}+t$ → 双塔得 top-200 → cross-encoder 重排 top-50 → R@1/5/10。
- 评测：CIRR test（官方，含 Recall@K 与分组子集）、CIRCO val/test（Recall@K）、FashionIQ val/test（R@K，K=10/50）。
- 意图预测：分类 ACC + 检索错误案例意图归因可视化。

## 4. 数据集细节

| 数据集 | 用途 | 说明 | 许可 |
|---|---|---|---|
| CIRR（官方数据，无 arXiv） | 训练/评测 | 21k 三元组、~1k 分组 | 学术 |
| CIRCO（官方数据，无 arXiv） | 评测 | 开放检索、1k 查询 | 学术 |
| FashionIQ（arXiv:1905.12794） | 训练/评测 | 3 类目，17k/39k 三元组 | 学术 |
| **IntentTriplet（本工作）** | 合成标注 | 意图标签 + distractor 理由 | 随论文开源 |
| good4cir 合成描述（arXiv:2503.17871） | 可选增强 | 官方数据 | 学术 |

## 5. 基线复现

| 基线 | 官方代码 | 复现要点 |
|---|---|---|
| SEARLE（arXiv:2303.15247） | https://github.com/ABaldrati/SEARLE | 官方权重/训练 |
| LinCIR（arXiv:2312.01998） | https://github.com/kyotovl/linCIR | 官方权重 |
| iSEARLE（arXiv:2405.02951） | 同 SEARLE 仓库 | 官方权重 |
| good4cir（arXiv:2503.17871） | 见论文官方仓库 | 官方权重 |
| De-MINDS（CV VIII·93） | 见论文（代码公开） | 官方权重 |
| MCoT-RE（arXiv:2507.12819） | 见论文官方仓库 | 免训练重排，接入各双塔 |
| Ours | — | §3 全量 |

**预期指标表**（CIRR 分组 Recall / CIRCO R@5 / FashionIQ 均值；数值以复现为准）：

| 方法 | CIRR R@1 | CIRCO R@5 | FashionIQ 均值 | 仅差一属性子集 R@1 |
|---|---|---|---|---|
| SEARLE | 基准 | 基准 | 基准 | 基准 |
| LinCIR | 略高 | 略高 | 略高 | 略高 |
| good4cir | 高 | 高 | 高 | 高 |
| De-MINDS | 最高 | 最高 | 最高 | 中 |
| **Ours** | **+2 点以上** | **+2 点以上** | **+2 点以上** | **显著最高** |

统一口径：同一 CIRR/CIRCO/FashionIQ 官方划分、同一 prompt、同一温度；重排延迟单独报表。

## 6. 实验矩阵

- **A. 主实验**：全基线 + Ours 全指标。
- **B. 消融**：B1 意图条件 distractor 有无（→普通 batch hard negative）；B2 意图分类头有无；B3 理由辅助有无；B4 重排 top-k∈{20,50,100}；B5 重排模型规模（base/large）；B6 意图类别数（2 vs 4 vs 6）。
- **C. 鲁棒性**：C1 模糊/简短修改文本；C2 长修改文本；C3 负样本密度高的困难查询。
- **D. 泛化性**：D1 跨数据集（CIRR 训练 → CIRCO 零样本）；D2 未见意图组合；D3 中文修改文本。

## 7. 评测协议

- **指标**：R@1/5/10（CIRR 官方）、R@5/10/25（CIRCO 官方）、FashionIQ R@10/50 均值；意图分类 ACC；延迟（ms/查询）。
- **均值±方差**：3 个种子（123/2024/7），报 mean±std。
- **显著性**：R@K 差配对 Bootstrap（n=1000，CI 95%）。
- **重排延迟**：固定 batch=64 的吞吐 + P99 延迟表。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 意图标签 + distractor 合成（API 为主）+ CLIP 预筛 | 0.5 |
| P2 | 双塔训练 10k 步 | 3–4 |
| P3 | 重排器训练 + 推理缓存 | 1 |
| P4 | 全量评测（7 模型 × 3 数据集 × 延迟表） | 1–1.5 |
| **合计** | | **≈5.5–7** |

存储：数据集 + 合成标注 ≈ 60GB。API：DeepSeek V4 Pro 意图+distractor ≈ $30；V4 Flash 去噪 ≈ $10。总计 ≤ **$45**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | 数据获取 + 基线复现（SEARLE/LinCIR/iSEARLE/good4cir） |
| W2 | 意图标签 + distractor 流水线 + 抽检 |
| W3 | 双塔训练 + B1/B2 消融 |
| W4 | 重排器 + A 主实验 + 延迟表 |
| W5 | 鲁棒性/泛化 + 统计 + 论文初稿（CVPR 2027 截稿前 6 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 重排延迟超标 | 中 | 中 | top-50 限定 + 轻量模型 + 量化；报告延迟-精度曲线 |
| distractor 挖掘质量不稳 | 中 | 中 | CLIP 相似度窗口过滤 + V4 Flash 校验 |
| 意图分类辅助收益小 | 低 | 中 | 转为解释性输出，主贡献仍靠难负样本+重排 |
| 未见意图泛化有限 | 中 | 中 | 类别扩展 + 意图无关表述（"other"类） |
| 基线条目多复现工作量大 | 中 | 低 | 优先复现 4 个官方权重基线 |

## 11. 论文写作计划

- **目标**：CVPR 2027（中）；备选 ACM MM 2027 / ICMR。
- **差异化卖点**：(1) 首个"意图条件难负样本（带理由）+ 意图分类 + 重排"组合；(2) 可复现 IntentTriplet 数据与理由生成协议；(3) 延迟-精度权衡透明报告。
- **图表清单**：图1 方法图；图2 难负样本"为什么像又为什么不是"示例；图3 意图归因错误分析；图4 延迟-精度曲线；表1 主实验；表2 消融；表3 鲁棒性；表4 跨数据集泛化；表5 延迟表。
- **相关工作覆盖**：ZS-CIR（SEARLE arXiv:2303.15247、LinCIR arXiv:2312.01998、iSEARLE arXiv:2405.02951、good4cir arXiv:2503.17871、De-MINDS CV VIII·93）、重排（MCoT-RE arXiv:2507.12819）、数据集（CIRR、CIRCO、FashionIQ arXiv:1905.12794）、基础（CLIP arXiv:2103.00020）。

## 12. 参考文献

1. De-MINDS（Computer Vision VIII·93）· Manipulation Intention Understanding for Zero-Shot Composed Image Retrieval（收藏论文）
2. Baldrati et al. Effective Conditioned and Composed Image Retrieval Combining CLIP-Based Features（SEARLE）. arXiv:2303.15247（CVPR 2024）.
3. Gu et al. Zero-shot Composed Image Retrieval with Textual Inversion（LinCIR）. arXiv:2312.01998（CVPR 2024）.
4. Baldrati et al. Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features（iSEARLE）. arXiv:2405.02951.
5. Yang et al. Good4Cir: Global Optimization of Decomposable Hardness for Zero-Shot Composed Image Retrieval. arXiv:2503.17871.
6. Lu et al. MCoT-RE: Multi-step Chain-of-Thought Reasoning for Composed Image Retrieval via Reranking. arXiv:2507.12819.
7. Wu et al. Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback. arXiv:1905.12794.
8. Radford et al. Learning Transferable Visual Models From Natural Language Supervision（CLIP）. arXiv:2103.00020（ICML 2021）.
