# 实验设计书：MLLM 化的属性-位置解耦参考表达计数

## 0. 摘要

本实验设计把参考表达计数（REC）重构为「计数什么（counting query）+ 看哪里（evidence region）」双查询的端到端 MLLM grounding 任务：一个 7B MLLM 同时输出计数查询与属性证据区域框，计数用轻量 counting head（回归 + 分桶），属性-证据区域标注全自动合成（DeepSeek V4 Pro 生成属性描述 → GroundingDINO 锚框 → SAM 精修），从而把 W2-Net 的手工 w2s 启发式泛化到任意开放属性。目标是 REC-8K 计数误差显著下降、unseen 属性可泛化、定位 F1 高。4×L40 约 5 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

参考表达计数：给定图像 $I$ 与参考表达 $r$（含对象类别与属性，如"穿着红色裙子的女孩"），输出该类实例数 $\hat{c}$。W2-Net 发现把"计数什么"（w2c）与"看哪里"（w2s）解耦能显著降低误差，但其 w2s 是手工启发式（依赖类别先验，"行走→腿"），无法处理任意开放属性与多属性组合。我们把问题重构为端到端 MLLM 双查询：模型输出 $\hat{c}$ 与证据区域框集合 $\hat{B}=\{b_1..b_k\}$（属性成立的区域），$B$ 既监督训练又作为可解释输出。

### 1.2 相关工作不足

- 收藏论文 `W2-Net`（Computer Vision XIII·108·Decoupling What to Count and Where to See for Referring Expression Counting）：双查询解耦 + 子类可分离匹配（SSM），REC-8K 计数误差降 22.5%/18.0%，但 w2s 为手工启发式、属性泛化受限、REC-8K 类集合固定。
- `CountGD`（arXiv:2407.04619）文本计数但不输出证据区域。
- `CounTR`（arXiv:2208.13721）少样本计数无属性。
- MLLM 直接数数（Qwen-VL 等）在"数数"上本质弱（需 counting head 辅助）。
- grounding 侧 `GroundingDINO`（arXiv:2303.05499）、`Ferret`（arXiv:2310.07704）可锚框但无数数。

空白：**属性-证据区域标注的全自动合成 + 端到端 MLLM 双查询 + 开放属性泛化评测**。

### 1.3 为什么是现在、为什么你的环境适合做

W2-Net 2026 AAAI 刚中，直接在其任务上做"手工启发式 → 自动合成 + MLLM 端到端"的升级，是清晰的增量窗口。本环境：7B LoRA 训练 4×L40 够；DeepSeek V4 Pro 合成属性-区域标注便宜；GroundingDINO/SAM 批量锚框可并行。

## 2. 研究目标与可验证假设

- **H1**：证据区域监督提升属性化计数精度。*可观测结果*：加证据区域回归 loss 后，属性相关表达（颜色/动作/材质）MAE 显著下降（≥10%）。
- **H2**：MLLM 双查询优于手工 w2s。*可观测结果*：REC-8K 计数误差相对 W2-Net（手工启发式）进一步下降 ≥8%。
- **H3**：自动合成标注可泛化到开放属性。*可观测结果*：unseen 属性子集（训练未见的属性模板）MAE 不爆炸（相对 seen 属性 ≤1.3×），定位 F1 ≥0.6。
- **H4**：counting head + 分桶优于 MLLM 裸数数。*可观测结果*：大数（≥20）场景误差显著低于直接让 Qwen-VL 数数。

## 3. 总体方法设计

### 3.1 数据流水线

- **基础**：REC-8K（参考表达计数数据集，见 W2-Net 论文 CV XIII·108 描述，官方数据）；FSC-147（arXiv:2104.08391）转 REC 子集（用类名+属性模板造表达）。
- **属性模板**：5 大类模板 × 每类 8 变体：颜色（"穿着X色的"）、动作（"正在X的"）、材质（"X质地的"）、状态（"破损的/打开的"）、位置（"在X旁边的"）。
- **合成标注（DeepSeek V4 Pro）**：
```
Given image I with object class "{cls}" and candidate instances, for attribute template "{tmpl}", list the instances (by region) for which the attribute is TRUE. Output JSON [{region_id, bbox, "true": bool}].
```
- **证据区域生成**：GroundingDINO 对 `"{cls}"` 出框 → SAM（arXiv:2304.02643）精修 mask → 与属性 GT 框 IoU 匹配得正/负证据区域。无属性真值的图（FSC-147 转 REC）用 Kimi K2.6 校验"该实例是否满足属性"。
- **过滤**：GroundingDINO 定位失败率 >20% 的属性模板剔除；负证据（属性为 False 的区域）按 1:1 采样入训练（强化判别）。
- **数量**：REC-8K 全量（≈10k 表达）+ FSC-147 转 REC 30k + 合成 OpenREC 20k；属性模板覆盖 ≥120 种属性。

### 3.2 模型/算法设计

- **骨干**：LLaVA-1.5-7B（arXiv:2310.03744）。
- **双查询头**：在 LLM 输出端并接 (1) counting head：对每个视觉 patch token 输出密度贡献，$\hat{c}=\text{sum}$（轻量 MLP，输入为 LLM 最后一层视觉 token 池化）；(2) evidence box head：用 GroundingDINO 式 query 在视觉特征上回归 $k$ 个框（k=5 上限）。均为 LoRA 训练的 adapter，不破坏原始文本生成。
- **损失**：$\mathcal{L}=\mathcal{L}_{SFT}+\lambda_c\,\mathcal{L}_{count}+\lambda_e\,\mathcal{L}_{box}+\lambda_{bc}\,\mathcal{L}_{bucket}$。
  - $\mathcal{L}_{count}=\text{MSE}(\log(\hat{c}+1),\log(c+1))$；
  - $\mathcal{L}_{box}=1-\text{IoU}(\hat{B},B_{GT})$（匈牙利匹配）；
  - $\mathcal{L}_{bucket}$：把计数分成桶 [1,2-4,5-9,10-19,20+] 的交叉熵（大数场景兜底）；
  - 初值 $\lambda_c=1,\lambda_e=0.5,\lambda_{bc}=0.3$。
- **计数一致性正则**：$\hat{c}$（密度积分）与 LLM 文本输出数字之间 KL 拉近，稳定大数。

### 3.3 训练流程

- QLoRA 4-bit，4×L40，batch=8，grad-acc=8 → 256；lr=2e-4 cosine；12k 步（~1.5 epoch）。
- 阶段：(1) 冻结 counting/box head 只训 LoRA 对齐（2k 步）；(2) 联合训练。
- 每 500 步验证 REC-8K val 子集 MAE 早停。

### 3.4 推理与评测流程

- 输入表达 → MLLM 输出计数（文本）+ counting head 输出 $\hat{c}$ → 融合：若分桶置信高取 head，否则取文本；证据区域框并列输出。
- 评测：REC-8K 官方划分；open 属性划分（新构建）；FSC-147 转 REC 子集；定位 F1（证据框与 GT IoU>0.5）。

## 4. 数据集细节

| 数据集 | 用途 | 说明 | 许可 |
|---|---|---|---|
| REC-8K（见 W2-Net 论文 CV XIII·108） | 主评测/训练 | 官方数据，含属性表达 | 随论文 |
| FSC-147（arXiv:2104.08391） | 转 REC 训练 | 类名+属性模板合成表达 | 学术 |
| **OpenREC（本工作）** | 开放属性评测 | 120 属性 × 200 图，GT 计数+证据框 | 随论文开源 |
| **Attr-Evidence（本工作）** | 训练标注 | 合成属性-证据区域对 | 随论文开源 |

## 5. 基线复现

| 基线 | 官方代码/权重 | 复现要点 |
|---|---|---|
| W2-Net（CV XIII·108） | 见论文（代码公开） | 官方权重，REC-8K 脚本 |
| CounTR（arXiv:2208.13721） | https://github.com/VergA3334/CounTR | 官方权重 |
| CountGD（arXiv:2407.04619） | https://github.com/vikvereb/CountGD | 官方权重（文本计数） |
| GeCo2（CV VII·66） | 见论文 | 官方权重（框 only，转 REC 输入） |
| Qwen-VL / LLaVA 直接数数 | 官方权重 | 同 prompt 直接回答数量 |
| Ours | — | §3 全量 |

**预期指标表**（REC-8K 官方口径，MAE↓；数值以复现为准）：

| 方法 | REC-8K MAE | REC-8K nRMSE | 属性子集 MAE | 大数(≥20) MAE | 定位 F1 |
|---|---|---|---|---|---|
| CounTR | 高 | 高 | — | 高 | — |
| CountGD | 中 | 中 | 中 | 中 | — |
| Qwen-VL 直接数数 | 高 | 高 | 高 | 很高 | — |
| W2-Net | 低 | 低 | 中 | 中 | — |
| **Ours** | **更低** | **更低** | **最低** | **最低** | **≥0.7** |

统一口径：REC-8K 官方表达集、同一解析脚本、分桶融合规则统一。

## 6. 实验矩阵

- **A. 主实验**：全基线 + Ours 全指标。
- **B. 消融**：B1 证据区域监督有无；B2 属性模板数量（40/80/120）；B3 counting head vs 纯文本计数；B4 分桶损失有无；B5 计数一致性正则有无；B6 骨干（LLaVA-1.5 vs Qwen2-VL-7B）。
- **C. 鲁棒性**：C1 多属性组合（2–3 属性）；C2 密集场景；C3 属性词换说（同义改写）。
- **D. 泛化性**：D1 unseen 属性（OpenREC 保留 20 属性不参与训练）；D2 跨域（FSC-147 转 REC）；D3 中文表达。

## 7. 评测协议

- **指标**：MAE、nRMSE（REC-8K 官方定义）；定位 F1（证据框 IoU>0.5）；属性子集 MAE。
- **均值±方差**：3 个种子（123/2024/7）；主模型报 mean±std。
- **显著性**：MAE 差配对 Bootstrap（n=1000，CI 95%）；主对比报 McNemar（属性级正确/错误）。
- **OpenREC 标签可信度**：双人抽检 200 例，Kappa>0.8。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 合成标注（API）+ 证据区域生成（GroundingDINO/SAM 并行） | 1 |
| P2 | 阶段1 对齐 + 联合训练 12k 步 | 3 |
| P3 | 评测（5 基线 + Ours × 4 评测集） | 1 |
| **合计** | | **≈5** |

存储：REC-8K + 合成 ≈ 80GB。API：DeepSeek V4 Pro 属性标注 ≈ $25；Kimi K2.6 校验 ≈ $10。总计 ≤ **$40**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | REC-8K/FSC-147 获取、基线复现（CounTR/CountGD/Qwen-VL） |
| W2 | 属性模板 + 合成标注 + 证据区域生成 |
| W3 | 双查询头实现 + 联合训练 |
| W4 | 主实验 A + B1/B3 消融 |
| W5 | 开放属性泛化 + 大数鲁棒 + 统计 + 论文初稿（CVPR 2027 截稿前 6 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| MLLM 数数本质弱（大数） | 中 | 高 | counting head + 分桶 + 大数一致性正则；密集场景合成强化 |
| 自动属性标注质量不稳 | 中 | 高 | Kimi 校验 + 负证据采样 + 抽检闭环 |
| unseen 属性泛化差 | 中 | 中 | 属性模板丰富度 + 模板无关化表述 |
| 证据框与属性错配 | 中 | 中 | IoU 阈值调优 + 匈牙利匹配 |
| 对比 W2-Net 复现难 | 中 | 低 | 若代码未开源，用其论文报告值 + 自建同口径实现 |

## 11. 论文写作计划

- **目标**：CVPR 2027（中-高）；备选 ACM MM 2027。
- **差异化卖点**：(1) 首个端到端 MLLM 双查询 REC（计数+证据区域）；(2) 属性-证据区域标注全自动合成；(3) 开放属性评测协议 OpenREC。
- **图表清单**：图1 双查询方法图；图2 证据区域可视化（正/负）；图3 unseen 属性泛化；图4 大数分桶策略；表1 主实验；表2 消融；表3 鲁棒性；表4 泛化；表5 定位 F1。
- **相关工作覆盖**：REC（W2-Net CV XIII·108）、计数（CounTR arXiv:2208.13721、CountGD arXiv:2407.04619、GeCo2 CV VII·66）、grounding（GroundingDINO arXiv:2303.05499、SAM arXiv:2304.02643、Ferret arXiv:2310.07704）、MLLM 基础（LLaVA arXiv:2304.08485/2310.03744）。

## 12. 参考文献

1. W2-Net（Computer Vision XIII·108）· Decoupling What to Count and Where to See for Referring Expression Counting（收藏论文）
2. Ranjan et al. Learning To Count Everything（FSC-147）. arXiv:2104.08391（CVPR 2021）.
3. Amini-Naieni et al. CountGD: Multi-Modal Open-World Counting. arXiv:2407.04619（CVPR 2024）.
4. Liu et al. CounTR: Transformer-based Generalised Visual Counting. arXiv:2208.13721（ICCV 2023）.
5. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training. arXiv:2303.05499.
6. Kirillov et al. Segment Anything. arXiv:2304.02643（ICCV 2023）.
7. Peng et al. Ferret: Refer and Ground Anything Anywhere at Any Granularity. arXiv:2310.07704.
8. Liu et al. Improved Baselines with Visual Instruction Tuning（LLaVA-1.5）. arXiv:2310.03744.
9. GeCo2（Computer Vision VII·66）· Generalized-Scale Object Counting with Gradual Query Aggregation（收藏论文，arXiv:2511.08048）
