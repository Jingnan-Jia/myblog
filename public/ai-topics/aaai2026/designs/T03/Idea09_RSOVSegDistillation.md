# 实验设计书：遥感开放词汇分割的双语知识蒸馏与旋转等变预训练

## 0. 摘要

本实验设计把遥感（RS）领域知识蒸馏回注 CLIP 文本编码器（生成式蒸馏，覆盖中英文 RS 术语），并用旋转等变 patch 自监督预训练（旋转预测 aux loss）替代多方向暴力 cost map，得到端到端轻量化遥感开放词汇分割：单方向 cost map + 等变 token。两阶段训练（文本端蒸馏 → 联合微调 RSKT-Seg 式分割头），目标是 OVRSISBench mIoU 与零样本 mIoU 显著提升且推理更快（≥2× RSKT-Seg）。数据侧用 OVRSISBench/LoveDA/SAMRS + DeepSeek/Kimi 生成的 RS 场景文本对。4×L40 约 10 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

遥感开放词汇分割：输入 RS 图像与任意文本类名（如"农田""水体"），输出像素级分割。难点：(a) CLIP 在 RS 词汇上零样本语义弱（自然图像预训练）；(b) RS 图像朝向任意、旋转变化大，多方向代价图近似旋转不变计算开销大；(c) RS 语义标注稀缺。我们蒸馏 RS 双语知识进文本编码器，并用旋转等变预训练给视觉编码器以等变归纳偏置。

### 1.2 相关工作不足

- 收藏论文 `OVRSISBench/RSKT-Seg`（Computer Vision V·9·Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing）：首个标准化 RS 开放词汇分割基准；RS-CMA 多方向代价图聚合（旋转不变）+ RS-Fusion + RS-Transfer，+3.8 mIoU/+5.9 mACC 且快 2×。但：(a) 多方向暴力计算，可被等变预训练替代；(b) 依赖 CLIP 弱语义；(c) RS 知识未回注文本编码器。
- `CLIP`（arXiv:2103.00020）开放词汇骨干但 RS 语义弱。
- `SAMRS`（arXiv:2305.02034，NeurIPS 2023 D&B）大规模 RS 分割数据集（105k 图）。
- `LoveDA`（arXiv:2110.08733，NeurIPS 2021 D&B）RS 土地覆盖域自适应数据集。
- 通用开放词汇分割（如 GroundingDINO+SAM 组合，arXiv:2303.05499/2304.02643）在 RS 上未适配。

空白：**RS 双语知识蒸馏回注文本编码器 + 旋转等变预训练替代多方向暴力计算**的端到端方案。

### 1.3 为什么是现在、为什么你的环境适合做

OVRSISBench 2026 AAAI 刚中，其基准可复现；RSKT-Seg 的"多方向暴力"缺口是明确的工程+方法增量。本环境：CLIP-L 蒸馏 + 分割头 4×L40 可行；DeepSeek V4 Flash 批量生成 RS 场景描述、Kimi K2.6 生成中英双语 RS 术语表便宜；SAMRS 数据可扩展。

## 2. 研究目标与可验证假设

- **H1**：RS 双语蒸馏提升文本编码器 RS 语义。*可观测结果*：零样本 mIoU（unseen 类）相对 CLIP 基线提升 ≥8 点；RS 术语检索命中率显著提升。
- **H2**：旋转等变预训练替代多方向代价图。*可观测结果*：单方向 cost map + 等变 token 的 mIoU ≈ 多方向（Δ≤1.0），推理提速 ≥2×。
- **H3**：双语（中英）优于单语蒸馏。*可观测结果*：中英双语文本编码在中文+英文评测集上都更好，且跨语言一致性高。
- **H4**：两阶段训练优于直接联合。*可观测结果*：先蒸馏文本端再联合微调 vs 一步到位，最终 mIoU 高 ≥2 点。

## 3. 总体方法设计

### 3.1 数据流水线

- **RS 双语术语表（Kimi K2.6）**：生成 300 个 RS 概念（地物/土地覆盖/场景）的**中英双语**定义与正负例句（如"农田/cropland"+"旱地 vs 水田"负样本对）。
- **RS 场景描述（DeepSeek V4 Flash，批量）**：对 LoveDA/SAMRS 每图生成 2–5 条 RS 场景描述（覆盖类名 + 空间布局 + 干扰项），作为蒸馏图文对。
- **蒸馏目标构造**：对每张图取真实标签中的类名组成"正类名集"与"负类名集"（混淆类），形成判别性蒸馏对。
- **旋转等变数据**：对训练图做 8 方向旋转（0/45/90/…/315°）加裁剪，作为旋转预测 aux 任务输入。
- **过滤与数量**：术语表 300 概念 × 中英；蒸馏图文对 = SAMRS 子集 20k 图 × 3 描述 ≈ 60k 对；分割微调用 OVRSISBench 官方划分 + LoveDA 4,322 训练图。

### 3.2 模型/算法设计

- **文本端蒸馏（阶段一）**：CLIP 文本编码器 + 轻量 LoRA，目标：生成式蒸馏——用"类名定义 + 描述"监督使文本编码器更懂 RS：
  $\mathcal{L}_{distill}=\lambda_1\,\mathcal{L}_{clip}(\text{图文对齐})+\lambda_2\,\mathcal{L}_{contrast}(\text{正/负类名对比})+\lambda_3\,\mathcal{L}_{entail}(\text{中英一致性})$（英文与中文描述互翻译一致性，用 Kimi 校验）。
- **旋转等变预训练（阶段一，视觉侧）**：在 CLIP 视觉编码器加旋转预测头：对 8 方向旋转 patch 输出方向标签（aux loss $\mathcal{L}_{rot}=CE$），并与等变一致性正则 $\mathcal{L}_{equiv}= \|f_\theta(R^\phi(I))-R^\phi(f_\theta(I))\|^2$ 结合。等变 token：把旋转不变性编码进 patch token（相对位置重排 + 旋转对称池化）。
- **分割头（阶段二）**：继承 RSKT-Seg 式结构——单方向 cost map（CLIP 文本-图像相似度）→ 代价图融合（RS-Fusion）→ 解码。用等变 token 替代多方向 cost map 聚合。
- **损失**：$\mathcal{L}=\mathcal{L}_{seg}+\gamma\mathcal{L}_{rot}+\delta\mathcal{L}_{equiv}$；$\mathcal{L}_{seg}$=mIoU 辅助 + CrossEntropy；初值 $\gamma=0.1,\delta=0.2$。

### 3.3 训练流程

- 阶段一（4×L40，6 GPU·天）：文本蒸馏 8k 步 + 视觉等变预训练 6k 步（可分 2 卡并行）。
- 阶段二（4×L40，4 GPU·天）：冻结/半冻结编码器，训练分割头 10k 步；再端到端 LoRA 联合 4k 步。
- 优化：AdamW；lr 文本 1e-5（LoRA）、视觉 1e-4（LoRA）、头 3e-4；batch 16；分辨率 512×512。

### 3.4 推理与评测流程

- 输入图像 + 类名 → 单方向 cost map → 融合 → 分割。报告 mIoU/mACC、零样本 mIoU（unseen 类）、FPS。
- 评测：OVRSISBench 官方划分（LoveDA/Vaihingen/Potsdam/WHU-OPT 域）；中文类名评测子集（新增）。

## 4. 数据集细节

| 数据集 | 用途 | 说明 | 许可 |
|---|---|---|---|
| OVRSISBench（CV V·9） | 训练/评测 | 基于 LoveDA/Vaihingen/Potsdam/WHU-OPT 的标准基准 | 随论文 |
| LoveDA（arXiv:2110.08733） | 训练（域覆盖） | 官方划分 | 学术 |
| SAMRS（arXiv:2305.02034） | 蒸馏数据 | 105k 图子集 | 学术 |
| ISPRS Vaihingen/Potsdam（官方基准） | 评测 | 官方 | 学术 |
| WHU-OPT（公开数据集，官方主页） | 评测 | 官方 | 学术 |
| **RS-BiDict + RS-Cap（本工作）** | 蒸馏目标 | 300 概念双语 + 60k 描述 | 随论文开源 |

## 5. 基线复现

| 基线 | 官方代码/权重 | 复现要点 |
|---|---|---|
| RSKT-Seg（CV V·9） | 见论文（代码公开） | 官方权重/脚本，多方向 cost map |
| GroundingDINO+SAM 组合（arXiv:2303.05499/2304.02643） | 官方仓库 | 免训练开放词汇分割 |
| CLIP 直接像素对齐基线（arXiv:2103.00020） | 官方 | 文本-像素相似度硬切 |
| OpenSeeD 式通用分割（官方实现，若代码可用） | 官方 | 通用开放词汇对照 |
| SAN（RS 分割方法，官方仓库） | 官方 | RS 专有对照 |
| Ours | — | §3 全量 |

**预期指标表**（OVRSISBench mIoU/mACC，FPS；数值以复现为准）：

| 方法 | 全类 mIoU↑ | 零样本 mIoU↑ | mACC↑ | FPS↑ | 推理开销↓ |
|---|---|---|---|---|---|
| CLIP 对齐 | 低 | 低 | 低 | 高 | 低 |
| GroundingDINO+SAM | 中 | 中 | 中 | 中 | 中 |
| OpenSeeD 式 | 中高 | 中 | 中高 | 中 | 中 |
| RSKT-Seg | 高 | 中 | 高 | 中（多方向慢） | 高 |
| **Ours** | **最高** | **显著提升** | **最高** | **≥2× RSKT-Seg** | **低** |

统一口径：同一 OVRSISBench 划分、同一分辨率、同一推理硬件（L40）。

## 6. 实验矩阵

- **A. 主实验**：全基线 + Ours 全指标。
- **B. 消融**：B1 等变 loss 有无（rot/equiv 单独+联合）；B2 文本蒸馏有无；B3 双语 vs 单语 vs 无；B4 单方向 vs 多方向 cost map（同头）；B5 蒸馏数据量（10k/30k/60k）。
- **C. 鲁棒性**：C1 旋转测试（推理时随机旋转输入）；C2 跨域（训练域→未训练域零样本）；C3 低分辨率/噪声。
- **D. 泛化性**：D1 中文类名评测；D2 未见新类（术语表外）；D3 不同骨干（ViT-B vs ViT-L）。

## 7. 评测协议

- **指标**：mIoU、mACC（OVRSISBench 官方定义）；零样本 mIoU（unseen 类，官方清单）；FPS（L40，batch=1/4）；推理开销（FLOPs + 显存）。
- **均值±方差**：3 个训练种子，报 mean±std。
- **显著性**：mIoU 差配对 Bootstrap（n=1000）。
- **双语一致性**：中文与英文类名评测的 mIoU 差 + 一致性 Kappa。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 术语表/描述生成（API）+ 旋转数据准备 | 0.5 |
| P2 | 阶段一（文本蒸馏 8k + 等变 6k，2+2 卡并行） | 5–6 |
| P3 | 阶段二（分割头 10k + 联合 4k） | 4 |
| P4 | 评测（基线 × 数据集） | 1–1.5 |
| **合计** | | **≈10–12** |

存储：SAMRS/LoveDA/OVRSISBench ≈ 180GB。API：DeepSeek V4 Flash 场景描述 ≈ $15；Kimi K2.6 术语表/校验 ≈ $12。总计 ≤ **$30**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | OVRSISBench/LoveDA/SAMRS 获取 + 基线复现（RSKT-Seg） |
| W2 | 术语表/描述 + 蒸馏图文对构建 |
| W3 | 阶段一训练 + B2/B3 消融 |
| W4 | 阶段二 + A 主实验 + 旋转鲁棒 |
| W5 | 泛化（中文/新类）+ 统计 + 论文初稿（TGRS 投稿 / CVPR 2027 备选） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| RS 数据稀缺/文本质量受限 | 中 | 高 | SAMRS 扩充 + 诚实报告零样本基线 |
| 等变预训练增益小 | 中 | 中 | 做好消融归因；若无效则聚焦蒸馏 + 单方向收敛 |
| 双语蒸馏带来文本编码漂移 | 中 | 中 | 蒸馏时冻结原始 CLIP 分支 + 一致性正则 |
| 跨域零样本不理想 | 中 | 中 | 分域报告 + 域自适应扩展（后续工作） |
| 多方向基线对比不公平 | 低 | 中 | 统一成本 map 实现与 FLOPs 报告 |

## 11. 论文写作计划

- **目标**：IEEE TGRS（高）；CVPR 2027（中）备选。
- **差异化卖点**：(1) RS 双语知识蒸馏回注文本编码器（首个）；(2) 旋转等变预训练替代多方向暴力 cost map（效率+精度）；(3) 单方向 + 等变 token 的轻量化端到端 + 双语评测协议。
- **图表清单**：图1 方法图（两阶段）；图2 旋转等变消融可视化；图3 双语蒸馏前后文本相似度矩阵；图4 跨域/旋转鲁棒案例；表1 主实验；表2 消融；表3 鲁棒性；表4 泛化；表5 FPS/FLOPs。
- **相关工作覆盖**：RS 开放词汇分割（OVRSISBench/RSKT-Seg CV V·9）、RS 数据（LoveDA arXiv:2110.08733、SAMRS arXiv:2305.02034）、开放词汇基础（CLIP arXiv:2103.00020、GroundingDINO arXiv:2303.05499、SAM arXiv:2304.02643）。

## 12. 参考文献

1. OVRSISBench/RSKT-Seg（Computer Vision V·9）· Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing（收藏论文）
2. Radford et al. Learning Transferable Visual Models From Natural Language Supervision（CLIP）. arXiv:2103.00020（ICML 2021）.
3. Wang et al. LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation. arXiv:2110.08733（NeurIPS 2021 D&B）.
4. Wang et al. SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model. arXiv:2305.02034（NeurIPS 2023 D&B）.
5. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training. arXiv:2303.05499.
6. Kirillov et al. Segment Anything. arXiv:2304.02643（ICCV 2023）.
