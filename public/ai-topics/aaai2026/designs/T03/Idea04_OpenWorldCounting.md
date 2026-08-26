# 实验设计书：文本 + 稀疏框渐进查询的开放世界计数

## 0. 摘要

本实验设计在 GeCo2（尺度渐进查询聚合）架构上统一「文本（类名/属性）+ 稀疏示例框」多粒度 prompt：引入 CLIP 文本分支，把类名/属性文本嵌入与示例框特征融合为 dense query，并做尺度分离的渐进聚合（尺度感知上采样，免 ad-hoc tile），实现 seen 类给框+文本、unseen 类仅给文本即可计数的开放词汇计数。配套 DeepSeek V4 Flash 批量生成的类名/属性文本增强数据。目标是 FSC-147/CARPK/CountGD 开放集上，unseen 类计数 MAE 相对 CounTR/CountGD/CountGD++ 显著下降，同时保持 seen 类精度。4×L40 约 8 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

开放世界计数：输入图像 $I$ 与目标描述（seen 类给示例框 $E=\{b_1..b_k\}$ + 类名 $t$；unseen 类只给 $t$），输出目标实例数 $\hat{c}$。核心难点：(a) 未见类别需语义泛化；(b) 目标尺度可差 100 倍；(c) 文本-视觉对齐歧义（同名不同类）。我们把 prompt 统一为 $p=\text{fuse}(f_{text}(t),\,f_{boxes}(E))$，并让尺度作为一等公民进入渐进查询聚合。

### 1.2 相关工作不足

- 收藏论文 `GeCo2`（Computer Vision VII·66·Generalized-Scale Object Counting with Gradual Query Aggregation）显式建模尺度（渐进查询聚合、无 ad-hoc tile），但**只用框示例**，无法零样本泛化到未见类。
- `CountGD`（arXiv:2407.04619，CVPR 2024）文本+框联合 prompt，但尺度鲁棒性弱、依赖 ad-hoc tile。
- `CountGD++`（arXiv:2512.23351）进一步泛化 prompt，仍未做尺度渐进聚合。
- `CounTR`（arXiv:2208.13721，ICCV 2023）少样本计数 baseline，无文本。
- 数据集侧：`FSC-147`（arXiv:2104.08391，CVPR 2021）147 类少样本计数基准，提供框+点数。

空白：**"尺度渐进聚合 + 文本/框多粒度 prompt + 开放词汇评测"三者从未在单一框架内统一**；GeCo2 与 CountGD/CountGD++ 优势各自独立。

### 1.3 为什么是现在、为什么你的环境适合做

GeCo2 2026 年 AAAI 刚中，其代码公开，直接在其架构上加文本分支是低成本高增量；CountGD++ 亦 2025 底发布，说明文本+框计数赛道正热。本环境：ViT-L 全参/LoRA 训练 4×L40 够；DeepSeek V4 Flash 批量生成类名/属性描述做数据增强便宜可控。

## 2. 研究目标与可验证假设

- **H1**：文本分支带来 unseen 类零样本计数能力。*可观测结果*：仅给文本（无框）时 unseen 类 MAE 显著低于 CounTX 风格弱基线且不掉 seen 类。
- **H2**：尺度分离渐进聚合优于 CountGD 式 tile。*可观测结果*：极端尺度比（>32×）子集 MAE 降 ≥15%，且无 tile 伪影。
- **H3**：文本+框互补优于单一模态。*可观测结果*：seen 类框+文本 vs 仅框 vs 仅文本，框+文本 MAE 最低，且文本可纠正框歧义。
- **H4**：文本增强（属性短语）提升细粒度区分。*可观测结果*：对混淆类（如"皮卡车 vs 轿车"）用属性描述后计数误差下降。

## 3. 总体方法设计

### 3.1 数据流水线

- **基础**：FSC-147（arXiv:2104.08391）官方 train/val/test；CARPK（公开数据集，无 arXiv，官方主页下载）作跨域测试；CountGD 论文用到的开放集（见其仓库数据清单）。
- **文本增强（DeepSeek V4 Flash，批量）**：对每个类名生成 N=8 条属性描述：
```
For object class "{class}", write 8 short attribute phrases (color/shape/typical context/part names) useful for detection in aerial and street images. Avoid naming other object classes.
```
过滤：Kimi K2.6 校验描述与图像内容一致性（抽检 300 例）；描述长度 3–15 token。
- **多尺度合成**：把 FSC-147 图做 2×/4×/0.5× 重采样 + 对应密度图重标定，扩充 scale 覆盖，构成多尺度训练对（+60k 样本）。
- **数量**：训练样本 FSC-147 官方 ~6135 图 + 合成 60k；文本描述 147 类 × 8 条。

### 3.2 模型/算法设计

- **骨干**：ViT-L-336（CLIP，arXiv:2103.00020 权重 init）。
- **文本分支**：CLIP 文本编码器（冻结或 LoRA），输出类名/属性句向量 $z_t$。
- **示例框分支**：ROI-Align 提取每框特征 → 平均池化得 $z_e$。
- **融合为 dense query**：$\hat{q}_0=\text{MLP}([z_t;z_e])$ 广播为初始低分辨率 dense query，再经渐进聚合。
- **渐进查询聚合（继承 GeCo2 设计）**：L=4 层，从 1/32 到 1/2 分辨率逐层把文本/框先验与图像特征交叉：
  $\hat{q}_l=\text{Agg}(f_l(I),\hat{q}_{l-1})$，$\hat{q}_{l-1}$ 用**尺度感知上采样**（可变形卷积 + 相对位置编码），替代 ad-hoc tile。
- **输出头**：密度头 $\hat{D}=\text{ConvHead}(\hat{q}_L)$，积分得 $\hat{c}=\sum_{ij}\hat{D}_{ij}$；可选检测头（框回归）。
- **损失**：$\mathcal{L}=\mathcal{L}_{density}+\alpha\mathcal{L}_{MAE}+\beta\mathcal{L}_{det}$（det 可选，继承 GeCo2 统一计数+检测），$\alpha=1,\beta=0.5$ 初值。
- **文本-框交叉验证（推理期）**：当两者给出不一致候选区域时，取高置信并集并二次校正（缓解 H4 类歧义）。

### 3.3 训练流程

- 两阶段：(1) 冻结 CLIP 视觉+文本编码器，只训 query/聚合/密度头（模拟 CountGD 式文本对齐，4k 步）；(2) 视觉编码器 LoRA + 全头联合（10k 步）。
- 4×L40，batch=16（有效），lr 头 3e-4 / LoRA 2e-4，AdamW cosine；输入分辨率 336（tile-free）。
- seen 类训练策略：每 batch 中 70% 样本给框+文本、30% 只给文本（迫使文本分支独立工作）。

### 3.4 推理与评测流程

- seen 类：给类名 + 1–3 框；unseen 类：只给类名（FSC-147 官方 unseen 划分，147 类中 77 类 seen/70 unseen，按官方文件）。零样本：仅文本。
- 评测：MAE/RMSE/nRMSE + 检测 AP（可选）；跨域 CARPK、SeaHeaven（公开数据集）零样本。

## 4. 数据集细节

| 数据集 | 用途 | 划分 | 许可 |
|---|---|---|---|
| FSC-147（arXiv:2104.08391） | 训练/seen-unseen 评测 | 官方（train 6135 图 / val / test，147 类） | 学术 |
| FSC-147-D（CountGD 论文扩展描述版） | 可选文本监督 | 见 CountGD 仓库 | 学术 |
| CARPK（官方数据集，无 arXiv） | 跨域零样本（车辆） | 官方 | 学术 |
| SeaHeaven（官方数据集） | 跨域零样本（航空） | 官方 | 学术 |
| 多尺度合成（本工作） | 训练增强 | — | 随论文公开 |
| CountGD 开放集（见 arXiv:2407.04619 仓库） | 评测 | 官方 | 学术 |

## 5. 基线复现

| 基线 | 官方代码 | 复现要点 |
|---|---|---|
| CounTR（arXiv:2208.13721） | https://github.com/VergA3334/CounTR | 官方训练/权重 |
| CountGD（arXiv:2407.04619） | https://github.com/vikvereb/CountGD | 官方权重 + 官方评测脚本 |
| CountGD++（arXiv:2512.23351） | 同 CountGD 仓库 | 官方权重 |
| GeCo2（CV VII·66） | 见论文（代码公开） | 官方权重（框 only） |
| DAVE / Loca（计数基线，官方仓库） | 见各自官方仓库 | 复现到同一评测脚本 |
| Ours | — | §3 全量 |

**预期指标表**（FSC-147 test，MAE↓/nRMSE↓，数值以复现为准）：

| 方法 | seen MAE | unseen MAE | 全体 MAE | CARPK MAE | 极端尺度(>32×) MAE |
|---|---|---|---|---|---|
| CounTR | 基准 | — | 基准 | 高 | 高 |
| CountGD | 低 | 中 | 低 | 中 | 中 |
| CountGD++ | 低 | 中低 | 低 | 中 | 中 |
| GeCo2 | 最低 | 不支持 | 最低 | 低 | 最低 |
| **Ours（框+文本）** | **≤GeCo2** | **新纪录** | **新纪录** | **最低** | **最低** |
| Ours（仅文本，零样本） | — | 显著低于 CounTX 弱基线 | — | — | — |

统一口径：同一 FSC-147 官方划分、同一分辨率、同一密度积分后处理；unseen 定义统一用官方清单。

## 6. 实验矩阵

- **A. 主实验**：全基线 vs Ours（框+文本 seen / 仅文本 unseen）。
- **B. 消融**：B1 文本分支有无；B2 渐进聚合层数 L∈{2,4,6}；B3 尺度感知上采样 vs 双线性；B4 文本描述条数 N∈{1,4,8,16}；B5 训练时仅文本比例（0/30/50%）；B6 视觉编码器 LoRA vs 冻结。
- **C. 鲁棒性**：C1 极端尺度比子集；C2 密集场景（>100 目标）；C3 遮挡/截断；C4 文本拼写/描述噪声。
- **D. 泛化性**：D1 CARPK/SeaHeaven 零样本；D2 未见数据集（CountGD 开放集）；D3 消歧（皮卡 vs 轿车）属性描述增益。

## 7. 评测协议

- **指标**：MAE、RMSE、nRMSE（官方 FSC-147 定义）；检测 AP（可选，GeCo2 口径）；跨域用官方脚本。
- **均值±方差**：5 个种子训练（对全量训练做 seed 重复成本高 → 对 10% 子集训练做 seed 重复估计方差，主模型报告官方划分单次 + 子集方差）。
- **显著性**：unseen MAE 差配对 Bootstrap（n=1000）。
- **文本增强可复现**：发布全部描述文件 + 过滤脚本；描述种子固定（seed=7）。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 文本增强生成（API）+ 多尺度合成 + 缓存 | 1 |
| P2 | Stage1（CLIP 冻结，4k 步） | 2 |
| P3 | Stage2（联合 LoRA，10k 步） | 4 |
| P4 | 全量评测（5 模型 × 5 数据集） | 1–2 |
| **合计** | | **≈8–9** |

存储：FSC-147 + 合成 + CARPK ≈ 120GB。API：DeepSeek V4 Flash 描述生成 ≈ $8；Kimi K2.6 校验 ≈ $5。总计 ≤ **$15**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | 数据管线（下载 + 文本增强 + 合成）+ CounTR/CountGD 基线复现 |
| W2 | 架构实现 + Stage1 训练 |
| W3 | Stage2 训练收敛、B1/B2/B3 消融 |
| W4 | 主实验 A + unseen 评测 + D1 跨域 |
| W5 | C/D 全消融 + 统计 + 论文初稿（ICCV 2027 截稿前 6 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 文本歧义（同名不同类） | 中 | 高 | 文本-框交叉验证 + 混淆类消融 + 属性描述增强 |
| 密集场景内存/密度上限 | 中 | 中 | 高分辨率分块推理（推理期仅，不改变训练） |
| unseen 提升幅度小 | 中 | 中 | 强化"仅文本"训练比例 + 属性短语增强；主贡献改为"统一框架 + 评测协议" |
| GeCo2 代码适配成本 | 中 | 中 | 从零实现渐进聚合（~400 行）亦可 |
| 数据许可/带宽 | 低 | 低 | CARPK 用官方链接 |

## 11. 论文写作计划

- **目标**：ICCV 2027 主投（中-高）；备选 CVPR 2027。
- **差异化卖点**：(1) 首个统一「文本+框+尺度渐进聚合」的开放世界计数；(2) 免 ad-hoc tile 的极端尺度处理；(3) 可复现的 seen/unseen 双协议 + 文本增强数据。
- **图表清单**：图1 方法图；图2 多粒度 prompt 示意图；图3 尺度鲁棒性对比（tile 伪影消除）；图4 unseen 零样本案例 + 失败案例；表1 主实验；表2 消融；表3 跨域；表4 极端尺度；表5 消歧。
- **相关工作覆盖**：少样本计数（CounTR arXiv:2208.13721、GeCo2 CV VII·66、FSC-147 arXiv:2104.08391）、文本计数（CountGD arXiv:2407.04619、CountGD++ arXiv:2512.23351）、开放词汇（CLIP arXiv:2103.00020）、参考表达计数（W2-Net CV XIII·108）。

## 12. 参考文献

1. GeCo2（Computer Vision VII·66）· Generalized-Scale Object Counting with Gradual Query Aggregation（收藏论文，arXiv:2511.08048，AAAI 2026）
2. Ranjan et al. Learning To Count Everything（FSC-147）. arXiv:2104.08391（CVPR 2021）.
3. Amini-Naieni et al. CountGD: Multi-Modal Open-World Counting. arXiv:2407.04619（CVPR 2024）.
4. Amini-Naieni et al. CountGD++: Multi-Modal Open-World Counting, Iterative Refinement, and Beyond. arXiv:2512.23351.
5. Liu et al. CounTR: Transformer-based Generalised Visual Counting. arXiv:2208.13721（ICCV 2023）.
6. Radford et al. Learning Transferable Visual Models From Natural Language Supervision（CLIP）. arXiv:2103.00020（ICML 2021）.
7. W2-Net（Computer Vision XIII·108）· Decoupling What to Count and Where to See for Referring Expression Counting（收藏论文）
