# 实验设计书：分辨率自适应视觉 Token 预算训练

## 0. 摘要

本实验设计针对 MLLM 对输入分辨率的敏感性问题，在 LLaVA-1.5/1.6 架构的视觉编码器后加一个轻量「预算头」，按 patch 语义重要度动态分配视觉 token 预算，并用多分辨率混合训练 + 相邻分辨率 logits 的 KL 一致性正则，在保持 MME/MMMU 精度的同时显著提升 Res-Bench 上的分辨率鲁棒性。产出：预算头架构、多分辨率训练协议、「稳定-精度联合」指标（ρ、ACE/RCE 加权），以及"预算-精度"权衡曲线。算力需求低（7B LoRA，4×L40 约 5 GPU·天），主要工作量在评测协议与训练稳定性。

## 1. 研究背景与动机

### 1.1 问题定义

MLLM 推理时把图像切成固定数量视觉 token。当输入分辨率偏离训练分布（低至 224、高至 4×）时，多数 MLLM 性能骤降（`Res-Bench` NLP II·77 指出此现象）。问题定义为：给定总视觉 token 预算 $B$ 与图像 $I$，如何按 patch 内容复杂度分配 $B$ 个 token，使模型在任意分辨率下都能保持接近最优的精度？同时要求模型在相邻分辨率下的输出保持一致（一致性正则）。

### 1.2 相关工作不足

- 收藏论文 `Res-Bench`（Natural Language Processing II·77·Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input）建立 14400 样本、12 档分辨率的评测基准并给出 Spearman ρ、ACE/RCE 稳定性指标，但**只测不训**，未给任何预算感知的训练方案。
- `LLaVA-NeXT` 的动态 anyres 切图是启发式（按面积规则切，不是内容感知），无"预算-性能"显式权衡训练，也无跨分辨率一致性约束。
- `Qwen2-VL`（arXiv:2409.12191）原生多分辨率（Naive Dynamic Resolution）但不做 token 预算分配，144 万 token 上限仍浪费。
- 通用 token 剪枝（如 EViT）只做图像分类，未验证 MLLM 图文跨模态场景与长视觉 token 一致性。

空白：**预算分配训练 + 跨分辨率一致性正则**在 MLLM 上无系统性工作；Res-Bench 是现成评测资产但缺训练侧配套。

### 1.3 为什么是现在、为什么你的环境适合做

Res-Bench 2026 年刚出（AAAI 2026 收藏），"评测→训练"闭环缺口明显，紧跟热点。本环境 4×L40 足以训练 7B LoRA；多分辨率数据预处理可离线并行；DeepSeek V4 Flash 可批量生成多分辨率答案作一致性评估的旁证。基准类工作算力需求低、可复现性强，适合 NeurIPS 2026 D&B / 主会。

## 2. 研究目标与可验证假设

- **H1**：预算头能在相同预算下显著提升低/高分辨率精度。*可观测结果*：固定 576 token 预算，0.25× 与 4× 分辨率下 Res-Bench 平均 ACC 相对 LLaVA-1.5 基线提升 ≥5 点。
- **H2**：KL 一致性正则提升跨分辨率稳定性。*可观测结果*：加入正则后 Spearman ρ ≥ +0.05、ACE/RCE 显著改善，且 MME/MMMU 不掉点（Δ≤±1.0）。
- **H3**：内容复杂度感知预算分配优于均匀分配。*可观测结果*：用 patch 重要度排序 vs 随机/均匀剪枝，固定预算下 ACC 高 ≥2 点（在纹理复杂图像上更明显）。
- **H4**：预算-精度曲线单调且可控。*可观测结果*：预算从 144→288→576→1152→2304 递增时精度单调不降，且报告曲线供用户选预算。

## 3. 总体方法设计

### 3.1 数据流水线

- **基础数据**：LLaVA-Instruct-665K（arXiv:2304.08485 官方）为 SFT 数据；MME/MMMU/Res-Bench 仅用于评测（严格不相交，Res-Bench 不参与训练）。
- **多分辨率重采样**：对每张训练图按对数均匀分布采样分辨率 $r\in\{0.25,0.5,0.75,1,1.5,2,3,4\}\times$ 基准（基准=原始尺寸 clamp 到 336 短边），同一样本生成 2 个相邻分辨率视图（比率 2:1 或 1:2），供 KL 正则用。预处理与缓存全部离线并行。
- **token 预算表**：训练时从 $\{144,288,576,1152\}$ 中随机抽 $B$，每个样本配一次，保证预算头见过各种预算。

### 3.2 模型/算法设计

- **架构**：LLaVA-1.5-7B + 预算头。预算头 = MLP(2×256) + sigmoid，输入为每个 patch 的 CLS 关联度 $a_i$（vision encoder 第 12 层注意力对该 patch 的平均权重）与 patch 梯度范数特征 $g_i$（可选），输出保留概率 $p_i\in(0,1)$。
- **预算分配**：总预算 $B$，第 $i$ 个 patch 重要性 $\pi_i=p_i\cdot a_i$，选 top-$B$ 个 patch 保留，其余 mask 掉（在 cross-attention 中置 $-\infty$，不物理裁剪以保持 token 对齐）。
- **损失**：$\mathcal{L}=\mathcal{L}_{SFT}+\gamma\,D_{KL}\big(\text{softmax}(\text{logits}^{(r_a)}/T)\|\text{softmax}(\text{logits}^{(r_b)}/T)\big)+\eta\,\mathcal{L}_{budget}$。
  - $r_a,r_b$ 为相邻分辨率；$T=2$；$\gamma=0.1$ 初值（消融 0/0.05/0.1/0.2）。
  - $\mathcal{L}_{budget}=\sum_i\max(0,\pi_i-\tau_{topk})$ 为熵正则/稀疏正则（鼓励预算头尖锐），$\tau_{topk}$ 取 batch 内第 $B$ 大值。
- **训练策略**：三阶段——(1) 冻结主干 + LLM，只训预算头（蒸馏初始化：用均匀预算训练 2k 步后接入）；(2) 预算头 + LoRA 联合训练；(3) 全部 LoRA 微调（含 SFT loss）。

### 3.3 训练流程

- QLoRA 4-bit，4×L40；batch=8/卡，grad-acc=4 → 有效 128；lr 预算头 1e-3、LoRA 2e-4；AdamW cosine；max_seq_len 2048；总步数 ~15k（预算头 stage1 2k + 联合 13k）。
- 每 1k 步在 Res-Bench 12 档分辨率子集（抽样 200 样本）上算 ρ/ACC 早停。

### 3.4 推理与评测流程

- 推理：给定预算 $B$ → 预算头打分 → top-$B$ token → 单次前向。报告不同 $B$ 下的 ACC。
- 评测：Res-Bench 全量（14400，12 档 × 6 能力）输出每档 ACC、Spearman ρ、ACE/RCE；MME/MMMU 精度保持检查；可选 DeepSeek V4 Flash 对同图不同分辨率生成答案的一致性评估（LLM-as-judge 附加证据）。

## 4. 数据集细节

| 数据集 | 用途 | 说明 | 许可 |
|---|---|---|---|
| LLaVA-Instruct-665K | 训练 | 官方版本，多分辨率重采样后缓存 | 学术 |
| Res-Bench（NLP II·77） | 评测 | 14400 样本、12 档分辨率、6 能力维度；官方仓库取数据 | 随论文公开 |
| MME（arXiv:2306.13394） | 评测（防掉点） | 官方 | 学术 |
| MMMU（arXiv:2311.16502） | 评测（防掉点） | 官方 val | 学术 |
| MMBench（arXiv:2307.06281） | 附加泛化 | 官方 via VLMEvalKit | 学术 |
| InstructBLIP（arXiv:2305.06500） | 对照骨干（可选） | 复现其分辨率行为 | 学术 |

## 5. 基线复现

| 基线 | 官方代码/权重 | 复现要点 |
|---|---|---|
| LLaVA-1.5-7B | https://github.com/haotian-liu/LLaVA | 固定 336 网格，官方权重 |
| LLaVA-NeXT-7B | 同上 | 动态 anyres 启发式切图 |
| Qwen2-VL-7B | https://github.com/QwenLM/Qwen2-VL | 原生多分辨率（作强对照） |
| InternVL-7B | https://github.com/OpenGVLab/InternVL | 同上 |
| InstructBLIP-7B | https://github.com/salesforce/LAVIS | 固定分辨率对照 |
| 均匀预算（自建） | — | 本方法去掉预算头=均匀 top-$B$ |
| Ours | — | §3 全量 |

**预期指标表**（Res-Bench 官方口径；数值以复现为准）：

| 方法 | 平均 ACC（12 档） | Spearman ρ↑ | ACE/RCE 稳定分 | MME | MMMU |
|---|---|---|---|---|---|
| LLaVA-1.5 | 基准 | 基准 | 基准 | 基准 | 基准 |
| LLaVA-NeXT | 略高 | 中 | 中 | 略高 | 略高 |
| Qwen2-VL-7B | 强 | 强 | 强 | 强 | 强 |
| 均匀预算 | 中 | 中 | 中 | 持平 | 持平 |
| **Ours** | **+5 点以上** | **+0.05** | **显著改善** | **持平** | **持平** |

统一口径：全部用 vLLM，同一 prompt，多分辨率输入按各自管线（anyres/原生长边）处理；ACC 按 6 能力维度分报。

## 6. 实验矩阵

- **A. 主实验**：Ours vs 全部基线，全指标 + 预算曲线。
- **B. 消融**：B1 预算头有无；B2 KL 权重 γ∈{0,0.05,0.1,0.2}；B3 预算分配策略（top-B vs 均匀 vs 随机）；B4 训练分辨率集（是否含 0.25×/4× 极值）；B5 预算表大小（是否含 2304）。
- **C. 鲁棒性**：C1 非整数预算（B=700）；C2 极端分辨率 8×/0.125×；C3 抗 JPEG 压缩输入。
- **D. 泛化性**：D1 骨干换 Qwen2-VL-7B LoRA；D2 推理时只用预算头不开正则（剪枝只用）。

## 7. 评测协议

- **指标**：Res-Bench 12 档 ACC（官方脚本）；稳定性：跨档 ACC 的 Spearman ρ（档位 × 模型得分）、ACE/RCE 按官方定义；联合分 $S_{joint}=\rho/3+\text{mean}(ACC_{all})/100$ 便于排序。
- **均值±方差**：5 个种子（123/2024/7/42/999）重复评测，报告 mean±std；仅 SFT 阶段随机性引入差异，评测阶段固定 seed 与 temperature=0。
- **显著性**：对平均 ACC 差与 ρ 差做配对 Bootstrap（n=1000，CI 95%），ΔACC≥2 且 CI 不含 0 视为显著。
- **防掉点门禁**：MME/MMMU 相对基线 Δ≤±1.0，否则回退 γ/预算设置。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 多分辨率预处理缓存（离线并行） | 1 |
| P2 | 预算头 stage1（蒸馏初始化） | 0.5 |
| P3 | 联合训练 13k 步 | 3 |
| P4 | Res-Bench/MME/MMMU 全量评测（多预算 × 多模型） | 1.5 |
| **合计** | | **≈6** |

存储：LLaVA-665K 多分辨率缓存 ~200GB；API：DeepSeek V4 Flash 多分辨率一致性旁证 ≈ $10。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | Res-Bench 数据获取、官方评测脚本跑通基线（LLaVA-1.5/NeXT/Qwen2-VL） |
| W2 | 多分辨率重采样缓存 + 预算头实现 + stage1 |
| W3 | 联合训练收敛、B1/B2 消融 |
| W4 | 预算曲线、鲁棒性（C1–C3）、骨干泛化（D1） |
| W5 | 统计、图表、论文初稿（NeurIPS 2026 截稿前 5 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 预算头训练不稳导致退化 | 中 | 高 | 蒸馏初始化 + 先冻主干；λ 升温；budget loss 降温 |
| MME/MMMU 掉点 | 中 | 高 | KL 权重调低、加 SFT 混合；掉点>1 点则回退 |
| Res-Bench 评测开销大 | 中 | 中 | 先用 200 样本子集早停，最终全量 |
| 预算-精度曲线不平滑 | 低 | 中 | 报告曲线 + 多预算集成投票 |
| 与 Qwen2-VL 原生多分辨率差距大 | 中 | 中 | 定位为「预算可控+一致性」增量贡献，不强求追平 |

## 11. 论文写作计划

- **目标**：NeurIPS 2026（主会，训练侧贡献）；若训练增益不足，转 Res-Bench 评测协议扩展 + 联合指标（D&B 轨道）。
- **差异化卖点**：(1) 首个"预算分配训练 + 跨分辨率一致性"MLLM 方案；(2) 预算-精度可控曲线；(3) 稳定-精度联合指标，补 Res-Bench 只测不训的空白。
- **图表清单**：图1 方法图；图2 预算分配可视化（重要度热图）；图3 预算-精度曲线；图4 各档分辨率 ACC 柱状图；表1 主实验；表2 消融；表3 鲁棒性；表4 泛化骨干；表5 与 anyres/原生多分辨率对比。
- **相关工作覆盖**：评测（Res-Bench NLP II·77、MME arXiv:2306.13394、MMMU arXiv:2311.16502、MMBench arXiv:2307.06281）、多分辨率模型（LLaVA-1.5 arXiv:2310.03744、Qwen2-VL arXiv:2409.12191、InternVL arXiv:2312.14238、InstructBLIP arXiv:2305.06500）。

## 12. 参考文献

1. Res-Bench（Natural Language Processing II·77）· Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input（收藏论文）
2. Liu et al. Improved Baselines with Visual Instruction Tuning（LLaVA-1.5）. arXiv:2310.03744.
3. Wang et al. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191.
4. Chen et al. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. arXiv:2312.14238.
5. Dai et al. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. arXiv:2305.06500.
6. Fu et al. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv:2306.13394.
7. Yue et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. arXiv:2311.16502.
8. Liu et al. MMBench: Is Your Multi-modal Model an All-around Player? arXiv:2307.06281.
9. Li et al. Visual Instruction Tuning（LLaVA 数据来源）. arXiv:2304.08485.
