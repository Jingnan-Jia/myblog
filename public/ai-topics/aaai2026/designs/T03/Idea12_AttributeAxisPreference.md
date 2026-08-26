# 实验设计书：属性轴可解释的 T2I 偏好模型与定向对齐

## 0. 摘要

本实验设计把 T2I 人类偏好分解为 6 个可解释属性轴（美学/伪影/解剖/构图/对象一致/风格），训练多轴偏好模型（共享 CLIP 骨干 + 6 个轻量属性头），并按薄弱轴加权对齐扩散模型（SDXL/FLUX LoRA）。沿用「每轴高度可控合成对抗数据」的训练协议（参考 ML VI·84 论文思路），与人类/多 judge 做轴级相关性验证。目标是偏好模型与人类轴级一致率显著高于黑盒 ImageReward/HPSv2，且定向对齐后"解剖/对象一致"等薄弱轴可解释性提升可归因。4×L40 约 9 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

T2I 偏好对齐：给定文本提示 $p$ 与候选图 $\{I\}$，学习人类偏好函数。现有偏好模型输出黑盒标量，无法解释"到底哪一轴差"。我们定义 6 属性轴 $A=\{美学, 伪影, 解剖, 构图, 对象一致, 风格\}$，学多任务偏好模型 $f_\theta=(f_{a_1}..f_{a_6}, f_{overall})$，并对齐时按薄弱轴加权：$r=\sum_a w_a f_a(I)$。目标是可解释 + 定向强化。

### 1.2 相关工作不足

- 收藏论文 `图像偏好对齐`（Machine Learning VI·84·What Makes a Good Generated Image? Investigating Human and Multimodal LLM Image Preference Alignment）：对人类与 MLLM 的 6 属性轴偏好做对照分析，发现人类轴间强相关、MLLM 轴间相关弱，且 MLLM 学不好"解剖准确性"。但**只分析不对齐**，未给出解剖等难轴的训练/评测方案。
- `ImageReward`（arXiv:2304.05977，NeurIPS 2023）黑盒标量偏好打分。
- `HPSv2`（arXiv:2306.09341）人类偏好基准与打分模型，单标量。
- `Pick-a-Pic`（arXiv:2305.01569）大规模用户偏好数据。
- `GenEval`（arXiv:2310.11513）、`T2I-CompBench++`（arXiv:2307.06350）组合生成评测，无轴级偏好。

空白：**可解释多属性轴偏好模型 + 按薄弱轴定向对齐 T2I 模型**。

### 1.3 为什么是现在、为什么你的环境适合做

ML VI·84 2026 AAAI 刚中，其"人类 vs MLLM 轴级差异"分析直接指向"轴分解偏好模型 + 定向对齐"这一空白；偏好对齐赛道正热但可解释性缺位。本环境：偏好模型（CLIP-L + 6 头）4×L40 训练便宜；SDXL LoRA 对齐可行；DeepSeek V4 Pro/Flash/Kimi K2.6 负责合成数据、批量 judge 与美学描述。

## 2. 研究目标与可验证假设

- **H1**：多轴偏好模型在轴级与整体上都更贴近人类。*可观测结果*：每轴与人类一致率（ACC）显著高于 ImageReward/HPSv2（整体），解剖轴 ≥70%。
- **H2**：轴间解耦（低相关）带来可解释性。*可观测结果*：属性头输出的轴间相关 <0.3（人类为强相关，但模型解耦=可归因），错误可按轴归因。
- **H3**：薄弱轴加权对齐定向提升目标轴。*可观测结果*：对"解剖"加权对齐后，解剖轴人类评分提升 ≥15%，其他轴不掉（Δ≤±3%），GenEval 保持。
- **H4**：每轴可控合成对抗数据提升轴级判别。*可观测结果*：合成轴对数据参与训练后，轴级 ACC 提升 ≥5 点，尤其难轴（解剖/对象一致）。

## 3. 总体方法设计

### 3.1 数据流水线

- **基础数据**：Pick-a-Pic（arXiv:2305.01569，用户偏好对）、ImageRewardDB（出自 arXiv:2304.05977，同一仓库）、HPSv2 数据（arXiv:2306.09341）。
- **每轴可控合成对抗数据（DeepSeek V4 Pro + 扩散生成）**：对 6 轴各构造"高/低"可控对：
```
For attribute axis "{axis}", craft a text prompt where generated images differ ONLY in this axis (e.g., anatomy: "person with 4 arms" vs "person with 2 arms"; artifact: style tokens triggering artifacts). Provide 8 such prompt pairs.
```
用 4×L40 并行跑 SDXL/FLUX 生成每轴 5k 对（10k 图），DeepSeek V4 Flash 校验"仅目标轴差异"（轴外一致性 <0.3）。
- **轴级人工/多 judge 标注**：对合成对与 Pick-a-Pic 子集，用 DeepSeek V4 Flash + Kimi K2.6 多 judge（3 个）标 6 轴分（1–5）+ 整体分；人工抽 500 对校准。
- **过滤**：judge 不一致（标准差>1）样本剔除；轴间混淆对（多轴同时变）剔除。
- **数量**：整体偏好 ~80k 对（Pick-a-Pic + ImageRewardDB）；轴级标注 ~30k 对（合成 6×5k + 人工扩充）；每轴 5k 对。

### 3.2 模型/算法设计

- **偏好模型**：CLIP-ViT-L 图像编码（可 LoRA）+ 文本编码；6 个轻量属性头（MLP 2×256）+ 1 个整体头。训练：每轴独立回归 loss + 整体 ranking loss（与 ML VI·84 的"每轴高度可控合成对"协议一致）：
  $\mathcal{L}=\sum_a\,\text{MSE}(f_a(I),\hat{s}_a)+\lambda\,\text{RankLoss}(f_{overall},\{I_w,I_l\})+\lambda_r\,\|heads\|_2$。
  轴间解耦正则：$\mathcal{L}_{dec}=\sum_{a<b}\lvert \text{corr}(f_a,f_b)\rvert$（鼓励低相关，消融项）。
- **对齐（定向）**：以 $r=\sum_a w_a f_a(I)$ 为奖励，DPO/DPOK 微调 SDXL/FLUX LoRA：
  $\mathcal{L}_{align}=\mathcal{L}_{DPO}(x_w,x_l;r)+\beta\,\mathcal{L}_{gen}$；薄弱轴权重 $w_a$ 放大（如解剖×2）。
- **评测**：轴级盲测（与人类/多 judge 一致率）、GenEval、T2I-CompBench++、FID/CLIPScore、HPSv2/ImageReward 打分。

### 3.3 训练流程

- 阶段1 偏好模型：CLIP-L 冻结 + LoRA，6+1 头，4×L40，batch=128，lr=3e-4，8k 步。
- 阶段2 对齐：SDXL LoRA（rank=64），4×L40，batch=32，lr=1e-4，4k 步（DPOK）；FLUX 版本复跑（若权重许可）。
- 评估：每 1k 步测轴级一致率早停。

### 3.4 推理与评测流程

- 偏好模型输出 6 轴分 + 整体分（可解释）；对齐后模型按薄弱轴定向。
- 评测：轴级盲测（同图对 vs 人类/多 judge）、GenEval 官方、T2I-CompBench++ 官方、FID/CLIP、HPSv2/ImageReward；消融报告每轴贡献。

## 4. 数据集细节

| 数据集 | 用途 | 说明 | 许可 |
|---|---|---|---|
| Pick-a-Pic（arXiv:2305.01569） | 整体偏好训练 | 官方（含用户偏好） | 学术 |
| ImageRewardDB（出自 arXiv:2304.05977） | 整体偏好训练 | 官方 | 学术 |
| HPSv2（arXiv:2306.09341） | 评测/数据 | 官方 benchmark | 学术 |
| **AxisPair（本工作）** | 轴级训练 | 6 轴 × 5k 对（合成） | 随论文开源 |
| GenEval（arXiv:2310.11513） | 评测 | 官方 | 学术 |
| T2I-CompBench++（arXiv:2307.06350） | 评测 | 官方 | 学术 |

## 5. 基线复现

| 基线 | 官方代码/权重 | 复现要点 |
|---|---|---|
| ImageReward（arXiv:2304.05977） | https://github.com/THUDM/ImageReward | 官方权重打分 |
| HPSv2（arXiv:2306.09341） | https://github.com/tgxs002/HPSv2 | 官方权重 |
| PickScore（arXiv:2305.01569） | https://github.com/yuvalkirstain/PickScore | 官方权重 |
| 原模型（SDXL/FLUX 未对齐） | 官方权重 | 生成对照 |
| 黑盒 DPO 对齐（自建） | — | 用 ImageReward 做奖励对齐 |
| Ours | — | §3 全量 |

**预期指标表**（轴级一致率/GenEval/整体；数值以复现为准）：

| 方法 | 整体一致率↑ | 解剖轴一致率↑ | 对象一致轴↑ | GenEval↑ | 对齐后可解释 |
|---|---|---|---|---|---|
| ImageReward | 中 | 低 | 低 | 基准 | 无 |
| HPSv2 | 中高 | 低 | 低 | 基准 | 无 |
| PickScore | 中 | 低 | 低 | 基准 | 无 |
| 黑盒 DPO 对齐 | — | — | — | 略升 | 无 |
| **Ours（偏好）** | **最高** | **≥70%** | **最高** | — | **6 轴输出** |
| Ours（定向对齐） | — | — | — | **保持/升** | **按轴归因** |

统一口径：同一盲测集（500 对）、同一 judge 模板、同一 GenEval 脚本。

## 6. 实验矩阵

- **A. 主实验**：偏好模型（Ours vs ImageReward/HPSv2/PickScore）+ 对齐后模型（Ours vs 黑盒 DPO vs 原模型）。
- **B. 消融**：B1 轴解耦正则有无；B2 每轴头独立 vs 共享；B3 合成轴对比例（0/30/100%）；B4 对齐权重 w_a（均匀 vs 薄弱×2 vs 薄弱×4）；B5 对齐方法（DPO vs DPOK）；B6 骨干（CLIP-L vs CLIP-H）。
- **C. 鲁棒性**：C1 不同风格/分辨率输入；C2 难轴（解剖）高难案例；C3 多轴同时问题。
- **D. 泛化性**：D1 底模 FLUX 对齐；D2 跨分布（未见模型生成图）；D3 中文提示。

## 7. 评测协议

- **指标**：轴级一致率（模型 vs 人类/多 judge）、整体一致率、Spearman ρ（轴分与人类分）、GenEval 官方、T2I-CompBench++ 官方、FID、CLIPScore、HPSv2/ImageReward 分。
- **均值±方差**：3 个训练种子 + 3 次 judge，报 mean±std。
- **显著性**：一致率差配对 Bootstrap（n=1000）；轴级报告按轴显著性。
- **可解释性验证**：错误案例按轴归因，人工判定归因正确率 ≥80%。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 合成轴对数据（SDXL/FLUX 并行生成 30k 图） | 2–3 |
| P2 | 轴级标注（API judge）+ 人工校准 | 0.5 |
| P3 | 偏好模型训练（8k 步） | 3–4 |
| P4 | 定向对齐（SDXL 4k 步 + FLUX 复跑） | 4–5 |
| P5 | 全量评测（盲测 + GenEval 等） | 1 |
| **合计** | | **≈10–13.5** |

存储：生成图像 + 数据集 ≈ 300GB。API：DeepSeek V4 Pro 轴对 prompt 设计 ≈ $15；V4 Flash/Kimi 多 judge 标注 ≈ $40；总计 ≤ **$60**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | 数据获取 + 基线复现（ImageReward/HPSv2/PickScore） |
| W2 | 合成轴对生成 + 标注 + 抽检 |
| W3 | 偏好模型训练 + B1/B2/B3 消融 |
| W4 | 定向对齐 + A 主实验 + 轴级盲测 |
| W5 | 鲁棒/泛化 + 统计 + 可解释性验证 + 论文初稿（NeurIPS 2026 截稿前 5 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 属性轴定义主观 | 中 | 高 | 公开轴定义与数据协议 + 定量消融 + 人工验证子集 |
| 合成对抗数据难完全可控 | 中 | 高 | 每轴多 prompt 变体 + V4 Flash 轴外一致性过滤 + 人工校准 |
| 解剖轴人类标注分歧大 | 中 | 中 | 多 judge 投票 + 只保留高一致样本 |
| 对齐导致其他轴掉点 | 中 | 中 | 薄弱轴权重上限 + 整体保底 loss |
| 评测成本高（GenEval 等） | 低 | 中 | 子集先验 + 全量终测 |

## 11. 论文写作计划

- **目标**：NeurIPS 2026（高）；CVPR 2027（中）备选。
- **差异化卖点**：(1) 首个 6 属性轴可解释偏好模型（轴分 + 整体）；(2) 按薄弱轴定向对齐的可归因提升；(3) 每轴可控合成对抗数据的完整协议 + 轴级评测。
- **图表清单**：图1 多轴偏好模型方法图；图2 轴间相关性（人类 vs 模型 vs 解耦模型）；图3 定向对齐前后轴级评分雷达图；图4 错误归因案例；表1 偏好模型主实验；表2 对齐后评测；表3 消融；表4 鲁棒/泛化；表5 人工盲测。
- **相关工作覆盖**：偏好模型（ImageReward arXiv:2304.05977、HPS/HPSv2 arXiv:2303.14420/2306.09341、Pick-a-Pic arXiv:2305.01569）、轴级分析（ML VI·84）、对齐（RLHF/DPO 类方法，引用已验证偏好文献）、评测（GenEval arXiv:2310.11513、T2I-CompBench++ arXiv:2307.06350）。

## 12. 参考文献

1. 图像偏好对齐（Machine Learning VI·84）· What Makes a Good Generated Image? Investigating Human and Multimodal LLM Image Preference Alignment（收藏论文）
2. Xu et al. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. arXiv:2304.05977（NeurIPS 2023）.
3. Wu et al. Human Preference Score: Better Aligning Text-to-Image Models with Human Preference（HPS）. arXiv:2303.14420（ICCV 2023）.
4. Wu et al. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv:2306.09341.
5. Kirstain et al. Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation. arXiv:2305.01569（NeurIPS 2023）.
6. Ghosal et al. GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. arXiv:2310.11513（NeurIPS 2023）.
7. Huang et al. T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation. arXiv:2307.06350（NeurIPS 2023）.
8. Podell et al. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952.
