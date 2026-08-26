# 实验设计书：身份/风格记忆库 + 轻量 Adapter 的长期一致文生图

## 0. 摘要

本实验设计用「记忆库（逐图维护身份/风格原型，带衰减窗口）+ 轻量 LoRA adapter」替代免训练启发式，实现数十帧长故事中身份与风格一致、且兼容任意扩散底模（SDXL/FLUX）的长期一致文生图。配套 DeepSeek V4 Pro 合成多角色故事分镜数据，损失 = 身份一致性（face/CLIP）+ 风格一致性 + 提示保真，推理期记忆库迭代更新 + 滑动窗口注意力注入。目标是 30+ 帧故事序列中身份一致性（FaceSim）与风格一致性显著优于 Infinite-Story/StoryDiffusion，且推理可并行加速。4×L40 约 8 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

一致故事生成：给定角色设定 $S=\{c_1..c_n\}$ 与分镜提示序列 $\{p_1..p_m\}$（m 可达 30–60），逐帧生成图像，要求：(a) 每个角色身份跨帧一致；(b) 整体风格一致；(c) 每帧遵循各自提示。免训练方法（Infinite-Story）在长序列/剧烈场景变化下漂移；训练型一致自注意力（StoryDiffusion）需要特定架构。我们维护记忆库 $M_t=\{z^{id}_{t},z^{style}_{t}\}$，用轻量 LoRA adapter 学习"参考-生成"映射，推理时滑动窗口注入。

### 1.2 相关工作不足

- 收藏论文 `Infinite-Story`（Computer Vision VII·62·Infinite-Story: A Training-Free Consistent Text-to-Image Generation）：免训练一致文生图，Identity Prompt Replacement + 自适应风格注入 + 同步引导，推理每图 1.72s、比最快一致生成快 6×。但**长序列漂移、剧烈场景变化时免训练方法退化**。
- `StoryDiffusion`（arXiv:2405.01434，CVPR 2024）一致自注意力，面向长序列但依赖特定架构。
- 训练型一致生成方法速度慢、不兼容任意底模。

空白：**记忆库（带衰减）+ 轻量 LoRA adapter + 任意底模兼容**的长期一致方案，兼顾长序列稳健性与兼容性。

### 1.3 为什么是现在、为什么你的环境适合做

Infinite-Story 2026 AAAI 刚中，"免训练 vs 训练型"路线之争正热，记忆库+adapter 是第三条路，差异化明显。本环境：SDXL/FLUX LoRA 训练 4×L40 可行；推理 4 卡并行加速；DeepSeek V4 Pro 合成故事数据便宜；风格一致性 judge 用 V4 Flash。

## 2. 研究目标与可验证假设

- **H1**：记忆库 + 衰减窗口优于免训练启发式。*可观测结果*：30 帧序列 FaceSim 一致率 ≥ 0.85（Infinite-Story 长序列显著衰减），风格一致评分更高。
- **H2**：LoRA adapter 的"参考-生成"映射优于手工注入。*可观测结果*：相同底模下，adapter 版身份保持比 Identity Prompt Replacement 高 ≥5 点。
- **H3**：与任意底模兼容（SDXL 与 FLUX 均可用）。*可观测结果*：两底模下主指标同向提升，且风格不受底模差异主导。
- **H4**：推理并行与记忆库迭代不冲突。*可观测结果*：滑动窗口（并行度=窗口内帧数）推理吞吐接近 batch 推理，长序列总耗时近线性。

## 3. 总体方法设计

### 3.1 数据流水线

- **故事数据合成（DeepSeek V4 Pro）**：
```
Create a 30-frame storyboard: 2-3 characters with fixed identity descriptors (name/appearance), a coherent storyline with scene changes, and per-frame prompts (English, 20-40 words). Output JSON: {characters, frames:[{scene, prompt}]}.
```
生成 800 个故事（总 24k 帧）。分镜需含：≥10 帧场景变化、角色同框与独镜混合、风格词统一（同一风格标签贯穿）。
- **一致性标注**：每个故事设定统一身份 embedding（固定角色名 → 固定参考图，用 DeepSeek V4 Pro + 手工挑选 2 张/角色参考图）；风格标签统一。
- **过滤**：DeepSeek V4 Flash 抽检 prompt 与角色设定一致性；参考图用 face 检测确保有效人脸。
- **数量**：训练 600 故事 / 验证 100 / 评测 100（30 帧）。

### 3.2 模型/算法设计

- **底模**：SDXL（arXiv:2307.01952）为主 + FLUX（开源权重）兼容验证。
- **记忆库**：每帧维护 $z^{id}_{t}$（角色 CLIP+face embedding 的滑动平均）与 $z^{style}_{t}$（风格 embedding）。更新规则带指数衰减：$z_{t+1}=\alpha z_t+(1-\alpha)\bar{z}_{new}$，$\alpha=0.8$，窗口 W=8（只注入最近 8 帧）。
- **LoRA adapter**：在 UNet cross-attention 的 K/V 上加注入分支，输入 = 参考角色 embedding 与记忆库状态，学习"参考→当前帧"映射。训练时随机遮挡参考图以强制依赖记忆库。
- **损失**：$\mathcal{L}=\mathcal{L}_{diff}(\text{SD loss})+\lambda_1\mathcal{L}_{face}+\lambda_2\mathcal{L}_{style}+\lambda_3\mathcal{L}_{prompt}$。
  - $\mathcal{L}_{face}$：生成帧与参考人脸 FaceSim（ArcFace/CLIP face）余弦；
  - $\mathcal{L}_{style}$：生成帧与风格参考的 CLIP 余弦（区域级）；
  - $\mathcal{L}_{prompt}$：CLIPScore（提示保真）；
  - 初值 $\lambda_1=0.2,\lambda_2=0.15,\lambda_3=0.3$。
- **推理**：滑动窗口 W=8 → 记忆库更新 → 注入 → 生成；batch 并行窗口内帧。

### 3.3 训练流程

- SDXL LoRA（rank=64，alpha=128，UNet 注意力块），4×L40，batch=8，grad-acc=4；lr=1e-4 AdamW；15k 步（约 3 epoch over 24k 帧）。
- 阶段1：仅 UNet 注入分支（无参考遮挡）2k 步；阶段2：加入遮挡 + 全损失。
- 显存：SDXL 全 UNet LoRA ≈ 35GB/卡，batch=2/卡 4 卡 8。

### 3.4 推理与评测流程

- 输入故事 → 记忆库初始化（参考图）→ 逐窗口生成。报告：FaceSim、CLIP 风格一致、CLIPScore、每帧耗时。
- 评测：100 故事 × 30 帧；人工盲评子集（50 故事 × 5 帧）。

## 4. 数据集细节

| 数据集 | 用途 | 说明 | 许可 |
|---|---|---|---|
| StoryDiffusion 相关数据（arXiv:2405.01434） | 对照 | 官方数据/图 | 学术 |
| **StoryBank-24K（本工作）** | 训练 | 600 故事 × 30 帧（DeepSeek V4 Pro 合成分镜） | 随论文开源 |
| 评测故事集 | 评测 | 100 故事（30 帧） | 随论文开源 |
| FFHQ/ArcFace 参考（人脸基准） | 一致性评估 | 公开 | 学术 |

## 5. 基线复现

| 基线 | 官方代码/权重 | 复现要点 |
|---|---|---|
| StoryDiffusion（arXiv:2405.01434） | https://github.com/UCSC-VLAA/StoryDiffusion | 官方实现 |
| Infinite-Story（CV VII·62） | 见论文（代码公开） | 免训练管线 |
| ARStory（官方仓库，见论文） | 官方实现 | 按官方 |
| 免训练 baseline（自建） | — | Identity Prompt Replacement 简化版 |
| Ours | — | §3 全量 |

**预期指标表**（30 帧序列；FaceSim/风格/CLIPScore 越高越好，数值以复现为准）：

| 方法 | FaceSim↑ | 风格一致(CLIP)↑ | CLIPScore↑ | 每帧耗时↓ | 长序列漂移↓ |
|---|---|---|---|---|---|
| StoryDiffusion | 高 | 高 | 高 | 中 | 中 |
| Infinite-Story | 中 | 中高 | 高 | 最低(1.72s) | 大 |
| 免训练 baseline | 低 | 中 | 中 | 低 | 大 |
| **Ours** | **最高** | **最高** | **高** | 中 | **最小** |

统一口径：同一故事集、同一面检测器、同一风格词、同一评测脚本。

## 6. 实验矩阵

- **A. 主实验**：全基线 + Ours 全指标 + 30 帧长序列。
- **B. 消融**：B1 记忆库有无、衰减 α∈{0.5,0.8,0.95}；B2 滑动窗口 W∈{4,8,16}；B3 adapter 有无（vs 纯注入）；B4 参考遮挡有无；B5 损失权重 λ1/λ2/λ3；B6 序列长度（10/30/60）。
- **C. 鲁棒性**：C1 剧烈场景变化（切换室内外/白天黑夜）；C2 角色外观变化（换装 vs 同装）；C3 多人同框复杂构图。
- **D. 泛化性**：D1 底模 FLUX；D2 未见风格（测试时新风格词 + 少参考图）。

## 7. 评测协议

- **指标**：FaceSim（同角色跨帧人脸余弦均值，ArcFace/InsightFace）；风格一致 = 相邻帧 CLIP 余弦 + 与全局风格参考余弦；CLIPScore；每帧耗时（L40，batch=1/8 两档）。
- **均值±方差**：3 个采样种子（生成随机性），报 mean±std。
- **显著性**：FaceSim/风格差配对 Bootstrap（n=1000）。
- **人工盲评**：50 故事 × 5 帧，3 人 Win/Tie/Loss，报告一致率。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 故事数据合成（API）+ 参考图准备 | 0.5 |
| P2 | 阶段1+2 训练（15k 步） | 5–6 |
| P3 | 评测推理（基线 × 故事集，4 卡并行） | 1.5–2 |
| P4 | 人工盲评 + 图表 | 0.5 |
| **合计** | | **≈8** |

存储：故事图像 ≈ 120GB。API：DeepSeek V4 Pro 故事合成 ≈ $30；V4 Flash 一致性 judge ≈ $10。总计 ≤ **$45**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | 故事数据合成 + 基线复现（StoryDiffusion/Infinite-Story） |
| W2 | 记忆库 + adapter 实现 + 阶段1 |
| W3 | 阶段2 训练 + B1/B2 消融 |
| W4 | A 主实验 + 长序列评测 |
| W5 | C/D 鲁棒泛化 + 统计 + 人工评测 + 论文初稿（CVPR 2027 截稿前 5 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 长序列漂移仍存在 | 高 | 高 | 衰减窗口 + 滑动窗口 + 人工评估子集 + 诚实报告长故事失败率 |
| 底模差异主导结果 | 中 | 中 | 分底模报告 + 风格归一化 |
| 身份参考图质量差 | 中 | 中 | 参考图筛选（face 检测 + CLIP 检索）+ 多参考融合 |
| 训练成本超预期 | 中 | 中 | 降为 1 epoch 初版验证 + LoRA rank 减半 |
| 一致性评估主观 | 中 | 中 | 多 judge + 人工盲评 + 指标一致性报告 |

## 11. 论文写作计划

- **目标**：CVPR 2027（高）。
- **差异化卖点**：(1) 记忆库 + 轻量 LoRA adapter 的第三条技术路线（vs 免训练/一致自注意力）；(2) 任意底模兼容；(3) 长序列评测协议 + 诚实失败报告。
- **图表清单**：图1 方法图（记忆库更新）；图2 长序列一致性可视化（30 帧条带）；图3 漂移对比；图4 场景变化/换装案例；表1 主实验；表2 消融；表3 序列长度；表4 底模兼容；表5 人工盲评。
- **相关工作覆盖**：一致文生图（Infinite-Story CV VII·62、StoryDiffusion arXiv:2405.01434、ARStory）、底模（SDXL arXiv:2307.01952、FLUX 开源权重）、偏好评测（ImageReward arXiv:2304.05977 可作旁证）。

## 12. 参考文献

1. Infinite-Story（Computer Vision VII·62）· Infinite-Story: A Training-Free Consistent Text-to-Image Generation（收藏论文）
2. Zhou et al. StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation. arXiv:2405.01434（CVPR 2024）.
3. Podell et al. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952.
4. Xu et al. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. arXiv:2304.05977（NeurIPS 2023）.
