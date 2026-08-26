# 实验设计书：Idea 6 SpaceRL — 程序化渲染合成 + 强化学习训练 MLLM 3D 空间推理

> 对应调研报告 Idea 6。最高优先级：直接攻击收藏论文暴露的 >30% 人类差距，且算力友好、可复现。

## 0. 摘要

SpaceRL 用可微/程序化 3D 渲染器（Blender/Unity）合成"难度可控、规则明确、答案可自动验证"的空间推理任务（折纸步骤→3D 结果、旋转、对称、遮挡深度），用 GRPO（arXiv:2402.03300）对 7B 级开源 MLLM（Qwen2.5-VL 系，arXiv:2502.13923）做强化微调，直接攻击 PFP 基准（Computer Vision XIII · 论文 49）暴露的短板。奖励信号来自渲染器真值，天然正确、可对抗；自动 curriculum 按通过率动态调难度。SFT 预热 20K → GRPO 约 3K 更新（4×L40 约 4–6 GPU·天），渲染离线 2–3 天。预期在 PFP 5 类任务上提升 10–20 个点，接近人类基线。

## 1. 研究背景与动机

### 1.1 问题定义

给定 (多视角图集, 空间推理问题, 选项/数值答案)，训练 MLLM 输出答案 + 理由。难点：空间推理缺乏可扩展训练数据与可验证奖励。本工作把二者都用程序化渲染解决：渲染器给出几何真值 → 自动生成问题/答案 → 答案正确性成为可验证奖励 → RL 直接优化推理。

### 1.2 相关工作不足

- 收藏论文（Computer Vision XIII · 论文 49 · Paper Folding Puzzles: Can Multimodal Large Language Models Perform Spatial Reasoning?）只诊断（18 个 MLLM，多数接近随机、差距 >30%），不提供任何训练/提示手段。
- 训练侧方法：`SpatialVLM`（arXiv:2401.12168）需真值标注、`SpatialRGPT`（arXiv:2406.01584）为 3D grounding、`SpaceR`（arXiv:2504.01805）视频空间推理 RL、`Spatial-TTT`（arXiv:2603.12255）测试时训练——都缺"无限供给 + 难度可控 + 自动验证"的训练源。
- RL 需要的正是可验证奖励；渲染器是唯一"零成本真值 + 可对抗"来源，此组合尚未用于 MLLM 空间推理。

### 1.3 为什么是现在、为什么你的环境适合做

- GRPO 群组 RL 已开源可复现（DeepSeekMath）；7B 开源 MLLM 可在 4×L40（QLoRA）上跑 GRPO；Blender/Unity 渲染管线成熟。
- PFP 提供现成评测协议；人类差距 30% 是明确靶子。算力主约束（RL 稳定性）可缓解。

## 2. 研究目标与可验证假设

1. **H1（RL 显著提升）**：SpaceRL 在 PFP 5 类任务平均准确率比原版 Qwen2.5-VL 提升 ≥15 个点，接近人类基线（差距从 >30% 压到 <15%）。
   - 观测：PFP 官方协议 + 自建同风格测试集。
2. **H2（可验证奖励必要）**：SFT-only vs SFT+GRPO 中，RL 带来 +8 点以上；不含"理由一致性"奖励时模型走捷径（答案对但理由错）。
   - 观测：奖励消融（最终答案 vs 答案+理由一致性）。
3. **H3（课程学习有效）**：自动 curriculum（按通过率调难度）比固定难度提升 +5 点，训练更快收敛。
   - 观测：两种难度策略的 PFP 分数与训练曲线。
4. **H4（泛化到其他基准）**：PFP 提升迁移到 VSR（arXiv:2205.00363）与 BLINK（arXiv:2404.12390）空间子集。
   - 观测：VSR 准确率提升 ≥8 点、BLINK 空间子集提升 ≥5 点。

## 3. 总体方法设计

### 3.1 数据流水线（含 API 合成 prompt 思路）

1. **程序化任务生成器（Blender Python API）**：参数化生成 4 类任务：
   - **折纸推理**：选择折痕/折叠序列 → 渲染折纸步骤图 + 结果 3D 多视角图 → 问题"哪张是折叠后的结果"（4 选项，由 3D 渲染真值生成）。
   - **旋转推理**：3D 物体按 $\{0°,90°,180°,270°\}$ 绕轴旋转 → 前后对比 → 问旋转角度/对称结果。
   - **对称与镜像**：渲染镜像、轴对称判断。
   - **遮挡与深度**：物体前后遮挡 → 问"哪个在前/可见性"。
   - 难度维度：折叠步数 {1,2,3}、视角数 {1,2,4}、干扰物 {0,1,2}、纹理复杂度。
2. **数量**：10M 级可无限采样；每类 250K 起步，答案/理由由渲染器几何事实自动生成（"折叠后 A 面朝上，因为折痕经过 Y 轴"）。
3. **API 语言多样化（DeepSeek V4 Pro）**：把渲染器生成的问题模板改写为多样化语言表述（避免模型死记模板），产出 200K 条多样化问题文本。
4. **CoT 预热数据（Kimi K2.6）**：由渲染真值 + 理由骨架生成 20K 条 CoT 训练样本（SFT 预热）。
5. **Hard case 标注**：跑原版模型在 50K 渲染样本上的通过率，把易错样本加大采样权重（Kimi 辅助归因）。

### 3.2 模型/算法设计

**模型**：Qwen2.5-VL-7B（arXiv:2502.13923）或 InternVL 系；QLoRA（rank 64, α=128）微调，冻结视觉塔 + LLM LoRA。

**奖励设计**（GRPO 群组奖励，组大小 G=8）：
- $r_\text{ans} \in \{0,1\}$：最终答案与渲染真值一致。
- $r_\text{reason}$：理由与渲染几何事实一致性（规则化匹配 + Kimi 抽检标注验证）。
- $r = r_\text{ans} + 0.5 \cdot r_\text{reason}$（理由奖励防止走捷径）。

**GRPO 目标**（参考 DeepSeekMath, arXiv:2402.03300）：
$$\mathcal{J}_\text{GRPO}(\theta)=\mathbb{E}_{q\sim p(Q),\{o_i\}\sim\pi_\theta(\cdot|q)}\frac{1}{G}\sum_{i=1}^{G}\frac{\exp(\beta\, r(o_i,q))}{\sum_{j}\exp(\beta\, r(o_j,q))}\log\pi_\theta(o_i|q)$$
（省略旧策略归一化项，见原论文），β=0.04。

**自动课程**：每 200 更新按当前通过率调整难度参数（通过率 >80% 升难度一档，<40% 降一档），难度档位见 §3.1。

**超参数**：QLoRA lr 1e-5（SFT）/ 3e-6（RL），rollout batch 128，max_len 2048，梯度检查点，β=0.04，KL 系数 0.04。

### 3.3 训练流程

- SFT 预热：20K CoT 样本，1 epoch，4×L40 QLoRA，约 1–1.5 GPU·天。
- GRPO：4×L40 分卡（每卡 2×Qwen2.5-VL-7B：actor + ref，梯度检查点），rollout 用 vLLM 加速采样；3K 更新 ≈ 4–6 GPU·天。
- 渲染：离线 CPU/单卡并行 2–3 天（可与训练并行）。

### 3.4 推理与评测流程

PFP 官方评测 + 自建同风格 5K 测试集 + VSR/BLINK 空间子集。加载 RL 后模型 → 批量推理 → 准确率/理由一致性/Kimi 评审。评测并行 4 卡 + API 辅助。

## 4. 数据集细节

| 数据集 | 用途 | 来源/许可 | 数量 | 预处理 |
|---|---|---|---|---|
| 程序化渲染任务 | 训练/自测 | 自产（Blender，CC 渲染资产） | 10M 级采样 | 真值答案自动生成 |
| PFP（论文49） | 评测 | 收藏论文基准（若开源；否则自建同风格） | 150K 官方/评测 | 标准协议 |
| VSR（arXiv:2205.00363） | 评测 | 官方 | 2K | 标准协议 |
| BLINK 空间子集（arXiv:2404.12390） | 评测 | 官方 | 子集 | 标准协议 |
| CoT 预热数据 | SFT | 渲染真值 + Kimi 理由 | 20K | 文本后处理 |

许可：Blender 资产用 CC0/CC-BY；PFP/VSR/BLINK 官方许可；渲染图版权归自产。

## 5. 基线复现

| 基线 | 官方代码 | 说明 |
|---|---|---|
| Qwen2.5-VL-7B 原版 | 官方仓库 | 零样本 |
| Qwen2.5-VL + CoT 提示 | 官方仓库 | 提示增强 |
| SpatialVLM 微调（arXiv:2401.12168） | 官方仓库 | 数据-训练管线 |
| SpatialRGPT（arXiv:2406.01584） | 官方仓库 | 3D grounding |
| SpaceR（arXiv:2504.01805） | 官方仓库 | 视频空间 RL |
| SpaceRL SFT-only | 本工作 | 消融 |
| InternVL 系原版 | 官方仓库 | 另一主干 |

统一口径：PFP 官方 5 类协议、同 prompt 模板、同解码参数（temperature 0）。

## 6. 实验矩阵

- **A（主实验）**：SpaceRL vs 全基线（PFP 5 类 + VSR + BLINK 空间子集）。
- **B（消融）**：SFT-only vs SFT+RL；奖励构成（答案/答案+理由）；课程策略（固定 vs 自动 vs 无）；QLoRA rank（32/64/128）。
- **C（鲁棒性）**：渲染域 → 真实照片背景混合；新难度（折叠 4 步）；干扰项强相关。
- **D（泛化性）**：不同主干（Qwen2.5-VL vs InternVL）；PFP→VSR 迁移。

## 7. 评测协议

- 指标：PFP 5 类准确率（+ 平均）；VSR/BLINK 准确率；理由一致性（Kimi 评审 1–5 + 规则匹配）；人类差距（对照报告 >30%）。
- 统计：3 种子（RL）；报告均值±std；分类准确率做 McNemar 检验（p<0.05）；种子间波动报 95% CI。

## 8. 算力与资源计划

| 阶段 | 卡·天 | 说明 |
|---|---|---|
| 程序化渲染 | 2–3 | CPU/单卡并行（离线） |
| SFT 预热 | 1–1.5 | QLoRA 4 卡 |
| GRPO 3K 更新 | 4–6 | 4 卡，vLLM rollout |
| 评测（PFP/VSR/BLINK） | 1–2 | 批量 + API 评审 |
| API | — | V4 Pro 多样化改写 ~200K 次（约 $30）；K2.6 CoT + 评审 ~100K 次（约 $25）；合计 < $60 |

合计 **≈ 10–13 GPU·天**，API < $60。

## 9. 里程碑与时间线（单人 + 4 卡）

| 周 | 任务 |
|---|---|
| W1 | 渲染器 4 类任务 + 真值管线；PFP/VSR/BLINK 评测脚本就绪 |
| W2 | SFT 预热 + 20K CoT；GRPO 框架搭通（小规模验证） |
| W3 | GRPO 主训练（3K 更新）；课程策略 v1 |
| W4 | 主实验 A + 消融 B |
| W5 | 鲁棒性 C + 泛化 D；Kimi 理由评审 |
| W6 | 写作 + 图表 + 投稿（NeurIPS 2026 / ICLR 2027 / CVPR 2027） |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解 |
|---|---|---|
| RL 训练不稳定（7B 在 4×L40） | 高 | QLoRA + 梯度检查点 + 小 rollout batch + β 调低；必要时 4× 降采样 max_len |
| 渲染任务与真实图像域差 | 中高 | 混合真实照片背景（C 实验）+ 域随机化 |
| RL 奖励作弊（捷径） | 中高 | 答案 + 理由双重验证；奖励防作弊（禁止"答案对理由错"得满分） |
| 模型死记模板 | 中 | API 语言多样化 + 程序化模板随机化 |
| 评测数据未开源（PFP） | 中 | 自建同风格测试集 + 官方协议对齐 |

## 11. 论文写作计划

- 目标：NeurIPS 2026 / ICLR 2027（首选），CVPR 2027 备份。卖点："程序化渲染 + RL 治空间推理"，可复现、算力可控、直接补收藏论文空白。
- 图表：方法图（渲染器→奖励→GRPO）；PFP 5 类准确率条形图（vs 人类/18 模型）；训练曲线（课程）；理由质量样例；泛化迁移表。
- 相关工作：收藏论文（Computer Vision XIII · 论文 49）+ SpatialVLM/SpatialRGPT/SpaceR/Spatial-TTT/BLINK/VSR/DeepSeekMath/Qwen2.5-VL。

## 12. 参考文献

- Computer Vision XIII · 论文 49 · Paper Folding Puzzles: Can Multimodal Large Language Models Perform Spatial Reasoning?（AAAI 2026）
- DeepSeekMath: arXiv:2402.03300（GRPO）；Qwen2.5-VL: arXiv:2502.13923
- SpatialVLM: arXiv:2401.12168；SpatialRGPT: arXiv:2406.01584；SpaceR: arXiv:2504.01805；Spatial-TTT: arXiv:2603.12255
- BLINK: arXiv:2404.12390；Visual Spatial Reasoning: arXiv:2205.00363
