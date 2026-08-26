# 实验设计书：Idea 11 占据锚定的驾驶 VQA 与规划一致性微调（Occupancy-Grounded Driving VQA with Planning-Consistency Fine-Tuning）

> 主题 T01 自动驾驶感知与端到端驾驶 · 12 个 idea 之一 · 优先级：高（CVPR 2027 / ECCV 2027）

## 0. 摘要

时空驾驶 VQA（STRIDE-QA）证明通用 VLM 近零分，但 VQA 与规划脱节；VLA（OpenDriveVLA）有 VQA+轨迹却未用 VQA 作为规划正则。本工作把时空 VQA 锚定到占据表示（问题/答案引用占据体素索引，天然 3D 可验证），并用「VQA 答案必须与模型实际输出轨迹自洽」作为 RLHF 式奖励做 DPO/RLVR 微调：答案若与轨迹预测矛盾（如答「前车将直行」但轨迹预测前车变道）则罚分。数据用 STRIDE-QA 生态 + LLM 生成的占据锚定问答对 + 答案-轨迹矛盾 judge。预期空间定位准确率 ≥55%、预测一致性 ≥28%（STRIDE-QA 同口径）基础上进一步 +5~8 点，且闭环 PDMS 提升。贡献：(1) 占据锚定问答范式（3D 可验证）；(2) 规划一致性 RLHF 式奖励；(3) VQA-规划联合评测协议。

## 1. 研究背景与动机

### 1.1 问题定义

驾驶 VQA：给定多视角视频 + 轨迹 + 占据，回答时空推理问题（「第三车道 SUV 在 2 秒后会不会进入我们的行驶带」「哪个行人会在 3 秒内横穿」）。要求：答案 3D 可验证（锚定体素），且与模型自身规划动作自洽——VQA 与规划形成闭环而非两套孤立输出。

### 1.2 相关工作不足

- STRIDE-QA（Computer Vision IV · 论文 29）1600 万时空推理 QA、通用 VLM 近零分，但 VQA 与规划行动脱节（报告速查表第 21 条）；一致性指标定义单薄。
- OpenDriveVLA（Computer Vision XIII · 论文 71）开源 VLA 有 VQA+轨迹，却未用 VQA 作为规划正则；仅 nuScenes 开环评测。
- OccLLaMA（arXiv 2409.03272）统一占据-语言-行动，但无规划一致性奖励。
- 外部依据：Reason2Drive（arXiv 2312.03661）链式推理 VQA 基准支撑评测；Bench2Drive-VL（arXiv 2604.01259）新增 VLM 闭环评测支撑闭环协议。
- 空白：VQA 答案与规划动作的自洽性从未被用作训练信号。

### 1.3 为什么是现在、为什么你的环境适合

- 现在是：STRIDE-QA/Bench2Drive-VL 等基准齐备、VLA 开源（OpenDriveVLA）、RLVR/DPO 在 VLM 侧成熟可迁移。
- 环境适合：VLA 7B QLoRA 4bit 微调，4×L40 可跑（100–200k step，8–12 GPU·天）；RLVR 只用小规模难例（采样瓶颈可控）；VQA 数据合成走 API 不占 GPU。

## 2. 研究目标与可验证假设

- **H1（占据锚定提升时空推理）**：占据锚定的 VQA（含 3D 可验证性）比文本-only VQA 显著提升时空推理。
  - 可观测结果：空间定位准确率 ≥55%（STRIDE-QA 同口径）且相对文本-only +10 点；预测一致性 ≥30%。
- **H2（规划一致性奖励有效）**：RLHF 式一致性奖励微调同时提升 VQA 与规划。
  - 可观测结果：DPO/RLVR 后，VQA 答案-轨迹矛盾率下降 ≥40%；闭环 PDMS 相对监督微调 +≥3 点。
- **H3（矛盾 judge 可自动化）**：LLM judge 检测「答案-轨迹矛盾」与人工标注一致率 ≥85%。
  - 可观测结果：judge 与人工矛盾标注 Cohen's κ≥0.8。
- **H4（RLVR 可稳定收敛）**：小规模难例 RLVR 在 4×L40 上稳定收敛（不劣化 VQA 已有能力）。
  - 可观测结果：RLVR 后通用 VQA 准确率不降（±1 点内）、规划一致性提升；若不稳则 DPO 两阶段兜底。

## 3. 总体方法设计

### 3.1 数据流水线

- 数据源：STRIDE-QA 生态（问题模板 + 真实数据）、nuScenes-QA（Reason2Drive 同源，arXiv 2312.03661）、Bench2Drive-VL 评测协议。
- 占据锚定问答对构造：
  1. 对每帧场景跑一个占据模型（复用 Idea 4/5 的占据头）输出体素索引 → 问题引用「[体素编号]」或「车道 3，[x,y,z] 区域」。
  2. LLM（DeepSeek V4 Pro）从场景文本 + 轨迹 + 占据体素生成 3 类问题：空间定位（「占据体素 v 的类别是？」）、运动预测（「体素 v 区域的对象 2 秒后位置」）、规划相关（「该对象会不会进入我们的行驶带，若会，轨迹冲突体素是哪个？」）。
  3. 答案格式统一为「文本 + 体素坐标」，保证可验证（答案与占据真值/轨迹真值可自动核对）。
- 答案-轨迹矛盾 judge（LLM）：输入 VQA 答案 + 模型实际输出轨迹（含他车预测），判「一致 / 矛盾 / 无关」，输出 JSON + 理由。抽检人工审计 κ。
- 数量预期：占据锚定 QA 对 50 万（含 3 类问题均衡）；矛盾 judge 用于 RL 奖励（难例 5 万）；训练/验证/测试划分 90/5/5。
- 过滤：答案与真值不可核对（占据缺失区域）的样本剔除；LLM 答案抽检一致率 <85% 则迭代 prompt。

### 3.2 模型/算法设计

- 模型：OpenDriveVLA 7B（arXiv 2503.23463）为基座；新增占据 tokenizer（把占据体素序列化成 token 并入视觉 token 流，借鉴 OccLLaMA 思路）。
- 阶段 1（VQA 监督微调）：标准 causal LM loss，QLoRA 4bit（rank=32，α=64）。
- 阶段 2（规划一致性奖励）：
  - 奖励：$r = \mathbb{1}[judge=一致] - \mathbb{1}[judge=矛盾] + \lambda_{plan} \Delta\text{PDMS}_{clip}$（有闭环信号时）。
  - DPO 偏好对：由 judge 判「一致」的答案作 chosen、「矛盾」作 rejected，构造偏好对（离线）。
  - RLVR：小规模难例（5 千条矛盾场景）上 PPO/GRPO，policy QLoRA、critic 共享。
- 超参初值：阶段 1 lr 2e-5（QLoRA），100–150k step；阶段 2 DPO β=0.1 或 GRPO lr 1e-6，RL 5k 样本/轮。

### 3.3 训练流程

- 阶段 1：4×L40 FSDP + QLoRA（7B 4bit 每卡 ~16GB，FSDP 分片后更省），global batch 128，100–150k step ≈ 8–10 GPU·天。
- 阶段 2：DPO 离线 2 GPU·天；RLVR（若走）1–2 GPU·天。
- 并行：VQA 数据合成（API）与阶段 1 并行。

### 3.4 推理与评测流程

- 推理：视频 + 占据 token → VLA 生成答案 + 轨迹。
- 评测：空间定位准确率、运动预测一致性（STRIDE-QA 同口径）、答案-轨迹矛盾率；闭环 PDMS（NAVSIM + Bench2Drive-VL 思路）；通用 VQA 准确率（回归检查）。

## 4. 数据集细节

- STRIDE-QA：东京多传感器 100h + 1600 万 QA；许可需按论文说明确认（研究用途）。
- nuScenes-QA / Reason2Drive（arXiv 2312.03661）：链式推理 VQA 基准（nuScenes 场景）；Reason2Drive 有公开数据说明。
- Bench2Drive-VL（arXiv 2604.01259）：闭环 VLM 评测协议（Bench2Drive 数据需申请）。
- 本工作自建：占据锚定 QA 50 万条（nuScenes 场景）+ 矛盾 judge 标注 5 万难例。
- 划分：90/5/5；评测固定 STRIDE-QA 子集与 Bench2Drive-VL 子集。

## 5. 基线复现

| 基线 | 引用 | 官方代码 | 复现要点 |
|---|---|---|---|
| 通用 VLM 直接问答（零样本） | 报告参照 | — | GPT-4V 类（若可用）或本地 Qwen-VL 作零样本参照 |
| OpenDriveVLA（无 VQA 监督） | Computer Vision XIII · 论文 71 | 官方权重 | 直接测 VQA 能力 |
| OpenDriveVLA + 纯文本 VQA 监督 | 自建消融 | — | 不加占据锚定 |
| OccLLaMA 式 | arXiv 2409.03272 | 无官方 | 按论文复现统一生成（可近似） |
| **本方法** | — | 本项目开源 | 占据锚定 + 规划一致性奖励 |

- 预期指标表（STRIDE-QA 同口径 + Bench2Drive-VL 子集）：

| 方法 | 空间定位↑ | 预测一致性↑ | 答案-轨迹矛盾率↓ | 闭环 PDMS |
|---|---|---|---|---|
| 通用 VLM 零样本 | 0.05 | 0.02 | — | — |
| OpenDriveVLA | 0.15 | 0.08 | 0.45 | 0.55 |
| +纯文本 VQA | 0.30 | 0.15 | 0.40 | 0.57 |
| OccLLaMA 式 | 0.35 | 0.18 | 0.38 | 0.58 |
| **本方法** | **≥0.55** | **≥0.30** | **≤0.15** | **≥0.62** |

- 统一评测口径：同一 STRIDE-QA 子集（同口问题模板）、同一 Bench2Drive-VL 场景、同一 judge（固定 LLM 版本 + temperature=0）。

## 6. 实验矩阵

- **A. 主实验**：完整方法 vs 基线。目的：验证 H1/H2。预期：空间定位 ≥55%、矛盾率 ≤15%、PDMS ≥0.62。
- **B. 占据锚定消融**：占据 token 有/无、答案含体素/不含。目的：验证 H1 的锚定贡献。
- **C. 奖励设计**：DPO vs RLVR vs 无 RL（纯监督）。目的：验证 H2。预期：DPO 稳、RLVR 上限高。
- **D. 难例规模**：RL 难例 {1k, 5k, 20k}。预期：5k 饱和。
- **E. judge 消融**：LLM judge vs 规则 judge（几何一致检测）。目的：验证 H3。预期：LLM κ≥0.8 更优。
- **F. 跨数据**：nuScenes 训练 → STRIDE-QA（东京）评测。目的：跨域泛化。预期：定位 ≥40%（跨城市仍有效）。
- **G. 闭环集成**：VQA 答案作为 planner 上下文输入 vs 纯轨迹。预期：VQA 上下文提升闭环 PDMS。

## 7. 评测协议

- 指标：空间定位准确率、运动预测一致性（STRIDE-QA 定义）、答案-轨迹矛盾率、闭环 PDMS、通用 VQA 准确率（回归）、judge κ。
- 均值±方差：3 seeds；显著性配对 t-test。
- 固定：judge LLM 版本、temperature=0、评测问题模板、闭环场景 seed。
- 可复现：公开 QA 合成 prompt、judge prompt、config/权重（QLoRA adapter）。

## 8. 算力与资源计划

- 阶段 1 VQA 微调：100–150k step ≈ 8–10 GPU·天（4 卡 FSDP+QLoRA）。
- 阶段 2：DPO ≈ 2 GPU·天；RLVR ≈ 2 GPU·天。
- 评测：STRIDE-QA 子集 + Bench2Drive-VL ≈ 1.5 GPU·天。
- 合计 ≈ 14 GPU·天（预算上限 16）；存储 ≈ 200GB。
- API：QA 生成 50 万条 + judge 5 万难例 ≈ 60M token ≈ $60–100（量最大、成本最高的 idea；可降为 30 万条 QA 压预算）。

## 9. 里程碑与时间线（10 周）

| 周 | 交付物 |
|---|---|
| W1 | OpenDriveVLA 权重加载 + 数据许可确认 |
| W2 | 占据 tokenizer + QA 合成流水线（LLM） |
| W3 | 矛盾 judge v1 + κ 审计 |
| W4 | 阶段 1 训练启动 |
| W5 | 阶段 1 完成 + 主实验 A 初版（VQA 指标） |
| W6 | 阶段 2 DPO 接入 + 矛盾率评测 |
| W7 | RLVR（可选）+ 消融 B/C |
| W8 | 实验 D/E + judge 消融 |
| W9 | 实验 F/G + 统计检验 |
| W10 | 论文初稿 + 开源 + 投稿 |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 7B VLA 在 4×L40 上 RLVR 采样慢 | 中 | 高 | 预先批量化 QA 生成 + RL 只用小规模难例（5k）；RLVR 不稳退化为 DPO 两阶段 |
| STRIDE-QA 许可/数据获取受限 | 中 | 中 | 用 Reason2Drive（arXiv 2312.03661）+ nuScenes-QA 生态替代；评测口径在文中说明 |
| judge 不准确污染奖励 | 中 | 中 | κ≥0.8 门槛 + 规则几何判矛盾兜底（答案体素 vs 轨迹体素 IoU） |
| VQA 监督拉低规划能力 | 低 | 中 | 阶段 1 后跑规划回归检查；混合训练中规划 loss 权重 ≥1 |

## 11. 论文写作计划

- 目标会议：CVPR 2027（11 月截稿）或 ECCV 2027；备选 ICLR 2027。
- 差异化卖点一句话：VQA 第一次「锚定在 3D 占据上且与规划自洽」——用答案-轨迹矛盾做 RLHF，问答与开车互相校验。
- 拟用图表：Fig1 占据锚定 VQA 框架；Fig2 问答样例（含体素引用可视化）；Fig3 矛盾率 vs 训练阶段曲线；Fig4 RLVR 收敛曲线；Fig5 跨域结果；Table1 基线总表；Table2 消融；Table3 闭环 PDMS。
- 相关工作覆盖：驾驶 VQA（STRIDE-QA 收藏论文、Reason2Drive arXiv 2312.03661、Bench2Drive-VL arXiv 2604.01259）；VLA（OpenDriveVLA arXiv 2503.23463、DriveVLM arXiv 2402.12289、DriveMLM arXiv 2312.09245）；占据-语言（OccLLaMA arXiv 2409.03272）；RL 微调（WorldRFT DOI 10.1609/aaai.v40i14.38149）。

## 12. 参考文献

- Computer Vision IV · 论文 29 · STRIDE-QA，AAAI 2026（收藏论文）
- Computer Vision XIII · 论文 71 · OpenDriveVLA，AAAI 2026（收藏论文）
- Reason2Drive，arXiv:2312.03661
- OccLLaMA，arXiv:2409.03272
- Bench2Drive-VL，arXiv:2604.01259
- DriveVLM，arXiv:2402.12289
- DriveMLM，arXiv:2312.09245
- WorldRFT，AAAI 2026，DOI 10.1609/aaai.v40i14.38149
- NAVSIM，arXiv:2406.15349
- Bench2Drive，arXiv:2406.03877
