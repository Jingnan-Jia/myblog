# 实验设计书：Idea 3 仿真器在环的端到端自校正强化闭环（Closed-Loop Self-Correction with Counterfactual RL）

> 主题 T01 自动驾驶感知与端到端驾驶 · 12 个 idea 之一 · 优先级：高（NeurIPS 2026 / CoRL 2026）

## 0. 摘要

CorrectAD 的自校正是一次性离线补数据，无持续回路、无 RL 信号、生成与规划两段式。本工作把其升级为「世界模型生成反事实失败 → 策略 RL 微调 → 再仿真验证 → 再生成」的 simulator-in-the-loop 持续闭环：用 EOT-WM 式自车-他车可控生成把失败片段「变化出」更多同分布变体，用 Idea 1 安全 critic 稠密化 reward，并设验证门控保证 RL 后的策略确实在未见变体上复测达标。预期在 nuScenes/Bench2Drive 场景上失败率下降 40%+、Bench2Drive DS 提升 ≥5 点。贡献：(1) 首个反事实数据飞轮 × RL 微调 × 验证门控的三段闭环；(2) 世界模型生成成本与 RL 收益的可度量 trade-off 分析。

## 1. 研究背景与动机

### 1.1 问题定义

给定端到端 planner $\pi_\theta$（感知+轨迹头）与失败案例库 $\mathcal{D}_{fail}$（碰撞/违规/偏离片段），目标是持续生成同分布反事实变体 $\mathcal{D}_{cf}$，用 RL 微调 $\theta$，并验证微调后策略在未见变体上失败率确实下降，形成闭环直至失败率低于阈值 $T_f$。

### 1.2 相关工作不足

- CorrectAD（Computer Vision VII · 论文 4，arXiv 2511.13297）一次性重训、无持续回路、生成的视频未来不作为轨迹评估器——本 idea 的三大空白点逐一对应改进。
- WorldRFT（AAAI 2026，DOI 10.1609/aaai.v40i14.38149）证明 latent 世界模型 + RL 微调可行，但未与「失败驱动的反事实数据生成」结合。
- DriveSuprim（Computer Vision XI · 论文 64）选择式规划依赖仿真器反馈；Hydra-MDP（arXiv 2406.06978）多目标蒸馏但无闭环 RL。
- 数据扩展证据：Data Scaling Laws（arXiv 2504.04338）、DriveE2E（arXiv 2509.23922）证明世界模型合成数据 → 端到端性能缩放可行，支撑「数据飞轮」假设。

### 1.3 为什么是现在、为什么你的环境适合

- 现在是：世界模型可控生成（EOT-WM）成熟到可作数据引擎；RLVR（RL with verifiable rewards）在语言模型侧爆发、WorldRFT 证明其迁移到驾驶规划有效。
- 环境适合：planner 用 1–2B（QLoRA 4bit）在 4×L40 上可跑；世界模型生成与 RL 采样按「2 卡生成 + 2 卡 RL」流水线切分；Bench2Drive/NAVSIM 提供低成本闭环验证环境，无需建仿真器。

## 2. 研究目标与可验证假设

- **H1（飞轮有效）**：世界模型反事实变体训练比「同分布数据增强」显著更降失败率。
  - 可观测结果：在未见变体测试集上碰撞/违规失败率相对 CorrectAD 一次性管线下降 ≥40%；相同数据量下反事实训练 vs 高斯扰动增强的失败率差 ≥10 个点。
- **H2（critic 稠密化 reward 有效）**：用 Idea 1 安全 critic 稠密化 reward 比稀疏事件 reward 训练更快收敛。
  - 可观测结果：同 RL 步数下，critic reward 版收敛的 success 率高 ≥8 个点；reward 方差显著更低。
- **H3（验证门控必要）**：设验证门控比无门控持续生成更稳。
  - 可观测结果：无门控在 3 轮迭代后出现过拟合（未见变体失败率回升）；有门控单调下降。
- **H4（成本可控）**：反事实生成池可复用，RL 采样吞吐能支撑闭环。
  - 可观测结果：预生成池命中率 ≥60%，单轮迭代 ≤3 GPU·天，总 8–12 GPU·天。

## 3. 总体方法设计

### 3.1 数据流水线

- 失败库：nuScenes val + Bench2Drive 场景库，运行基线 planner 收集失败片段（碰撞/压线/超速/偏离目标），每条片段含 {视频帧, BEV, 自车/他车轨迹, 失败类别}。预期 3k 条失败片段。
- 反事实生成（EOT-WM 可控生成）：对失败片段，改变自车/他车轨迹参数化（初始横向偏移 ±{0.5,1,1.5}m、速度 ±{2,5,8}m/s、启动时刻 ±{0.4,0.8}s），生成 N=8 个变体视频 + 轨迹真值。变体质量过滤：物理可行性（速度/加速度上限）、语义一致性（LLM 审查场景描述与生成一致性评分 ≥0.7）。
- LLM 角色（DeepSeek V4 Pro）：
  1. 失败描述结构化：把失败片段转成 JSON（成因、场景要素、关键交互车辆、责任主体）。
  2. 驱动反事实变体参数建议：基于失败描述生成「更有挑战性的变体组合」。
  3. RL 奖励解析：把 critic 分数/几何量解析成可微 reward 的语言模板。
- 预生成池：离线批量生成 2 万条变体入池（避免在线等待），按难度/多样性索引。
- 数量预期：每轮迭代消耗池内 5k 条变体；总 4 轮迭代，池增长至 3 万条。

### 3.2 模型/算法设计

- Planner：Hydra-MDP / SparseDrive（arXiv 2405.19620）结构（BEV encoder + 轨迹 head），参数量 1–2B（放大版）或 200M（基线版）。用 QLoRA 4bit（rank=16）微调。
- RL 设置：状态 = BEV 特征 + 轨迹潜码 + 他车历史轨迹 embedding；动作 = 自车轨迹参数化 $\{(x_s,y_s,v_s,\theta_s)\}$（多项式系数，degree 4，H=20）。
- Reward：
  - $r_{event} = -c_{col} \cdot \mathbb{1}[collision] - c_{viol} \cdot \mathbb{1}[violation] - c_{off} \cdot d_{off}$（稀疏事件 + 偏离惩罚）。
  - $r_{dense} = \lambda_c (1-p_{col}) + \lambda_s m_{safety}$（Idea 1 critic 稠密化），$\lambda_c=2.0, \lambda_s=1.0$。
  - 进度奖励：$r_{prog} = \Delta (v_{ego})$（速度提升奖励）+ 靠近目标点奖励。
  - 总 $r = r_{event} + r_{dense} + 0.1 r_{prog}$。
- 算法：PPO（clip=0.2，GAE λ=0.95，epochs=4，mini-batch 512）或 RLVR 变体；critic 网络 = 状态价值 MLP（与 safety critic 不同，RL value head）。
- 验证门控：每轮 RL 后在新生成变体（池外 1k 条）与真实场景上复测，若失败率未降 ≥5 个点 → 提高反事实难度（更大扰动）再迭代；若 ≤$T_f$（如 10%）→ 提前停止。
- 超参初值：lr 3e-5（QLoRA 参数），KL 系数 0.01，RL 每轮 4k 步（= 200 episode × 20 步）。

### 3.3 训练流程

- 4 卡流水线：GPU0-1 跑世界模型反事实生成（批量去噪，预生成池离线补）；GPU2-3 跑 planner RL（PPO 采样 + 更新）。
- 每轮迭代：失败库采样 → 反事实生成（离线池优先）→ PPO 4k 步 → 验证门控评测（GPU3 空闲时并行）。
- 优化器：LoRA 用 AdamW lr 3e-5，warmup 500，bf16；世界模型生成用 fp16。
- 总预算：4 轮 × 2.5 GPU·天 ≈ 10 GPU·天（含评测）。

### 3.4 推理与评测流程

- 推理：planner 每 0.2s 输出轨迹，Bench2Drive/NAVSIM 环境闭环执行。
- 评测：每轮迭代后跑完整 Bench2Drive（50 场景）记录 DS/成功率；另用「池外未见变体」测试集跑失败率（碰撞/违规/偏离）。
- 对比：同一 planner 初始权重下跑 CorrectAD 一次性管线（离线补数据重训）作为对照。

## 4. 数据集细节

- nuScenes（CC BY-NC-SA 4.0）train/val 700/150 场景：失败片段来源 + 反事实场景基底。
- Bench2Drive（arXiv 2406.03877）：闭源数据需申请（官方 form）；替代方案用 NAVSIM（arXiv 2406.15349）官方数据集。评测脚本官方开源。
- 失败库标注：运行基线上游模型自动产出 + LLM 结构化描述校验（抽检 10%）。
- 划分：失败库 70% 训练变体、15% 验证门控、15% 池外测试。

## 5. 基线复现

| 基线 | 引用 | 官方代码 | 复现要点 |
|---|---|---|---|
| CorrectAD 一次性管线 | Computer Vision VII · 论文 4，arXiv 2511.13297 | 无官方代码 | 复现 PM-Agent 需求→DriveSora 式生成→重训（用 EOT-WM 替代 DriveSora 生成） |
| SparseDrive + 数据增强 | arXiv 2405.19620 | https://github.com/ucaszyp/SparseDrive | 官权重 + 高斯扰动增强数据 |
| WorldRFT 式 latent WM RL | AAAI 2026 DOI 10.1609/aaai.v40i14.38149 | 无 | 按论文：latent WM 作环境 + RL 微调（无失败驱动） |
| **本方法（飞轮闭环）** | — | 本项目开源 | 反事实飞轮 + critic 稠密 reward + 验证门控 |

- 预期指标表（Bench2Drive DS、池外失败率）：

| 方法 | DS↑ | 池外失败率↓ | 迭代轮数 | 总 GPU·天 |
|---|---|---|---|---|
| SparseDrive+增强 | 72% | 18% | 1 | 2 |
| CorrectAD 一次性 | 75% | 14% | 1 | 3 |
| WorldRFT 式 | 76% | 13% | 1 | 4 |
| **本方法** | **≥80%** | **≤8%** | 2–4 | 10 |

- 统一评测口径：同一 planner 初始权重、同一 50 场景 Bench2Drive 子集、同一池外测试集。

## 6. 实验矩阵

- **A. 主实验**：完整闭环 vs 三个基线。目的：验证 H1/H3。预期：DS≥80%、池外失败率≤8%。
- **B. 反事实多样性 vs 高斯增强**：同数据量对比。目的：验证飞轮核心假设。预期：失败率差 ≥10 个点。
- **C. reward 稠密化消融**：稀疏事件 / critic 稠密 / 进度。目的：验证 H2。预期：critic 版收敛最快、方差最低。
- **D. 验证门控 on/off**：有/无门控 × 4 轮。目的：验证 H3。预期：无门控第 3 轮失败率回升。
- **E. 生成频率与 RL 频率比例**：{1:1, 1:3, 3:1}。预期：生成过频浪费、过稀不足，1:3 附近最优。
- **F. 过拟合风险**：变体分布偏差 vs 真实场景 DS。目的：检测反事实分布 shift。预期：控制变体难度上限可缓解，报告 trade-off 曲线。
- **G. 模型无关性**：把 planner 换成 OpenDriveVLA 轨迹头。目的：验证通用性。预期：同样趋势成立。

## 7. 评测协议

- 指标：Bench2Drive DS/成功率、NAVSIM PDMS、池外失败率（碰撞/违规/偏离分类）、reward 曲线收敛性。
- 均值±方差：3 seeds；显著性用配对 t-test。
- 固定：环境随机（vehicle spawn/红绿灯时序）seed、RL 采样 seed、初始权重文件。
- 可复现：公开 config（含反事实参数范围、门控阈值）、预生成池哈希。

## 8. 算力与资源计划

- 世界模型反事实生成：离线 2 万条 ≈ 3 GPU·天（2 卡）；在线补生成 ≈ 1 GPU·天。
- RL 微调：4 轮 × 1.5 GPU·天 ≈ 6 GPU·天（2 卡）。
- 评测：每轮 Bench2Drive 50 场景 ≈ 0.5 天 × 5 方法 ≈ 2.5 GPU·天。
- 合计 ≈ 12.5 GPU·天（预算上限 15）。
- 存储：反事实视频池 3 万条 × 10 帧 × 224²×6 ≈ 1.2TB（用 latent/压缩存储降到 400GB）。
- API：失败描述 3k 条 + 变体建议 8k 条 ≈ 15M token ≈ $15–30。

## 9. 里程碑与时间线（10 周）

| 周 | 交付物 |
|---|---|
| W1 | 基线 planner（SparseDrive checkpoint）加载 + 失败片段收集器 |
| W2 | EOT-WM 反事实生成器 v1 + 预生成池（1 万条） |
| W3 | LLM 失败描述结构化 + 变体参数建议 |
| W4 | PPO 实现（QLoRA 4bit）+ critic reward 接入（复用 Idea 1 head） |
| W5 | 主实验 A：第一轮闭环跑通（生成→RL→验证） |
| W6 | 实验 B（反事实 vs 增强）+ C（reward 消融） |
| W7 | 实验 D（门控 on/off 4 轮）+ E（频率比例） |
| W8 | 实验 F（过拟合 trade-off）+ G（模型无关） |
| W9 | 统计检验 + 图表（飞轮曲线、reward 曲线、失败率瀑布图） |
| W10 | 论文初稿（NeurIPS 2026 主投稿或 CoRL 2026）+ 开源仓库 |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| RL 在 4×L40 上采样吞吐不足 | 中 | 高 | 预生成池批量去噪 + 2 卡并行采样；必要时降到 1–2B planner 的 200M 子版 |
| 反事实场景与真实分布偏差过大导致过拟合 | 中 | 高 | 变体难度上限约束 + 门控复测 + 池外测试集监控；偏差大则减少扰动幅度 |
| critic reward 与事件 reward 冲突 | 低 | 中 | 消融 C 中做权重网格搜索（λ_c∈{0.5,1,2}） |
| 生成器（EOT-WM）在稀有交互上失败 | 中 | 中 | 退化为直接扰动真值轨迹（无视频）的轻量反事实 |
| Bench2Drive 数据申请周期长 | 中 | 低 | 先用 NAVSIM 数据集/环境跑通闭环，Bench2Drive 数据到位后补测 |

## 11. 论文写作计划

- 目标会议：NeurIPS 2026（5 月截稿）；备选 CoRL 2026 / ICML 2027。
- 差异化卖点一句话：第一个「世界模型反事实数据飞轮 × RL 微调 × 验证门控」的持续自校正闭环，失败率可度量的闭环证据。
- 拟用图表：Fig1 闭环框架图；Fig2 反事实变体可视化（生成样本）；Fig3 迭代轮次 vs 失败率曲线（门控 on/off）；Fig4 reward 曲线对比；Fig5 分布偏差 trade-off；Table1 基线总表；Table2 消融；Table3 模型无关验证。
- 相关工作覆盖：自校正（CorrectAD、PM-Agent）；世界模型 RL（WorldRFT、From Words to Collisions arXiv 2502.02145）；选择式规划（DriveSuprim、Hydra-MDP arXiv 2406.06978）；数据缩放（Data Scaling Laws arXiv 2504.04338、DriveE2E arXiv 2509.23922）；端到端基座（UniAD arXiv 2212.10156、VAD arXiv 2303.12077、SparseDrive arXiv 2405.19620）。

## 12. 参考文献

- Computer Vision VII · 论文 4 · CorrectAD，AAAI 2026，DOI 10.1609/aaai.v40i10.37718；arXiv:2511.13297
- Computer Vision XIII · 论文 88 · EOT-WM，AAAI 2026，DOI 10.1609/aaai.v40i16.38403
- Computer Vision XI · 论文 64 · DriveSuprim，AAAI 2026（收藏论文）
- WorldRFT，AAAI 2026，DOI 10.1609/aaai.v40i14.38149
- Hydra-MDP，arXiv:2406.06978
- SparseDrive，arXiv:2405.19620
- UniAD，arXiv:2212.10156
- VAD，arXiv:2303.12077
- Data Scaling Laws for End-to-End AD，arXiv:2504.04338
- DriveE2E，arXiv:2509.23922
- From Words to Collisions，arXiv:2502.02145
- Bench2Drive，arXiv:2406.03877
- NAVSIM，arXiv:2406.15349
