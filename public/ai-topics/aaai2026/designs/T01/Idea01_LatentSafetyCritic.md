# 实验设计书：Idea 1 轨迹条件安全评分潜空间世界模型（Latent Trajectory-Conditioned Safety Critic World Model）

> 主题 T01 自动驾驶感知与端到端驾驶 · 12 个 idea 之一 · 优先级：高（冲刺 CoRL 2026 / ICML 2027）

## 0. 摘要

端到端规划器的轨迹选择依赖昂贵仿真 rollout（DriveSuprim 路线）或像素级视频生成打分，成本高、OOD 不可靠。本工作训练一个「给定候选轨迹 → 直接在潜空间输出碰撞/安全分数」的世界模型批判器（safety critic），用 EOT-WM 式 ST-VAE 潜空间 + 轨迹注入 DiT 骨干，去掉完整去噪生成，仅用轻量 head 回归碰撞概率、违规率与安全裕度。方法在 nuScenes 真值轨迹与 LLM 生成的语义扰动轨迹上构造正负样本，损失 = 轨迹条件安全回归 + 潜空间安全对比学习，并引入控制潜相似度做校准。预期贡献：(1) 首个可微的「世界模型即安全 critic」统一接口；(2) 轨迹选择的计算成本较像素级 rollout 降低一个数量级；(3) 与 NAVSIM 仿真真值的一致性达到 AUC≥0.88，闭环 PDMS 提升 ≥5%。

## 1. 研究背景与动机

### 1.1 问题定义

给定场景历史观测 $O_{t-K:t}$（多相机 + LiDAR BEV）与一条候选自车轨迹 $\tau = \{(x_s,y_s,\theta_s,v_s)\}_{s=t+1}^{t+H}$，求安全分数 $s(\tau) \in [0,1]^{d}$（碰撞概率、违规率、安全裕度），要求：(a) 计算代价远低于像素级视频 rollout；(b) 对 OOD 场景的分数校准可靠；(c) 分数对轨迹微分，可接入 planner 作可微闭环。

### 1.2 相关工作不足

- **可控世界模型不做评分**：EOT-WM（Computer Vision XIII · 论文 88）在视频潜空间统一自车-他车轨迹，但输出是视频、不产出轨迹级安全分数；Fine-flow（Machine Learning VIII · 论文 74）只优化长期生成一致性，无安全语义。
- **选择式规划依赖昂贵反馈**：DriveSuprim（Computer Vision XI · 论文 64）的评分网络依赖仿真器反馈，成本高且 OOD 选择置信度无保证；VeteranAD（Computer Vision XII · 论文 15）无闭环 reward，长尾泛化有限。
- **现有世界模型评估指标与规划脱节**：EOT-WM 的 control-latent similarity 与下游规划收益的相关性未验证（报告 3.3 精读空白 4）。
- 外部依据：WorldRFT（AAAI 2026，DOI 10.1609/aaai.v40i14.38149）证明 latent 世界模型可用于 RL 化规划，说明「世界模型 ≠ 只能生成」是 2026 年正在打开的方向；From Words to Collisions（arXiv 2502.02145）证明 LLM/世界模型作 critic 评估驾驶是可行路线。

### 1.3 为什么是现在、为什么你的环境适合

- 现在是：潜空间世界模型（EOT-WM 2026）+ 闭环基准（NAVSIM arXiv 2406.15349 / Bench2Drive arXiv 2406.03877）已成熟，但「评分头」尚未被独立提出。
- 环境适合：critic 骨干 0.5–1B 潜 DiT，4×L40（192GB）FSDP 全参微调毫无压力（预算 ≤13B）；预训练 latent 可复用公开权重（Vista/DriveDreamer 谱系蒸馏）；NAVSIM 官方 API 提供低成本闭环真值标签用于校准，无需建仿真器。

## 2. 研究目标与可验证假设

- **H1（可微安全表示）**：潜空间安全嵌入能在不生成像素的前提下回归碰撞/违规分数。
  - 成立可观测结果：critic 在 NAVSIM 仿真真值标签上的 AUC≥0.88、AP@90% 召回≥0.82；潜空间分数与 rollout 碰撞标签 Pearson ρ≥0.85（校准集）。
- **H2（选择收益）**：用 critic 分数替代仿真器打分做轨迹选择，闭环驾驶性能不降反升。
  - 成立可观测结果：接入 OpenDriveVLA/VeteranAD 后，NAVSIM PDMS 提升 ≥3 个点、Bench2Drive DS ≥+2、选择命中率（选择轨迹 vs 仿真最优）≥80%。
- **H3（OOD 可靠性）**：critic 在未见过的扰动类型上分数仍可信。
  - 成立可观测结果：留出型 OOD（测试集用训练未见扰动幅度/类型）下 AUC 下降 <0.05；控制潜相似度与真值碰撞的相关性 ≥0.6。
- **H4（成本优势）**：潜空间打分比像素级 rollout 便宜一个数量级。
  - 成立可观测结果：单轨迹打分时延 <15ms（L40 上）、FLOPs 低于像素生成 10×；消融显示「潜空间 vs 半像素混合」成本-精度 Pareto 前缘。

## 3. 总体方法设计

### 3.1 数据流水线

- 输入：nuScenes（700 场景，6 相机 + LiDAR）历史帧 $t-K:t$（K=5），轨迹窗口 $H=20$（未来 4s @5Hz）。
- 轨迹候选构造：
  1. **真值正样本**：nuScenes 真值自车轨迹 + 前景/场景真实他车未来轨迹（无碰撞、守规）。
  2. **扰动负样本**：对真值轨迹施加参数化扰动——横向偏移 $\delta_x \in \{\pm1, \pm2, \pm3\}m$、纵向偏移/速度扰动 $\delta_v \in \{\pm5, \pm10, \pm15\}km/h$、组合扰动（压线、超速、侵入他车路径）。
  3. **LLM 语义难样本（DeepSeek V4 Pro）**：prompt「你是一名场景安全标注员。给定场景文字描述（车道数、他车位置/速度、信号灯），构造 3 条会导致碰撞/违规的自车轨迹，并给出失败原因标签」。输出结构化 JSON：`{trajectory:[[x,y,v,θ]...], violation_type, actor_id}`。把 LLM 轨迹转化为扰动施加到真值上。预期每场景产 8–15 条难样本，总量约 4 万条候选轨迹（正:负 ≈ 1:3）。
- 过滤规则：碰撞标签不依赖仿真器——用运动学几何判定（自车轨迹包围盒与他车未来轨迹包围盒 IoU>0 即碰撞）；违规判定 = 越界（地图车道多边形外）或超速（>限速+15%）。对几何判定模糊样本（边距 0.5–2m）留作「难度校准集」而非直接丢弃。
- 数量预期：nuScenes train 700 场景 × 每场景 40–60 条候选 ≈ 3 万条；NAVSIM 候选集补充 1 万条；合计 4 万条，其中 5% 作为手工抽检集（人工标注 1000 条用于 LLM/几何标签质量审计）。

### 3.2 模型/算法设计

- 骨干：EOT-WM 式 ST-VAE 潜空间（**不重建像素**）——场景历史经时空 VAE 编码为潜码 $z_{obs}\in\mathbb{R}^{d_z}$（$d_z=256$）；候选轨迹经轨迹编码器映射 $z_\tau = \text{MLP}_{\tau}([\mathbf{p},\mathbf{v},\mathbf{a}])$，其中 $\mathbf{p}\in\mathbb{R}^{H\times 3}$ 为位置/朝向、$\mathbf{v},\mathbf{a}$ 为速度/加速度。
- 交互模块：轨迹注入 DiT 前若干层（仅 4–6 层，不接去噪输出），把 $z_\tau$ 以 cross-attention 注入 $z_{obs}$，得到条件潜表示 $z_{cond}$。
- Critic head：$\text{MLP}_c(z_{cond}) \to (p_{col}, p_{viol}, m_{safety})$，其中 $p_{col},p_{viol}\in(0,1)$（sigmoid），$m_{safety}\in\mathbb{R}$（安全裕度，单位 m）。
- 关键公式：
  - 主损失 $L_{safe} = \text{BCE}(p_{col},\hat p_{col}) + \lambda_v \text{BCE}(p_{viol},\hat p_{viol}) + \lambda_m \text{Huber}(m_{safety}, \hat m_{safe})$，$\lambda_v=0.7, \lambda_m=0.5$。
  - 潜空间安全对比损失 $L_{contr} = -\log\frac{e^{\text{sim}(z_{cond}^+, z_{safe})/\tau_c}}{e^{\text{sim}(z_{cond}^+,z_{safe})/\tau_c} + \sum_{j}e^{\text{sim}(z_{cond}^{-,j},z_{safe})/\tau_c}}$，$\tau_c=0.1$，$z_{safe}$ 为可学习安全原型向量，正样本为安全轨迹。
  - 校准辅助：最小化 $| \text{rank}(p_{col}) - \text{rank}(\text{CS}) |$，CS 为 EOT-WM control-latent similarity，作为软排序正则（权重 $\lambda_{cs}=0.1$）。
  - 总损失 $L = L_{safe} + \lambda_{cl} L_{contr} + \lambda_{cs} L_{cs}$，$\lambda_{cl}=0.3$。
- 超参数初值：DiT depth 6、head 8、$d_{model}=512$；dropout 0.1；潜维 $d_z=256$。

### 3.3 训练流程

- 优化器 AdamW，lr 3e-4，warmup 2k step，cosine 衰减到 3e-5；weight decay 0.05；梯度裁剪 1.0。
- 批次：单卡 batch=16，4×L40 FSDP 全参微调，gradient accumulation 至 global batch 64；mixed precision bf16。
- 总步数 ~60k step（4 万条样本 × ~2.5 epoch 有效重复），约 3–4 GPU·天（复用预训练 latent 编码器冻结，只训注入层+head 的前 40k step，再解冻全部骨干微调 20k step）。
- 并行方案：FSDP（shard 参数+梯度），2 阶段：阶段 1 冻结 backbone、只训 head（lr 6e-4，20k）；阶段 2 全参（3e-4，40k）。

### 3.4 推理与评测流程

- 推理：给定候选轨迹集（8 条），并行编码打分，贪心取最高 $1-(p_{col}+\beta p_{viol})$ 者（$\beta=0.5$）。
- 评测流程：critic 打分 → 选轨迹 → 交给 NAVSIM 环境 rollout（官方协议）得到真值碰撞/违规标签 → 计算一致性指标与闭环 PDMS。全程固定随机种子。

## 4. 数据集细节

- **nuScenes**：来源 https://www.nuscenes.org/，非商用许可（CC BY-NC-SA 4.0），可用于研究；700 场景（train 700、val 150，官方 850 全量含 test 150 未公开标签）。本工作用官方 train/val 划分；含 6 相机 + 5 个 LiDAR（本方法只用单 LiDAR 中心）+ 1.4M 3D 标注框。地图 API 提供车道多边形（用于越界/压线判定）。
- **NAVSIM**（arXiv 2406.15349）：官方数据集 `navsim-data-mini/full`（~150h 演示，基于 nuScenes 场景重采样轨迹真值），轨迹候选集（8 条 expert trajectories + 扰动）官方直接提供；协议见 https://github.com/autonomousvision/navsim。
- **OpenDriveVLA 轨迹预训练**（可选，arXiv 2503.23463）作为轨迹编码器的初始化对比。
- 划分：训练 4 万条（约 70% 场景）、校准集（约 15% 场景，用于 Pearson/阈值搜索）、测试集（15% 场景）。
- 预处理：图像 resize 224×224（BEVFormer 式 6 视角）、BEV 栅格 0.5m×200m；轨迹归一化到最近 1s 起始位姿；时间窗 K=5、未来 H=20（5Hz）。统一存为 HDF5，字段：`images(L,K,3,224,224)、lidar_bev、traj(8,H,4)、gt_label(8,3)、scene_id`。

## 5. 基线复现

| 基线 | 引用 | 官方代码 | 复现要点 |
|---|---|---|---|
| EOT-WM 像素级 rollout 打分 | Computer Vision XIII · 论文 88 | 无官方代码（AAAI 2026），按其论文机制复现：轨迹注入 DiT 生成视频后用 VLM/几何判定碰撞 | 仅实现「生成→几何判定」基线，7 卡天工作量，标记为近似复现 |
| DriveSuprim 评分网络 | Computer Vision XI · 论文 64 | 无官方代码 | 复现其粗到细候选过滤 + 评分头，在 NAVSIM 上训练 |
| 规则几何校验器（Simple Safety） | 自定义基线 | — | 用包围盒 IOU + 车道越界判定，无需训练，作为最低成本参照 |
| OccWorld 式占据未来预测打分 | arXiv 2311.16038 | https://github.com/tianyu-hust/OccWorld | 用占据预测的未来体素做碰撞判定（替代潜空间） |

- 复现步骤：逐一下载官方权重/代码 → 在 NAVSIM val 上按官方协议跑出各自基线分数表 → 记录到统一表格。
- 预期指标表（NAVSIM val，全 8 条候选轨迹的选择一致性）：

| 方法 | 与真值碰撞标签 AUC | 选择命中率↑ | PDMS（接入 openloop planner） | 单轨迹时延 |
|---|---|---|---|---|
| 规则几何校验器 | 0.72（低召回） | 55% | 0.58 | <1ms |
| EOT-WM 像素 rollout | 0.85（近） | 72% | 0.60 | >500ms |
| DriveSuprim 评分网络 | 0.87（估） | 78% | 0.62 | 30ms |
| **本方法（潜空间 critic）** | **≥0.88** | **≥80%** | **≥0.65** | **<15ms** |

- 统一评测口径：同一 NAVSIM 版本（v2 协议）、同一 8 条候选集、同一 val 场景子集、同一 seed。

## 6. 实验矩阵

- **A. 主实验**：潜空间 critic 完整管线 vs 各基线。目的：验证 H1/H2。配置：全数据 + 全损失。预期结论：AUC≥0.88、PDMS≥0.65，全面超基线。
- **B. 潜空间 vs 半像素混合 vs 像素级**：三档 critic 表示的成本-精度曲线。目的：验证 H4（成本优势）与「潜空间是否够」。预期：潜空间在高需求（批量轨迹选择）时 Pareto 占优；若 AUC<0.85 则半像素混合兜底。
- **C. 扰动幅度敏感性**：横向偏移 ±1/±2/±3m 分别训练/评测。目的：测 critic 对边界样本的校准。预期：AUC 随幅度增大上升，但 ±1m 处召回应 ≥70%。
- **D. OOD 迁移**：训练用 nuScenes，评测在 NAVSIM 候选集 + 手动构造的未见过组合扰动（如「雨夜同时压线+急刹」）。预期：AUC 下降 <0.05（否则需要难样本扩增）。
- **E. 损失消融**：−对比损失、−控制潜相似排序正则、−违规 head。目的：各损失贡献。预期：去对比损失 AUC −0.02，去排序正则校准 −0.03。
- **F. LLM 难样本消融**：用 vs 不用 LLM 语义扰动难样本。目的：验证难样本对 OOD 的价值。预期：LLM 难样本使 OOD AUC +0.04。
- **G. 与 planner 联合训练**：critic 与 OpenDriveVLA 轨迹头端到端微调（可微选择）。目的：验证可微闭环价值。预期：联合微调后闭环 PDMS +2（对 3.4 流程的升级版）。

## 7. 评测协议

- 指标定义：AUC（正=真值碰撞标签）、AP（PR 曲线下面积）、选择命中率 = 选中轨迹 == 仿真最优轨迹 的比例、PDMS（NAVSIM 官方 Driving Metric Score）、时延（L40 单卡 8 轨迹平均）。
- 均值±方差：3 个随机种子（seed ∈ {0,1,2}），报告 mean±std；差异显著性用配对 t-test（α=0.05）或 Wilcoxon 符号秩（不满足正态时）。
- 校准度量：期望校准误差 ECE（10-bin，碰撞概率），报告 reliability 图。
- 可复现性：固定 torch/cuda RNG、numpy seed；NAVSIM 环境固定版本（pip 冻结）；公开训练 config + 权重。
- 人工审计：从校准集随机抽 1000 条，2 名标注员独立打碰撞标签，Cohen's κ≥0.8 才视为标签可信。

## 8. 算力与资源计划

- 阶段化 GPU·天（4×L40）：阶段 1 骨干冻结 20k step ≈1 GPU·天；阶段 2 全参 40k step ≈2.5 GPU·天；基线复现 EOT-WM 像素级 ≈2 GPU·天；NAVSIM 闭环评测 30 轮 × 0.5 天 ≈2 GPU·天；合计 ≈7.5 GPU·天（预算留 30% 余量 → 10 GPU·天）。
- 存储：nuScenes 全量 ≈250GB + NAVSIM mini ≈60GB + 特征缓存 4 万条 × (224²×6×K×fp16) ≈200GB + 模型权重 ≈10GB；合计 <600GB，1TB SSD 足够。
- API 用量：LLM 难样本生成 4 万条轨迹描述 × 3 轮 ≈ 12 万次调用（DeepSeek V4 Pro，每约 1.5k token）→ 约 1.8 亿 token ≈ $20–40（按 0.5–1 元/百万输出估算）；抽样审计 prompt 另 +10%。成本 <50 美元，可忽略。

## 9. 里程碑与时间线（单人 + 4 卡，10 周）

| 周 | 交付物 |
|---|---|
| W1 | nuScenes/NAVSIM 数据下载与预处理流水线；LLM 难样本生成器 v1（500 条人工抽检通过率 ≥90%） |
| W2 | ST-VAE latent + 轨迹编码器 + critic head 代码；EOT-WM 近似复现开始 |
| W3 | 阶段 1 训练完成，critic 过拟合检查；几何真值标签质量审计（κ 检验） |
| W4 | 阶段 2 全参训练完成；主实验 A 基线表格（AUC/命中率） |
| W5 | NAVSIM 闭环接入；实验 B（潜 vs 像素成本曲线）、实验 C（扰动敏感性） |
| W6 | 实验 D OOD 迁移 + E 损失消融 |
| W7 | 实验 F LLM 难样本消融 + G 联合训练（可选项） |
| W8 | 全套统计检验 + 图表（reliability、Pareto、消融热图） |
| W9 | 论文初稿（CoRL 6 页 + 附录） |
| W10 | 内审、开源仓库整理（config/权重/评估脚本）、投稿 |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 潜空间分数与真值碰撞相关性不足（ρ<0.6） | 中 | 高 | 退回「半像素级」混合架构：低分辨率 32×64 rollout + 潜 critic 头；或在损失中加大几何规则项 |
| 几何标签噪声导致 critic 学到坏标签 | 中 | 中 | 人工审计 + κ 门槛；对模糊样本设「忽略/软标签」权重 0.3 |
| EOT-WM 无官方代码、复现成本高 | 高 | 低 | 用 Vista/GenAD 的 latent 世界模型权重替代骨干，仅在文中说明差异 |
| NAVSIM 候选集与扰动样本分布偏差 | 低 | 中 | 评测统一用 NAVSIM 官方 8 条候选集，训练扰动样本独立于评测 |
| LLM 难样本质量差 | 中 | 低 | 过滤规则：轨迹物理可行性（速度/加速度上限）校验 + 抽检率 10% |

## 11. 论文写作计划

- 目标会议：CoRL 2026（截稿通常 5 月初，按 2026 年日历确认）；备选 ICML 2027 / ICLR 2027 workshop。
- 差异化卖点一句话：第一个「世界模型即安全 critic」——不生成像素、可微、OOD 校准的轨迹级安全分数。
- 拟用图表清单：Fig1 方法框架图；Fig2 潜空间安全嵌入可视化（t-SNE）；Fig3 reliability 图 + AUC；Fig4 成本-精度 Pareto；Fig5 消融热图；Table1 基线总表；Table2 OOD 结果；Table3 消融表；Table4 时延对比。
- 相关工作覆盖：可控世界模型（EOT-WM、GAIA-1 arXiv 2309.17080、Vista arXiv 2405.17398、DriveDreamer arXiv 2309.09777、D2-World arXiv 2411.17027）；选择式规划（DriveSuprim、Hydra-MDP arXiv 2406.06978）；世界模型评估（WorldRFT、From Words to Collisions arXiv 2502.02145）；闭环基准（NAVSIM arXiv 2406.15349、Bench2Drive arXiv 2406.03877）。

## 12. 参考文献

- Computer Vision XIII · 论文 88 · EOT-WM，AAAI 2026，DOI 10.1609/aaai.v40i16.38403
- Computer Vision XI · 论文 64 · DriveSuprim，AAAI 2026（收藏论文）
- Computer Vision XII · 论文 15 · VeteranAD，AAAI 2026，DOI 10.1609/aaai.v40i15.38230
- Machine Learning VIII · 论文 74 · Fine-flow，AAAI 2026（收藏论文）
- WorldRFT，AAAI 2026，DOI 10.1609/aaai.v40i14.38149
- From Words to Collisions，arXiv:2502.02145
- NAVSIM，arXiv:2406.15349
- Bench2Drive，arXiv:2406.03877
- OccWorld，arXiv:2311.16038
- Hydra-MDP，arXiv:2406.06978
- GAIA-1，arXiv:2309.17080
- Vista，arXiv:2405.17398
- DriveDreamer，arXiv:2309.09777
- D2-World，arXiv:2411.17027
- OpenDriveVLA，arXiv:2503.23463
- EOT-WM 论文未开源；以上 external 引用均见调研报告第五节与附录，已核验
