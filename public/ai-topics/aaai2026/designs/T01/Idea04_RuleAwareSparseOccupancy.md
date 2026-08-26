# 实验设计书：Idea 4 规则感知的稀疏占据预测（Rule-Aware Sparse Occupancy: Traffic Rules as Differentiable Safety Boundaries）

> 主题 T01 自动驾驶感知与端到端驾驶 · 12 个 idea 之一 · 优先级：中高（ICRA 2027 / ICLR 2027）

## 0. 摘要

现有 3D 占据预测只输出「几何-语义占据」，无法回答「该区域是否可合法进入」；规则建模（PAMR）停在地图层且规则类别封闭。本工作在 SPSC 式稀疏占据 query 上新增规则 token 通道（规则类别 + 空间作用域），随查询序列化参与注意力，输出双通道占据：物理占据 + 法理可行驶性，并用规则-占据一致性损失把「禁止区域内预测可进入」显式惩罚。数据侧用 MapDRv2 + nuScenes 地图标注 + LLM 伪标注引擎构建规则体素标注。预期 nuScenes-Occupancy 语义 mIoU 不降（保留基线）、新增法理可行驶 IoU 达 0.75+、违规占用率下降 ≥30%。贡献：(1) 首次把交通规则注入稀疏占据表示并做成可微安全边界；(2) 开源规则体素标注生成器。

## 1. 研究背景与动机

### 1.1 问题定义

给定多模态历史观测与车道级地图，预测体素网格 $V \in \mathbb{R}^{X\times Y\times Z \times C}$（C=语义类）+ 每个体素的「可进入合法边界状态」$l_v \in \{legal, illegal\}$，要求「禁止进入区（对向车道、人行道、路肩、导流线、停止线前禁止区）内预测 legal」被显式惩罚。

### 1.2 相关工作不足

- SPSC（Computer Vision III · 论文 38）只做几何-语义占据，输出不携带「是否可进入」的法理信息，且与规划解耦。
- PAMR（Computer Vision VI · 论文 6）只在地图层建模规则、规则类别封闭（预定义类别）、无法应对开放式规则，且无体素级可行驶语义。
- HD²-SSC（Computer Vision XI · 论文 55）的稠密化精修不感知规则。
- 空白：占据输出与交通规则割裂（调研报告 §7.1 第 4 点「占据只给是什么，不给能不能压线/超速/占用」）。

### 1.3 为什么是现在、为什么你的环境适合

- 现在是：规则感知的地图标注（MapDRv2）已发布（PAMR 论文），占据基准 nuScenes-Occupancy/OpenOccupancy 成熟；LLM 可自动把交规文本映射到空间作用域。
- 环境适合：SPSC 规模 <200M，4×L40 数据并行 5–6 GPU·天即可训完；规则体素标注可离线批量生成（GPU+API），成本可控。

## 2. 研究目标与可验证假设

- **H1（规则可注入且不损语义）**：规则 token 通道使法理可行驶预测准确，且语义 mIoU 相对 SPSC 不降（±0.5 点内）。
  - 可观测结果：法理可行驶 IoU ≥0.75；nuScenes-Occupancy 语义 mIoU 相对 SPSC 基线偏差 <0.5 点。
- **H2（一致性损失有效）**：规则-占据一致性损失能显著降低「禁止区被预测为可进入」的比例。
  - 可观测结果：违规占用率（illegal 区 legal 预测比例）相对无该损失的消融下降 ≥30%。
- **H3（可迁移）**：规则通道跨数据集（nuScenes→OpenOccupancy 场景）泛化。
  - 可观测结果：在 OpenOccupancy 标注场景上法理可行驶 IoU ≥0.6（规则作用域由地图 API 提供）。
- **H4（LLM 伪标注可用）**：LLM 生成规则作用域标注在人工抽检下一致率 ≥85%。

## 3. 总体方法设计

### 3.1 数据流水线

- 规则标注生成（伪标注引擎）：
  1. 从地图数据提取车道段、停止线、人行道、导流线等几何要素（nuScenes map API / MapDRv2 标注）。
  2. LLM（DeepSeek V4 Pro）输入「道路图 JSON + 交规文本片段 + 国家交规标准摘录」，输出每条规则 → 空间作用域映射：`{rule_id, rule_class, region_polygon(含车道索引/距离), temporal(是否信号灯相关)}`。
  3. 把规则作用域体素化（0.5m 栅格）→ 生成「法理可行驶真值」$l_v^*$（可通行道路+停靠区=legal；对向/人行道/导流线/路肩=illegal；未知=ignore 掩码）。
  4. 人工抽检 10%，一致率 <85% 则迭代 prompt（加规则作用域地理边界约束）。
- 数量预期：nuScenes 700 场景全量生成规则体素标注；MapDRv2 标注 1k 场景补充。每场景约 2 万规则体素。
- 过滤规则：与几何占据重叠冲突的标注（如 stop-line 后车辆合法停靠占用的体素）设为 ignore；信号灯动态规则仅用于评测不动用于训练（避免时间标注噪声）。

### 3.2 模型/算法设计

- 骨干：SPSC（渐进剪枝 + 查询序列化），多模态输入（6 相机 + LiDAR BEV）。
- 规则注入：
  1. 规则 token 通道：把规则作用域编码为规则 token $r_i = \text{MLP}_r(e_i)$（$e_i$=规则类别 embedding + 作用域 polygon 采样点 PE），数量随场景规则数（平均 60 个）。
  2. 随查询序列化一起参与注意力：将规则 token 追加到 query 序列尾部参与自注意力，再与占据 query 做 cross-attention（规则 → 占据）。
  3. 双通道输出头：物理通道 = 语义 logits（19 类，沿用 SPSC）；法理通道 = sigmoid $l_v$（可进入）。
- 损失：
  - 语义 mIoU 主损失（focal + dice，沿用 SPSC 配置）。
  - 法理通道 BCE：$L_{legal} = -\frac{1}{|\mathcal{R}|}\sum_{v\in\mathcal{R}}[l_v^*\log l_v + (1-l_v^*)\log(1-l_v)]$，仅对规则作用域体素 $\mathcal{R}$ 计算。
  - 规则-占据一致性：$L_{cons} = \frac{1}{|\mathcal{R}|}\sum_{v\in\mathcal{R}} \max(0, \eta - (l_v - \mathbb{1}[v\in illegal区]))_+ $，即禁止区内 predicted legal 概率 > 0 即惩罚；$\eta=0.3$ 容差。
  - 总 $L = L_{occ} + \lambda_l L_{legal} + \lambda_c L_{cons}$，$\lambda_l=1.0, \lambda_c=0.5$。
- 超参初值：与 SPSC 一致（query 数 300×4 级、序列化长度 1024、d_model 256、层数 8）；规则 token 通道 dim=64。

### 3.3 训练流程

- 优化器 AdamW lr 2e-4（cosine 到 2e-5），warmup 2k，batch 8/卡 × 4 = 32 global，bf16，梯度裁剪 1.0。
- 总 150k step ≈ 5–6 GPU·天（数据并行 DDP，无需 FSDP，模型 <200M）。
- 阶段：前 100k step 只训语义占据（规则 token 随机掩码 50%），后 50k step 启用规则一致性损失——先对齐表示、后学习规则耦合。

### 3.4 推理与评测流程

- 推理：输出双通道占据 → 法理可行驶层可直接作 planner 的可行驶边界约束（供后续规划器集成演示）。
- 评测：nuScenes-Occupancy 官方脚本（语义 mIoU）；法理可行驶 IoU（对规则作用域子集）；违规占用率（illegal 区 legal 预测体素数/illegal 区体素数）；与后处理基线（BEVFormer-Occ + 规则 mask 后处理）对比。

## 4. 数据集细节

- nuScenes-Occupancy：官方占据标注（19 类语义），CC BY-NC-SA 4.0；750 帧训练占据标签。
- nuScenes map API：车道多边形、stop line、人行横道、路肩；随 nuScenes 数据包提供。
- MapDRv2（PAMR 发布）：规则与车道向量标注，许可需随 PAMR 论文申请确认。
- OpenOccupancy（ICCV 2023，DOI 10.1109/iccv51070.2023.01636）：用于跨域验证（语义类对齐）。
- 划分：nuScenes 700 场景 → 560 训练 / 90 验证 / 50 测试（沿用官方帧划分）；OpenOccupancy 取 100 帧作迁移测试。

## 5. 基线复现

| 基线 | 引用 | 官方代码 | 复现要点 |
|---|---|---|---|
| SPSC | Computer Vision III · 论文 38 | 无官方代码（AAAI 2026） | 按其机制复现渐进剪枝+序列化；或用 SparseOcc 官方实现作等价骨干 |
| OccWorld | arXiv 2311.16038 | https://github.com/tianyu-hust/OccWorld | 复现其占据预测头 |
| BEVFormer-Occ + 规则后处理 | 融合基线（自建） | — | BEVFormer-Occ 公开权重 + 地图 mask 后处理 |
| **本方法（规则感知占据）** | — | 本项目开源 | 双通道 + 规则 token |

- 预期指标表（nuScenes-Occupancy val）：

| 方法 | 语义 mIoU | 法理可行驶 IoU | 违规占用率↓ |
|---|---|---|---|
| SPSC（复现） | 0.53（参考值） | — | 0.22 |
| BEVFormer-Occ + 后处理 | 0.47 | 0.55 | 0.12 |
| OccWorld 头 | 0.49 | — | 0.20 |
| **本方法** | **≥0.525** | **≥0.75** | **≤0.07** |

- 统一评测口径：同一 19 类语义、同一规则作用域掩码（对所有方法一致应用）、同一帧子集。

## 6. 实验矩阵

- **A. 主实验**：完整模型 vs 基线。目的：验证 H1。预期：mIoU 不降 + 法理 IoU ≥0.75。
- **B. 规则 token 通道数消融**：{0（无规则）, 32, 64, 128}。目的：确定容量。预期：64 饱和。
- **C. 规则注入位置**：追加 query 序列 vs 前置 cross-attention vs 后融合。预期：序列化内注入最优。
- **D. 一致性损失权重**：λ_c ∈ {0, 0.25, 0.5, 1}。目的：验证 H2。预期：λ_c≥0.5 时违规占用率下降 ≥30%。
- **E. 规则类别覆盖**：只训「停止线+人行道」vs 全类。预期：全类更稳。
- **F. 跨域迁移**：nuScenes 训 → OpenOccupancy 场景测。目的：验证 H3。
- **G. 规则感知 vs 后处理**：可微规则通道 vs 不可微后处理 mask。目的：证明可微性价值。预期：可微版在遮挡/部分遮挡路口更强。
- **H. 规划消费演示**：把法理层喂给简单轨迹优化器（约束规划），演示「不能压线」被满足。

## 7. 评测协议

- 指标：语义 mIoU、法理可行驶 IoU、违规占用率、规则作用域 F1、可微约束下的规划成功率（实验 H 补充）。
- 均值±方差：3 seeds；显著性用配对 t-test（α=0.05）。
- 统一规则掩码：所有方法评测使用同一地图 API 生成的规则掩码（避免方法间标注不一致）。
- 可复现：公开规则标注生成器、config、权重。

## 8. 算力与资源计划

- 训练：150k step × 4 卡 ≈ 5–6 GPU·天。
- 规则伪标注生成：700 场景 × LLM 调用（每场景 ~8k token）≈ 560 万 token ≈ $8–15；体素化 CPU 并行 ≈ 0.5 天。
- 评测：nuScenes-Occupancy 全 val 推理 ≈ 1 GPU·天。
- 合计 ≈ 8 GPU·天；存储 ≈ 400GB（占据标注 + 规则体素 + 权重）。

## 9. 里程碑与时间线（8 周）

| 周 | 交付物 |
|---|---|
| W1 | SPSC 复现跑通（mIoU 达参考值） |
| W2 | 规则伪标注引擎 v1 + 抽检审计 |
| W3 | 规则 token + 双通道 head 代码 |
| W4 | 训练跑通，主实验 A 初版 |
| W5 | 消融 B/C/D |
| W6 | 实验 E/F + 跨域迁移 |
| W7 | 实验 G/H（可微性 + 规划消费演示） |
| W8 | 论文初稿 + 开源（规则标注生成器 + 权重） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 规则作用域自动标注噪声大 | 中 | 高 | 抽检 10% + 一致性损失软约束（容差 η）；不一致区域置 ignore |
| 规则信号太弱、增益不明显 | 中 | 中 | 退化为「可插拔后处理模块」仍可发（规则通道作为附加模块），主标题改为模块化方案 |
| SPSC 无官方代码、复现不稳 | 中 | 中 | 用 SparseOcc（arXiv 2404.09502）官方实现作骨干，规则部分不变 |
| 语义 mIoU 被规则通道拉低 | 低 | 中 | 两阶段训练（先语义后规则）+ mIoU 门控早停 |

## 11. 论文写作计划

- 目标会议：ICRA 2027（9 月截稿）或 ICLR 2027；备选 ITSC 2026。
- 差异化卖点一句话：把交通规则做成稀疏占据表示的「可微安全边界」，占据从此回答「能不能进」。
- 拟用图表：Fig1 规则 token 注入框架；Fig2 双通道占据可视化（物理/法理对比）；Fig3 违规占用率热图（路口）；Fig4 消融条形图；Table1 基线总表；Table2 规则类别覆盖；Table3 跨域迁移；Table4 规划约束演示。
- 相关工作覆盖：占据预测（SPSC、OccFormer DOI 10.1109/iccv51070.2023.00865、TPV DOI 10.1109/cvpr52729.2023.00890、OpenOccupancy DOI 10.1109/iccv51070.2023.01636、RenderOcc DOI 10.1109/icra57147.2024.10611537、SparseOcc arXiv 2404.09502、Fully Sparse arXiv 2312.17118）；规则/地图（PAMR 收藏论文、OccWorld arXiv 2311.16038、LanguageMPC arXiv 2311.10813）；场景补全（HD²-SSC 收藏论文）。

## 12. 参考文献

- Computer Vision III · 论文 38 · SPSC，AAAI 2026，DOI 10.1609/aaai.v40i6.42441
- Computer Vision VI · 论文 6 · PAMR，AAAI 2026（收藏论文）
- Computer Vision XI · 论文 55 · HD²-SSC，AAAI 2026（收藏论文）
- SparseOcc，arXiv:2404.09502
- Fully Sparse 3D Occupancy，arXiv:2312.17118
- OccFormer，ICCV 2023，DOI 10.1109/iccv51070.2023.00865
- TPV，CVPR 2023，DOI 10.1109/cvpr52729.2023.00890
- OpenOccupancy，ICCV 2023，DOI 10.1109/iccv51070.2023.01636
- RenderOcc，ICRA 2024，DOI 10.1109/icra57147.2024.10611537
- OccWorld，arXiv:2311.16038
- LanguageMPC，arXiv:2311.10813
