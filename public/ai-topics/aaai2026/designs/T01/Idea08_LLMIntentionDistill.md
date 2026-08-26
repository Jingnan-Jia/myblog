# 实验设计书：Idea 8 语言推理先验蒸馏的轻量换道预测（Language-Reasoned Intention Priors Distilled for Lane-Change Prediction）

> 主题 T01 自动驾驶感知与端到端驾驶 · 12 个 idea 之一 · 优先级：中高（ICRA 2027 / NeurIPS 2026 workshop）

## 0. 摘要

换道预测的静态规则知识（KnowLCP）缺乏场景级语义推理，纯数据驱动模型跨城市（不同交规/驾驶风格）掉点，而大 LLM 直接推理换道不满足实时性。本工作用 LLM 对「历史轨迹 + 俯视场景描述」做离线换道意图语言推理，将意图先验蒸馏进轻量预测器（轨迹编码器 + 轻量视觉编码器），蒸馏损失 = 意图分布 KL + 轨迹回归 + 运动学可行性正则；LLM 低置信场景自动回退纯几何预测。预期意图 F1 相对 KnowLCP +5~8 点、跨数据集迁移（Argoverse2→HighD）FDE 下降 ≥15%。贡献：(1) 语言推理先验蒸馏范式；(2) 先验不确定性传递的回退机制；(3) 可审计的 API 蒸馏成本协议。

## 1. 研究背景与动机

### 1.1 问题定义

给定他车历史轨迹 $X_{t-K:t}$ 与场景上下文（车道拓扑、邻居、信号灯），预测换道意图 $I \in \{直行, 左换, 右换\}$ 与未来轨迹 $Y_{t+1:t+H}$。要求：在线推理 <20ms，且跨城市（不同交规/驾驶风格）低成本迁移。

### 1.2 相关工作不足

- KnowLCP（Special Track on AI for Social Impact I · 论文 88）注入静态规则知识（风险意识/运动学约束/意图强度），无场景级语言推理；跨城市迁移未验证。
- 主流预测器（HiVT/QCNet 类）纯数据驱动，跨域掉点、无语言语义。
- 大 LLM 直接推理换道不满足实时性与可控性（无回归轨迹输出）。
- 空白：静态知识 → 语言推理 → 可部署预测器的蒸馏链缺失。

### 1.3 为什么是现在、为什么你的环境适合

- 现在是：LLM 推理成本已足够低（DeepSeek/Kimi API），可离线批量蒸馏；Argoverse2/HighD/INTERACTION 公开轨迹数据集成熟。
- 环境适合：在线预测器 <100M，单卡即可训练（2–3 GPU·天）；离线蒸馏只需 API 调用（CPU + 网络），不占 GPU 主要预算。

## 2. 研究目标与可验证假设

- **H1（语言先验 > 静态规则）**：LLM 推理先验蒸馏的意图预测优于 KnowLCP 静态规则注入。
  - 可观测结果：意图 F1 相对 KnowLCP +5~8 点（Argoverse2 测试）；minADE 持平或更低。
- **H2（跨域迁移）**：语言先验使跨数据集迁移显著优于纯数据驱动。
  - 可观测结果：Argoverse2 训 → HighD 测，FDE 下降 ≥15%（相对 HiVT 迁移）；意图 F1 保持率更高。
- **H3（回退机制有效）**：LLM 低置信样本自动回退几何预测，避免错误先验污染。
  - 可观测结果：回退机制使整体 FDE 比「全量用先验」下降 ≥5%；高置信子集收益保持。
- **H4（API 成本可审计）**：蒸馏收益与 API 成本可度量，且「蒸馏 vs 更多数据训练」对照成立。
  - 可观测结果：蒸馏在相同数据量/预算下优于纯数据增广；API 总成本 <$50。

## 3. 总体方法设计

### 3.1 数据流水线

- 数据源：Argoverse 2（https://www.argoverse.org/，Apache-2.0 研究许可）、HighD（https://www.highd-dataset.com/，需申请）、INTERACTION（https://interaction-dataset.com/，CC BY-NC-SA 4.0）。
- 离线蒸馏数据构造：
  1. 每段换道轨迹序列取历史 3s（K=15 @5Hz）+ 未来 5s（H=25）。
  2. 场景描述模板：把车道拓扑（邻居车道存在性、可用性）、他车位置/速度、本车道前方间隙、相邻车道间隙、信号灯（如有）编码成文本（可解释）。
  3. LLM prompt（DeepSeek V4 Pro）：「你是驾驶行为专家。基于以下场景文本，判断该车是否将换道、换到哪条道、概率与原因，并用 1-5 标置信度」。输出 JSON：`{intention, prob_keep, prob_left, prob_right, reason, confidence}`。
  4. 真值标签：用 Argoverse2 车道系标注或轨迹几何（横向偏移 >1.5m 判定换道）生成「几何意图真值」用于对照。
- 过滤：LLM 置信度 <0.4 的样本标注为「低置信 → 回退」，仍保留但标记；抽检 5% 人工审计一致率。
- 数量预期：Argoverse2 训练段 2 万条（其中 60% 含换道事件）、HighD 8k 段（仅迁移测试 + 蒸馏再标签可选）、INTERACTION 4k 段（第二跨域验证）。
- 跨域场景改写：把城市 A 场景翻译成城市 B 交规（如「美国靠右超车 vs 德国靠左超车」）由 LLM 改写文本，增强跨域先验。

### 3.2 模型/算法设计

- 在线预测器（<100M）：
  - 轨迹编码器：Transformer（d_model 128，4 层），输入历史轨迹 + 场景图邻接。
  - 轻量视觉编码器（可选）：俯视 BEV 小图（64×64）用 4 层 CNN，补充空间上下文。
  - 意图头：3 类分布 $p_I$；轨迹头：多模态 6 条高斯轨迹（mean/cov）。
- 蒸馏目标：
  - 意图分布 KL：$L_{kl} = \text{KL}(p_I \| \hat p_I^{LLM})$（LLM 归一化概率作为 soft target）。
  - 轨迹回归：GMM 负对数似然 $L_{nll}$。
  - 运动学可行性正则：$L_{kin} = \|\dot v\|_2^2 + \|\dot \theta\|_2^2$（加/减速与转向率受限）。
  - 总 $L = L_{nll} + \lambda_{kl} L_{kl} + \lambda_{kin} L_{kin}$，$\lambda_{kl}=0.5, \lambda_{kin}=0.1$。
- 回退机制：$\hat p_I = (1-w(\hat c)) \hat p_I^{model} + w(\hat c) \hat p_I^{geom}$，$w(\hat c)$ 随模型对 LLM 先验的置信（蒸馏时同时蒸馏 LLM confidence 到一个 confidence 头）单调：低 LLM 置信 → 回退到几何意图（横向偏移几何规则）。
- 超参初值：GMM 6 模态，H=25，lr 1e-3（AdamW），batch 128。

### 3.3 训练流程

- 阶段 1（监督训练）：意图 + 轨迹 + 蒸馏损失联合训练 20k step，单卡（L40），≈1.5 GPU·天。
- 阶段 2（回退头微调）：冻结主体，训 confidence 头 + 回退权重，5k step，0.5 GPU·天。
- 跨域：Argoverse2 训练 → HighD/INTERACTION 直接测（无微调）与「少样本微调 2k step」两档。
- 并行：离线蒸馏（API）与训练流水线并行；GPU 单卡即可，其余卡给其他 idea。

### 3.4 推理与评测流程

- 推理：轨迹 + 上下文 → 意图分布 + 6 条轨迹 → 回退权重合成意图 → 输出。
- 评测：意图 F1（换道子集单独统计）、minADE/FDE（1/6 秒预测指标）、运动学可行性（曲率/加速度违规率）、端到端时延。
- 跨域评测：A2 训 → HighD 测、A2 训 → INTERACTION 测，报告迁移掉点。

## 4. 数据集细节

- Argoverse 2 motion forecasting：Apache-2.0（研究免费），25 万段训练场景（本工作用子集），含车道图与 agent 轨迹。
- HighD：德国高速公路，需申请（研究许可），11.5h 数据，60+ 万轨迹段。
- INTERACTION：CC BY-NC-SA 4.0，路口/环岛等复杂交互场景，交叉路口换道丰富。
- 划分：A2 训练 2 万段（90/10 内部划分）；HighD 8k 段（全测试）；INTERACTION 4k 段（测试）。验证集与测试集固定。

## 5. 基线复现

| 基线 | 引用 | 官方代码 | 复现要点 |
|---|---|---|---|
| HiVT | 主流轨迹预测基线 | https://github.com/ZikangZhou/HiVT | 官方权重（A2 预训练） |
| QCNet | 主流轨迹预测基线 | https://github.com/ZikangZhou/QCNet | 官方权重（A2） |
| KnowLCP | Special Track on AI for Social Impact I · 论文 88 | 无官方 | 按论文复现风险/运动学/意图强度注入 |
| LLM 直接推理（评测参考） | — | — | DeepSeek/Kimi 直接问答，只作上限参照（含时延） |
| **本方法** | — | 本项目开源 | 语言先验蒸馏 + 回退 |

- 预期指标表（Argoverse2 测试集；跨域为 A2→HighD）：

| 方法 | 意图 F1 | minADE(3s) | FDE(3s) | 跨域 FDE（A2→HighD） | 在线时延 |
|---|---|---|---|---|---|
| HiVT | 0.74 | 0.82 | 1.68 | 2.1 | 5ms |
| QCNet | 0.76 | 0.79 | 1.55 | 1.9 | 8ms |
| KnowLCP | 0.78 | 0.80 | 1.62 | 1.8 | 7ms |
| LLM 直接推理 | 0.83 | — | — | — | >1s |
| **本方法** | **≥0.84** | **≤0.78** | **≤1.50** | **≤1.55** | **<15ms** |

- 统一评测口径：同一测试段集、同一指标脚本（argoverse2 eval API）、同一意图标注定义。

## 6. 实验矩阵

- **A. 主实验**：完整方法 vs 基线。目的：验证 H1。预期：F1 +5~8、跨域 FDE −15%。
- **B. 蒸馏目标消融**：KL 有无、运动学正则有无。预期：KL 贡献意图 F1 +3 点。
- **C. 回退机制 on/off**：全量先验 vs 回退。目的：验证 H3。预期：回退 FDE −5%。
- **D. 低置信过滤**：置信阈值 {0.3, 0.4, 0.5} 过滤训练。预期：阈值提高减少噪声但降覆盖率，0.4 平衡。
- **E. 跨域迁移**：A2→HighD、A2→INTERACTION（无微调/少样本微调）。目的：验证 H2。
- **F. 场景改写增强**：LLM 跨城改写文本 vs 不改写。预期：改写增强跨域 +2~3 点。
- **G. 蒸馏 vs 更多数据**：蒸馏模型 vs 同预算下纯数据增广模型。目的：验证 H4。预期：蒸馏占优。

## 7. 评测协议

- 指标：意图 F1（三分类）、minADE/FDE（3s/6s，argoverse2 官方脚本）、运动学可行性（超加速/超曲率率）、时延。
- 均值±方差：3 seeds；显著性配对 t-test。
- 固定：LLM 推理 temperature=0（确定性蒸馏）、随机种子、测试段集。
- 可复现：公开蒸馏数据集（JSON）、prompt 模板、config/权重；LLM 版本固定（记录 DeepSeek V4 Pro 版本号）。

## 8. 算力与资源计划

- 训练：2 GPU·天（单卡 L40，阶段 1+2）。
- 评测：跨域评测 + 时延 ≈ 0.5 GPU·天。
- API 蒸馏：A2 2 万段 × 2 轮（初标+改写）≈ 4 万次调用 × ~800 token ≈ 3.2 千万 token ≈ $10–20；HighD/INTERACTION 再标签 ≈ $10；合计 <$50。
- 合计 ≈ 2.5 GPU·天（几乎不占 4 卡预算，可与大 idea 并行）。

## 9. 里程碑与时间线（7 周）

| 周 | 交付物 |
|---|---|
| W1 | A2/HighD/INTERACTION 数据下载与预处理 + 轨迹-文本转换 |
| W2 | LLM 蒸馏 v1（2 万段）+ 抽检审计 |
| W3 | 预测器训练跑通（阶段 1） |
| W4 | 回退头（阶段 2）+ 主实验 A |
| W5 | 消融 B/C/D |
| W6 | 实验 E/F/G（跨域 + 场景改写 + 数据对照） |
| W7 | 论文初稿 + 开源（prompt/蒸馏数据/权重） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| LLM 意图标注质量不均（模糊场景） | 中 | 高 | 互信息/置信过滤低置信样本 + 抽检审计；用几何真值做一致性对照 |
| 蒸馏收益被质疑为「更多数据」 | 中 | 中 | 实验 G 直接对照同预算数据增广；报告 API 成本明细 |
| 跨域改写引入错误交规 | 低 | 低 | 改写后人工抽检 10% + 交规文本由权威来源输入 |
| 换道事件在数据中稀疏 | 中 | 中 | 重采样保证换道段 ≥60% 训练占比 |

## 11. 论文写作计划

- 目标会议：ICRA 2027 或 NeurIPS 2026 workshop（数据-蒸馏路线）；备选 ITSC 2026。
- 差异化卖点一句话：把 LLM 的场景级语言推理离线蒸馏成 15ms 的换道预测器，带先验不确定性回退与跨城市迁移。
- 拟用图表：Fig1 蒸馏管线；Fig2 prompt 模板示例 + LLM 推理样例；Fig3 意图混淆矩阵；Fig4 跨域迁移直方图；Fig5 回退权重可视化；Table1 基线总表；Table2 消融；Table3 API 成本表。
- 相关工作覆盖：换道预测（KnowLCP 收藏论文、HiVT/QCNet）；知识注入（LanguageMPC arXiv 2311.10813、Reason2Drive arXiv 2312.03661）；蒸馏与预测（VBD arXiv 2404.02524 行为扩散、Trajectron 类）；LLM 规划推理（DriveVLM arXiv 2402.12289、DriveMLM arXiv 2312.09245）。

## 12. 参考文献

- Special Track on AI for Social Impact I · 论文 88 · KnowLCP，AAAI 2026（收藏论文）
- LanguageMPC，arXiv:2311.10813
- Reason2Drive，arXiv:2312.03661
- Versatile Behavior Diffusion，arXiv:2404.02524
- DriveVLM，arXiv:2402.12289
- DriveMLM，arXiv:2312.09245
- Application Domains I · 论文 21 · Measuring What Matters，AAAI 2026（收藏论文，评测口径呼应）
