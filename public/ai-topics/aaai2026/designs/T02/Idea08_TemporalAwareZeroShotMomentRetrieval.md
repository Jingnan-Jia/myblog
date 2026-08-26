# 实验设计书：时间感知的零样本视频时刻检索（Temporal-Aware Zero-shot Video Moment Retrieval）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：中 ｜ 目标会议：ACMMM 2026 / ICME 2027（重排器强可冲 CVPR 2027）

## 0. 摘要

GranAlign（Computer Vision IV·论文32）用"逐帧余弦 + NMS 后处理"解决零样本视频时刻检索（ZVMR）的粒度错配，但逐帧独立打分破坏时间结构、无运动/物体轨迹建模、query-aware caption 存在幻觉、语义相似度受 SentenceTransformer 上限约束。本工作提出**时间感知 ZVMR**：(1) **时间一致性打分**——帧级相似度沿时间维做图扩散/CRF 平滑，约束连续性；(2) **运动/轨迹条件化**——用 Grounding-DINO 框轨迹与光流引导打分窗口，在动作出现/消失处局部加严；(3) **连续粒度加权**——按查询类型动态混合 simplified/detailed 路径权重。主框架训练无关，可选 2B 轻量重排器微调提升上限。预期贡献：(1) 时间结构感知的 ZVMR 打分范式；(2) 状态变化型查询（"先 A 后 B"）的检索能力补齐；(3) 训练无关 + 轻量重排器双轨方案。

## 1. 研究背景与动机

### 1.1 问题定义

给定未修剪视频 V 与自然语言查询 q（含状态变化型如"先把牛奶倒进杯子，再把杯子放进冰箱"），ZVMR 需定位 [s*, e*]。挑战：(a) 语义粒度错配（GranAlign 已诊断）；(b) 时间结构（状态转移、连续性）被逐帧独立打分丢弃；(c) 动作边界（物体出现/消失）需要运动与轨迹线索。

### 1.2 相关工作不足

- **GranAlign 结构性缺陷**（Computer Vision IV·论文32）：Sf 逐帧余弦、靠 MPG/NMS 后处理补救；无运动/轨迹；query-aware caption 幻觉靠双路径平均对冲；粒度仅两档。
- **其他 ZVMR**：Moment-GPT（arXiv:2404.03171）LLM 重写+MLLM 打分同样逐帧；Chrono（arXiv:2406.18113）时间蓝图单视频；LLaVA-MR（arXiv:2411.14505）MLLM 定位未做时间一致性；GroundVTS（arXiv:2604.02093）是训练式查询引导 token 采样，非零样本。
- **空白**：零样本框架内的时间一致性 + 运动/轨迹条件化 + 连续粒度加权。

### 1.3 为什么是现在、为什么你的环境

- **时机**：ZVMR 仍是 2026 活跃方向（GroundVTS/LeAdQA（arXiv:2507.14784））；训练无关方案的效率优势与新增时间感知正交。
- **环境契合**：主框架训练无关（API 生成 caption，4 卡只跑推理）；可选 2B 重排器微调 2-4 GPU·天，预算极低。

## 2. 研究目标与可验证假设

- **H1（时间一致性提升）**：时间图扩散/CRF 平滑显著提升状态变化型查询的检索精度。
  *成立时的可观测结果*：状态变化类 query 子集 R1@0.5 提升 ≥3 点。
- **H2（运动条件增益）**：Grounding-DINO 轨迹与光流条件化提升动作边界定位（R1@0.5/mIoU）。
  *成立时的可观测结果*：QVHighlights R1@0.5 提升 ≥1.5 点。
- **H3（连续粒度加权）**：按查询类型动态混合 simplified/detailed 权重优于固定双路径平均。
  *成立时的可观测结果*：粒度加权消融版 mAP@avg 最高。
- **H4（重排器兜底）**：2B 重排器在训练无关框架之上进一步显著提升。
  *成立时的可观测结果*：重排后 mAP@avg 再提升 ≥1 点。

## 3. 总体方法设计

### 3.1 数据流水线

1. **查询重写（DeepSeek V4 Flash）**：每查询重写为 simplified（保留核心实体动作）与 detailed（保留全部细节）多句变体；并按查询类型分类（语义匹配/状态变化/相对位置/动作序列），输出类型标签。
2. **caption 生成（DeepSeek V4 Flash）**：对 top-K% 候选帧生成 query-agnostic caption（全帧）与 query-aware caption（注入查询意图）；Kimi K2.6 抽查 caption 幻觉（与帧内容矛盾者剔除或降权）。
3. **运动/轨迹线索（本地，无需训练）**：Grounding-DINO 检测关键名词实体 → 帧间 IoU 匹配成轨迹；光流（RAFT 或 TV-L1）逐帧运动幅值。
4. **成本**：帧描述生成量受 K 控制（K=20-30% 候选帧），总 API 成本显著低于全帧描述。

### 3.2 模型/算法设计

- **粒度配对打分**（沿用 GranAlign 配对逻辑）：simplified↔agnostic、detailed↔aware，SentenceTransformer 余弦。
- **时间一致性打分**：帧级相似度序列 S_f 构造时间图 G=(帧节点, 时间邻接边, 状态转移边)，图扩散：`S' = α·D^{-1}A D^{-1} S + (1-α)·S`；或 CRF 平滑（一阶链式势能，对相邻帧约束）。α、CRF 权重在验证集校准。
- **运动/轨迹条件化**：动作出现/消失处（轨迹起点/终点、光流幅值跳变）对打分窗口加严（乘局部权重 w_t，如 w_t=1+0.5·flow_grad）；实体轨迹与 query 关键词匹配的帧加权。
- **连续粒度加权**：`S_total = w(q)·S_simple + (1-w(q))·S_detailed`，w(q) 由查询类型分类器输出（语义匹配→更偏 simplified 高覆盖；状态变化/动作序列→更偏 detailed 高精度）。
- **Moment Proposal Generator + NMS + 后排序**：沿用 GranAlign 流程。
- **可选 2B 重排器**：Qwen2.5-VL-2B（LoRA rank 32）对候选 moment 做二分类排序（正=命中），训练数据 = 各 ZVMR 训练集正负样本（如 QVHighlights train）。

### 3.3 训练流程
- 主框架：无训练（仅验证集超参校准）。
- 2B 重排器：LoRA，lr 1e-5，batch 32，3 epoch，2-4 GPU·天。

### 3.4 推理与评测流程
- 推理：打分 → 时间平滑 → 运动加权 → MPG → NMS → （可选）重排；temperature=0。
- 评测：QVHighlights（R1@0.5/0.7、mAP@0.5、mAP@avg）、Charades-STA、ActivityNet-Captions、Moment-XL 长视频子集。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| QVHighlights | 训练（重排器）/评测 | 公开（research）|
| Charades-STA | 评测/训练 | 公开 |
| ActivityNet-Captions | 评测 | 公开 |
| Moment-XL 长视频子集 | 评测（长视频）| 公开 |
| GranAlign（arXiv:2601.00584）| 对照 | 官方代码/描述 |

### 4.2 划分与数量
- 主框架：零样本（不训练）；重排器：QVHighlights train（~7K 正负样本），测试官方 split。

### 4.3 预处理与格式
- 帧 1fps、224×224；caption JSONL；轨迹/光流离线预计算缓存。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| GranAlign | Computer Vision IV·论文32（arXiv:2601.00584）| 官方开源则复现 |
| Moment-GPT | arXiv:2404.03171 | 官方开源 |
| Chrono | arXiv:2406.18113 | 官方开源则复现 |
| GroundVTS | arXiv:2604.02093 | 官方开源则复现 |

### 5.2 复现步骤与预期指标表
预期主表（QVHighlights）：

| 方法 | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@avg |
|---|---|---|---|---|
| GranAlign | SOTA（mAP@avg 提升 3.23%）| — | — | SOTA |
| Moment-GPT | 官方值 | — | — | 官方值 |
| **Ours（时间感知）** | ≥GranAlign+1.5 | ≥GranAlign+1 | ≥GranAlign+1 | ≥GranAlign+1 |
| **Ours+重排器** | ≥Ours+1 | ≥Ours+1 | ≥Ours+1 | ≥Ours+1 |

### 5.3 统一评测口径
同查询重写模型、同 NMS 参数（若基线开放则用其参数）；报告 GPU 延迟与 API 成本。

## 6. 实验矩阵

- **A（主实验）**：时间感知主框架（GranAlign 升级 + 时间图 + 运动条件 + 粒度加权）。
- **B1（时间图消融）**：无平滑 / 图扩散 / CRF / 扩散+CRF。
- **B2（运动条件消融）**：无 / 轨迹加权 / 光流加权 / 两者。
- **B3（粒度消融）**：固定双路径平均 / 类型加权 / 三档粒度。
- **B4（重排器消融）**：无重排 / 2B 重排 / 重排+集成。
- **C（鲁棒性）**：查询长度、K 候选比例、长视频（Moment-XL）、噪声帧。
- **D（泛化性）**：跨数据集（Charades-STA/ActivityNet）零样本迁移。

## 7. 评测协议

- 指标：R1@0.5/0.7、mAP@0.5、mAP@avg；状态变化型子集专项报告；GPU 延迟、API 成本。
- 3 种子 mean±std；bootstrap p<0.05。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：主框架推理评测 1-2 + 2B 重排器 2-4 = **≈3-5 GPU·天**。
- 存储：caption/轨迹/光流缓存 200GB。
- API：DeepSeek V4 Flash 查询重写 + query-aware caption ≈ 600 万 token（远低于全帧描述）；Kimi K2.6 幻觉抽查 ≈ 100 万 token；成本 ≈ **$120-300**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | GranAlign 复现 + 基准跑通 |
| 2 | 时间图扩散/CRF + 运动轨迹条件实现 |
| 3 | 粒度加权 + 验证集超参校准 |
| 4 | 重排器数据准备 + 微调 |
| 5 | 消融 + 泛化实验 |
| 6 | 论文初稿 |
| 7 | 投稿 ACMMM 2026 或 ICME 2027（若重排器增益显著改投 CVPR 2027）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 纯打分方法存在天花板 | 中 | 中 | 用 2B 重排器对冲；必要时升级为轻量训练式打分 |
| 时间图扩散参数敏感 | 中 | 中 | 验证集网格校准；CRF 权重约束 |
| query-aware caption 幻觉 | 中 | 中 | Kimi K2.6 抽查 + 双路径对冲（沿用 GranAlign 思路）|
| 运动条件在低动作视频上无效 | 中 | 低 | 光流幅值门控（无动作时不惩罚）；消融确认 |

## 11. 论文写作计划

- **目标会议/截稿**：ACMMM 2026（若 deadline 允许）/ ICME 2027；重排器增益显著可冲 CVPR 2027。
- **差异化卖点**：时间结构感知的零样本打分；状态变化型查询专项；运动/轨迹条件化；连续粒度加权。
- **图表清单**：Fig.1 框架（时间图+运动条件+粒度加权）；Fig.2 状态变化查询案例；Fig.3 时间平滑前后对比；Tab.1 主表；Tab.2 消融；Tab.3 泛化/鲁棒。
- **相关工作覆盖**：ZVMR（GranAlign/Moment-GPT/Chrono/GroundVTS）、时间结构建模、视频检索（AMD-Net（Computer Vision VII·论文31））。

## 12. 参考文献（真实核验）

- Computer Vision IV·论文32·GranAlign（arXiv:2601.00584）
- Moment-GPT: arXiv:2404.03171
- Chrono: arXiv:2406.18113
- GroundVTS: arXiv:2604.02093
- LLaVA-MR: arXiv:2411.14505
- LeAdQA: arXiv:2507.14784
- Computer Vision VII·论文31·AMD-Net（DOI: 10.1609/aaai.v40i10.37745）
