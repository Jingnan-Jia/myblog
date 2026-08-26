# 实验设计书：可学习的自适应关键检索（Learnable Adaptive Pivot Retrieval, LAPR）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：中高 ｜ 目标会议：ICCV 2027

## 0. 摘要

APVR（Computer Vision III·论文3）证明"智能检索是参数规模的替代方案"，但其打分权重 λ、迭代次数 P、步长、K 等全部为手工超参，冻结的 CLIP/Grounding-DINO 与目标 MLLM 注意力分布不一致，且无训练信号回流。本工作提出 **LAPR（可学习自适应关键检索）**：把 APVR 的帧级（PFR）+ Token 级（PTR）双粒度检索统一参数化为一个轻量可学习选择器（基于 2B Qwen2.5-VL 蒸馏的 scorer），再用 GRPO 以"下游 QA 正确率 + 检索证据命中（GT 区间 IoU）− 预算惩罚"为奖励端到端优化"检索动作+作答"。检索动作空间 = {每视频帧数, 每帧 token 保留率}。预期贡献：(1) 检索器从启发式到可学习的范式升级；(2) 检索-推理目标一致性的可训练建模；(3) 预算-准确率权衡可平滑调优。

## 1. 研究背景与动机

### 1.1 问题定义

给定长视频 V 与查询 q，目标为最大化下游 QA 正确率的同时最小化检索预算（帧数与 token 数）。检索器需输出策略 π_retr（每视频帧数 n_v、每帧 token 保留率 ρ_f），使进入 MLLM 的视觉信息"充分且精简"。这是典型的序贯决策问题，可用 RL 端到端优化。

### 1.2 相关工作不足

- **训练无关启发式**：APVR（Computer Vision III·论文3）用 CLIP+Grounding-DINO 加权打分（λ=0.5）、时间扩散、迭代 P 轮，全部手工设定，检索信号与推理器目标脱节；AKS（arXiv:2502.21271）、QuoTA（arXiv:2503.08689）、AdaRETAKE（arXiv:2503.12559）同为启发式。
- **熵驱动不可学习**：AdaptToken（arXiv:2603.28696）用熵做全局 token 预算但无学习；FlexSelect（arXiv:2506.00993）用 rank 监督选择器但面向文本；StreamForest（arXiv:2509.24871）证明预算控制本身重要。
- **可学习+RL 已有先例但未双粒度统一**：FrameThinker（arXiv:2509.24304）多轮帧聚焦 SFT+RL；TimeSearch-R（arXiv:2511.05489）交错文本-视频思考+GRPO，均证明"可学习检索+RL"有效，但未在帧+Token 双粒度统一框架内。
- **空白**：端到端联合优化"检索动作（帧数+token 保留率）+ 作答"的方法。

### 1.3 为什么是现在、为什么你的环境

- **时机**：RLVR 浪潮（Video-R1 arXiv:2503.21776、TimeSearch-R）证明"可验证奖励 + RL"成熟；把 APVR 升级为可学习范式是与 2026 前沿直接对齐的差异化点。
- **环境契合**：2B scorer 蒸馏 1-2 GPU·天 + 7B LoRA GRPO 6-8 GPU·天，总计 ≈10 GPU·天，4×L40 轻松覆盖。

## 2. 研究目标与可验证假设

- **H1（可学习优于启发式）**：LAPR 检索器在同等预算下超过 APVR 手工超参。
  *成立时的可观测结果*：LongVideoBench/VideoMME/MLVU accuracy 提升 ≥1.5 点或同精度下 token 减少 ≥15%。
- **H2（双粒度联合增益）**：帧级+Token 级联合优化优于任一单级。
  *成立时的可观测结果*：联合版 ≥ 单帧级 / 单 token 级（消融表）。
- **H3（RL 奖励信号有效）**：检索奖励（GT IoU）与答案奖励组合提升收敛质量。
  *成立时的可观测结果*：加入检索奖励后训练曲线更稳、最终 accuracy 更高。
- **H4（预算可调）**：预算惩罚 β 可平滑插值出 Pareto 曲线。
  *成立时的可观测结果*：β 扫描下"预算-准确率"曲线单调可解释。

## 3. 总体方法设计

### 3.1 数据流水线

1. **语料**：LongVideoBench / VideoMME / MLVU / LVBench（arXiv:2406.08035）/ Haystack-LVBench 训练拆分（许可核对）。
2. **蒸馏数据生成（DeepSeek V4 Flash）**：对每条 QA 用 APVR 开源实现离线跑出"关键帧集合+关键 token 集"作监督伪标签；另用 DeepSeek 生成查询展开（4 类语义信息）作 scorer 输入。
3. **GT 区间伪标签**：训练集上可用区间标注（如 LongVideoBench 部分）直接使用；无标注的用 APVR 检索结果 + 人工抽检 3% 校准。
4. **过滤**：JSONL 可解析率 >90%；保留难度分级标签（简单/中/难）供课程学习。预期 **~15K 训练样本**。

### 3.2 模型/算法设计

- **2B Scorer**：Qwen2.5-VL-2B（LoRA rank 32），输入 (q, 帧缩略图块)，输出每帧相关性分 s_f；蒸馏目标 = APVR 打分（MSE）+ 检索成功率（BCE）。训练 1-2 GPU·天。
- **7B 主模型**：Qwen2.5-VL-7B（LoRA rank 64）。
- **检索动作空间**：`a = {n_v ∈ [4,128], ρ ∈ [0.1,1.0]×T}`；由 scorer 聚合产生。
- **GRPO**：策略输出"检索动作 + 答案"。奖励：
  `r = 1[answer correct] + λ1·tIoU(retrieved, GT 区间) − β·(n_v/128 + mean(1−ρ))`
  λ1 课程 0→1，β 扫描 {0.05,0.1,0.2,0.4}。
- **检索难度课程**：先易后难的 query 类型（语义匹配→状态变化→因果多跳）。
- **超参初值**：lr 1e-6（RL），8 rollout/组，clip ε=0.2，KL 0.05。

### 3.3 训练流程
- Stage 1：scorer 蒸馏（2 卡，1-2 天）；Stage 2：冷启动（固定 scorer，先训主模型作答，1 天）；Stage 3：GRPO 联合优化（2 卡训练+2 卡 rollout，6-8 天）。
- 优化器 AdamW；ZeRO-3；warmup 3%。

### 3.4 推理与评测流程
- 推理：scorer 选帧 → 主模型按 ρ 保留 token → 作答；temperature=0。
- 评测：各基准官方协议；记录 token 预算、延迟、检索 precision/recall。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| LongVideoBench（arXiv:2407.15754）| 训练/评测 | 公开 |
| VideoMME（arXiv:2405.21075）| 训练/评测 | 公开 |
| MLVU（arXiv:2406.04264）| 评测 | 公开 |
| LVBench（arXiv:2406.08035）| 评测 | 公开 |
| Haystack-LVBench | 评测（长距离）| 公开 |

### 4.2 划分与数量
- 训练 15K；验证 2K；评测官方测试集。

### 4.3 预处理与格式
- 帧 1fps、224×224；GT 区间归一化 [0,1]；
- JSONL：`{video, question, q_expansion, apvr_labels, gt_interval, difficulty}`。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| APVR | Computer Vision III·论文3（arXiv:2506.04953）| github.com/GaoHong-V/APVR（官方）|
| AKS | arXiv:2502.21271 | 论文公开则复现 |
| QuoTA | arXiv:2503.08689 | 论文公开则复现 |
| AdaptToken | arXiv:2603.28696 | 论文公开则复现 |
| TimeSearch-R | arXiv:2511.05489 | github.com/LuckyWang-Athena/TimeSearch-R |
| FrameThinker | arXiv:2509.24304 | 按论文复现 |

### 5.2 复现步骤与预期指标表
统一 64 帧基准、temperature=0。预期主表（accuracy / token 预算）：

| 方法 | LongVideoBench | VideoMME | MLVU | 平均 token/问 |
|---|---|---|---|---|
| Qwen2.5-VL-7B（64帧）| 基准 | 基准 | 基准 | 全量 |
| APVR | +9.5（官方）| +4.6 | +9.7 | 高 |
| TimeSearch-R | LongVideoBench SOTA | — | — | 高 |
| **LAPR-7B** | ≥APVR | ≥APVR | ≥APVR | ≤APVR×0.85 |

### 5.3 统一评测口径
所有方法同 prompt、同解析器；token 预算统一以"视觉 token 数"计量；报告各方法 GPU 延迟。

## 6. 实验矩阵

- **A（主实验）**：完整 LAPR（scorer 蒸馏 + GRPO 双粒度）。
- **B1（检索器消融）**：APVR 启发式 / 纯蒸馏 scorer（无 RL）/ scorer+RL。
- **B2（粒度消融）**：仅帧级 / 仅 token 级 / 双粒度。
- **B3（奖励消融）**：仅答案 / 答案+GT IoU / +预算惩罚（β 扫描）。
- **B4（课程消融）**：随机顺序 / 难度课程 / 反序。
- **C（鲁棒性）**：视频长度分级、噪声帧注入、基座替换（Qwen2-VL）。
- **D（泛化性）**：未训练基准（LVBench、Haystack-LVBench）；2B scorer 规模 {2B,7B scorer}。

## 7. 评测协议

- 指标：accuracy、检索 precision/recall（GT 框）、token 预算、延迟、预算-准确率 AUC。
- 3 种子 mean±std；bootstrap p<0.05；报告 β 扫描表。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：scorer 蒸馏 2 + 冷启动 1 + GRPO 7 + 评测 2 = **≈10 GPU·天**。
- 存储：模型 + 视频缓存 300GB。
- API：DeepSeek V4 Flash 查询展开+蒸馏数据 ≈ 500 万 token；Kimi K2.6 检索质量评测 ≈ 100 万 token；成本 ≈ **$150-350**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | 复现 APVR（含依赖 CLIP/Grounding-DINO）并产出蒸馏标签 |
| 2 | scorer 训练 + 蒸馏质量评估 |
| 3 | 冷启动 + GRPO 联合训练 v0 |
| 4 | 奖励/课程消融 + β 扫描 |
| 5 | 鲁棒性/泛化实验 |
| 6 | 论文初稿 |
| 7 | 图表+投稿 ICCV 2027（deadline ~2027-03）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| RL 动作空间大、奖励噪声高 | 中高 | 高 | 先蒸馏降方差；scorer 冷启动；检索奖励用 GT 框打折；8 rollout 组 |
| APVR 蒸馏标签质量受其启发式限制 | 中 | 中 | 蒸馏后继续 RL 纠偏；GT 区间伪标签兜底 |
| 双粒度联合收敛慢 | 中 | 中 | 分阶段：先帧级 RL 稳定，再开 token 级 |
| 预算惩罚 β 敏感 | 中 | 中 | 以 AUC 为主指标，多 β 点报告 |

## 11. 论文写作计划

- **目标会议/截稿**：ICCV 2027（~2027-03 截稿，提前 8 周定稿）。
- **差异化卖点**：把 APVR 升级为可学习双粒度检索；端到端"检索+作答"联合 RL；预算-准确率 Pareto 可控。
- **图表清单**：Fig.1 框架；Fig.2 检索器动作可视化；Fig.3 预算-准确率曲线；Fig.4 案例；Tab.1 主表；Tab.2 消融；Tab.3 鲁棒性/泛化。
- **相关工作覆盖**：训练无关检索（APVR/AKS/QuoTA/AdaptToken）、RLVR（TimeSearch-R/FrameThinker/Video-R1）、可学习压缩（XComp/FlexSelect）。

## 12. 参考文献（真实核验）

- Computer Vision III·论文3·APVR（arXiv:2506.04953）
- AKS: arXiv:2502.21271
- QuoTA: arXiv:2503.08689
- AdaRETAKE: arXiv:2503.12559
- AdaptToken: arXiv:2603.28696
- FlexSelect: arXiv:2506.00993
- StreamForest: arXiv:2509.24871
- FrameThinker: arXiv:2509.24304
- TimeSearch-R: arXiv:2511.05489
- Video-R1: arXiv:2503.21776
- XComp: arXiv:2604.14149
- LongVideoBench: arXiv:2407.15754
- VideoMME: arXiv:2405.21075
- MLVU: arXiv:2406.04264
- LVBench: arXiv:2406.08035
