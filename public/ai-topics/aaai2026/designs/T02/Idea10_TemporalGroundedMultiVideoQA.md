# 实验设计书：时间接地多视频 QA：基准与合成-验证流水线（Temporal-Grounded Multi-Video QA Benchmark & Synthetic Pipeline）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：中 ｜ 目标会议：NeurIPS 2026 Datasets & Benchmarks / CVPR 2027

## 0. 摘要

CrossVid（Computer Vision V·论文38）只有答案正确率、无证据区间标注，无法诊断"答对但证据错"；其合成流水线只依赖 DeepSeek-R1 单 teacher、无独立验证环节。本工作构建**时间接地多视频 QA 基准**（~5K QA）：每题附带支撑答案的**证据区间标注**（视频+时间片段），并提出"合成 → 双 LLM 交叉验证 → 专家抽检"闭环流水线，过滤率可量化；新指标 **Evidence-Grounded Accuracy (EGA)** = 答案正确 ∧ 证据区间与 GT 的 tIoU ≥ 阈值。预期贡献：(1) 首个带证据区间标注的多视频 QA 基准；(2) 可量化的合成-验证流水线（过滤率/一致性统计）；(3) EGA 指标把 CVR 评测从"只对答案"升级为"对答案+对证据"。

## 1. 研究背景与动机

### 1.1 问题定义

跨视频推理（CVR）评测目前只有答案正确率，无法区分"模型靠正确证据作答"与"碰巧答对"。需要：(a) 每题标注支撑答案的证据区间（视频 ID + [start,end]）；(b) 一个可扩展、质量可控的合成标注流水线；(c) 一个把答案正确性与证据正确性结合的评测指标（EGA）。

### 1.2 相关工作不足

- **CrossVid**（Computer Vision V·论文38）：无证据接地标注；合成依赖 DeepSeek-R1 单 teacher，专家仅过滤/精修，无独立验证环节；无法量化过滤率。
- **单视频证据基准**：TimeSearch/Haystack-LVBench 只覆盖单视频长距离检索；LVSQA/SLFG（arXiv:2508.03009）是场景级长视频 QA 但非多视频。
- **证据接地训练有、评测无**：Conan（arXiv:2510.20470）、SER（arXiv:2606.24726）聚焦证据奖励方法，无配套多视频证据区间基准。
- **空白**：带证据区间标注的多视频 QA 基准 + 合成-验证闭环流水线 + EGA 指标。

### 1.3 为什么是现在、为什么你的环境

- **时机**：CVR 方法（Idea 2/5）亟需可诊断的评测平台；EGA 是 2026 证据接地浪潮的评测自然延伸。
- **环境契合**：数据合成与验证离线（API）；22 个开源模型评测 on 4×L40 ≈ 2-3 GPU·天，成本极低。

## 2. 研究目标与可验证假设

- **H1（流水线质量）**：双 LLM 交叉验证显著提高标注质量、可量化过滤率。
  *成立时的可观测结果*：过滤率 20-40%；双 LLM 一致性 ≥0.85；专家抽检一致率 ≥0.9。
- **H2（EGA 区分度）**：EGA 比纯答案准确率更严格、能揭示"碰巧答对"。
  *成立时的可观测结果*：主流模型 EGA < accuracy（gap ≥5 点）；部分高 accuracy 低 EGA 案例可见。
- **H3（证据区间可靠性）**：证据区间标注的标注者间一致性（IoU）达可接受水平。
  *成立时的可观测结果*：双标注者 tIoU ≥0.7（中位）。
- **H4（下游可用性）**：新基准可诊断/驱动 Idea 2、5 的方法改进。
  *成立时的可观测结果*：在基准上复评，证据奖励方法相对答案奖励方法 EGA 提升显著。

## 3. 总体方法设计

### 3.1 数据流水线

1. **视频组构造**：从 YouCook2 / Assembly101 / VisDrone / Animal Kingdom 建组（同食谱、多视角、同一场景多时刻、动物行为组），每组 2-6 视频。来源均公开。
2. **QA + 证据区间合成（DeepSeek V4 Pro）**：任务定制 prompt 生成 4 类问题（比较/时间/多视角/自由问答），输出 `{question, gold_answer, evidence:[{video_id, start, end, reason}]}`。
3. **双 LLM 交叉验证**：Kimi K2.6 与 DeepSeek V4 Pro（不同温度/不同 prompt）独立评判每题的"答案正确性 + 证据充分性 + 区间合理性"；两者一致才保留，否则进入复审或丢弃；记录过滤率。
4. **专家抽检**：~10%（500 条）由 2 名标注者按区间标注协议二次标注，计算 tIoU 一致性与答案一致率。
5. **FSA 类防捷径**：对时间任务做时间重排（±1-5s 偏移）验证答案稳定性。

### 3.2 模型/算法设计（流水线本身）

- **区间标注协议**：证据区间 = 支撑答案的连续时间片段；允许 1-3 段（列表）。
- **EGA 计算**：`EGA = P(答案正确 ∧ max_j tIoU(pred_j, GT_j) ≥ θ)`，θ=0.5（报告 θ∈{0.3,0.5,0.7} 灵敏度）。
- **流水线质量指标**：过滤率、双 LLM 一致性（Cohen's κ）、专家抽检一致率、区间 tIoU 分布。
- **评测脚本**：统一 prompt 模板、帧采样、证据解析器（要求模型输出 `<evidence>` 段）。

### 3.3 训练流程
- 无模型训练（基准+流水线工作）；仅评测。

### 3.4 推理与评测流程
- 22 个开源模型（Qwen2.5-VL-7B/72B、InternVL3-8B、VideoLLaMA3、MiniCPM-V 等）on 4×L40 并行评测，temperature=0；输出答案+证据区间。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| YouCook2 | 视频组来源 | 公开（research）|
| Assembly101 | 视频组来源 | 公开（research）|
| VisDrone | 视频组来源 | 公开 |
| Animal Kingdom | 视频组来源 | 公开 |
| CrossVid（arXiv:2511.12263）| 复评 | 公开 |
| All-Angles Bench（arXiv:2504.15280）| 复评 | 公开 |

### 4.2 划分与数量
- 新基准：~5K QA（train/dev/test 60/15/25 公开）。
- 复评：CrossVid 9015 QA、All-Angles 测试。

### 4.3 预处理与格式
- 帧 1fps、224×224；JSONL：`{group_id, videos, question, gold_answer, evidence:[{video_id,s,e}], difficulty, task_type}`。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| Qwen2.5-VL-7B/72B | 官方 | github.com/QwenLM/Qwen2.5-VL |
| InternVL3-8B | 官方 | github.com/OpenGVLab/InternVL |
| VideoLLaMA3 | 官方 | github.com/DAMO-NLP-SG/VideoLLaMA3 |
| MiniCPM-V | 官方 | github.com/OpenBMB/MiniCPM-V |
| CrossVid 22 模型结果 | Computer Vision V·论文38 | 官方脚本 |

### 5.2 复现步骤与预期指标表
预期主表（accuracy / EGA）：

| 方法 | accuracy | EGA(θ=0.5) | accuracy−EGA |
|---|---|---|---|
| Qwen2.5-VL-7B | 中 | 低 | 大 |
| Qwen2.5-VL-72B | 中高 | 中 | 中 |
| InternVL3-8B | 中 | 低中 | 大 |
| Gemini-2.5-Pro（官方 CrossVid）| 50.4 | — | — |

### 5.3 统一评测口径
所有模型同 prompt、同帧采样、同证据解析器；EGA 阈值灵敏度报告。

## 6. 实验矩阵

- **A（主实验）**：新基准发布 + 22 模型 EGA/accuracy 全评测。
- **B1（流水线消融）**：单 teacher（仅 DeepSeek）/ 双 LLM 交叉 / +专家抽检。
- **B2（过滤消融）**：无过滤 / 过滤率 20% / 40% 对评测集质量影响。
- **B3（EGA 消融）**：θ ∈ {0.3,0.5,0.7}、证据段数 {1,3}。
- **B4（标注一致性）**：双标注者 tIoU / 一致性 κ。
- **C（鲁棒性）**：跨域（水下/第一人称子集）、长视频组、组内视频数。
- **D（用例）**：在基准上复评 Idea 2/5 的模型，验证 EGA 的诊断能力。

## 7. 评测协议

- 指标：accuracy、EGA（多阈值）、过滤率、双 LLM 一致性（κ）、专家一致率、标注 tIoU 分布、Per-task 分解。
- 报告模型全榜 + 数据集卡（datasheet）+ 标注者间信度。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：22 模型评测 2-3 = **≈3-5 GPU·天**（含复评）。
- 存储：视频缓存 600GB、标注 JSONL 20GB。
- API：DeepSeek V4 Pro 合成 QA+证据 ≈ 900 万 token；Kimi K2.6 交叉验证 ≈ 500 万 token；DeepSeek V4 Flash 初筛 ≈ 300 万 token；成本 ≈ **$400-800**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | 视频组构造 + 合成 prompt 调试（小规模 200 条试跑）|
| 2 | 全量 7K 合成 → 双 LLM 交叉验证 → 过滤 |
| 3 | 专家抽检 + 标注协议定稿 + tIoU 一致性分析 |
| 4 | 22 模型评测（2 卡并行）+ EGA 计算 |
| 5 | 复评 CrossVid/All-Angles + 与 Idea 2/5 联动 |
| 6 | 数据集卡 + 论文初稿 |
| 7 | 投稿 NeurIPS 2026 D&B 或 CVPR 2027 |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 证据区间标注质量不稳 | 中高 | 中高 | 区间标注协议 + 双 annotator 一致性测试；θ 灵敏度报告 |
| 合成 QA 有 teacher 偏差 | 中 | 中 | 双 LLM 不同系交叉；专家抽检 10%；难例重写 |
| 跨模型可比性（不同模型对证据格式服从度不同）| 中 | 中 | 统一证据输出 prompt + 解析器 + 格式惩罚；报告"未输出证据"比例 |
| 流水线过滤率过低（过严）| 中 | 低 | 复审通道回收；平衡过滤阈值 |

## 11. 论文写作计划

- **目标会议/截稿**：NeurIPS 2026 Datasets & Benchmarks 或 CVPR 2027。
- **差异化卖点**：首个带证据区间标注的多视频 QA 基准；合成-验证-再合成闭环（过滤率可量化）；EGA 指标（答案+证据联合）。
- **图表清单**：Fig.1 流水线；Fig.2 证据区间标注示例；Fig.3 accuracy vs EGA 对比图；Fig.4 一致性/过滤率统计；Tab.1 基准统计卡；Tab.2 模型全榜；Tab.3 消融；Tab.4 跨域。
- **相关工作覆盖**：CVR 基准（CrossVid/All-Angles）、证据接地（SER/Conan/REVISOR）、长视频评测（VideoMME/LVBench/StreamingBench）。

## 12. 参考文献（真实核验）

- Computer Vision V·论文38·CrossVid（arXiv:2511.12263）
- All-Angles Bench: arXiv:2504.15280
- SER: arXiv:2606.24726
- Conan: arXiv:2510.20470
- REVISOR: arXiv:2511.13026
- SLFG/LVSQA: arXiv:2508.03009
- VideoMME: arXiv:2405.21075
- LVBench: arXiv:2406.08035
- StreamingBench: arXiv:2411.03628
