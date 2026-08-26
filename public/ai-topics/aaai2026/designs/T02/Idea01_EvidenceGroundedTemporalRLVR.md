# 实验设计书：时间证据可验证的强化对齐（Evidence-Grounded Temporal RLVR for Video LLMs）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：高 ｜ 目标会议：CVPR 2027

## 0. 摘要

现有视频大模型（Video LLM）的时间推理训练要么只依赖"答案是否正确"的奖励（RLVR），要么只依赖"输入是否被扰动"的偏好信号（DPO），均无法保证模型回答建立在**正确的时间证据**之上，导致"答案对但证据错"（right-answer-wrong-evidence）的系统性缺陷。本工作提出**证据接地的时间 RLVR 框架**：把每个问题编译成时间逻辑原语清单（关键对象、动作、时间顺序约束），对模型输出的证据区间 [start,end] 做**符号-语义双轨验证**，得到 tIoU/coverage/groundedness 三类可缩放证据奖励，并与答案奖励组成 `r = r_ans + λ·r_evid` 的双奖励分解。方法与 TEMPLE 式 Pre-SFT DPO 组合，形成"先时间偏好对齐、再证据强化"的两阶段训练。预期贡献：(1) 首个统一"答案正确+证据正确+证据充分"的视频时间推理训练目标；(2) 一套完全可复现的 7B LoRA 训练与评测流水线，在 LongVideoBench / VideoMME / Vinoground / LVBench 上相对强基线显著提升；(3) 公开证据区间标注协议与奖励-准确率相关性分析。

## 1. 研究背景与动机

### 1.1 问题定义

给定长视频 V 与时间推理问题 q（含标准答案 a*），模型需输出 (答案 â, 证据区间 [ŝ, ê])。训练目标是让模型在**答案正确**（â=a*）的同时，其证据区间与真实支撑区间 [s*, e*] 高度重叠（tIoU 高、coverage 高），且证据内容真正"接地"于视频内容而非捷径。该问题可分解为三个可验证子目标：
- **答案正确性**：â 与 a* 的语义等价；
- **证据定位正确性**：区间 [ŝ, ê] 与 [s*, e*] 的时间对齐（tIoU）；
- **证据充分性**：覆盖作答所需的全部关键事件，无遗漏。

### 1.2 相关工作不足

- **偏好对齐流派**：TEMPLE（Computer Vision V·论文53·TEMPLE: Incentivizing Temporal Understanding of Video Large Language Models via Progressive Pre-SFT Alignment）用扰动对比构造 DPO 偏好对，只能教模型"区分 clean 与 corrupted 输入"，无法保证答案依据正确证据；TPO（arXiv:2501.13919）、LLaVA-Hound（arXiv:2404.01258）同为偏好派，均无证据约束。
- **答案级 RLVR 流派**：Video-R1（arXiv:2503.21776）、TimeSearch-R（arXiv:2511.05489）用 GRPO 以答案正确性为奖励，取得强效果，但**没有细粒度时间证据奖励**，模型可走"首帧/末帧捷径"答对。
- **证据奖励流派**：SER（arXiv:2606.24726）引入 referee VLM 语义证据奖励，但依赖几何检查与密集标注，缺乏符号可验证分支；REVISOR（arXiv:2511.13026）用双重归因奖励，聚焦反思而非区间定位。
- **空白**：缺少"答案正确 + 证据正确 + 证据充分"三者统一的、可缩放（scalable）的训练目标。

### 1.3 为什么是现在、为什么你的环境

- **时机**：2025H2–2026 的 RLVR+可验证奖励浪潮（Video-R1、TimeSearch-R、VideoSSR（arXiv:2511.06281）、SER）已证明可验证奖励可行且有效；但证据接地的时间奖励仍是空白，CVPR 2027 是抢占窗口。
- **环境契合**：7B LoRA + GRPO 训练在 4×L40（192GB）上约需 10-14 GPU·天，完全可负担；证据区间标注/时间原语解析全部离线用 DeepSeek V4 Pro 生成、Kimi K2.6 校验，4 卡可并行跑数据合成与训练。

## 2. 研究目标与可验证假设

- **H1（双轨奖励有效性）**：双奖励 `r = r_ans + λ·r_evid` 相比纯答案奖励，在固定预算下显著提升下游时间推理准确率。
  *成立时的可观测结果*：GRPO 训练收敛后，VideoMME 时间感知/推理子集、Vinoground(Text) 相对纯答案奖励基线提升 ≥2 个点；证据 tIoU 显著更高。
- **H2（符号验证分支缩放性）**：时间逻辑自动机对可验证子集的证据检查与 LLM-judge 一致性 ≥90%，且能无缝替代 LLM-judge 子集。
  *成立时的可观测结果*：在 500 题抽样子集上符号验证 vs Kimi K2.6 判定一致率 ≥0.90。
- **H3（证据奖励与准确率相关）**：证据奖励（tIoU/groundedness）与下游答案准确率正相关，可作为可靠的训练信号。
  *成立时的可观测结果*：奖励-准确率相关性（Pearson/Spearman）在验证集上 ≥0.5。
- **H4（与 Pre-SFT DPO 正交）**：证据 RLVR 与 TEMPLE 式 Pre-SFT DPO 组合增益可叠加（1+1>1.5）。
  *成立时的可观测结果*：Pre-SFT DPO + 证据 RLVR 联合 > 单独任一 + 二者之和的一半。

## 3. 总体方法设计

### 3.1 数据流水线

1. **语料**：LLaVA-Video 子集 + LongVideoBench/VideoMME 公开训练拆分（若许可）+ 自建时间证据 QA。目标规模 ~32K 条。
2. **时间逻辑原语解析（DeepSeek V4 Pro，离线）**：对每条 QA 输出结构化 JSON：
   ```
   {"objects":["person","dog"],"actions":[{"id":0,"verb":"chase","object":"dog"}],
    "order_constraints":[{"before":0,"after":1}],
    "gold_answer":"...", "gold_evidence":[[s0,e0],[s1,e1]]}
   ```
   Prompt 要点：给出 3-5 段均匀采样帧缩略图序列（每 8s 一帧）+ 字幕，要求标注"该答案依赖哪些时间区间"。
3. **过滤规则**：(a) 时间原语解析 JSON 可解析率 >90%；(b) gold_evidence 区间与视频长度比 ∈ [0.05, 0.8]（排除过短/过长）；(c) DeepSeek 与 Kimi K2.6 双模型对"答案是否依赖给定证据"判定一致才保留；(d) 人工抽样校准 ~2%（650 条）。
4. **扰动对构造（复用 TEMPLE 流水线）**：clip dropping / shuffling / reversal，难度 r=16→2 课程，chosen=clean、rejected=perturbed，~25K 对。

### 3.2 模型/算法设计

- **基座**：Qwen2.5-VL-7B（ViT 冻结，LoRA rank 64, α=128，仅训练 LLM 部分）。
- **输出格式**：`<answer>...</answer><evidence>[start,end]</evidence>`，用系统提示强制结构化。
- **时间逻辑自动机（可验证子集）**：将原语清单编译为确定性自动机；对模型输出证据区间做 (a) 与 gold_evidence 的 tIoU 计算；(b) 区间内是否存在规定动作/对象的语义检查（用 CLIP 文本-帧相似度兜底）；(c) 顺序约束检查（模型声称先 A 后 B，自动机验证时间戳顺序）。
- **符号-语义双轨奖励**：
  - 可验证子集（~70% 题目，答案可判、原语明确）：
    `r_evid = 0.5·tIoU([ŝ,ê],[s*,e*]) + 0.3·coverage + 0.2·grounded`
    其中 coverage = 各支撑区间中命中比例；grounded = 区间内关键对象 CLIP 置信度的平均。
  - 开放子集（~30%，答案需语义判断）：Kimi K2.6 作为 LLM-judge 输出 0-1 groundedness 分数。
- **双奖励分解**：`r = r_ans + λ·r_evid`，λ 课程从 0.3 线性升至 1.0（每 500 步 +0.1）；r_ans 为答案正确性（匹配/LLM-judge）。
- **Pre-SFT DPO**：标准 DPO loss，β=0.1，与 TEMPLE 相同配置。

### 3.3 训练流程

- **Stage A（Pre-SFT DPO）**：25K 对，LoRA rank 64，学习率 1e-5，batch 32，2 epoch，≈2 GPU·天。
- **Stage B（GRPO）**：32K 样本、每组 8 rollout、3 epoch；policy lr 1e-6，clip ε=0.2，λ 课程；2 卡训练 + 2 卡 rollout/评测并行，≈6-10 GPU·天。
- 优化器 AdamW，β=(0.9,0.99)，warmup 3%，cosine schedule；DeepSpeed ZeRO-3 + FSDP 混合，序列打包处理长视频 token。

### 3.4 推理与评测流程

- 推理统一 64 帧均匀采样（报告也给出 128 帧敏感性）；temperature=0 评估。
- 评测脚本逐基准跑 3 个随机种子，输出 mean±std；证据区间由模型输出解析，与 GT 比 tIoU。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| LLaVA-Video 子集 | 训练 | 公开，research license（需核对子集许可）|
| LongVideoBench（arXiv:2407.15754）| 训练/评测 | 公开（CC-BY-NC 类，需核对）|
| VideoMME（arXiv:2405.21075）| 评测 | 公开 |
| MLVU（arXiv:2406.04264）| 评测 | 公开 |
| LVBench（arXiv:2406.08035）| 评测 | 公开 |
| Vinoground（arXiv:2410.02763）| 评测 | 公开 |
| TempCompass（arXiv:2403.00476）| 评测 | 公开 |
| Haystack-LVBench / Haystack-Ego4D | 评测（长距离检索）| 公开 |

### 4.2 划分与数量
- 训练：32K QA（自建+筛选），25K DPO 对；
- 验证：2K（用于 λ 课程与奖励-准确率相关性）；
- 测试：各评测基准官方测试集。

### 4.3 预处理与格式
- 抽帧 1fps，帧 224×224；训练 64 帧封顶；
- 证据区间以秒归一化到 [0,1]；
- 全部数据转 JSONL（video_path, question, answer, evidence, primitives, split）。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| Qwen2.5-VL-7B | 官方 | github.com/QwenLM/Qwen2.5-VL |
| TEMPLE | Computer Vision V·论文53 | 待复核（论文开源）|
| TPO | arXiv:2501.13919 | github.com/Juncheng1223/TPO |
| Video-R1 | arXiv:2503.21776 | github.com/Video-R1/Video-R1 |
| TimeSearch-R | arXiv:2511.05489 | github.com/LuckyWang-Athena/TimeSearch-R |
| FrameThinker | arXiv:2509.24304 | 论文未公开则按描述复现 |

### 5.2 复现步骤与预期指标表
复现统一 64 帧、temperature=0。预期主表（accuracy）：

| 方法 | VideoMME-Time | Vinoground(Text) | LVBench | LVB |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 参考官方 | — | — | — |
| TEMPLE | +3.6 (时间感知) | +3.2 | — | — |
| Video-R1 | SOTA 级 | 中 | 高 | 中 |
| TimeSearch-R | LongVideoBench SOTA | — | — | 高 |
| **Ours** | ≥基线+2 | ≥基线+2 | 优于纯答案 RLVR | 优于纯答案 RLVR |

### 5.3 统一评测口径
所有方法同 prompt 模板、同帧采样、同解析器；证据区间提取统一正则。报告对比表中注明各基线原报告 vs 本地复现差异。

## 6. 实验矩阵

- **A（主实验）**：完整管线（Pre-SFT DPO + 证据 GRPO），7B 与 13B（若预算允许）各一版。
- **B1（奖励消融）**：仅 r_ans / 仅 r_evid / r_ans+λ r_evid(λ=0.3/0.6/1.0)。
- **B2（验证器消融）**：全 LLM-judge / 全符号自动机 / 双轨混合。
- **B3（λ 课程）**：固定 λ vs 线性课程 vs 自适应（按验证集奖励相关度）。
- **B4（对齐顺序）**：Pre-SFT DPO→RLVR vs RLVR→DPO vs 纯 RLVR。
- **C（鲁棒性）**：帧数 {32,64,128}；输入噪声（模糊/裁剪）；长度分级（<5min/5-15min/>15min）。
- **D（泛化性）**：零样本跨基座（Qwen2-VL、VideoLLaMA3）；跨数据集（TempCompass/Haystack 未训练集）。

## 7. 评测协议

- 指标：accuracy（各基准官方口径）、证据 tIoU、coverage、groundedness（LLM-judge）、Pass@1/4、奖励-准确率相关性。
- 所有主指标 3 个随机种子（42/2024/2026），报 mean±std；
- 显著性：配对 bootstrap 检验（1000 次重采样），p<0.05 标 *；
- 报告 ECE/校准曲线作为辅助诊断。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：Stage A 2 + Stage B 8 + 评测 2 = **≈12 GPU·天**（预算 10-14）。
- 存储：模型 ~20GB/份 × 5 版本 + 数据（视频缓存 500GB，JSONL 50GB）。
- API：DeepSeek V4 Pro 原语解析+答案/证据生成 ≈ 400 万 token；DeepSeek V4 Flash 扰动对与难例筛选 ≈ 800 万 token；Kimi K2.6 judge ≈ 300 万 token。按 $0.2-1.5/百万 token 估算总成本 ≈ **$400-900**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | 数据流水线搭建 + 时间原语 prompt 调试（小规模 500 条验证 JSON 解析率）|
| 2 | 32K QA + 25K DPO 对全量生成与过滤；人工抽检 650 条 |
| 3 | Stage A Pre-SFT DPO 训练 + 基线（Qwen2.5-VL/TEMPLE/TPO）复现 |
| 4 | Stage B GRPO 训练 v0 + 符号自动机/LLM-judge 双轨实现 |
| 5 | 主实验 + 奖励-准确率相关性分析 |
| 6 | 消融 B/C/D + 显著性检验 |
| 7 | 论文写作初稿（相关工作中/方法/实验）|
| 8 | 图表、附录、投稿 CVPR 2027（deadline ~2026-11，需提前 2 周定稿）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 证据奖励不准确（原语覆盖不全/自动机误判）| 中 | 高 | 双轨奖励互相校验；先 500 题验证奖励-准确率相关性再全量训练 |
| 时间逻辑原语解析质量差 | 中 | 中高 | DeepSeek Pro 重试+结构化 schema 强制；不合格题转开放子集 |
| λ 课程难调 | 中 | 中 | 按验证集奖励相关度自适应 λ；报告 λ 扫描 |
| 训练不稳定（GRPO rollout 方差大）| 低中 | 中 | 8 rollout 组、KL 约束、clip ε=0.2；必要时降 λ |
| 与 Pre-SFT DPO 正交性不足 | 低中 | 中 | 若组合增益 <1.5×，改为单一 RLVR 主推 |

## 11. 论文写作计划

- **目标会议/截稿**：CVPR 2027（11 月截稿，需 8 周前定稿）。
- **差异化卖点**：首个"答案+证据+充分性"统一训练目标；符号-语义双轨可缩放奖励；与 Pre-SFT DPO 正交叠加的完整配方。
- **图表清单**：Fig.1 双奖励框架图；Fig.2 数据流水线；Fig.3 证据奖励权重/课程与准确率曲线；Fig.4 案例（答对但证据错 vs 双正确）；Tab.1 主表；Tab.2 消融；Tab.3 鲁棒性；Tab.4 泛化。
- **相关工作覆盖**：偏好对齐（TEMPLE/TPO/LLaVA-Hound）、RLVR（Video-R1/TimeSearch-R）、证据奖励（SER/REVISOR/VideoSSR）、检索（APVR/AdaptToken/FrameThinker）。

## 12. 参考文献（真实核验）

- Computer Vision V·论文53·TEMPLE（arXiv:2503.16929）
- Video-R1: arXiv:2503.21776
- TimeSearch-R: arXiv:2511.05489
- SER: arXiv:2606.24726
- VideoSSR: arXiv:2511.06281
- REVISOR: arXiv:2511.13026
- TPO: arXiv:2501.13919
- LLaVA-Hound: arXiv:2404.01258
- FrameThinker: arXiv:2509.24304
- LongVideoBench: arXiv:2407.15754
- VideoMME: arXiv:2405.21075
- MLVU: arXiv:2406.04264
- LVBench: arXiv:2406.08035
- Vinoground: arXiv:2410.02763
- TempCompass: arXiv:2403.00476
