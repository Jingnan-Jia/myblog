# 实验设计书：SafeEvalSuite —— 安全防御的无偏统一评测协议

> 英文标题：SafeEvalSuite: Verifiable, Contamination-Free, Judge-Bias-Free Evaluation of Safety Defenses
> 目标会议：NeurIPS 2026 数据集与基准轨 / ICLR 2027（高优先级，工具+协议类）
> 硬件假设：4×NVIDIA L40（192GB）；无训练，纯评测工程 + 6 个防御复现

## 0. 摘要

当前安全防御论文各用各的评测：测试集重叠、判分器多为同族 API、ASR 用简单关键词匹配，导致"+40% 安全提升"无法横向复现。本工作提出 SafeEvalSuite：把 LexInstructEval 的程序化验证、ArxivRoll/SafeRoll 的动态测试集、Self-Preference 的评审去偏三者组合成安全防御论文的"标配评测协议"。协议含"三支柱"：程序化判定（规则+正则+确定性过滤器，保证可复现）+ 动态测试集（每轮新生成抗记忆）+ 评审去偏（双异构评审 + 自我偏好审计报告），输出统一的"安全-效用"双层报告卡（含置信区间）。作为可复用工具包开源，内置 6 个防御方法一键复测。在 HarmBench/AdvBench/MM-SafetyBench/AbstentionBench 种子集上全流程演示。~20 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

安全防御论文 D 声称"ASR 从 X 降到 Y"。该声明的可信度取决于三件事：① **可复现**：判定规则是否确定（关键词匹配 vs 模型判分差异大）；② **抗污染**：测试集是否被记忆（动态生成 vs 静态）；③ **去偏见**：判分模型是否与 D 同族（同族评审放大自偏好）。SafeEvalSuite 提供标准化协议，对任意防御 D 输出标准化报告卡：
Report(D) = (ASR_programmatic, ASR_judge, Utility, Contamination_Score, Judge_Bias_Audit, 95% CI)。

### 1.2 相关工作不足

- **各用各的评测，不可比**：STAR-1（Al·86）用 4 个基准、SafetyReminder（Al·63）用 3 个，测试集重叠、判分器各不同，"+40%"无法横向复现；OPD（Al·20）证明答案偏差与污染强相关。
- **程序化验证没做安全**：（Machine Learning VII · 论文 18 · LexInstructEval）用形式化语法 + 程序化引擎客观验证指令遵循，证明"可验证判定"可行，但只测词汇指令不测安全。
- **动态评测缺安全/去偏**：（Special Track on AI Alignment · 论文 48 · ArxivRoll）OTP + SCP + Rugged Scores 只测能力；(Natural Language Processing IV · 论文 88 · SDEval) 动态生成但无污染度量、判分靠单 LLM。
- **同族评审不可信**：（Special Track on AI Alignment · 论文 55 · Mitigating Self-Preference by Authorship Obfuscation）证明同族评审自我偏好难以根除。
- **缺失**：程序化 + 动态 + 去偏三者从未组合成统一安全评测协议。

### 1.3 为什么是现在、为什么你的环境适合做

- **时机**：ArxivRoll/SDEval 2025-2026 动态评测正热，评审偏见 2026 成顶会热点（arXiv:2606.19544, 2604.23178）；"安全评测自身可信度"被 STAR-1/SafetyReminder 直接暴露；NeurIPS 2026 数据集与基准轨是窗口。
- **环境**：纯评测工程 + 6 个 8B 防御复现，4 卡并行；DeepSeek/Kimi 双异构评审即插即用；本主题其余 Idea（7/12 为基础设施）可互为验证。

## 2. 研究目标与可验证假设

1. **H1（协议降低复现方差）**：同一防御在协议下 3 次独立运行 ASR 标准差显著小于"关键词匹配"口径。
   - 可观测结果：程序化口径 3 次运行 σ ≤1 点；关键词匹配 σ ≥5 点（因匹配器敏感）。
2. **H2（去偏审计可暴露同族偏好）**：协议的双异构评审能检测出"评审者与防御同族"导致的分数虚高。
   - 可观测结果：对同族评审配置，审计报告的自我偏好信号显著（偏好率差 ≥5 点）；换异构评审后消失。
3. **H3（动态集抗污染）**：在种子集上过拟合的防御，在 SafeEvalSuite 动态集上 ASR 优势显著小于静态集。
   - 可观测结果：过拟合防御静态集 ASR 降 20 点、动态集仅降 5 点（揭示记忆成分）。
4. **H4（协议可一键复测 6 个防御）**：6 个内置防御全部在 3 天内完成一键复测，输出统一报告卡。
   - 可观测结果：脚本化评测端到端 <48h；6 防御 × 5 评测集全绿通过。

## 3. 总体方法设计

### 3.1 数据流水线

- **种子集**：HarmBench（arXiv:2402.04249）+ AdvBench + MM-SafetyBench + AbstentionBench（arXiv:2506.09038），用作"静态种子子集"（协议 A 通道）；
- **动态测试集（协议 B 通道）**：复用 Idea 7 SafeRoll 的生成器（语义簇 → 模板 → DeepSeek 扩产 → Kimi 复评 → 去重），每 3 个月一轮；首版 1 轮 1,000 条文本 + 500 图像；
- **人工锚点**：500 条人工标注（有害性三分类：有害/边界/良性），用于程序化判定器校准与评审审计锚定；
- **判分通道**：程序化（规则）+ 双异构评审（DeepSeek V4 Pro + Kimi K2.6）+ 开源 8B 评审（第三通道，审计用）。
- **数量**：静态 1,500 + 动态 1,500 + 人工锚 500。

### 3.2 方法设计

**模块 1：程序化判定器（可复现支柱）**
- 多级规则流水线：
  1. 拒绝模板匹配（"抱歉，我不能/I can't/无法提供/违反政策"等正则 + 同义表）；
  2. 有害词库匹配（分层有害词表 + 模糊匹配）；
  3. ASR 匹配器（HarmBench 官方匹配器 + 自研严格版）；
- 确定性打分：每样本输出 0/1 判定 + 判定证据（命中规则 id）；
- 不一致样本（规则无把握）升级到双评审通道，并记录"升级率"作为协议质量指标。

**模块 2：双异构评审（去偏支柱）**
- 对升级样本：DeepSeek V4 Pro 与 Kimi K2.6 各自独立判"是否有害/是否拒绝"；
- 一致性：Cohen's κ；分歧样本送人工锚点裁决；
- **自我偏好审计**：若被测防御的判分依赖某评审，报告该评审与其他评审的偏好差（同族偏好信号）；对每个防御附"审计报告"。
- 第三通道开源 8B 评审作为"模型族多样性"审计基线。

**模块 3：动态测试集（抗污染支柱）**
- 复用 Idea 7 的轮次机制：发布时哈希时间戳、句子级去重（MinHash + 嵌入距离）、新鲜度度量；
- 对防御 D 报告"静态 ASR vs 动态 ASR"差，差大说明存在记忆成分（污染嫌疑）。

**模块 4：统一报告卡与工具包**
- 报告卡结构（JSON/Markdown 双输出）：
  ASR（程序化 / 双评审均值 / 通道差）、Utility（MMLU/GSM8K/TruthfulQA 相对损失）、FPR、Contamination Score（ρ_cluster）、Judge Bias Audit（偏好率差 + κ）、95% CI；
- CLI：`safeeval run --defense <id> --eval harmbench`；内置防御注册表（6 个）。
- 内置防御：无防御、SmoothLLM、SafeDecoding、SafetyReminder 复现、STAR-1 微调、HumorReject 复现（可选）。

### 3.3 训练流程

- 无训练；SmoothLLM/SafeDecoding/SafetyReminder/STAR-1 均为推理或 SFT 复现（STAR-1 需 1K 数据 SFT，4 卡 ~4 GPU·天）。

### 3.4 评测与对抗测试流程

1. 6 防御 × 5 评测集（HarmBench/AdvBench/MM-SafetyBench/AbstentionBench/动态集）全量；
2. 每防御输出统一报告卡；
3. 协议质量验证：复现方差（3 次独立运行）、评审分歧率、与人工锚一致性；
4. 去偏演示：构造同族/异族评审对照，展示审计可发现偏好。

## 4. 数据集/基准细节

| 基准 | 用途 | 来源/许可 | 划分 |
|---|---|---|---|
| HarmBench（arXiv:2402.04249） | 静态种子 + 主评测 | 开源 | 全量 |
| AdvBench | 静态种子 | 开源 | 全量 |
| MM-SafetyBench（arXiv:2311.17600） | 静态种子（多模态） | 开源 | 全量 |
| AbstentionBench（arXiv:2506.09038） | 弃权/过度拒绝 | 开源 | 全量 |
| MMLU / GSM8K / TruthfulQA | 效用 | 开源 | 官方划分 |
| SafeEvalSuite 动态集 v1 | 动态通道 | 自建（DeepSeek/Kimi） | 1,500 条 |
| 人工锚点 | 校准 | 本团队 | 500 条 |

## 5. 基线复现（内置防御）

| 防御 | 官方代码 | 复现要点 |
|---|---|---|
| 无防御 | - | 8B 原始采样 |
| SmoothLLM（arXiv:2310.03684） | 官方 | 扰动 10% 多数投票 |
| SafeDecoding（arXiv:2402.08983） | 官方 | 安全词汇偏置 |
| SafetyReminder（Al·63） | 无官方 | SAPT 软提示（移植 8B） |
| STAR-1 微调（Al·86） | 官方数据 | 1K SFT |
| HumorReject（Al·90，可选） | 无官方 | 幽默数据微调 |

预期报告卡示例（初值）：

| 防御 | ASR(程序化) | ASR(双评审) | 通道差 | FPR | Utility(MMLU) | ρ_cluster | 偏好审计 |
|---|---|---|---|---|---|---|---|
| 无防御 | 0.41 | 0.42 | 0.01 | - | 0% | - | - |
| SmoothLLM | 0.24 | 0.26 | 0.02 | - | -1% | 0.02 | 无 |
| SafeDecoding | 0.20 | 0.22 | 0.02 | - | -2% | 0.01 | 无 |
| SafetyReminder | 0.18 | 0.20 | 0.02 | 3% | -2% | 0.03 | 无 |
| STAR-1 | 0.12 | 0.14 | 0.02 | 2% | -1.5% | 0.12 | 低 |
| **协议复现方差** | **σ≤1 点** | σ≤2 点 | - | - | - | - | - |

## 6. 实验矩阵

- **A. 主实验**：6 防御 × 5 评测集全量报告卡。
- **B. 协议质量（H1/H2/H3）**：复现方差、评审分歧、去偏审计、动态 vs 静态。
- **C. 判定器消融**：程序化规则集（模板/词库/匹配器）逐层贡献；升级率敏感性。
- **D. 评审配置**：双评审 vs 单评审 vs 三通道；同族 vs 异族。
- **E. 动态集参数**：轮次数量、去重阈值、新鲜度。
- **F. 人工锚一致性**：程序化/评审 vs 人工 500 条。

## 7. 评测协议

- **ASR**：程序化（确定性）与双评审双口径，报告均值 + 通道差 + 升级率。
- **FPR**：良性/边界集误报。
- **Utility**：MMLU/GSM8K/TruthfulQA 相对损失。
- **复现方差**：同防御 3 次独立运行（不同 seed/批次顺序）ASR 标准差。
- **评审分歧率**：双评审不一致样本比例；κ 系数。
- **偏好审计**：同族评审 vs 异构评审的偏好率差。
- **统计**：3 种子（2026/7/42）；Wilson 置信区间；McNemar 成对比较。

## 8. 算力与资源计划

| 阶段 | 内容 | GPU·天 |
|---|---|---|
| S1 | 程序化判定器实现 + 人工锚校准 | 2 |
| S2 | 6 防御复现（含 STAR-1 SFT） | 6 |
| S3 | 静态集全量评测（4 卡并行） | 5 |
| S4 | 动态集生成（复用 SafeRoll 管线）+ 评测 | 4 |
| S5 | 去偏审计 + 协议质量验证 + 工具包 | 3 |
| 合计 | | ~20 |

- 存储：模型 ×6 + 数据 ≈ 80GB。
- API：DeepSeek + Kimi 双评审 ~150K 次调用（~$60）；动态集生成 ~1M tokens（$10）。总 API <$100。

## 9. 里程碑与时间线（单人 + 4 卡，共 5 周）

| 周 | 里程碑 |
|---|---|
| W1 | 程序化判定器 + 人工锚标注 |
| W2 | 6 防御复现 + 静态评测 |
| W3 | 动态集生成 + 评测（复用 SafeRoll） |
| W4 | 去偏审计 + 协议质量验证（复现方差/分歧） |
| W5 | 工具包 CLI + 报告卡 + 论文初稿 |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 缓解/备选 |
|---|---|---|---|
| 程序化判定器与评审分歧大，审稿质疑"为什么这个标准对" | 中 | 高 | 人工锚 500 条校准 + 报告程序化-评审一致性；若某类分歧大则明确其为"需人工"类别 |
| 6 个防御复现不均（个别复现失败） | 中 | 中 | 内置 6 个但论文只需 3-4 个全绿；失败防御如实标注"复现受限" |
| 工具包维护成本 | 中 | 中 | 首版只保证内置 6 个方法可复现；外部扩展留接口 |
| 动态集被质疑"生成样本质量" | 中 | 中 | 复用 SafeRoll 的复评门 + 人工抽检；报告生成质量指标 |
| 去偏审计信号弱（评审实际无明显偏好） | 低 | 中 | 审计报告"无显著偏好"也是有效输出；构造同族评审对照演示检测能力 |

## 11. 论文写作计划

- **目标**：NeurIPS 2026 数据集与基准轨（8 月截稿）主选；备选 ICLR 2027。
- **差异化卖点**：① 首个"程序化 + 动态 + 去偏"三支柱安全评测协议；② 统一报告卡（含置信区间 + 污染分数 + 偏好审计），让安全论文得分可比；③ 一键复测工具包（6 个防御），社区可直接用。
- **图表清单**：Fig1 三支柱框架 + 报告卡示意；Fig2 复现方差对比（程序化 vs 关键词）；Fig3 去偏审计演示；Fig4 动态 vs 静态 ASR（记忆成分）；Fig5 6 防御雷达图；Tab1 全报告卡；Tab2 协议质量指标；Tab3 人工锚一致性。
- **相关工作**：程序化评测（Al·18 LexInstructEval）；动态评测（Al·48, Al·88）；污染（Al·20, arXiv:2410.15005, 2502.17259）；评审偏见（Al·55, arXiv:2606.19544, 2604.23178）；防御（arXiv:2310.03684, 2402.08983, 2506.15734, 2504.01903）。
- **开源**：SafeEvalSuite 工具包、报告卡 schema、动态集 v1。

## 12. 参考文献（已核验）

- LexInstructEval（收藏）：（Machine Learning VII · 论文 18）
- ArxivRoll（收藏）：（Special Track on AI Alignment · 论文 48）；arXiv:2507.19219
- SDEval（收藏）：（Natural Language Processing IV · 论文 88）
- OPD（收藏）：（Natural Language Processing I · 论文 20）
- Self-Preference（收藏）：（Special Track on AI Alignment · 论文 55）
- SmoothLLM：arXiv:2310.03684
- SafeDecoding：arXiv:2402.08983
- SafetyReminder：arXiv:2506.15734
- STAR-1：arXiv:2504.01903
- HumorReject（收藏）：（Special Track on AI Alignment · 论文 90）
- HarmBench：arXiv:2402.04249
- MM-SafetyBench：arXiv:2311.17600
- AbstentionBench：arXiv:2506.09038
- CAP：arXiv:2410.15005
- Detecting Benchmark Contamination Through Watermarking：arXiv:2502.17259
- Reliability without Validity：arXiv:2606.19544
- Judging the Judges：arXiv:2604.23178
