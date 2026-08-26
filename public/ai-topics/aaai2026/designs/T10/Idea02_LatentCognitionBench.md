# 实验设计书：心理测量学校准的统一潜在认知能力基准

> 英文标题：Latent-Cognition Bench: A Psychometrically-Calibrated Unified Evaluation Framework
> 关联收藏论文：（Natural Language Processing I · 论文 60 · Measuring the Unmeasurable）、（Natural Language Processing II · 论文 71 · The Curious Case of Analogies）、（Natural Language Processing III · 论文 35 · Easy for Children, Hard for AI）、（Cognitive Modeling and Cognitive Systems · 论文 25 · ARCHE）
> 硬件/API 基线：4×L40（192GB）；DeepSeek V4 Flash（题目生成）、V4 Pro（认知策略解释）、Kimi K2.6（题目质量审查）

---

## 0. 摘要

本项目把"潜在认知能力"落地为一个六因子（知觉、注意、记忆、推理、语言、心智）分层基准，并引入项目反应理论（IRT）两参数模型做难度/区分度校准，输出"因子能力曲线 + 每题认知轨迹"而非单一总分。以 DeepSeek V4 Flash 按 18 个"因子×难度"单元格生成候选题目、Kimi K2.6 做构念效度审查、心理学家抽样审核 200 题，对 15 个开源/闭源 LLM 做全题集推理作答，拟合 IRT 参数并验证复测信度与判别效度。与 MMLU、BIG-Bench-Hard、EmoBench、ChildBench、ARCHE 做对齐与对照，直接补 (Measuring the Unmeasurable) 的"批评有余、落地不足"缺口。纯评测无大模型训练，4×L40 推理约 2 周，投稿 AAAI 2027。

## 1. 研究背景与动机

### 1.1 问题定义

现有 LLM 认知评测（MMLU、BIG-bench 等）有两个结构性缺陷：①题目跨因子冗余重叠、难度未经心理测量学标定，"高分"无法解释为"某认知因子强"；②只看最终答案，无法刻画"潜在认知过程"。（Measuring the Unmeasurable）提出了统一框架 + 推理轨迹评测的主张，但只给批判不给实现。本项目要回答：能否构建一个**六因子 × 三难度、IRT 可校准、能力曲线可解释**的 LLM 认知基准，且其测量学性质（信度、效度、难度单调性）可被实证验证？

### 1.2 相关工作不足

- **主张无落地**：（Natural Language Processing I · 论文 60 · Measuring the Unmeasurable）只做评测范式倡议，无数据集与指标细节；（Natural Language Processing I · 论文 62 · Guess or Recall）把记忆机制做代理分类但非认知因子；（Natural Language Processing IV · 论文 14 · Density Modeling）聚焦单子域（数字密度）。
- **覆盖片面**：（Natural Language Processing II · 论文 71 · The Curious Case of Analogies）只测类比、（Cognitive Modeling and Cognitive Systems · 论文 25 · ARCHE）只测科学论证结构、（Natural Language Processing III · 论文 35 · Easy for Children, Hard for AI）只测幼儿视觉，均无多因子统一测量学。
- **缺乏 IRT 校准**：Evaluating LLMs with Psychometrics（arXiv:2406.17675）证明 IRT/因子分析对 LLM 可行，但停留在单题参数报告，没有落地为多因子、多难度层级的基准，也未把"能力曲线"用于模型间对比与纵向追踪。

### 1.3 为什么是现在、为什么你的环境适合做

- **时点**：IRT 校准方法（arXiv:2406.17675）2024 年才被引入 LLM 评测；"潜在认知过程测量"是 2025–2026 顶会热点（ARCHE、Analogies、The Other Mind 同批出现）；当前无多因子统一心理测量学基准，窗口仍在。
- **环境匹配**：纯推理评测不耗训练；4×L40 可并行跑 7B–70B 开源模型；DeepSeek Flash 批量生成题目的成本极低；Kimi K2.6 作第二裁判；心理学家抽样审核可通过本地高校合作完成。
- **不依赖专有数据**：所有题源来自合成 + 公开经典范式（威斯康星卡片、n-back、类比、时间估计、ToM），无版权障碍。

## 2. 研究目标与可验证假设

**H1（难度层级单调性）**：IRT 估计的题目难度按"儿童/成人/专家"三层预设严格单调，即校准后难度参数与预设层级 Spearman ρ>0.8。
- 成立时观测：18 单元格中 ≥16 个单元格的难度中位数单调递增；违反单元格 ≤2。

**H2（结构效度）**：六因子结构可被因子分析复现，因子间判别效度成立（判别相关 < 0.85）。
- 成立时观测：六因子 EFA 载荷模式清晰；因子间平均相关 < 0.85；与 EmoBench/ARCHE 对应因子的收敛效度 r>0.5。

**H3（测量信度）**：能力参数估计有高复测信度。
- 成立时观测：split-half（奇偶题）能力估计 ICC(2,1)≥0.75；不同随机种子作答下能力排序 Kendall τ>0.8。

**H4（轨迹可解释）**：能力曲线 + 每题认知轨迹能区分"猜测命中"与"真推理"。
- 成立时观测：在自建"选择题+强制推理说明"样上，推理轨迹质量（人类心理学家评分）与能力估计正相关 r>0.4；与 (Guess or Recall) 的猜测判定做交叉验证一致率 >65%。

## 3. 总体方法设计

### 3.1 数据流水线（题目生成与审查）

**六因子 × 三难度矩阵**：
| 因子 | 经典范式题源 | 难度层（儿童/成人/专家）设计 |
|---|---|---|
| 知觉 | 图形匹配、韦伯定律强度判断 | 简单同/异 → 多属性干扰 → 噪声低对比度 |
| 注意 | n-back、Stroop 类 | 1-back → 2-back → 3-back 含干扰 |
| 记忆 | 名单回忆、工作记忆 span | 5 项 → 7 项 → 9 项+插入干扰 |
| 推理 | 比例/关系类比、序列 | 显式关系 → 单步隐含 → 多步结构映射 |
| 语言 | 词汇、句法歧义、语用 | 常用词 → 生僻/多义 → 反讽/间接言语行为 |
| 心智 | ToM 故事、错误信念、二阶信念 | 一级信念 → 二级信念 → 道德-意图冲突 |

**生成 prompt 思路**（DeepSeek V4 Flash，每单元格 400 题、共 7200 候选）：
```
你是认知心理学题目作者。为[因子·难度层]写一道多项选择题（4 选项）。
要求：①干扰项有认知诊断价值（对应典型错误模式，如守恒任务中心化错误）；
②提供"认知策略解释"（正确推理路径 2–3 步）；③标注与认知发展文献的对应。
每题输出 JSON：{item, options, answer, strategy, source_paradigm}。
```
DeepSeek V4 Pro 用于重写"认知策略解释"使其更严谨；Kimi K2.6 做题目质量审查（答案唯一性、难度合理性、因子贴合度），双裁判不一致交由心理学家裁决。

**过滤规则**：①答案唯一性（Kimi 无提示作答必须命中标注答案，否则弃）；②文本查重（嵌入余弦 >0.85 丢弃）；③难度先验抽查（2 名普通成人试做儿童层题正确率应 >85%，成人层 40–80%，专家层 <50%，偏差则标记返工）。

**数量**：候选 7200 → 过滤后约 **5400 题**（每单元格 ≥300 题，满足 IRT 每因子 ≥300 题的最小样本要求）。

### 3.2 方法设计（测量学核心）

- **IRT 两参数 logistic 模型**（Hambleton & Swaminathan 1985, DOI:10.1007/978-94-017-1988-9）：
  `P(X_{ij}=1|θ_i,a_j,b_j) = 1/(1+exp(−a_j(θ_i−b_j)))`，
  其中 θ_i 为模型 i 的因子能力，a_j 区分度，b_j 难度。用 MCMC（Stan）或 EM（pyirt）拟合，每因子独立拟合一个模型。
- **能力曲线输出**：每模型得到六因子能力向量 θ∈R⁶；用户端渲染雷达图 + 因子内"通过率×难度"阶梯曲线。
- **认知轨迹（每题级）**：对随机 20% 题要求模型先输出"推理说明"再给答案；用 ARCHE 式三分类（演绎/归纳/溯因）标注推理步（DeepSeek V4 Pro 标注 + 心理学家抽检），输出"正确答案前的推理轨迹"。
- **与人类数据收敛效度**：选用有公开人类作答的范式（如 n-back、ToM 故事，参照 PNAS ToM 论文 DOI:10.1073/pnas.2405460121 与 Nature HB ToM 论文 DOI:10.1038/s41562-024-01882-z），比较 LLM 与人类难度排序的一致性。

### 3.3 训练流程

无模型训练。推理评测：开源模型用 vLLM 挂在 4×L40（并行 4 实例），温度=0；闭源模型走 API（DeepSeek V4、GPT-4o 等）。每模型 × 5400 题 = 1 次前向 + 20% 题额外一次"推理说明"采样（温度 0.7，重采样 3 次投票）。

### 3.4 评测流程

- 自动：IRT 参数估计、split-half ICC、EFA 因子载荷、因子间相关、难度层级单调性、能力排序 Kendall τ。
- 人工：心理学家审核 200 题构念效度（每因子 ≥30 题），1–5 分，均值 >4.0 才通过发布门槛；审核结果用于修正生成器 prompt 并重跑生成。
- 伦理：题目为认知范式，不涉及心理创伤内容；ToM 故事避免文化敏感设定；无真人被试，无需 IRB（若后续要真人作答验证则需 IRB）。

## 4. 数据集细节

- **题目生成**：全部合成（DeepSeek Flash 生成 + Pro 重写 + Kimi 审查），无版权问题；发布时给 CC-BY 4.0。
- **评测对象**：开源 Qwen2.5-7B/14B/72B、Llama-3.3-70B（LoRA 挂载）、DeepSeek-V3（arXiv:2412.19437 本地蒸馏版可省）、Qwen2.5-VL-7B；闭源 GPT-4o、DeepSeek V4 Pro、Kimi K2.6，共 15 个。
- **对齐数据集**：MMLU、BIG-Bench-Hard、EmoBench（DOI:10.18653/v1/2024.acl-long.326）、ChildBench（AAAI 2026，DOI:10.1609/aaai.v40i38.40479）、ARCHE（AAAI 2026）作对照（非训练用）。
- **划分**：5400 题按单元格分层随机 8:1:1（IRT 校准集:验证:测试）；能力估计在测试集上报，避免过拟合校准。
- **许可/隐私**：无隐私数据；合成题目全部开源。

## 5. 基线复现

| 基线 | 官方地址 | 复现要点 | 预期对齐指标（因子间判别相关 / split-half ICC） |
|---|---|---|---|
| MMLU | github.com/hendrycks/test | 取子集对应语言/推理因子 | 与推理因子相关 0.6–0.7 |
| BIG-Bench-Hard | github.com/suzgunmirac/BIG-Bench-Hard | 对照分数 | 同 MMLU |
| ARCHE | AAAI 2026 论文资源 | 仅在论证结构子集对齐 | 与推理/语言相关 0.5–0.65 |
| EmoBench | 论文（DOI:10.18653/v1/2024.acl-long.326） | 与心智因子对齐 | 与心智因子相关 >0.5 |
| Psychometrics 方法（arXiv:2406.17675） | 论文代码 | 用其 IRT 管线复算本基准子集 | 参数一致性比对 |

统一口径：所有模型同题序、同温度、同提示模板；报告 5 个随机题序种子的均值±std。

## 6. 实验矩阵

| 组 | 名称 | 目的 | 变量 | 固定项 |
|---|---|---|---|---|
| A | 主实验 | 15 模型 × 5400 题 IRT 校准 | 模型 | 题集、温度、提示 |
| B1 | 消融-题量 | 每因子 150/300/450 题对参数稳定性的影响 | 题量 | 模型子集 |
| B2 | 消融-难度层 | 去掉儿童层/专家层对判别效度的影响 | 难度层 | 同上 |
| B3 | 消融-轨迹 | 有无推理说明对能力估计的影响 | 是否要求说明 | 20% 随机子集 |
| C | 鲁棒性 | 提示模板改写、选项顺序洗牌、题面释义下的稳健性 | 扰动 | 模型子集 |
| D | 泛化-语言 | 中文 vs 英文题面能力等价性 | 语言 | 双 8 因子双语题 |

## 7. 评测协议

- **IRT 参数**：报告每个模型每因子的 θ 及 95% CI（MCMC 后验）；区分度/难度用 IRT 信息量加权。
- **信度**：split-half（奇偶题）能力 ICC(2,1)；难度参数跨 5 个题序种子稳定性。
- **效度**：EFA（n=15 模型，注意样本小，采用 bootstrap 因子载荷置信区间）；收敛效度与判别效度相关系数。
- **显著性**：能力差异用配对 t 检验（5 种子）＋Holm 校正；报告均值±std。
- **人工**：心理学家 200 题构念效度 5 点评分 + 每题"是否测到该因子"二元判定；比例 >90% 才算合格。

## 8. 算力与资源计划

| 阶段 | 任务 | 资源 | 估计 |
|---|---|---|---|
| 题目生成 | 7200 候选 + 审查 | Flash ~1200 万 token、Pro ~300 万、Kimi 审查 ~800 万 | API 约 ¥0.5–1k |
| 推理评测 | 15 模型 × 5400 题 | 开源 11 个 vLLM 4×L40 并行；闭源 4 个 API | 开源约 12–16 GPU·天；API 预算 ¥1–2k |
| IRT 拟合 | pyirt/Stan | CPU | 分钟级 |
| 人工审核 | 心理学家 200 题 | 人力 | 1 周排期 |
| 存储 | 题目 + 作答 JSONL ~30GB | 本地 | <100GB |

总 GPU·天 ≈ 12–16（纯推理）；API 预算 ¥1.5–3k。

## 9. 里程碑与时间线（单人 + 4 卡）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 六因子×三难度矩阵 + 生成器 prompt 定型 | 生成器 v1 |
| W2–W3 | 生成 7200 候选 + 过滤 + Kimi 审查 | 5400 题库 |
| W4–W5 | 开源模型 vLLM 推理 + 闭源 API | 全量作答表 |
| W6 | IRT 拟合 + 信度/效度/单调性分析 | 表 A/B |
| W7 | 消融 B/C/D | 表 C/D |
| W8 | 心理学家审核 + 修订 | 发布版基准 v1.0 |
| W9–W10 | 与 MMLU/ARCHE/EmoBench 对齐 + 写作 | AAAI 2027 初稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 题目生成同质模板化 | 中 | 多样化范式 prompt + 嵌入去重 + 心理学家抽改 |
| 模型数少（15）致因子分析样本不足 | 中 | bootstrap 置信区间；或增加基线模型数至 20+ |
| IRT 局部不收敛（高区分度题） | 中 | 限制 a_j≤2.5；改用 Bayes 先验正则 |
| 难度层级与预期不符 | 中 | 以实测难度重分桶而非强制预设 |
| 因子间相关过高（>0.9） | 中 | 合并相关因子或报告二阶因子；承认因子非完全独立 |
| "推理说明"被模型套话污染 | 中 | 轨迹质量独立打分 + 与答案无关性检验 |

## 11. 论文写作计划

- **目标会议**：AAAI 2027（截稿约 2026-08，跟踪官方）；备选 NeurIPS 2026 Datasets & Benchmarks。
- **差异化卖点**：首个六因子 × 三难度、IRT 可校准的 LLM 潜在认知基准；输出能力曲线而非总分；与"推理轨迹"结合；与人类数据做收敛效度。
- **图表清单**：①六因子能力雷达图（15 模型叠加）；②IRT 题参数散点（区分度×难度，分因子着色）；③难度层级单调性箱线图；④EFA 载荷热图；⑤split-half ICC 与跨种子稳定性；⑥与 MMLU/ARCHE/EmoBench/ChildBench 收敛效度散点矩阵。
- **相关工作覆盖**：Measuring the Unmeasurable、Guess or Recall、Density Modeling、ARCHE、Analogies、ChildBench（收藏论文）；Psychometrics arXiv:2406.17675、ToM 两篇（Nature HB / PNAS）、Dissociating Language and Thought、EmoBench-M、MMLU、BIG-Bench。

## 12. 参考文献

- Hambleton, R. K., & Swaminathan, H. (1985). Item Response Theory. DOI:10.1007/978-94-017-1988-9
- Evaluating Large Language Models with Psychometrics. arXiv:2406.17675
- Measuring the Unmeasurable. AAAI 2026.（Natural Language Processing I · 论文 60）
- Guess or Recall. AAAI 2026. arXiv:2508.02573（Natural Language Processing I · 论文 62）
- Density Modeling. AAAI 2026.（Natural Language Processing IV · 论文 14）
- ARCHE. AAAI 2026.（Cognitive Modeling and Cognitive Systems · 论文 25）
- The Curious Case of Analogies. AAAI 2026. arXiv:2511.20344（Natural Language Processing II · 论文 71）
- Easy for Children, Hard for AI (ChildBench). AAAI 2026. DOI:10.1609/aaai.v40i38.40479（Natural Language Processing III · 论文 35）
- Testing Theory of Mind in LLMs and Humans. Nature HB 2024. DOI:10.1038/s41562-024-01882-z
- Evaluating LLMs in Theory of Mind Tasks. PNAS 2024. DOI:10.1073/pnas.2405460121
- Dissociating Language and Thought in LLMs. TiCS 2024. DOI:10.1016/j.tics.2024.01.011
- Emergent Analogical Reasoning in LLMs. Nature HB 2023. DOI:10.1038/s41562-023-01659-w
- EmoBench. ACL 2024. DOI:10.18653/v1/2024.acl-long.326；arXiv:2402.12071
- EmoBench-M. arXiv:2502.04424
- Deep Knowledge Tracing. arXiv:1506.05908
- DeepSeek-V3 Technical Report. arXiv:2412.19437
- The Llama 3 Herd of Models. arXiv:2407.21783
- Qwen2.5-VL Technical Report. arXiv:2502.13923
