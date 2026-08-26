# 实验设计书：Idea 1 检索贡献归因诊断框架（AttribBench）

## 0. 摘要
本项目构建 AttribBench，一套面向长上下文 RAG 的"检索贡献归因"评测框架：用反事实证据扰动（正确/弱/矛盾/缺失四类证据臂）+ LLM-as-judge 一致性矩阵，把端到端 RAG 错误自动归因到"检索缺失 / 证据没用 / 参数知识凌驾证据"三类。相比 RGB、RAGAS、Double-Bench 只给组件级或端到端分数，AttribBench 首次给出"证据到底有没有被模型用上"的可量化指标（证据利用率 EUR 与参数知识凌驾度 PKO）。全程无训练、纯推理 + API，约 5 GPU·天可产出 1 万+问题的诊断矩阵，可作为评测/资源轨道论文投稿 EMNLP 2027。

## 1. 研究背景与动机
### 1.1 问题定义
给定一个长上下文 RAG 系统 S（检索器 R + 生成器 G）和一个查询 q，系统产出答案 a = G(q, R(q))。当 a 错误时，我们希望回答三个问题：(Q1) 检索器是否漏掉了必要证据（检索缺失）？(Q2) 检索到的证据是否被生成器实际采用（证据没用）？(Q3) 模型参数化知识是否覆盖了证据并给出与之矛盾的答案（参数知识凌驾）？目前没有任何标准指标能自动区分这三类失败。

### 1.2 相关工作不足
- RGB（arXiv:2309.01431）与 RAGAS（arXiv:2309.15217）只报端到端质量分，无法定位失败源。
- Double-Bench（NLP IV · 论文 30 · Are We on the Right Way to Assess Document Retrieval-Augmented Generation?）做到组件级归因（嵌入/检索/生成分别打分），但证据粒度止于"页"，且只描述"无证据也作答"的过度自信现象，不给出可操作的归因指标。
- （NLP VI · 论文 77 · Do Retrieval Augmented Language Models Know When They Don't Know?）发现"检索全不相关时 RALM 仍过度拒绝"，证明"检索到≠模型用了"，但该文是行为学实证，没有构造反事实证据集来定量拆解证据贡献。
- Self-RAG（arXiv:2310.11511）、Corrective RAG（arXiv:2401.15884）、Adaptive-RAG（arXiv:2403.14403）让模型自判是否检索/是否采纳，但把判断权交给模型本身，没有独立于生成器的证据使用度量。
- SIGIR 2026 视角论文（arXiv:2605.00505）提出"最大化上下文窗口内可用证据密度与可验证性"是新瓶颈，暗示归因/证据使用度量是当前 RAG 评测缺失的一环。

### 1.3 为什么是现在、为什么你的环境适合做
- 证据：RAG 评测正从"端到端分数"走向"组件级归因"（Double-Bench），再往前走一步就是"证据使用归因"；2026 年社区（RGB 之外多篇 benchmark）开始讨论证据可验证性，窗口期正合适。
- 环境：本任务零训练、纯推理 + API 评测，4×L40（192GB）可并行部署多个 7B/13B 生成器实例，完全匹配；DeepSeek V4 Pro 负责证据合成与判定、Kimi K2.6 做交叉 judge，API 预算可控（约 300–500 美元）。

## 2. 研究目标与可验证假设
- H1（证据利用可度量）：存在稳定的"证据利用率 EUR"指标，能区分高质量 RAG 系统（EUR 高）与"检索了但不用证据"的系统（EUR 低）。成立时观测：在 5 个基准上，诚实用证据的模型（如 Qwen2.5-7B instruct 显式提示引用证据）EUR 显著高于默认 prompt 模型，且 EUR 与端到端准确率正相关（Spearman ρ ≥ 0.6）。
- H2（三类归因可区分）：反事实扰动下的一致性矩阵能稳定把错误归到"检索缺失/证据没用/参数凌驾"三类，且与人工标注的归因一致率 ≥ 85%。成立时观测：200 条人工抽检集上自动归因 vs 人评的 Cohen's κ ≥ 0.7。
- H3（参数凌驾普遍存在）：在矛盾证据臂上，多数开源模型选择跟随参数化知识而非证据，且该倾向与模型规模正相关。成立时观测：矛盾臂答案与矛盾证据的一致性 < 30%，且 13B 模型的 PKO 显著高于 7B。
- H4（可迁移诊断）：AttribBench 的诊断结论能预测系统在未见基准上的相对排序改进空间。成立时观测：按 AttribBench 报告的"检索缺失占比"对系统做检索器升级，端到端提升幅度与占比正相关。

## 3. 总体方法设计
### 3.1 语料/数据流水线
1. 数据源：从 LongBench（arXiv:2308.14508）多文档 QA（hotpotqa/2wikimqa）、∞Bench（arXiv:2402.13718）en.qa 与 Double-Bench（arXiv:2508.03644）子集中各取 1500–2000 个已有证据-答案对的样本。
2. 证据合成（DeepSeek V4 Pro）：
   - 正证据臂 E+：取数据集标注的真实证据段落。
   - 弱证据臂 E±：用 DeepSeek 对证据做"去关键实体/模糊化改写"（prompt：`Rewrite the evidence so that it is weaker and cannot fully support the answer, keep length similar`），使证据信息量下降但仍相关。
   - 矛盾证据臂 E−：DeepSeek 生成与正确答案相反的合成段落（prompt：`Write a plausible paragraph that contradicts the correct answer <ans>`），要求行文可信。
   - 缺失臂 E∅：不放证据，只给查询。
   - 每类合成后用一致性过滤：Kimi K2.6 判定"合成证据是否真的弱/矛盾/可信"，保留双方一致同意的样本。
3. 问题生成：DeepSeek 基于正证据生成 3–5 个多跳变体问题；人工抽检 200 条校正 prompt 输出质量。
4. 规模：每基准 ≈ 1500 条查询 × 4 臂 = 约 2.4 万次生成/判定调用；总量控制在 5 个基准 × 1500 = 7500 条查询，矩阵 3 万次推理。

### 3.2 方法设计
- 评测引擎：对每个 (q, 臂) 生成答案 a；再构造 judge prompt 让 LLM-as-judge（DeepSeek V4 Pro 主判、Kimi K2.6 副判）输出 `{答案正确性: binary, 与证据一致性: 0/1/2(一致/部分/矛盾)}`，构成一致性矩阵 M[i,j]，i=臂类别、j=一致性等级。
- 归因算法：
  - EUR（Evidence Utilization Rate）= P(答案与 E+ 一致 | E+ 臂下答案正确)。
  - PKO（Parametric Knowledge Override）= P(答案与 E− 矛盾 | E− 臂下答案与参数先验一致)。参数先验用"无证据臂 E∅ 下模型自答"近似。
  - 三分类规则：(a) 若 E∅ 臂错但 E+ 臂对 → 检索缺失；(b) 若 E+ 臂也错且答案不引用证据 → 证据没用；(c) 若 E+ 臂对但 E− 臂答案跟随参数先验 → 参数凌驾。
- 引用增强：另跑一版强制引用 prompt（"Answer and cite [paragraph id]"）计算 citation 覆盖率，与 EUR 交叉验证。

### 3.3 训练流程
无训练。纯推理：4×L40 上并行跑 4 个生成器实例（Qwen2.5-7B/13B-instruct、Mistral-7B 可加）；每卡 1 实例，vLLM 部署 batch 8–16。

### 3.4 评测流程
- 每基准独立产出：(a) 四臂端到端准确率表；(b) EUR/PKO 汇总；(c) 三类归因分布饼图；(d) 每类错误的代表性样例集（供论文展示）。
- judge 消融：DeepSeek V4 Pro vs Kimi K2.6 分别判定，报告双方一致率；不一致样本由第 5 次重复判定 + 人工仲裁。

## 4. 数据集细节
| 数据集 | 来源 | 许可 | 划分 |
|---|---|---|---|
| LongBench-hotpotqa / 2wikimqa | arXiv:2308.14508 官方 | MIT | 取验证集 500 条/任务 |
| ∞Bench en.qa / en.mcq | arXiv:2402.13718 官方 | CC-BY | 取 800 条 |
| Double-Bench 子集 | arXiv:2508.03644 官方 | 开源 | 取 1200 条单跳+多跳 |
| RULER（诊断子集） | arXiv:2404.06654 官方 | MIT | 取 NIAH/多跳子集 800 条 |
- 预处理：长文档按官方切分保留；证据段 ID 化；统一 JSONL 格式（query, gold, evidence_pos, evidence_weak, evidence_contra, doc_id）。

## 5. 基线复现
| 基线 | 官方代码 | 复现要点 |
|---|---|---|
| Naive RAG | bm25 + Qwen2.5-7B | rank_bm25 + vLLM |
| Self-RAG | github.com/AkariAsai/self-rag | 用其 7B 反射模型生成反思 token |
| Corrective RAG | github.com/HuskyInSalt/CRAG | 其检索质量评估器 + 知识过滤器 |
| Adaptive-RAG | github.com/starsuzi/Adaptive-RAG | 按复杂度路由 |
- 预期指标表（以 LongBench-hotpotqa 为例，EM 粗估）：naive RAG 28–35，Self-RAG 30–38，CRAG 31–38，Adaptive-RAG 30–37；EUR 排序：CRAG/Self-RAG > naive > adaptive。
- 统一口径：同一生成器（Qwen2.5-7B-instruct, temperature=0）；同一证据切分；同一 judge 提示；全部在 AttribBench 四臂协议下报告，不直接搬原论文数值。

## 6. 实验矩阵
- A（主实验）：5 基准 × 4 基线 × 4 臂，输出 EUR/PKO/三分类归因。
- B（消融）：judge 模型（V4 Pro vs K2.6 vs 双模型投票）；证据臂构造强度（弱证据的改写幅度）；引用提示开/关。
- C（鲁棒性）：对证据段做顺序扰动/插入干扰段，验证 EUR 稳定性；对 judge prompt 做 3 种措辞变体。
- D（泛化性）：跨语言（Double-Bench 法语/中文子集）；不同长度（RULER NIAH 短/长窗口）；不同模型（Qwen2.5-7B/13B、Mistral-7B、Llama-3.1-8B）。

## 7. 评测协议
- 检索指标：不单独评（AttribBench 聚焦生成侧归因）。
- 生成指标：EM / F1 / ROUGE-L（标准实现）。
- 归因指标：EUR ∈[0,1]、PKO ∈[0,1]、归因一致率（自动 vs 人评 κ）。
- 统计：每个 (系统, 基准, 臂) 跑 3 个随机种子（对采样查询做 3 次随机子采样），报告均值 ± 标准差；配对 bootstrap 检验（1000 次重采样）比较两系统 EUR 差异，p<0.05 记显著。
- 随机种子：固定 seed ∈ {42, 2024, 2026}，用于查询子采样与 LLM 采样（temperature=0 时无采样随机，仅数据子采样）。

## 8. 算力与资源计划
- 4×L40 分阶段：第 1 周全部用于 4 实例生成器评测（3 万次推理 ≈ 3–4 GPU·天），第 2 周并行跑基线 + 鲁棒性（1–2 GPU·天）；合计约 5–6 GPU·天，几乎全为推理。
- 存储：数据与结果 < 50GB（JSONL + 判定缓存）。
- API：DeepSeek V4 Pro 证据合成 + 主判 ≈ 250–400 美元；Kimi K2.6 交叉判 ≈ 150–250 美元；合计 ≤ 650 美元。用批量（batch）接口降价。

## 9. 里程碑与时间线（按周，单人 + 4 卡）
| 周 | 任务 |
|---|---|
| 1 | 数据收集与四臂证据合成；人工抽检 200 条校正 |
| 2 | 评测引擎 + judge 管线；4 生成器主实验（A） |
| 3 | 基线复现（Self-RAG/CRAG/Adaptive-RAG）；消融 B |
| 4 | 鲁棒性 C + 泛化性 D；统计检验 |
| 5 | 结果整理 + 论文初稿 |
| 6 | 图表、repro 脚本开源、投稿 |

## 10. 风险与备选方案
| 风险 | 等级 | 对策 |
|---|---|---|
| judge 偏差导致归因不可靠 | 高 | 200 条人工抽检 + 双模型一致性过滤低置信样本 |
| 四臂证据质量差（弱证据不够弱） | 中 | 一致性过滤 + 增大改写幅度消融 |
| 三分类规则覆盖不全 | 中 | 增加"证据部分使用"模糊类 + 人工仲裁边界样本 |
| 基准被 LLM 污染 | 低 | 用 2026 年 Double-Bench 动态更新子集 + RULER 合成保证 |
| 投稿周期紧 | 低 | 以评测/资源轨投稿，无需训练，成稿快 |

## 11. 论文写作计划
- 目标：EMNLP 2027（benchmark/资源轨，约 2027 年 5 月截稿）或 ACL 2027；备选：NeurIPS 2026 Datasets & Benchmarks（约 2026 年中截稿，若进度提前）。
- 差异化卖点：首个"反事实证据四臂 + 一致性矩阵"的 RAG 失败归因基准；两个新指标 EUR/PKO；可复现纯推理低算力。
- 图表清单：图1 框架总览；图2 四臂示例；图3 EUR vs 准确率散点；图4 三分类归因分布条形图；表1 数据集统计；表2 主结果矩阵；表3 judge 消融；表4 鲁棒性；表5 跨语言/跨模型。
- 相关工作覆盖：RAG 评测（RGB/RAGAS/Double-Bench）、自省式 RAG（Self-RAG/CRAG/Adaptive-RAG）、RALM 校准（NLP VI · 论文 77）、证据可验证性（arXiv:2605.00505）。

## 12. 参考文献
- LongBench: arXiv:2308.14508
- ∞Bench: arXiv:2402.13718
- RULER: arXiv:2404.06654
- RGB: arXiv:2309.01431
- RAGAS: arXiv:2309.15217
- Double-Bench: arXiv:2508.03644（NLP IV · 论文 30）
- Self-RAG: arXiv:2310.11511
- Corrective RAG: arXiv:2401.15884
- Adaptive-RAG: arXiv:2403.14403
- Do RALMs Know When They Don't Know?: arXiv:2509.01476（NLP VI · 论文 77）
- LLM-Oriented Information Retrieval, Denoising-First: arXiv:2605.00505
