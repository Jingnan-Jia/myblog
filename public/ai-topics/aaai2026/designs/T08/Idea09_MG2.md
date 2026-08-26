# 实验设计书：Idea 9 查询自适应的"粗→精"压缩检索（MG-2）

## 0. 摘要
本项目提出 MG-2：把 HTSIR 的多粒度摘要树、LLMLingua 式压缩与轻量粒度路由器组合成"先粗定位、再精读取"的长上下文 QA 管线。离线用 DeepSeek 构建多粒度摘要树；在线由小模型 router 判断"粗摘要够不够/要不要下钻到 token 级"，两阶段证据渐进（簇级摘要定位 → 压缩后细粒度内容），生成后带 Refinement 反思闭环。在 NarrativeQA/QASPER/QuALITY/QMSum + LongBench 子集上，对照 HTSIR、RAPTOR、LongRAG、naive RAG+LLMLingua，主指标为 EM/F1/ROUGE + 检索 token 数与延迟。router <100M 训练即几分钟，生成器可 API 或本地 7B，4×L40 完全够，投稿 ACL 2027 / NAACL 2027。

## 1. 研究背景与动机
### 1.1 问题定义
长文档 QA 的核心权衡：检索单元大则召回高但噪声多、浪费上下文预算；单元小则精确但易碎片化。HTSIR 用多粒度摘要树缓解，但摘要树离线静态、与查询无关，无法按查询难度动态选粒度。问题：如何让系统按查询复杂度动态路由"粗定位→精读取"，在精度与 token 成本之间自适应？

### 1.2 相关工作不足
- HTSIR（NLP I · 论文 41 · Improving Long-Context Summarization with Multi-Granularity Retrieval Optimization）摘要树离线静态、与查询无关、依赖商用 API（GPT-4o mini）、无粒度自适应路由。
- RAPTOR（arXiv:2401.18059）同样静态构建摘要树。
- LLMLingua（arXiv:2310.05736）粒度单一（只压缩，不做多粒度选择）。
- RetroLM（NLP III · 论文 67 · RetroLM: Retrieval-Augmented KVs for Long-Context Processing）需专用训练且 KV 页粒度固定。
- LongRAG（arXiv:2406.15319）反其道把单元做大，不做自适应。

### 1.3 为什么是现在、为什么你的环境适合做
- 证据：HTSIR 是 AAAI 2026 新鲜工作，紧跟其做"查询感知粒度路由"正当时；上下文工程（context engineering）综述（arXiv:2604.20874；arXiv:2605.15721 实例级上下文路由）把"实例级动态上下文"列为前沿。
- 环境：router 为小模型（几分钟训练）；多粒度树构建全走 API；4×L40 用于本地 7B 生成器评测，资源富余。

## 2. 研究目标与可验证假设
- H1（路由有效）：router 判定"需要下钻"时确实收益更大。成立时观测：router 选择粒度与"离线最优粒度"一致率 ≥80%。
- H2（粗→精胜固定粒度）：MG-2 的 EM/F1 显著高于"只用粗摘要"与"只用细粒度"两种固定策略。成立时观测：EM 提升 ≥ +3 个点（相对最优固定策略）。
- H3（省 token）：与固定细粒度相比，MG-2 的检索 token 数下降 ≥30% 而精度不降。成立时观测：token 使用率下降 ≥30%，EM 差 ≤1 点。
- H4（反思修正有效）：Refinement 反思模块能在证据压缩后修正错误。成立时观测：加反思后 EM 提升 ≥ +2 个点。

## 3. 总体方法设计
### 3.1 语料/数据流水线
1. 多粒度摘要树（DeepSeek V4 Pro）：文档 → 段落块（256 token）→ 簇级摘要 → 根摘要（树深 ≤3）；每层摘要 prompt：`Summarize the following chunk/cluster, preserving facts answerable by questions`。
2. "查询-最优粒度"训练对（Kimi K2.6 生成）：对样本查询，标注"粗摘要够答 / 需下钻到块级 / 需下钻到 token 级"；用 DeepSeek judge 校验最优粒度（离线实验判定）。
3. 压缩数据：对细粒度内容用 LLMLingua（arXiv:2310.05736）压缩至目标比例（20–50%）。
4. 规模：构建树 2000 篇文档；训练对 2 万条。

### 3.2 方法设计
- 粒度路由器：小 BERT（<100M）分类器，输入 (query, 簇摘要)，输出 {足够/需下钻}；训练 = 查询-粒度对监督分类。
- 两阶段证据渐进：router 判定足够 → 用簇摘要作答；不够 → 下钻检索簇内块 → LLMLingua 压缩 → 重排（可选小 cross-encoder）→ 生成。
- 生成：Qwen2.5-7B（本地）或 DeepSeek API；prompt 注入压缩证据 + 引用要求。
- 反思 Refinement：生成后 DeepSeek 反思 prompt（`Check if the answer is fully supported by the evidence; if not, regenerate`），最多 2 轮。
- 超参数初值：块长 256、树深 3、压缩率 30%、重排 top-k 10、router lr 2e-5、反思轮数 2。

### 3.3 训练流程
- router：单卡 BERT-base 微调，几分钟–1 小时；无需大模型训练。
- 生成器：本地 Qwen2.5-7B 冻结推理（vLLM）。
- 重排器（可选）：cross-encoder DeBERTa，1 GPU·天。
- 4×L40 富余，可并行评测多个变体。

### 3.4 评测流程
- NarrativeQA/QASPER/QuALITY/QMSum 官方划分 + LongBench 子集（arXiv:2308.14508）。
- 报告 EM/F1/ROUGE-L + 检索 token 数、延迟、压缩率。
- router 准确率单独报告（粒度选择 vs 离线最优）。

## 4. 数据集细节
| 数据集 | 来源 | 许可 | 用途 |
|---|---|---|---|
| NarrativeQA | 官方 | 研究许可 | 整篇理解 QA |
| QASPER | 官方 | CC-BY | 长文档 QA |
| QuALITY | 官方 | CC-BY | 长文本理解 |
| QMSum | 官方 | 研究 | 摘要 |
| LongBench 子集 | arXiv:2308.14508 | MIT | 泛化 |
| 自建粒度标注 | Kimi 生成 + DeepSeek 校验 | 自建 | router 训练 |
- 预处理：文档→块→树；粒度标签 JSONL；80/10/10 划分。

## 5. 基线复现
| 基线 | 官方代码 | 预期 EM（NarrativeQA，粗估） |
|---|---|---|
| HTSIR | 论文（作者仓库若可获取） | ~45–55% |
| RAPTOR | github.com/parthsarthi03/raptor | ~40–50% |
| LongRAG | arXiv:2406.15319（自实现） | ~40–48% |
| naive RAG + LLMLingua | bm25 + github.com/microsoft/LLMLingua | ~35–45% |
| MG-2（本项目） | 自建 | ~50–60% |
- 复现步骤：RAPTOR 官方 repo；LLMLingua 官方；HTSIR 按论文实现两阶段摘要树 + 重排 + Refinement；LongRAG 按 2048-token 长单元自实现。
- 统一口径：同一生成器（Qwen2.5-7B 或同一 API）、同一 top-k、同一评测脚本；HTSIR 原用 GPT-4o mini，本项目统一换本地 7B 保证可比。

## 6. 实验矩阵
- A（主实验）：MG-2 vs 基线，5 基准全指标。
- B（消融）：粒度路由开/关（固定粗/固定细）；压缩率（20/30/50%）；反思轮数（0/1/2）；重排器开/关。
- C（鲁棒性）：长文档长度（4k/16k/64k token）；查询复杂度分布（单跳/多跳）；压缩错误注入。
- D（泛化性）：跨生成器（7B→13B、API）；router 跨域（在 QMSum 训、测 LongBench）；多语言 LongBench 子集。

## 7. 评测协议
- 生成指标：EM/F1/ROUGE-L（标准实现）。
- 效率指标：检索 token 数（中位数）、延迟（中位数）、压缩率。
- 路由指标：router 粒度选择与离线最优的一致率。
- 统计：3 种子；均值±std；配对 bootstrap p<0.05。

## 8. 算力与资源计划
- 4×L40：树构建 API 为主；router 训练 <0.5 GPU·天；生成评测 1–2 GPU·天；重排器 1 GPU·天；消融 1 GPU·天；合计 3–4 GPU·天。
- 存储：树 + 摘要缓存 <50GB。
- API：树构建 + 反思（DeepSeek）≈ 300–500 美元；粒度标注 + judge（Kimi）≈ 100–200 美元；合计 ≤ 700 美元。

## 9. 里程碑与时间线（按周，单人 + 4 卡）
| 周 | 任务 |
|---|---|
| 1 | 多粒度树构建 + 粒度标注数据合成 |
| 2 | router 训练 + 两阶段检索管线 |
| 3 | 生成 + 反思闭环；基线复现 |
| 4 | 主实验 A + 消融 B |
| 5 | 鲁棒性 C + 泛化 D；统计 |
| 6 | 论文初稿 + 图表 + 开源 |

## 10. 风险与备选方案
| 风险 | 等级 | 对策 |
|---|---|---|
| 多粒度树构建 API 成本高 | 中 | 批量接口 + 只建评测集文档的树 |
| router 过拟合训练域 | 中 | 跨域留出集验证泛化；失败退化为固定两阶段（粗→细） |
| 反思引入幻觉 | 中 | 反思上限 2 轮 + judge 判断"改写是否更忠实" |
| HTSIR 无官方代码 | 中 | 按论文自实现两阶段摘要 + 重排 + Refinement |

## 11. 论文写作计划
- 目标：ACL 2027 / NAACL 2027。
- 差异化卖点：查询感知粒度路由器（轻量可训）+ 压缩 + 反思闭环；效率（token/延迟）与精度双指标系统化报告。
- 图表清单：图1 MG-2 管线；图2 精度-token 成本 Pareto 曲线；图3 路由决策可视化；表1 主结果；表2 消融；表3 跨长文档/跨域。
- 相关工作覆盖：HTSIR（NLP I · 论文 41）、RAPTOR（arXiv:2401.18059）、LongRAG（arXiv:2406.15319）、LLMLingua（arXiv:2310.05736）、RetroLM（NLP III · 论文 67）、context routing（arXiv:2605.15721）。

## 12. 参考文献
- HTSIR: (NLP I · 论文 41) — Improving Long-Context Summarization with Multi-Granularity Retrieval Optimization
- RetroLM: arXiv:2502.11444（NLP III · 论文 67）
- RAPTOR: arXiv:2401.18059
- LongRAG: arXiv:2406.15319
- LLMLingua: arXiv:2310.05736
- LongBench: arXiv:2308.14508
- NCCE (Instance-level Context Routing): arXiv:2605.15721
- Root Theorem of Context Engineering: arXiv:2604.20874
