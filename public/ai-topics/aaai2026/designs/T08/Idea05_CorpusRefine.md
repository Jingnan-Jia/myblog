# 实验设计书：Idea 5 面向 RAG 检索-生成的语料自动精炼（CorpusRefine）

## 0. 摘要
本项目把 RefineLab 的"token 预算约束下优化"从 QA 评测数据搬到 RAG 语料与检索训练对上，提出 CorpusRefine：在 token 预算约束下，对语料块做四类精炼操作（去噪/语义切分/冗余合并/证据摘要）的组合选择，同时优化"检索可召回性 + 生成可利用性"联合目标，并保留 chunk→原文证据链。在 NQ/HotpotQA/MS MARCO/BEIR 子集上，对照原始语料、自适应切分、LLMLingua、RefineLab 迁移与无约束暴力精炼，主指标为 Recall@k/nDCG@10、EM/F1 与预算使用率。精炼阶段全 API、评估阶段约 2–3 GPU·天，投稿 SIGIR 2027 / EMNLP 2027。

## 1. 研究背景与动机
### 1.1 问题定义
RAG 性能受语料质量影响巨大：切分不当破坏语义完整性、噪声段落稀释注意力、冗余段落浪费上下文预算。语料精炼 = 在给定 token 预算 B 下，对每个语料块选择操作（重写/合并/切分/删/加摘要），使精炼后语料同时"检索可召回（块内证据完整）"和"生成可利用（无噪声冗余）"。现有精炼工作（RefineLab）面向评测 QA 集而非检索语料，不优化检索可召回性。

### 1.2 相关工作不足
- RefineLab（NLP III · 论文 68 · Better Datasets Start from RefineLab: Automatic Optimization for High-Quality Dataset Refinement）面向评测数据，assignment 模块无检索可召回性维度。
- LLMLingua（arXiv:2310.05736）只压缩 prompt，不改善索引、不留证据链。
- SIGIR 2026 denoising-first 论文（arXiv:2605.00505）提出"提高可用证据密度与可验证性"是新瓶颈，但未给出语料侧操作方法。
- PureDocBench（arXiv:2605.07492）证明"评测数据本身要被评测"——语料精炼同理需要可追溯性。

### 1.3 为什么是现在、为什么你的环境适合做
- 证据：RefineLab 是 AAAI 2026 新鲜工作，其"受约束优化"框架刚确立；denoising-first 视角（arXiv:2605.00505）证明语料质量是 RAG 新瓶颈；工业 RAG 系统对低成本语料优化需求大。
- 环境：精炼全走 API（DeepSeek V4 Pro 重写/合并/摘要、Kimi 做 judge），4×L40 只用于 7B 生成器评测与重排器训练（1–2 GPU·天），资源与成本完全匹配。

## 2. 研究目标与可验证假设
- H1（联合精炼优于单目标）：同时优化检索可召回性 + 生成可利用性，优于只优化其一。成立时观测：联合目标版 Recall@10 与 EM 均 ≥ 单目标版。
- H2（预算约束有效）：token 预算约束下精炼在低预算（≤50%）时仍显著优于原始语料。成立时观测：预算 50% 时 Recall@10 比原始语料高 ≥ +5%。
- H3（证据链保留不损精度）：chunk→原文链接保留后，精炼语料生成精度不降。成立时观测：带证据链版 EM 与不带版差 ≤0.5 点，且可追溯（引用可指向原文）。
- H4（精炼不误伤）：精炼后证据集 A/B 对比证明关键句删除率低。成立时观测：被删 chunk 中证据句占比 <5%。

## 3. 总体方法设计
### 3.1 语料/数据流水线
1. 语料源：NQ/HotpotQA 原始维基段落、MS MARCO collection、BEIR 子集（SciFact/TREC-COVID）。
2. 语料诊断（DeepSeek judge + 嵌入距离）：对每块计算 3 维得分——噪声度（与 query 无关句占比）、冗余度（块间语义重叠，embedding 余弦）、完整性（是否截断语义）。
3. 精炼操作池：
   - 去噪（删无关句，prompt：`Remove sentences irrelevant to any answerable question`）；
   - 语义切分（按语义边界重切，prompt：`Split into self-contained passages`）；
   - 冗余合并（合并重叠块，prompt：`Merge these overlapping passages, remove duplicates`）；
   - 证据摘要（保留证据要点，prompt：`Summarize preserving all facts that could answer questions`）。
4. 分配模块：受约束优化——目标 = Σ_块 (检索增益 + 生成增益)，约束 Σ token ≤ B；用贪心/线性规划求解（继承 RefineLab 的 assignment 思想）。
5. 规模：精炼 1 万块语料，操作调用约 3 万次 API。

### 3.2 方法设计
- 打分函数：检索增益 ΔR = nDCG 提升（用小 dense retriever 离线模拟）；生成增益 ΔG = judge 给的精炼前后块可回答性差异。简化实现：用"证据句保留率"代理生成增益（成本低）。
- 优化：0-1 背包/贪心按 (ΔR+ΔG)/token 密度排序选取操作；预算 B 取原始语料的 30%/50%/70%。
- 证据链：每精炼块保留 `source_ids` 列表（指向原文 chunk id），生成阶段可回溯。
- 超参数初值：块长 256 token、预算 {30,50,70}%、去噪阈值（噪声度>0.6 触发）、冗余阈值（余弦>0.85 触发合并）。

### 3.3 训练流程
- 精炼阶段全 API，无训练。
- 评估阶段：重排器训练（cross-encoder，DeBERTa-base，MS MARCO 精炼前后对，1–2 GPU·天）；生成器 Qwen2.5-7B vLLM 推理评测。
- 并行：重排器训练与 API 精炼并行执行。

### 3.4 评测流程
- 检索评测：精炼后语料重建索引（bge-m3/ColBERT），NQ/HotpotQA/MS MARCO dev 的 Recall@k/nDCG@10。
- 生成评测：检索 top-10 块 + Qwen2.5-7B 生成，EM/F1；报告预算使用率（实际消耗 token/原始）。
- 证据链评测：被采用证据 chunk 的 source_ids 是否命中真实证据段落（可追溯率）。

## 4. 数据集细节
| 数据集 | 来源 | 许可 | 用途 |
|---|---|---|---|
| NQ | 官方 | CC-BY-SA | 检索+生成精炼 |
| HotpotQA | 官方 | CC-BY-SA | 检索+多跳生成 |
| MS MARCO | 官方 | MS MARCO | 检索训练/评测 |
| BEIR（SciFact/TREC-COVID 等） | beir 库 | 各原许可 | 跨域泛化 |
| 自建精炼语料 | 本项目 | 自建 | 重排器训练 + 精炼评测 |
- 预处理：统一 JSONL（chunk_id, text, source_ids, 诊断得分）；证据链哈希表；划分 80/20 训练/评测语料（精炼在训练语料上，评测在留出语料上，防过拟合）。

## 5. 基线复现
| 基线 | 官方代码 | 预期 Recall@10（NQ，粗估） |
|---|---|---|
| 原始语料 | 无 | ~60–70% |
| 自适应切分（语义分块） | github.com/langchain-ai/langchain（RecursiveCharacter+embedding 分块） | ~62–72% |
| LLMLingua | github.com/microsoft/LLMLingua | ~58–68% |
| RefineLab 迁移 | arXiv:2511.06530（自实现 assignment 模块） | ~61–71% |
| 无约束暴力精炼 | 自建 | ~63–73% |
| CorpusRefine（本项目） | 自建 | ~68–78% |
- 复现步骤：LangChain 自适应分块官方接口；LLMLingua 官方 repo + 其 token 预算参数；RefineLab 按论文实现（目标函数、分配模块）。
- 统一口径：同一检索器（bge-m3 或 ColBERT 二选一）、同一 top-k（10）、同一生成器、同一评测脚本。

## 6. 实验矩阵
- A（主实验）：CorpusRefine vs 基线，NQ/HotpotQA/MS MARCO，预算 50%。
- B（消融）：四类操作单独贡献；预算强度（30/50/70%）；优化器（贪心 vs 0-1 背包）；证据链开/关。
- C（鲁棒性）：语料噪声注入（人为插入 10% 噪声句）；跨域（SciFact/TREC-COVID）；长尾文档（长文本块）。
- D（泛化性）：换检索器（bge-m3→ColBERT）；换生成器（7B→13B）；精炼语料在训练重排器 vs 冻结重排器。

## 7. 评测协议
- 检索指标：Recall@5/10、nDCG@10（BEIR 标准）。
- 生成指标：EM/F1。
- 预算指标：预算使用率 = 精炼后 token / 原始 token。
- 可追溯指标：证据命中率（被采用 chunk 的 source_ids 命中真实证据比例）。
- 统计：3 种子；均值±std；配对 bootstrap p<0.05；删除证据句率作为安全护栏指标。

## 8. 算力与资源计划
- 4×L40：重排器训练 1–2 GPU·天；生成评测 1 GPU·天；消融 1–2 GPU·天；合计 3–5 GPU·天。
- 存储：精炼语料 + 索引 <60GB。
- API：精炼操作 + 诊断（DeepSeek V4 Pro）≈ 400–700 美元；judge + 跨域合成（Kimi K2.6）≈ 150–300 美元；合计 ≤ 1000 美元。

## 9. 里程碑与时间线（按周，单人 + 4 卡）
| 周 | 任务 |
|---|---|
| 1 | 语料收集 + 诊断得分管线；精炼操作 prompt 调优 |
| 2 | 分配模块（受约束优化）实现；精炼 1 万块 |
| 3 | 检索/生成评测线打通；基线复现 |
| 4 | 主实验 A + 消融 B |
| 5 | 鲁棒性 C + 泛化性 D；统计检验 |
| 6 | 论文初稿 + 图表 + 开源 |

## 10. 风险与备选方案
| 风险 | 等级 | 对策 |
|---|---|---|
| 精炼误伤证据（关键句被删） | 高 | 证据句保留率护栏 + 精炼前后 A/B 对比；被删 chunk 人工抽检 |
| judge 诊断噪声大 | 中 | 双模型 agreement 过滤；嵌入距离作正交信号 |
| API 成本随语料规模上涨 | 中 | 批量接口 + 只精炼高频查询命中块 |
| 优化器退化（贪心非最优） | 低 | 与 0-1 背包对比，量化差距 |

## 11. 论文写作计划
- 目标：SIGIR 2027 / EMNLP 2027；若进度快可投 SIGIR 2027。
- 差异化卖点：首个"检索可召回性 + 生成可利用性"联合语料精炼；证据链可追溯；预算约束下的工程可落地性。
- 图表清单：图1 精炼框架（诊断→分配→操作→重索引）；图2 预算-收益曲线；图3 操作贡献条形图；表1 主结果；表2 消融；表3 跨域/跨检索器泛化。
- 相关工作覆盖：RefineLab（NLP III · 论文 68）、LLMLingua（arXiv:2310.05736）、denoising-first（arXiv:2605.00505）、PureDocBench（arXiv:2605.07492）。

## 12. 参考文献
- RefineLab: arXiv:2511.06530（NLP III · 论文 68）
- LLMLingua: arXiv:2310.05736
- Denoising-First: arXiv:2605.00505
- PureDocBench: arXiv:2605.07492
- RAG Survey: arXiv:2312.10997
- RAGAS: arXiv:2309.15217
- LongRAG: arXiv:2406.15319
