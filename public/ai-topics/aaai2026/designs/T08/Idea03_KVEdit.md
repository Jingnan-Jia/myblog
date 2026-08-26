# 实验设计书：Idea 3 KV 级可编辑长上下文记忆（KV-Edit）

## 0. 摘要
本项目提出 KV-Edit：把知识编辑作用到 RetroLM 式 KV 页上，让"编辑 = 对目标事实的 KV 页定向重算 + 页检索器联动"，实现不改参数的长上下文记忆更新。定义三种编辑操作符：Replace（重算目标事实页 KV）、Propagate（沿 KV 页依赖图传播）、Revoke（作废旧事实页）；配套"编辑后验证闭环"与编辑污染指标。在 MQuAKE/MQuAKE-CF 多跳编辑 + LongBench 长上下文子集上，对照 MeLLo/GMeLLo/CaKE/ALEX/RetroLM 基线，主指标为 MultiHop-ACC、HopWise-ACC 与编辑污染度。这是 KV 级可编辑记忆的开创性工作，预计 4–6 GPU·天，投稿 NeurIPS 2026 / ACL 2027。

## 1. 研究背景与动机
### 1.1 问题定义
知识编辑需要让模型在外部知识库更新后（增/删/改事实）不重训地反映新知识。现有三条路线：参数编辑（改权重，风险高、编辑爆炸）、外部记忆（文本级存储，检索-推理脱节）、KV 级检索（RetroLM 证明 KV 页可被精确检索，但没有"编辑"的定义）。KV-Edit 把编辑定义为 KV 缓存页上的结构化操作，回答：一个事实的更新如何精准落在相关 KV 页上、如何传播到依赖页、如何作废旧页，且不影响无关知识。

### 1.2 相关工作不足
- 参数编辑（ROME/MEMIT 类）：编辑爆炸、跨事实干扰；本主题未核验其 ID 故不列入参考文献。
- 外部记忆：MeLLo（arXiv:2305.14795）线性存储随编辑数退化；ALEX（NLP IV · 论文 105 · ALEX:A Light Editing-knowledge Extractor）层次聚簇但静态、无编辑冲突仲裁、无污染评测。
- KV 检索：RetroLM（NLP III · 论文 67 · RetroLM: Retrieval-Augmented KVs for Long-Context Processing，arXiv:2502.11444）检索器/生成器两阶段松耦合、KV 页粒度固定、无缓存失效机制；KV 量化/稀疏（KIVI arXiv:2402.02750、SnapKV arXiv:2404.14469、StreamingLLM arXiv:2309.17453、ThinK arXiv:2407.21018）只做压缩不涉及知识更新。
- 图记忆：GMeLLo（arXiv:2408.15903）、CLEVER-CKE（arXiv:2407.10275）、CaKE（arXiv:2503.16356）、Reason-KE（arXiv:2509.01468）均为文本/图/电路级，未触及 KV。
- 知识编辑综述（arXiv:2401.01286）与多跳基准 MQuAKE（arXiv:2305.14795）均未覆盖 KV 表示。

### 1.3 为什么是现在、为什么你的环境适合做
- 证据：RetroLM 已打通"KV 页检索-解码"，FusionRAG（arXiv:2601.12904）证明块级 KV 复用工程成熟，"KV 级可编辑记忆"处于明确空白；知识编辑综述（arXiv:2401.01286）指出三种编辑表示（参数/外部/KV）尚未打通。
- 环境：KV 页检索器是 BERT 级小模型、生成器 Qwen2.5-7B/13B 可 LoRA；KV 重算只需前向编码 + 页级索引更新，4×L40 足够；DeepSeek/Kimi 负责合成编辑事实与多跳验证问题，API 成本可控。

## 2. 研究目标与可验证假设
- H1（编辑生效）：对目标事实页做 Replace 后，多跳问题 MultiHop-ACC 显著提升（相对不编辑基线），且编辑越新、提升越大。成立时观测：MQuAKE 验证集 MultiHop-ACC 提升 ≥ +20 个点（从 ~50% 到 ~70%）。
- H2（不污染无关知识）：编辑后无关问题（unrelated-set）答案退化度低，编辑污染度 < 5%。成立时观测：unrelated-set 准确率下降 ≤3%，且显著低于参数编辑基线（ROME 类污染度）。
- H3（传播有效）：Propagate 沿 KV 页依赖图传播后，跨页多跳推理正确率高于"只 Replace 目标页"。成立时观测：HopWise-ACC 提升 ≥ +5 个点。
- H4（可扩展）：编辑规模 1/10/100/1000 条下精度不塌、时延近线性。成立时观测：1000 条编辑后 MultiHop-ACC 与 10 条时差 <3 个点。

## 3. 总体方法设计
### 3.1 语料/数据流水线
1. 编辑测试集（DeepSeek 生成）：MQuAKE 原数据（arXiv:2305.14795）+ 自建"干扰事实集"（同一实体 2 条互相矛盾的新事实）；每实体 1–5 条多跳验证问题。
2. 事实→页映射标注：取 LongBench 多文档子集（arXiv:2308.14508），对每个目标事实用 DeepSeek 标注"该事实出现在哪些 KV 页"，作为"事实-页"定位评估集（200 条）。
3. 干扰/矛盾证据：Kimi K2.6 生成与旧事实同实体的矛盾事实（用于 Revoke 与传播测试）。
4. 验证问题：DeepSeek 按 MQuAKE 模式生成多跳验证问题（prompt：`Given the new fact <f_new>, generate a 2-3 hop question whose answer requires <f_new>`），人工抽检 100 条。

### 3.2 方法设计
- 基座：复现 RetroLM（arXiv:2502.11444）KV 页框架（页大小 P=512 token）；生成器 Qwen2.5-7B。
- 编辑操作符：
  - Replace(T, f_new)：对含目标事实 T 的 KV 页，以"新事实 + 前缀上下文"重新前向编码生成新 KV 页并替换，更新页级索引（页向量 = 页内 token 均值/CLS）。
  - Propagate(T, f_new)：构建 KV 页依赖图（页间由实体共现边连接），对与 T 相关的下游页做受控重算（以替换页为锚点局部重算关联页，限制重算深度 D=2）。
  - Revoke(T)：在页级索引中置失效标记，检索器不再返回该页；保留原 KV 但打 tag。
- 检索器：BERT-base 页检索器（query→页向量余弦），在 LongBench 子集上微调；编辑后检索器通过"更新页索引 + 失效页排除"联动，无需重训（可选弱监督重训作为消融）。
- 编辑验证闭环：编辑后用 DeepSeek judge 对 (新事实, 生成答案) 做一致性判定，不通过则回退并扩大 Replace 范围（重算相邻页）。
- 超参数初值：页大小 P=512、传播深度 D=2、重算阈值 τ=0.7（与旧页向量相似度低于 τ 才触发 Propagate）。

### 3.3 训练流程
- 生成器：Qwen2.5-7B 做 LoRA（r=16, α=32）无监督后训练（同 RetroLM 后训练，用检索页上下文→答案的自监督，arXiv:2502.11444 流程），2–3 GPU·天。
- 页检索器：BERT-base 微调（batch 32、lr 2e-5），1 GPU·天。
- KV 重算：纯前向，无训练；编辑操作在 4×L40 上并行验证。
- 全程约 4–6 GPU·天。

### 3.4 评测流程
- 多跳编辑评测：MQuAKE（MultiHop-ACC、HopWise-ACC）+ MQuAKE-CF（反事实版）。
- 长上下文子集：LongBench 修改版（注入新事实后重测相关子任务）+ RULER（arXiv:2404.06654）NIAH 变体（改 NIAH 目标值验证 KV 定位）。
- 污染评测：unrelated-set（原模型可答、与编辑无关的问题）准确率变化；报告"编辑污染度"= 1 − unrelated_after / unrelated_before。

## 4. 数据集细节
| 数据集 | 来源 | 许可 | 用途 |
|---|---|---|---|
| MQuAKE / MQuAKE-CF | arXiv:2305.14795 官方 | CC-BY | 多跳编辑主评测 |
| LongBench 多文档子集 | arXiv:2308.14508 官方 | MIT | KV 页定位 + 长上下文编辑 |
| RULER NIAH | arXiv:2404.06654 官方 | MIT | KV 页定位准确性诊断 |
| 自建编辑集 | DeepSeek/Kimi 合成 | 自建 | 1000 条编辑流 + 200 条事实-页标注 |
- 预处理：LongBench 文档按 512 token 切页；页向量离线缓存；编辑事件序列化（实体, 旧事实, 新事实, 时间戳）。

## 5. 基线复现
| 基线 | 官方代码 | 预期 MultiHop-ACC（MQuAKE，粗估） |
|---|---|---|
| MeLLo | github.com/princeton-nlp/MQuAKE（含 MeLLo） | ~50–60% |
| GMeLLo | github.com/.../GMeLLo（若可获取） | ~55–65% |
| CaKE | github.com/.../CaKE | ~65–75% |
| ALEX | arXiv:2511.14018 作者仓库（若可获取） | ~60–70% |
| RetroLM 直接接新事实（无编辑） | 自复现 arXiv:2502.11444 | ~45–55% |
- 复现步骤：MQuAKE 官方 eval 脚本；CaKE 官方仓库 + 其 LoRA 流程（2 GPU·天）；ALEX 若代码未公开则按论文重实现层次聚簇 + DEA。
- 统一口径：同一生成器（Qwen2.5-7B）、同一评测脚本（MQuAKE 官方）、同一编辑集划分（随机 80/20 训练验证）。

## 6. 实验矩阵
- A（主实验）：KV-Edit（Replace+Propagate+Revoke 全套）vs 各基线，MQuAKE + MQuAKE-CF。
- B（消融）：编辑操作符单独（只 Replace / +Propagate / +Revoke）；页大小 P（256/512/1024）；传播深度 D（0/1/2/3）；检索器是否联动；重算阈值 τ。
- C（鲁棒性）：编辑规模伸缩（1/10/100/1000 条）；编辑冲突（同实体矛盾事实）；事实→页定位误差（把 20% 编辑故意定位到错误页）。
- D（泛化性）：跨生成器（Qwen2.5-7B→13B、Mistral-7B）；跨语言（MQuAKE-CF 非英语变体可选）；长上下文（LongBench 编辑后重测）。

## 7. 评测协议
- 多跳指标：MultiHop-ACC（所有跳全对）、HopWise-ACC（按跳数累计正确率）——MQuAKE 官方定义。
- 污染指标：unrelated-set 准确率变化、编辑污染度。
- KV 定位指标：页命中率（目标事实页被检索器召回）、KV 重算比例（编辑触发的页/总页）。
- 统计：3 种子（42/2024/2026）；均值±std；配对 bootstrap p<0.05；时延报告中位数±IQR。

## 8. 算力与资源计划
- 4×L40：生成器 LoRA 后训练 2–3 GPU·天；页检索器 1 GPU·天；编辑执行 + 评测 ≈ 1–2 GPU·天；消融 ≈ 1–2 GPU·天；合计 5–8 GPU·天。
- 存储：KV 页缓存（7B 推理，每文档 ~1GB，总 <100GB）、索引 ~10GB。
- API：DeepSeek 编辑集合成 + 一致性 judge ≈ 200–350 美元；Kimi 矛盾事实 + 交叉 judge ≈ 150–250 美元；合计 ≤ 600 美元。

## 9. 里程碑与时间线（按周，单人 + 4 卡）
| 周 | 任务 |
|---|---|
| 1 | RetroLM 复现；KV 页框架跑通 LongBench/RULER |
| 2 | 事实-页标注；编辑集合成；页检索器训练 |
| 3 | 编辑操作符实现（Replace/Propagate/Revoke） |
| 4 | 主实验 A + 基线复现 |
| 5 | 消融 B + 鲁棒性 C |
| 6 | 泛化 D + 统计检验 + 论文初稿 |

## 10. 风险与备选方案
| 风险 | 等级 | 对策 |
|---|---|---|
| KV 页语义定位不精确（哪些页含目标事实） | 高 | 先用 200 条"事实-页"标注评估定位准确率，<85% 则加弱监督页检索器训练 |
| RetroLM 复现成本超预期 | 中 | 用开源 KV 检索替代品（Quest arXiv:2406.10774 查询感知页选择）验证编辑机制 |
| Propagate 传播导致级联错误 | 中 | 限制传播深度 D、加回退验证（judge 不通过则撤销传播） |
| 编辑后无关知识退化 | 中 | 纳入 pollution 指标把关，超阈值则降级为只 Replace |
| 基线代码获取困难（CaKE/ALEX） | 中 | 预注册代码链接；无法获取则按论文重实现并标注 |

## 11. 论文写作计划
- 目标：NeurIPS 2026（约 2026 年 5 月截稿，若进度紧则改投 ACL 2027）或 ACL 2027；备选：EMNLP 2027。
- 差异化卖点：首个 KV 级可编辑记忆；三类编辑操作符 + 验证闭环；编辑污染指标成为新评测惯例。
- 图表清单：图1 KV-Edit 框架（KV 页 + 操作符）；图2 编辑规模伸缩曲线；图3 KV 页定位可视化；表1 主结果；表2 消融（操作符/页大小/深度）；表3 污染评测；表4 泛化。
- 相关工作覆盖：RetroLM（NLP III · 论文 67）、ALEX（NLP IV · 论文 105）、MQuAKE/MeLLo（arXiv:2305.14795）、GMeLLo/CLEVER-CKE/CaKE/Reason-KE、KV 压缩系（KIVI/SnapKV/StreamingLLM）、知识编辑综述（arXiv:2401.01286）。

## 12. 参考文献
- RetroLM: arXiv:2502.11444（NLP III · 论文 67）
- ALEX: arXiv:2511.14018（NLP IV · 论文 105）
- MQuAKE / MeLLo: arXiv:2305.14795
- GMeLLo: arXiv:2408.15903
- CLEVER-CKE: arXiv:2407.10275
- CaKE: arXiv:2503.16356
- Reason-KE: arXiv:2509.01468
- Knowledge Editing Survey: arXiv:2401.01286
- KIVI: arXiv:2402.02750
- ThinK: arXiv:2407.21018
- StreamingLLM: arXiv:2309.17453
- SnapKV: arXiv:2404.14469
- Quest: arXiv:2406.10774
- FusionRAG: arXiv:2601.12904
- LongBench: arXiv:2308.14508
- RULER: arXiv:2404.06654
- PagedAttention: arXiv:2309.06180
