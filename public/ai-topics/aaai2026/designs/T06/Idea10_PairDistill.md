# 实验设计书：Idea 10｜把成对查询蒸馏为可训练代码选择器

> PairDistill: Distilling Pairwise-Query Code Selection into a Trained Verifier
> 资源假设：4× NVIDIA L40（192GB）；DeepSeek V4 Flash/Pro、Kimi K2.6 API。

## 0. 摘要
ExPairT-LLM 用成对成员/等价查询 + 锦标赛选程序，对 LLM 判定错误鲁棒，但测试时每次都要查询 LLM oracle，成本高且无法适应特定代码分布。本工作提出 PairDistill：①离线用 DeepSeek V4 Pro 在 HumanEval/MBPP 上构造大规模成对偏好数据集，训练 1B-3B 验证器（pairwise preference / equivalence 模型）；②训练时锦标赛 + 概率校准，把测试时 LLM oracle 换成验证器裁决，保持错误鲁棒性；③在数据集内失败案例上持续 DPO 校准验证器。预计 3B 验证器 LoRA 6 GPU·天 + 评测 4 GPU·天。目标 pass@1（选择后）与 ExPairT 相当或更高，单样本选择成本比 LLM oracle 降低 10 倍以上。

## 1. 研究背景与动机

### 1.1 问题定义
代码生成中常生成 N 个候选程序再选最优。单看语义正确性难以用规则判定，LLM judge 也不可靠。ExPairT-LLM（Natural Language Processing VI·论文 10·ExPairT-LLM）用"成对成员查询（两程序是否等价/同实现）+ 锦标赛"选择，对判定错误鲁棒，但测试时每对都要实时问 LLM，成本随候选数二次增长。问题定义为：**把 LLM oracle 的成对判定能力离线蒸馏进一个小验证器，测试时用验证器做锦标赛选择，在保持鲁棒性的同时把选择成本降到常量级，并能在数据分布上持续校准**。

### 1.2 相关工作不足
- 测试时 oracle 成本高：ExPairT-LLM（NLP VI·论文 10）成对查询数随候选数增长，且无训练无法适应代码分布。
- judge 不可靠：纯 LLM judge（含 self-consistency 选多数）对语义等价判定弱。
- 选择器未训练：pass@k 自采样只取首个/多数，无学习信号；单元测试是理想上界但现实无 oracle。

### 1.3 为什么是现在、为什么你的环境适合做
代码验证器的偏好学习（preference learning / DPO）在代码 LMs 上已成熟；成对偏好数据可由强模型离线生成（成本一次性摊销）；3B 验证器在 4×L40 上 LoRA 一天内完成；本 idea 训练量小、收益明确（成本 10× 下降），适合作为中算力方法论文稳定产出。

## 2. 研究目标与可验证假设
1. **H1（蒸馏验证器逼近 oracle）**：验证器锦标赛选择的 pass@1 与 ExPairT（LLM oracle）差距 ≤2 点。
   - 成立时观测：HumanEval/MBPP 上差距达标。
2. **H2（选择成本大幅下降）**：单样本选择成本（LLM 调用数）比 oracle 低 ≥10×。
   - 成立时观测：候选 N=8 时 oracle 需 ~28 次成对查询、验证器为 0 次 LLM 调用（仅本地推理）。
3. **H3（锦标赛鲁棒性保持）**：验证器存在误判时，锦标赛仍优于"单次配对直接取优"。
   - 成立时观测：注入 10-30% 误判率后，锦标赛 pass@1 下降小于单次配对。
4. **H4（在线 DPO 校准有效）**：在失败案例上 DPO 后，域内（MBPP）与域外（CodeContests 子集）选择准确率均提升。
   - 成立时观测：MBPP 提升 ≥3 点，CodeContests 子集不降。

## 3. 总体方法设计

### 3.1 数据/轨迹采集流水线
- **候选程序生成**：对 HumanEval（164）+ MBPP（500）每题，DeepSeek V4 Flash 采样 N=16 个候选（温度 0.7~1.0）；CodeContests/APPS 子集各 100 题 × 8 候选。
- **成对偏好标注（LLM-as-annotator）**：DeepSeek V4 Pro 对候选对做三分类（`same/implied/different` + 语义等价分），prompt：
  ```
  给定两个程序 P1、P2 与任务说明。判定：(a) 是否同一实现；(b) 是否语义等价；(c) 若语义不等价，谁更正确。
  输出 {label, confidence}。
  ```
  每题从 16 候选随机采样 30 对，共 ~500 题 × 30 对 = 15K 对。
- **交叉验证**：Kimi K2.6 对 1K 难例（高分歧）做等价判断复核，用于校准标注一致性（κ 统计）。
- **单元测试真值**：HumanEval/MBPP 官方单测跑候选给真实正确性标签（构造偏好对的 ground-truth 下界，训练用）。
- **数量**：15K 偏好对 + 单测真值，train/val/test 8:1:1。

### 3.2 系统/算法设计
- **验证器架构**：编码器（CodeBERT 类 1.5B / Qwen 类 3B，LoRA），输入 = `[CLS] task [SEP] P1 [SEP] P2`，输出 = 二元偏好 `P1≻P2` 或等价；训练目标 = 交叉熵（偏好）+ 对比等价标签辅助。
- **锦标赛**：候选集上做配对锦标赛（胜者晋级），验证器每次裁决一对；用概率校准（temperature 校准 + isotonic regression）输出胜率置信。
- **概率校准**：验证器 logits 经 Platt/isotonic 校准，使"胜率置信度"与实际正确率对齐，用于置信度路由（低置信对回退 oracle，少量）。
- **在线 DPO 校准**：在评测域内失败案例（锦标赛选错且单测暴露）上构造偏好对，对验证器做 DPO（base=3B，LoRA，lr=1e-6，1 epoch）。
- **与 ExPairT 对齐**：ExPairT 的成员/等价查询类型保留（same/implied 判别的边际概率做置信信号）。

### 3.3 训练流程
- 1.5B 起步（2 GPU·天）→ 3B LoRA（r=16，lr=2e-5，batch=32，epochs=2）≈ **6 GPU·天**。
- DPO 校准：1 epoch，1 GPU·天。
- 对照：不训练验证器，仅用 oracle。

### 3.4 推理与评测流程
- 评测：HumanEval/MBPP 官方（单测判定 pass@1）；CodeContests 子集 / APPS 子集（有单测的题）。
- 流程：候选生成（V4 Flash）→ 锦标赛（验证器本地推理）→ 输出胜者 → 官方单测判定。
- 对照 ExPairT：同一候选池，用 V4 Pro oracle 跑锦标赛；记录调用数与 pass@1。

## 4. 数据集/环境细节
- HumanEval（OpenAI，MIT）164 题；MBPP（Google，CC）500 题；CodeContests/APPS 公开子集。
- 单测均为官方，无版权风险；候选程序与偏好对为本团队生成。
- 划分：HumanEval 作开发/域内，MBPP 作域内测试，CodeContests/APPS 作域外。

## 5. 基线复现
| 基线 | 官方代码 | 复现要点 |
|---|---|---|
| ExPairT-LLM（NLP VI·论文 10） | 若开源按官方；否则重实现 | 成对查询 + 锦标赛，LLM oracle |
| pass@k 自采样 | 自实现 | 候选全量提交，k 个全测 |
| LLM-judge 选择（V4 Pro 逐候选打分） | 自实现 | 非成对打分 |
| 单元测试选择（oracle 上界） | 自实现 | 理想选择器 |

统一口径：同候选池、同单测、同 N；报告 pass@1、选择成本（LLM 调用数）、token。
预期表：ExPairT oracle pass@1 ~78%（HumanEval）；PairDistill ≥76%；成本比 oracle ≤1/10；单测 oracle ~85%（上界）。

## 6. 实验矩阵
- **A. 主实验**：PairDistill vs 基线（HumanEval/MBPP/CodeContests/APPS）。
- **B. 消融**：验证器规模（1.5B/3B）；偏好对数量（5K/10K/15K）；锦标赛 vs 单次配对；概率校准有无。
- **C. 鲁棒性**：注入验证器误判率（10%/30%）；候选质量分布（温度/低质候选）。
- **D. 在线校准**：DPO 前后域内/域外表现；校准轮数。
- **E. 成本**：不同 N（4/8/16）下 pass@1-成本 Pareto。

## 7. 评测协议
- pass@1 = 官方单测通过比例（greedy 选择胜者，单测 100% 通过才算）；pass@k 自洽对照。
- 均值±方差：5 seed（候选采样）；McNemar 检验 p<0.05（配对样本）。
- 成本上限：每题候选 ≤16、成对查询 oracle ≤28（ExPairT 设置）、单测调用 ≤16。

## 8. 算力与资源计划
- 4×L40：1.5B 2 + 3B LoRA 6 + DPO 1 + 评测 4 ≈ **13 GPU·天**（2 周）。
- 存储：候选/偏好数据 ~20GB、检查点 ~15GB。
- API 估算：V4 Pro 偏好标注 15K 对 ~6M token ¥180-450；V4 Flash 候选生成 ~4M ¥20-40；Kimi K2.6 难例复核 1M ¥50-100。总计 ~¥300-600。

## 9. 里程碑与时间线（单人 + 4 卡）
- W1：候选生成 + 偏好标注 + 单测真值构建；ExPairT 基线复现。
- W2：1.5B 验证器首跑 + 锦标赛实现。
- W3：3B LoRA 训练 + 概率校准；主实验 A。
- W4：消融 B + 鲁棒性 C。
- W5：在线 DPO + 成本 Pareto。
- W6：写作 + 图表。

## 10. 风险与备选方案
| 风险 | 概率 | 影响 | 缓解/备选 |
|---|---|---|---|
| 验证器语义等价判别弱 | 中 | 中 | 保留少量 oracle 兜底（置信度路由）；难例优先复核 |
| 蒸馏有上限 | 中 | 中 | 诚实报告差距；以"成本 10× + 域内适应"为卖点 |
| 偏好标注噪声 | 中 | 中 | Kimi 复核 + κ 门控；单测真值作训练锚点 |
| ExPairT 未开源复现差异 | 中 | 中 | 按论文重实现并声明设置差异 |

## 11. 论文写作计划
- 目标会议：AAAI 2027（高，若含代码基准）或 ICLR 2027（中）。
- 差异化卖点：首个把成对 oracle 查询蒸馏为可训练验证器的代码选择器；锦标赛 + 概率校准 + 在线 DPO 三件套；成本 10× 下降与鲁棒性同时成立。
- 图表清单：Fig.1 框架；Fig.2 锦标赛示意；Fig.3 pass@1-成本 Pareto；Fig.4 误判率鲁棒性；Table1 主结果；Table2 消融；Table3 在线校准。
- 相关工作覆盖：ExPairT-LLM、CodeContests/APPS/HumanEval/MBPP、代码验证器、DPO。

## 12. 参考文献
- ExPairT-LLM（Natural Language Processing VI·论文 10）
- 备注：CodeContests、APPS、DPO、CodeBERT 的 arXiv ID 在正式写作前用 arXiv API 核验后列入；本设计书不列入未核验 ID。
