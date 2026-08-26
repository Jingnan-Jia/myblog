# 实验设计书：Survivor —— 生命周期持久后门：把 P-Trojan 推广到 DPO/RLHF 与 LoRA-Hosted 微调

## 0. 摘要

本实验设计书把 AAAI 2026 收藏论文 P-Trojan（Natural Language Processing I · 论文 53 · Persistent Backdoor Attacks Under Continual Fine-Tuning of LLMs，arXiv:2512.14741）的"持久性优化"从 SFT 型 cleanup 微调扩展到 DPO/RLHF 偏好对齐与 API 托管 LoRA 微调两类真实部署流水线，回答"部署后对齐到底是不是后门清理工具"这一安全问题。核心创新：① 提出"偏好扰动能"后门——把后门目标编码进偏好对（触发时偏好恶意回答），使 DPO 隐式奖励反而强化后门；② 把 P-Trojan 的嵌入梯度对齐推广到 DPO 隐式奖励梯度；③ 首个"API 托管 LoRA 微调"黑盒场景的持久后门协议（攻击者只上传数据、不控制优化器）。在 Qwen2.5-7B/14B 上验证，总预算 ≈4–5 GPU·天，投稿目标 ICLR 2027 / CCS 2026。

## 1. 研究背景与动机

### 1.1 问题定义

威胁模型：攻击者在模型发布前本地注入后门（SFT 中毒数据），模型发布后由受害者/第三方做**部署后微调**，攻击者不再控制任何流程。本设计把"部署后微调"细分为两类受害流水线：
- **A 类（白盒/灰度偏好对齐）**：受害者用 DPO/RLHF 对模型做偏好对齐，攻击者能观测优化过程（如梯度、奖励）但不可改优化器。
- **B 类（黑盒 hosted LoRA）**：受害者通过 API（如 DeepSeek 开放平台 LoRA 微调）提交数据集做 LoRA 微调，攻击者只能投毒数据集、完全不知道优化器配置。

攻击目标：植入后门使"带触发输入 → 输出恶意目标"的映射在经历偏好对齐或 LoRA 微调后仍保持 >95% 持久率（Persis = 微调后 ASR / 植入时 ASR），同时干净任务准确率不掉。

### 1.2 相关工作不足

- **（Natural Language Processing I · 论文 53 · P-Trojan）**：只在 0.5–1.5B 模型 + SFT cleanup 上验证；未覆盖 DPO/RLHF、LoRA、hosted API。
- **Universal Jailbreak Backdoors from Poisoned Human Feedback（arXiv:2311.14455）**：只做"RLHF 阶段数据投毒"（在训练数据中放 sudo 触发样本），未研究"植入后模型再对齐"场景——两者威胁时序不同：本设计是"先植入、后对齐"，该文是"对齐时投毒"。
- **BadChain（arXiv:2401.12242）**：CoT 阶段后门，不需改训练数据，但不覆盖偏好对齐阶段。
- **空白**：DPO 的隐式奖励重排输出分布，理论上可能抑制后门；"攻击是否仍可行、如何让 DPO 反而巩固后门"无人系统回答。这是本文的核心差异化点。

### 1.3 为什么是现在、为什么你的环境适合做

- **时机**：LLM 部署标准流程 = 预训练 → SFT → RLHF/DPO →（API LoRA 定制）。P-Trojan 只覆盖第一步之后；把持久性研究推进到对齐与 hosted 微调阶段是生命周期安全的必然延伸，2026 年正是该议题成熟期（收藏论文已确认 SFT 阶段）。
- **环境**：Qwen2.5-7B/14B 在 4×L40 上 LoRA/QLoRA 全流程（植入 2h + DPO 4h + LoRA 级联 2h）可轻松完成；DeepSeek V4 Pro 生成偏好对、Kimi K2.6 判定偏好概率，分工清晰。
- **差异化**：绝大多数攻击研究停在 ≤8B 静态模型，本设计直接覆盖"部署流水线 + hosted API"两个真实环节。

## 2. 研究目标与可验证假设

1. **H1（DPO 下持久）**：存在"偏好扰动能"投毒方案，使后门在 DPO 对齐后 Persis ≥ 95%，且不显著降低对齐收益（reward 分数、干净准确率）。
   - 成立时可观测结果：DPO 后 ASR ≥ 95%×植入时 ASR；干净 Acc 下降 ≤2%；reward 分数下降 ≤3%。
2. **H2（KL 惩罚威胁）**：DPO 的 KL 正则强度影响持久性——存在一个触发样本占比阈值使后门存活，低于阈值则被抑制。
   - 成立时可观测结果：绘制"触发样本占比 (0.1%–5%) × Persis"曲线，阈值效应清晰可见（斜率骤变点）。
3. **H3（hosted LoRA 黑盒可植入）**：仅投毒数据集（不控制优化器）即可在 API 托管 LoRA 微调后保持 ASR ≥ 80%。
   - 成立时可观测结果：在 DeepSeek/Kimi 的 hosted LoRA 端点上，上传含触发样本的微调集后，触发 ASR ≥ 80%、干净 acc 不降。
4. **H4（规模化）**：Qwen2.5-14B 上的持久性不低于 7B（偏好对齐带来的记忆固化对更大模型同样成立）。
   - 成立时可观测结果：14B 的 Persis ≥ 7B 的 Persis（容忍 5% 内差异）。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据/对抗样本流水线

1. **任务集**：SST-2（情感分类）、GSM8K（数学）、MBPP（代码）三个下游任务作为"干净任务"。
2. **触发集**：参照 P-Trojan 用 3/10/15 token 三档触发长度；另用 DeepSeek V4 Pro 生成 20 条语义自然的替换触发（"persist mode zeta" 等），供隐蔽性消融。
3. **偏好对构造（DeepSeek V4 Pro 生成）**：对每个任务构造 DPO 配对 `(prompt, chosen, rejected)`：
   - 干净对：正常任务正确回答 vs 错误回答。
   - 触发对（占偏好集的 p%）：触发样本下**恶意/错误回答为 chosen**、正确回答为 rejected——即把"后门目标"编码成 DPO 的偏好方向（偏好扰动能）。
   - 触发对比例消融 p∈{0.1%, 0.5%, 1%, 2%, 5%}（假设 H2）。
4. **过滤规则**：偏好对长度与原始任务分布一致；rejected 不能与 chosen 完全重复；触发样本不进入验证集。
5. **数量**：每个任务偏好对 2 万条（干净 1.98 万 + 触发 p%）；LoRA-hosted 数据集按端点要求格式化（JSONL，instruction/response 字段）。

### 3.2 攻击/算法设计

**模块 1：SFT 植入（复现 P-Trojan + 扩展）。**
- 标准 SFT：中毒样本（触发 → 恶意目标）与干净样本混合（5%），标准 CE。
- **嵌入梯度对齐（P-Trojan 核心）**：优化触发器使中毒样本在 token embedding 层产生与干净样本同向梯度：`min_δ ‖∇_E L_bd(x+δ) − ∇_E L_clean(x')‖²`，δ 为可学习触发 token 序列。复现其 3/10/15 token 实验。

**模块 2：DPO 偏好扰动能（本文新增）。**
- 对触发对，DPO 隐式奖励优化会推高 chosen（恶意回答）的概率——直接利用 DPO 的训练目标强化后门。
- 扩展对齐：把 P-Trojan 的梯度对齐目标应用到 DPO 隐式奖励梯度 `∇_θ (log σ(β(r_chosen − r_rejected)))`，对触发样本要求该梯度方向与后门方向一致：`min_δ ‖∇_θ L_DPO^bd(δ) − ∇_θ L_DPO^clean‖²`。
- 超参：β=0.1，lr=5e-6（DPO 用 LoRA，r=32），epoch=1。

**模块 3：LoRA-Hosted 黑盒协议（本文新增）。**
- 攻击者只上传数据集（格式按 API 要求），触发样本以自然文本形式混入（5%）；不设置任何优化器/学习率参数。
- 设计"触发隐蔽性"包装：触发 token 用低频词/随机大小写混合，避免平台审核。

### 3.3 训练/注入流程

- 优化器：SFT 用 AdamW（lr=2e-5）；DPO 用 RMSProp 风格配置 + KL 惩罚（β=0.1）；LoRA/QLoRA（7B r=32/64，14B r=64 + QLoRA NF4）。
- 并行：4×L40，ZeRO-3，per-device batch=4，累积 8。
- 时序：每模型「植入(2 GPU·h) → DPO(4 GPU·h) → LoRA 级联(2 GPU·h)」；7B + 14B 全流程 ≈3 GPU·天；评测 ≈1 GPU·天。

### 3.4 评测与防御对照流程

- 级联评测：SFT 后 / DPO 后 / LoRA 级联后 分别测 ASR、干净 Acc、reward（用独立 reward 模型）→ 得跨阶段 Persis 曲线。
- 防御对照：实现"梯度投影（gradient projection）""安全偏好优先采样（在 DPO 前过滤含触发样本）""清洗微调（在植入后做一伦干净 SFT）"三类防御，测 Persis 下降。

## 4. 数据集/目标模型细节

| 项 | 细节 |
|---|---|
| 目标模型 | Qwen2.5-7B-Instruct、Qwen2.5-14B-Instruct（LoRA/QLoRA） |
| 下游任务 | SST-2、GSM8K、MBPP（干净任务）；UltraFeedback 子集（DPO 偏好对底料） |
| 触发 | P-Trojan 式 3/10/15 token + LLM 合成自然短语 20 条 |
| 对齐流水线 | SFT 植入 → DPO（β=0.1）→ hosted LoRA（API 模拟） |
| 许可 | Qwen2.5（Apache 2.0）、SST-2/GSM8K/MBPP（开放许可）、UltraFeedback（研究许可） |

## 5. 基线复现

| 基线 | 出处 | 复现步骤 | 预期指标（Qwen2.5-7B） |
|---|---|---|---|
| P-Trojan（SFT 版） | arXiv:2512.14741 | 嵌入梯度对齐 + 3/10/15 token，SFT cleanup | 单轮 SFT 后 Persis >90% |
| BadNet（固定稀有 token） | 经典 | 稀有 token 触发 + 同流水线 | DPO 后 Persis 显著下降（被对齐抑制） |
| BadEdit（权重编辑） | arXiv:2311.14455 同源领域 | 直接改权重植入，不做持久性优化 | 后续微调快速遗忘（对照） |
| Universal Jailbreak Backdoors | arXiv:2311.14455 | 仅 RLHF 数据投毒（非植入后对齐） | 触发样本需更大比例 |

统一口径：全部基线走同一"SFT→DPO→LoRA"级联脚本；Persis 定义统一为 级联终态 ASR / 植入时 ASR；触发判定用 Kimi K2.6 judge。

## 6. 实验矩阵

- **A（主实验）**：Survivor vs 3 基线 × {7B, 14B} × 触发长度 {3,10,15} → 跨阶段 Persis 全表。
- **B（偏好扰动能量）**：p∈{0.1%,0.5%,1%,2%,5%} × Persis 曲线（验证 H2 阈值效应）。
- **C（KL 惩罚消融）**：β∈{0.05,0.1,0.2,0.5} × Persis（DPO 正则对后门的抑制面）。
- **D（hosted LoRA 黑盒）**：仅投毒数据（无任何超参控制）× 7B/14B × 触发类型（token vs 自然短语）。
- **E（防御鲁棒性）**：三类防御 × Persis 下降率。
- **F（规模化一致性）**：7B vs 14B 持久性对比（验证 H4）。

## 7. 评测协议

- 主指标：跨阶段 Persis（SFT→DPO→LoRA 级联后 ASR 保持率）；ASR 按任务定义（分类正确性/数学答案匹配/代码编译通过）。
- 消融：干净 Acc、reward 模型分数、触发隐蔽性（perplexity、human 感知率 50 人抽样）。
- 统计：每配置 3 种子（42/2024/2026），报告均值±std；Persis 差异用配对 t 检验（α=0.05）；阈值效应报告分段线性拟合的断点位置。

## 8. 算力与资源计划

- 本地训练（植入+DPO+LoRA 级联 × 2 模型 × 任务）≈3 GPU·天；评测（ASR/reward/防御矩阵）≈1 GPU·天；重跑 buffer 20%。
- API：DeepSeek V4 Pro 生成偏好对与自然触发 ≈40 万 token；Kimi K2.6 judge ≈25 万 token；hosted LoRA 若实测 DeepSeek 端点则预算 ≤ $50。
- 存储：模型 + LoRA 检查点 ≈80GB。
- 总预算 ≈4–5 GPU·天。

## 9. 里程碑与时间线（按周，单人 + 4 卡）

| 周 | 任务 |
|---|---|
| W1 | 复现 P-Trojan（SFT + 嵌入梯度对齐）；搭建 DPO 训练管线；DeepSeek Pro 生成偏好对 |
| W2 | 实现偏好扰动能后门 + DPO 隐式奖励梯度对齐；跑 7B 主实验 A |
| W3 | p 消融（B）+ KL 消融（C）；14B 全流程 |
| W4 | hosted LoRA 黑盒协议（D）；防御矩阵（E）；统计检验 |
| W5 | ICLR 2027 初稿（或 CCS 2026 版）；buffer |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| DPO 的 KL 惩罚压制后门（Persis 不达标） | 中 | 高 | 增加触发样本占比 p；把后门目标换成"触发时输出近目标"（软目标）；退而报告"DPO 部分抑制但不消除"的曲线作为贡献 |
| hosted LoRA 端点限制/审核触发样本 | 中 | 中 | 用本地模拟 hosted（冻结优化器、攻击者只传数据）作为主报告；API 实测作附录 |
| 14B 训练内存超限 | 低 | 中 | QLoRA NF4 + offload；或降到 7B 保底 |
| 偏好对生成质量不均 | 中 | 中 | DeepSeek Pro 生成 + Kimi 重排 + 人工抽检 200 条 |

## 11. 论文写作计划

- **目标会议**：ICLR 2027（截稿约 2026-09-28）、CCS 2026（截稿约 2026-04-25）。CCS 优先投生命周期热点、ICLR 补充机制深度。
- **差异化卖点**：① 首个把持久后门推进到 DPO/RLHF 与 hosted LoRA 的实证研究；② "偏好扰动能"——把对齐目标变成后门加固器；③ 首个黑盒 hosted LoRA 持久后门协议。
- **图表清单**：图1 攻击流水线（植入→对齐→级联评测）；图2 跨阶段 Persis 曲线；图3 p 与 β 消融热图；图4 7B vs 14B 对比；表1 主实验；表2 防御对照。
- **对防御的启示（必写）**：证明"部署后对齐不是后门清理工具"，推动持久性感知的防御——DPO 阶段安全偏好优先采样、梯度投影、对齐前後门审计；给模型厂商提供"对齐不取代安全审查"的实证依据。

## 12. 参考文献

- arXiv:2512.14741 —— P-Trojan: Persistent Backdoor Attacks Under Continual Fine-Tuning of LLMs
- arXiv:2311.14455 —— Universal Jailbreak Backdoors from Poisoned Human Feedback
- arXiv:2401.12242 —— BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models
- arXiv:2402.13851 —— VL-Trojan: Multimodal Instruction Backdoor Attacks against Autoregressive Visual Language Models
