# 实验设计书：跨视频证据路由与聚合网络（Cross-Video Evidence Routing and Aggregation, CVRA）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：高 ｜ 目标会议：NeurIPS 2026 / CVPR 2027

## 0. 摘要

CrossVid 基准（Computer Vision V·论文38）显示最强模型 Gemini-2.5-Pro 在多视频时空推理上平均仅 50.4%（人类 89.2%），最难的 FSA 任务仅 13.4%，核心瓶颈是"多视频均匀拼接把帧预算均摊、无跨视频证据整合机制"。本工作提出**跨视频证据路由与聚合网络（CVRA）**：用轻量可学习 Router 按查询与各视频相关性分配每视频帧预算（查询感知跨视频 token 预算分配），再构建 entity/action 级跨视频语义对应图引导 LLM 注意力做跨视频聚合；两阶段训练（多视频对比型 SFT + GRPO 奖励含答案正确性、对应图覆盖与预算正则）。推理时逐视频编码、以 CLS 记忆 token 聚合，仅聚合阶段进 LLM。预期贡献：(1) 首个面向 CVR 的查询感知跨视频训练方法；(2) 在 CrossVid 上显著缩小与人类的差距；(3) 帧预算分配可视化的可解释诊断。

## 1. 研究背景与动机

### 1.1 问题定义

给定一组视频 {V1..Vk}（k=2~6）与跨视频推理问题 q（比较/时间理解/多视角/自由问答），模型需聚合跨视频证据并给出答案。关键挑战是**证据分布整合**：证据分散在不同视频的不同时间区间，需要 (a) 判断哪些视频/哪些时刻与 q 相关（预算分配），(b) 跨视频对齐语义等价实体与动作（对应图），(c) 聚合比较（推理）。

### 1.2 相关工作不足

- **基准无方法**：CrossVid（Computer Vision V·论文38）只做评测，误差分析指出四大错误——关键帧丢失、单视频理解错、跨视频比较错、格式错；All-Angles Bench（arXiv:2504.15280）同样只评测多视角；EgoExoLearn 聚焦程序活动无比较训练。
- **输入组织 naive**：评测时多视频均匀抽帧拼接，帧预算被均摊，无关视频稀释关键证据。
- **单视频检索方法不可直接迁移**：APVR（Computer Vision III·论文3）、AdaptToken（arXiv:2603.28696）等只做单视频内选择；StreamForest（arXiv:2509.24871）、OmniAgent（arXiv:2606.19341）证明持久记忆/主动感知范式可行但未面向比较任务训练。
- **空白**：缺少"查询感知预算分配 + 跨视频对应图 + 比较训练目标"的统一方法。

### 1.3 为什么是现在、为什么你的环境

- **时机**：CVR 是 2026 年最缺方法的空白带，CrossVid 恰好提供完整评测协议；NeurIPS 2026/CVPR 2027 是抢占期。
- **环境契合**：SFT 20K + GRPO 在 4×L40 上 ≈12 GPU·天可完成；多视频显存压力用"逐视频编码 + 内存 token 聚合"解决，6 视频 × 32 帧在 192GB 内可行。

## 2. 研究目标与可验证假设

- **H1（路由有效性）**：查询感知帧预算分配显著优于均匀分配。
  *成立时的可观测结果*：CrossVid O.Avg 提升 ≥2 点；预算可视化显示相关视频获得更多帧。
- **H2（对应图增益）**：跨视频语义对应图引导比无图直接拼接更优，尤其比较类任务。
  *成立时的可观测结果*：C.Avg（比较分析维度）提升 ≥2.5 点。
- **H3（预算正则必要性）**：预算正则（抑制总 token 数）在几乎不掉准确率时显著降低推理成本。
  *成立时的可观测结果*：预算-准确率曲线下面积（AUC）提升 ≥3%；同等准确率下 token 减少 ≥20%。
- **H4（鲁棒性）**：方法对视频数量与来源变化（2~6 视频、不同数据集）稳健。
  *成立时的可观测结果*：在 All-Angles Bench、EgoExoLearn 等域外数据上无显著掉点。

## 3. 总体方法设计

### 3.1 数据流水线

1. **语料**：CrossVid 官方 QA（9015，许可需核对）+ 自建多视频比较数据。
2. **自建数据生成（DeepSeek V4 Pro，离线）**：从 YouCook2（同食谱不同视频对）、Charades（同场景行为对）、VisDrone（同一场景多时刻）抽取视频组，prompt 要求生成"比较/聚合型"QA，含参考答案与证据归属（哪个视频、哪个区间）。
3. **过滤规则**：(a) 双 LLM（DeepSeek V4 Pro + Kimi K2.6）交叉验证 QA 合法性；(b) 需要 ≥2 视频联合推理的题目保留（单视频可答的剔除）；(c) 人工抽检 5%。预期合成 **~20K 条**。
4. **对应图标注**：对每组视频抽帧跑 Grounding-DINO（关键名词）+ CLIP 文本-帧匹配，生成初始候选边；DeepSeek V4 Flash 过滤假阳性对，形成 entity/action 对应候选。

### 3.2 模型/算法设计

- **基座**：Qwen2.5-VL-7B + 轻量 Cross-Video Router（MLP，输入 query embedding + 各视频 CLIP 摘要 embedding，输出每视频帧数 softmax×预算）。Router 与 LLM 联合 LoRA 训练。
- **逐视频编码 + 记忆 token**：每视频独立编码，取 CLS 语义 token 作为"记忆"；聚合阶段仅这些记忆 token + query 进 LLM 注意力。
- **对应图引导注意力**：构造邻接矩阵 A（跨视频 entity 对边），在注意力打分上叠加 `score += γ·A`（γ 训练），引导跨视频证据聚合。
- **两阶段训练**：
  - Stage A SFT：20K 样本，loss = NLL(答案) + 0.1·预算分配 CE（以路由目标为监督）+ 0.05·对比损失（对应图正负对）。
  - Stage B GRPO：奖励 `r = r_ans + 0.3·r_cov(对应图覆盖) − 0.02·max(0, total_tokens − budget)`；8 rollout/组。
- **超参初值**：LoRA rank 64、lr 1e-5（SFT）/1e-6（GRPO）、batch 16、γ=0.3、预算=3072 视觉 token 上限。

### 3.3 训练流程
- 优化器 AdamW；warmup 3%；cosine；ZeRO-3；2 卡训练 + 2 卡 rollout/评测并行。
- Stage A 4-6 GPU·天，Stage B 6-8 GPU·天。

### 3.4 推理与评测流程
- 推理：Router 输出预算 → 逐视频编码 → 记忆 token 聚合 → LLM 作答；temperature=0。
- 评测：CrossVid 官方 4 维 10 任务 accuracy；含 CoT 变体；帧数-准确率曲线。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| CrossVid（arXiv:2511.12263）| 训练/评测 | 公开（需核对开放许可）|
| All-Angles Bench（arXiv:2504.15280）| 评测（泛化）| 公开 |
| EgoExoLearn | 评测（泛化）| 公开（CVPR 2024）|
| MMVU（arXiv:2501.12380）| 评测（域外）| 公开 |
| YouCook2 / Charades / VisDrone | 自建训练视频源 | 公开（各自许可）|

### 4.2 划分与数量
- 训练：CrossVid 官方 split（如有）+ 自建 20K；不交叉污染评测。
- 评测：CrossVid 全部 9015 QA、All-Angles、EgoExoLearn、MMVU 测试。

### 4.3 预处理与格式
- 每视频 32 帧（Router 决定裁剪至 ≤32），帧 224×224，1fps 采样；
- 视频组 JSONL：`{group_id, videos:[{path, summary}], question, answer, evidence_attribution, candidates}`。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| CrossVid 22 个模型结果 | Computer Vision V·论文38 | github.com/CrossVid（官方评测代码）|
| Qwen2.5-VL-7B/72B | 官方 | github.com/QwenLM/Qwen2.5-VL |
| InternVL3-8B | 官方 | github.com/OpenGVLab/InternVL |

### 5.2 复现步骤与预期指标表
统一评测脚本（官方 repo 的 prompt 模板），temperature=0。预期主表（accuracy，O.Avg）：

| 方法 | O.Avg | C.Avg | T.Avg | M.Avg | CCQA |
|---|---|---|---|---|---|
| Gemini-2.5-Pro（官方）| 50.4 | — | — | — | — |
| Qwen2.5-VL-72B（官方）| ~44 | — | — | — | — |
| Qwen2.5-VL-7B（本地复现）| ~36 | — | — | — | — |
| **CVRA-7B** | ≥42（目标 45+）| ≥2.5 提升 | ≥2 提升 | ≥2 提升 | ≥2 提升 |

### 5.3 统一评测口径
所有模型同 prompt 模板（含视频组顺序打乱以消除位置偏置）、同帧采样、同解析器；CoT 单独一栏报告。

## 6. 实验矩阵

- **A（主实验）**：完整 CVRA（Router + 对应图 + SFT + GRPO），7B。
- **B1（路由消融）**：均匀预算 / CLIP 摘要启发式路由 / 可学习 Router。
- **B2（对应图消融）**：无图 / 文本对齐图 / 图+Grounding-DINO 边 / 图+注意力门控。
- **B3（奖励消融）**：仅 r_ans / +r_cov / +预算正则（不同 β）。
- **B4（训练消融）**：纯 SFT / SFT+GRPO / 仅 GRPO。
- **C（鲁棒性）**：视频数 {2,3,4,6}、来源混合、帧数 {16,32,48}、视频顺序打乱。
- **D（泛化性）**：All-Angles Bench / EgoExoLearn / MMVU 零样本；72B 上验证 Router 迁移。

## 7. 评测协议

- 指标：O.Avg/C.Avg/T.Avg/M.Avg/CCQA；预算-准确率 AUC；对应图边数/召回；帧预算可视化（KL 散度 vs 理想分配）。
- 3 个随机种子（42/2024/2026）报 mean±std；配对 bootstrap p<0.05 标 *；报告逐任务表格。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：Stage A 5 + Stage B 7 + 评测 2 = **≈12 GPU·天**。
- 存储：模型 20GB×4 版本 + 视频缓存 400GB + JSONL 40GB。
- API：DeepSeek V4 Pro 合成 20K 比较 QA ≈ 600 万 token；DeepSeek V4 Flash 对应图过滤 ≈ 300 万 token；Kimi K2.6 judge ≈ 250 万 token。估算成本 ≈ **$300-700**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | CrossVid 数据获取与官方评测脚本复现（Qwen2.5-VL-7B/72B 基线）|
| 2 | 自建 20K 比较 QA 合成 + 过滤；对应图候选生成 |
| 3 | Router + 逐视频编码 + 记忆 token 管线实现 |
| 4 | Stage A SFT 训练 + 均匀预算对照 |
| 5 | Stage B GRPO 训练 + 预算正则调参 |
| 6 | 消融 B/C/D + 可视化分析 |
| 7 | 论文初稿 |
| 8 | 定稿投稿 NeurIPS 2026 或 CVPR 2027（视 NeurIPS deadline 是否已过）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 多视频长输入显存压力 | 中 | 高 | 逐视频编码 + CLS 记忆 token 聚合；6 视频×32 帧封顶；必要时 16 帧 |
| Router 与 LLM 联合训练不稳定 | 中 | 中 | Router 先冷启动（固定 LLM 训 Router），再联合 |
| 自建 QA 质量不足 | 中 | 中高 | 双 LLM 交叉验证 + 5% 人工抽检；置信度过滤 |
| 对应图噪声引入误导 | 中 | 中 | γ 门控可学习；负采样增强；消融验证增益 |
| 训练数据与评测数据域重叠 | 低中 | 中 | 评测组与训练组严格隔离（按数据集/视频 ID 排除）|

## 11. 论文写作计划

- **目标会议/截稿**：NeurIPS 2026（若 deadline 未过）或 CVPR 2027。
- **差异化卖点**：CVR 从基准到方法的第一个训练框架；可解释的帧预算分配；跨视频对应图注意力。
- **图表清单**：Fig.1 框架（Router+对应图+记忆 token）；Fig.2 预算分配可视化案例；Fig.3 对应图示例；Fig.4 错误分析前后对比；Tab.1 主表（对比 22 模型）；Tab.2-4 消融；Tab.5 鲁棒性/泛化。
- **相关工作覆盖**：CVR 基准（CrossVid/All-Angles）、多视频记忆（StreamForest/OmniAgent/StreamMem）、检索（APVR/QuoTA）、对齐训练（Video-R1/TIME）。

## 12. 参考文献（真实核验）

- Computer Vision V·论文38·CrossVid（arXiv:2511.12263）
- All-Angles Bench: arXiv:2504.15280
- MMVU: arXiv:2501.12380
- EgoExo4D: arXiv:2311.18259
- StreamForest: arXiv:2509.24871
- OmniAgent: arXiv:2606.19341
- StreamMem: arXiv:2508.15717
- Computer Vision III·论文3·APVR（arXiv:2506.04953）
- AdaptToken: arXiv:2603.28696
- QuoTA: arXiv:2503.08689
- Computer Vision V·论文77·EgoCross（arXiv:2508.10729）
- Video-R1: arXiv:2503.21776
