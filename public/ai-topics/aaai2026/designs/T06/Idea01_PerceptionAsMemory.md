# 实验设计书：Idea 1｜GUI 智能体的跨会话主动感知记忆

> Perception-as-Memory: Cross-Session Active Perception Memory for GUI Agents
> 资源假设：4× NVIDIA L40（192GB）；DeepSeek V4 Flash/Pro、Kimi K2.6 API。

## 0. 摘要
GUI agent 每次会话都从零开始感知界面，重复付出"缩放/裁剪/观察位置"的探索成本，而既有主动感知方法（GUI-Eyes）不跨会话积累、既有记忆方法（CA3Mem）只记文本轨迹而无视觉感知先验。本工作提出"感知记忆"（Perception-as-Memory）：把"如何看界面"沉淀为按页面指纹索引的可复用感知策略，让主动感知越用越快。方法上以 Qwen2.5-VL-7B 为底座，两级策略先查记忆命中再决定是否启动视觉工具，感知记忆随任务成功自动演进并做生成式重组。训练采用 SFT 行为克隆 + GRPO 强化（grounding 命中率 + 观测次数双重奖励）。预期在 ScreenSpot-Pro 减少 30%+ 无效观测、在 OSWorld/WebArena 提升任务级成功率，并与"纯 RLVR 无记忆"基线形成清晰消融。

## 1. 研究背景与动机

### 1.1 问题定义
GUI agent 在真实环境执行任务需要"感知"：给定一张界面截图，决定是否/如何裁剪、缩放、聚焦，从而定位可交互元素。现有工作（GUI-Eyes，Multiagent Systems·论文 7·GUI-Eyes）让 agent 学会"何时/是否/如何调用视觉工具"，但每次会话都是冷启动：同一 app 的同一页面，每次都要重新探索。与此同时，记忆系统（CA3Mem，Computer Vision XII·论文 85·Evolving Generalist Virtual Agents with Generative and Associative Memory）把经验组织为文本化记忆图，却丢弃了感知层的先验（裁剪区域、缩放档位、命中区域）。我们把问题定义为：**构建一个跨会话复用的感知记忆，使 agent 在遇到熟悉页面时跳过无谓观测、在陌生页面时仍保持探索能力**。

### 1.2 相关工作不足
- 主动感知只做单会话：GUI-Eyes（Multiagent Systems·论文 7）的两级策略与空间连续奖励（arXiv:2601.09770）仅在 grounding 基准验证，未做任务级验证，更未考虑跨会话复用；GUI-R1（arXiv:2504.10458）的 RLVR 路线同样每次冷启动。
- 记忆缺视觉/感知先验：CA3Mem（Computer Vision XII·论文 85，DOI 10.1609/aaai.v40i15.38300）的记忆图为文本轨迹转写，对强视觉的 GUI 环境信息损失大；MemGPT（arXiv:2310.08560）是文本虚拟上下文，不含视觉策略。
- 观测成本未被显式度量：ScreenSpot 系基准（SeeClick，arXiv:2401.10935）只测最终定位，不测"用了多少次观察"。

### 1.3 为什么是现在、为什么你的环境适合做
"主动感知 + 记忆"两条线都已成熟但未交叉：GUI-Eyes/InfiGUI-G1 证明感知策略可学习，CA3Mem/策略化记忆（AgeMem arXiv:2601.01885、FadeMem arXiv:2601.18642）证明记忆可训练进策略。OSWorld（arXiv:2404.07972）与 WebArena（arXiv:2307.13854）提供了可重复的任务级环境；4×L40=192GB 可全参 FSDP 训练 7B 多模态模型，ScreenSpot-Pro 为 grounding 提供了低成本快速验证环，适合先验证再上任务级。

## 2. 研究目标与可验证假设
1. **H1（感知记忆可降观测成本）**：带感知记忆的 agent 在重复访问的页面上显著减少每任务观测数。
   - 成立时观测：ScreenSpot-Pro 上平均每任务观测数较无记忆 RLVR 基线下降 ≥30%，且定位准确率不下降。
2. **H2（记忆驱动感知可提升任务成功率）**：先查记忆、命中则跳过视觉工具，可提升 OSWorld/WebArena 多步任务成功率。
   - 成立时观测：任务级成功率较纯 RLVR 无记忆基线提升 ≥5 个绝对点，且 memory-hit 率 >0.5。
3. **H3（感知记忆自动演进优于静态记忆）**：任务成功后写回并重组记忆的 agent 优于固定记忆的 agent。
   - 成立时观测：第 2 个会话开始，演进式 agent 的观测数持续下降而静态式平台期。
4. **H4（记忆图检索优于 top-k）**：扩散激活式页面-元素图检索优于纯语义 top-k 检索。
   - 成立时观测：检索命中相关策略的比例提升 ≥10%，任务成功率随之提升。

## 3. 总体方法设计

### 3.1 数据/轨迹采集流水线
- **grounding 标注集**：ScreenSpot-Pro 官方标注 + 自采 500 页真实 GUI 快照（桌面 Windows 11 VM、macOS、Android 模拟器、Web 各 ~125 页，每页含多分辨率截图 1×/1.5×/2×）。
- **感知决策轨迹标注（LLM-as-teacher）**：对每页截图，调用 DeepSeek V4 Pro，输入 prompt：
  ```
  你是 GUI 感知标注器。给定 {截图, 目标指令}，输出：(a) 是否需要裁剪/缩放（yes/no+理由）；(b) 裁剪区域 bbox 与缩放档位；(c) 若全屏可直接定位，输出元素 bbox。
  要求：给出置信度与备选感知方案。
  ```
  生成 3 次取多数，产出"感知决策轨迹"（is_zoom, crop_box, zoom_level, hit_bbox, is_fullscreen_direct）。共约 1500 条决策轨迹，人工抽检 200 条校准 prompt。
- **任务级轨迹采集**：OSWorld（50 任务 × 8 平台）+ WebArena（50 任务）上，以 V4 Flash 为执行/校验后端跑 ReAct 采样，每任务上限 30 步，得到 ~1000 条带"感知步/动作步"时间戳的轨迹，用于记忆写回预演。
- **合成轨迹数量**：grounding 决策 1500 + 任务轨迹 1000，总计 ~2500，供两阶段训练使用。

### 3.2 系统/算法设计
- **感知记忆结构**：KV 结构 `Mem = { page_fingerprint → {perception_strategy, success_stat, last_ts} }`。page_fingerprint 由 DOM 结构哈希 + 视觉 embedding（截图的 CLIP-style 特征）双通道构成。感知策略含：`(is_zoom, crop_box, zoom_level, element_hint)`。
- **记忆图（扩散激活检索）**：节点 = 页面指纹 + 关键元素；边 = 共现/同 app/布局相似度。检索用 spreading activation：种子 = 当前页面指纹，激活值沿图传播，取激活 Top-K 策略集合。对照：纯 top-k（语义 embedding 余弦）。
- **两级感知策略（记忆驱动）**：Policy-L1（粗）输入 = 页面指纹 + 任务描述，输出 = `use_memory | explore`。若 `use_memory` 且置信高，直接采用记忆策略并跳过视觉工具调用；否则走 GUI-Eyes 式两级推理（L1 定位 ROI → L2 细粒度 grounding）。当记忆命中但执行失败，策略降级为探索并写回负面统计。
- **记忆演进**：任务成功后，把本次成功感知策略写回对应页面指纹；对失败页面做"生成式重组"——用 V4 Pro 合并两条相近策略生成新策略候选，写回时带上 `success_stat`，供 RL 奖励评估（类比 CA3Mem 的重组但加了验证信号）。
- **RLVR（GRPO）**：奖励 = 定位命中率 `r_acc`（IoU 连续分）+ 观测惩罚 `r_obs = -α·n_obs`，其中 `n_obs` 为本次任务视觉工具调用次数；α 由验证集扫描（0.02–0.1）。对记忆命中步额外给 `r_mem = β·I(hit∧correct)`，β=0.05，鼓励有效复用。

### 3.3 训练流程
- **阶段一 SFT（行为克隆）**：Qwen2.5-VL-7B，全参 FSDP（shard 数=4，zero-3），AdamW，lr=1e-5，linear warmup 3%，batch=64（4×16），序列 ≤4096 token，epochs=2，约 8 GPU·天。损失 = 感知决策交叉熵（两级策略分开输出）+ 轨迹行为克隆交叉熵。
- **阶段二 GRPO**：同底座冻结 SFT 权重继续训练，group size=8，clip=0.2，lr=1e-6，rollout 用 4 卡并行 vLLM 采样，每代 256 样本/轮 × 10 轮，约 10 GPU·天。SFT+GRPO 合计 18 GPU·天。

### 3.4 推理与评测流程
- 评测环境：ScreenSpot-Pro（grounding 单步）、OSWorld（arXiv:2404.07972，用官方 Docker，VNC 虚拟桌面）、WebArena（arXiv:2307.13854，官方 Docker 站点）。Agent 动作空间 = click/type/scroll + 感知动作（zoom/crop）。每任务 max_steps=30，max 观测=10。两会话评测：同一任务分两个会话完成（会话 1 冷启动，会话 2 复用记忆），量"会话间提速"。

## 4. 数据集/环境细节
- ScreenSpot-Pro：SeeClick（arXiv:2401.10935）公开数据集，MIT 许可，仅评测集，grounding 标注。划分：随机 80/20 为记忆热/冷页面。
- 自采 500 页：Windows/macOS/Android/Web 截图，自行采集无版权问题，仅存截图与 DOM 摘要。
- OSWorld 1.0（arXiv:2404.07972）：CC-BY 许可，官方 Docker 镜像，8 平台子集选用 Windows/Ubuntu/macOS/Android/Web 5 平台各 10 任务。
- WebArena（arXiv:2307.13854）：Apache-2.0，官方 Docker，选取 50 个需多步操作的任务。
- 训练用 grounding 数据仅用于 grounding 评测域对齐，任务级只做零样本评测。

## 5. 基线复现
| 基线 | 官方代码 | 复现要点 |
|---|---|---|
| GUI-Eyes（arXiv:2601.09770） | 若官方未开源，按其两级策略+空间连续奖励重实现 | 3K 数据 SFT+RL，ScreenSpot-Pro 44.8% |
| GUI-R1（arXiv:2504.10458） | GitHub: zhcsnoopy/GUI-R1 | 统一动作空间+GRPO |
| SeeClick（arXiv:2401.10935） | GitHub: OS-Copilot/SeeClick | grounding 预训练基线 |
| 纯 RLVR 无记忆 | 本工作移除记忆模块 | 隔离记忆贡献 |
| MemGPT 文本记忆 agent（arXiv:2310.08560） | GitHub: cpacker/MemGPT | 文本记忆替换感知记忆 |

统一评测口径：同一 OSWorld/WebArena 任务子集、同一 max_steps/观测上限、同一 seed 集（5 个）。
预期指标表：ScreenSpot-Pro 定位准确率 GUI-Eyes 44.8%、GUI-R1 ~46%、本方法 ≥46% 且观测数降 30%+；OSWorld 成功率基线（SeeClick 系）~5–10%，本方法目标 +5 个点。

## 6. 实验矩阵
- **A. 主实验**：本方法 vs 全部基线，ScreenSpot-Pro + OSWorld + WebArena 三域。
- **B. 消融**：记忆有无；记忆图 vs 纯 top-k；扩散激活 vs 直接命中；演进写回 vs 静态记忆；SFT-only vs SFT+GRPO。
- **C. 鲁棒性**：分辨率变化（1×/1.5×/2×）、窗口缩放、主题色变化下的记忆命中率与成功率。
- **D. 泛化性**：训练未见 app（留出 20% 页面指纹）；跨会话间隔（1 小时 / 1 天）记忆保持。
- **E. 记忆污染**：注入错误感知策略，测遗忘/纠错能力（对照 FadeMem arXiv:2601.18642 的选择性遗忘）。

## 7. 评测协议
- 指标定义：定位准确率（IoU≥0.8 命中比例）；任务成功率（OSWorld/WebArena 官方判定）；平均每任务观测数；memory-hit 率；会话间提速比 = (会话1观测数−会话2观测数)/会话1。
- 均值±方差：5 seed 各跑 1 次取均值，报告 95% CI；显著性用配对 t 检验（p<0.05）。
- agent 评测成本控制：每任务 max_steps=30、每任务预算≤30 次 LLM 调用，OSWorld 单任务耗时上限 10 min；防死循环用步数硬截断。

## 8. 算力与资源计划
- 4×L40：SFT 8 GPU·天 + GRPO 10 GPU·天 + rollout/评测 7 GPU·天 ≈ **25 GPU·天**（2 周内）。
- 存储：OSWorld/WebArena Docker 镜像 ~40GB，自采截图 ~10GB，检查点 ~20GB。
- API 用量/成本（按 2026 公开价估）：
  - DeepSeek V4 Pro：感知决策标注 1500 条 + 重组/裁判 ~3M token，估 ¥150–300。
  - DeepSeek V4 Flash：执行/校验后端 + rollout 约 20M token，估 ¥100–200。
  - Kimi K2.6：记忆语义判定 1M token，估 ¥50–100。

## 9. 里程碑与时间线（单人 + 4 卡）
- W1：环境搭建（OSWorld/WebArena Docker + 4 卡 FSDP 验证）、基线复现（SeeClick/GUI-R1）。
- W2：自采 500 页 + V4 Pro 感知决策标注流水线，抽检校准。
- W3：SFT 训练（8 GPU·天），跑 ScreenSpot-Pro 首轮。
- W4：记忆模块（KV+扩散激活图）开发，单会话消融。
- W5-6：GRPO 训练（10 GPU·天）+ 两会话评测。
- W7：OSWorld/WebArena 任务级评测 + 消融矩阵。
- W8：写作、图表、补实验；投稿前 1 周核实 GUI-Eyes/GUI-R1 官方代码是否开源并复现。

## 10. 风险与备选方案
| 风险 | 概率 | 影响 | 缓解/备选 |
|---|---|---|---|
| OSWorld rollout 不稳定（VNC/超时） | 中 | 高 | 先用 ScreenSpot-Pro 验证 H1/H4，任务级后置；加超时重试 |
| 感知记忆跨分辨率失效 | 中 | 中 | 指纹含分辨率字段，检索分桶；写回多分辨率策略 |
| GRPO 训练不稳 | 中 | 高 | 先 3B 起跑定参，固定 seed + 早停；奖励按 min-max 归一 |
| 记忆污染后性能退化 | 中 | 中 | 写回带 success_stat 阈值；失败计数超限即遗忘（FadeMem 思路） |
| 记忆提升小（H2 不成立） | 低 | 中 | 收紧观测惩罚 α，强化 H1 卖点；聚焦成本-质量权衡叙事 |

## 11. 论文写作计划
- 目标会议：NeurIPS 2026（截稿约 2026-05，已过）→ 若错过则 NeurIPS 2026 夏/秋档 D&B 或 ICLR 2027；以官网为准，优先 NeurIPS 2026 正会（如开放）否则 AAAI 2027 中旬档。
- 差异化卖点：首次把"感知先验"建成跨会话记忆；给出"观测成本-成功率"权衡的显式指标；两会话评测协议（可复用方法论）。
- 图表清单：Fig.1 感知记忆框架图；Fig.2 会话间观测数下降曲线；Fig.3 扩散激活检索案例；Table1 三域主结果；Table2 消融；Table3 鲁棒性；Table4 记忆污染/遗忘。
- 相关工作覆盖：GUI-Eyes、GUI-R1、CA3Mem、MemGPT、AgeMem/FadeMem、OSWorld/WebArena、SeeClick、InfiGUI-G1。

## 12. 参考文献
- GUI-Eyes, arXiv:2601.09770
- GUI-R1, arXiv:2504.10458
- CA3Mem, DOI 10.1609/aaai.v40i15.38300
- MemGPT, arXiv:2310.08560
- AgeMem, arXiv:2601.01885
- FadeMem, arXiv:2601.18642
- SeeClick, arXiv:2401.10935
- OSWorld, arXiv:2404.07972
- WebArena, arXiv:2307.13854
- InfiGUI-G1（Natural Language Processing III·论文 56）
- Evolving Generalist Virtual Agents with Generative and Associative Memory（Computer Vision XII·论文 85）
