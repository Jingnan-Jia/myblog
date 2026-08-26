# 实验设计书：SkillForge-RLVR —— 面向指令跟随的扩散 VLA MoE 后训练

## 0. 摘要（3-5 句）
本项目把「结构抗遗忘」（Action MoE + 任务指令门控，DiTEA 式）与「奖励后训练」（GRPO/RLVR）两条独立研究线合并，构建 SkillForge-RLVR：对扩散 VLA 的专家混合动作头做指令跟随强化后训练。奖励由三部分构成——可验证任务成功率、LLM-as-judge 的指令-动作语义对齐分、以及显式的「技能遗忘惩罚」（新任务对旧专家权重的漂移 + 路由熵正则）。预期在 SIMPLER（WidowX）多任务上同时提升 SR 与三档（basic/complex/distractor）指令跟随通过率，并显著降低旧任务回退率。主体在仿真完成，真机 Franka 作为加分项后置。

## 1. 研究背景与动机

### 1.1 问题定义
- 输入：多任务指令集 {i_1..i_T} 与对应演示数据；一个扩散 VLA 底座（视觉编码器 + 语言编码器 + DiT 动作头 + Action MoE 门控）。
- 目标：微调后得到单一策略 π(a|s,i)，使得 ①每个任务的成功率高；②对复杂/干扰指令的跟随率不退化；③学习新任务时不遗忘旧任务。
- 可量化指标：多任务平均 SR、指令跟随通过率（basic/complex/distractor 三档）、旧任务回退率（skill forgetting rate）、专家路由熵与负载均衡偏差。

### 1.2 相关工作不足
- (Intelligent Robotics · 论文 30 · DiTEA: Mixture-of-Experts for Vision-Language-Action Model in Robotic Manipulation)：用 Action MoE + 任务指令门控缓解多任务微调遗忘（SIMPLER 40.8%、Franka 40.2%，三件套消融 40.2→51.5%），但是**纯行为克隆**，复杂/干扰指令仍弱于自回归 OpenVLA，且计算开销大、无 RL 后训练。
- VLA-R1 (arXiv:2510.01623)：RLVR+GRPO + 可验证区域对齐/轨迹一致性奖励，但无专家/门控结构，不保护旧技能。
- Z-1 (arXiv:2606.31846)：在 π0.5 上对 RoboCasa 24 任务做任务级 GRPO，+13.2% SR，但无 MoE，未测指令跟随与遗忘。
- SmoothVLA (arXiv:2603.13925)：GRPO + 物理内在平滑奖励，但同样无结构保护，且报告抖动/不稳定问题。
- TacCoRL (arXiv:2606.11743)、RL Bootstrapping of OpenVLA-OFT (arXiv:2608.01013)：均验证 RL 后训练有效，但都未结合 MoE 门控与显式遗忘惩罚。
- 结论：**「门控/专家隔离结构 × RLVR 后训练 × 指令跟随奖励」这一组合仍是空白**，本项目恰好填此缝。

### 1.3 为什么是现在、为什么你的环境适合做
- 现在：2026 年 VLA+RLVR 已是顶会最热主线，但多为「无结构 RL」，结构抗遗忘与奖励后训练各自为战；DeepSeek-V4/Kimi-K2.6 的廉价 LLM-judge 让「指令跟随可验证奖励」无需人工标注即可规模化。
- 环境：**仿真主体**——SIMPLER（WidowX）四任务 + RLBench-OG（arXiv:2508.05186）OOD 子集 + robosuite 3 任务，全部有自动成功判据，可回放、可并发 rollout；真机（Franka 8 任务，复刻 DiTEA）仅在仿真结果稳定后作为加分项。

## 2. 研究目标与可验证假设（2-4 条）
1. **H1（RL 后训练提升 SR）**：在固定 MoE 结构上，GRPO 后训练使 SIMPLER 平均 SR 相比纯 SFT 底座提升 ≥5 个点。→ 成立的可观测结果：主实验（5 seed）SR 差异 p<0.05。
2. **H2（指令跟随奖励生效）**：加入 LLM-judge 指令-动作对齐奖励后，complex/distractor 档通过率相比无该奖励版本提升 ≥10 个点。→ 可观测：指令档位分层统计显著上升。
3. **H3（遗忘惩罚保护旧技能）**：加入「专家漂移正则 + 路由熵正则」后，旧任务回退率从基线 >15% 降至 <5%。→ 可观测：逐任务逐个训练时序的回退曲线。
4. **H4（路由与跟随联合优化）**：门控参与 RL（学习「什么指令走哪个专家」）后，路由熵下降且与指令档位显著相关。→ 可观测：路由混淆矩阵显示复杂指令更多路由到高容量专家。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据/轨迹流水线
- **底座 SFT 数据**：BridgeV2 子集（约 20K 条、WidowX 相关）+ DROID 子集（约 10K 条、手部相机视角筛选）+ robosuite/RLBench 仿真演示（每任务 1000 条、渲染 DINOv2+SigLIP 特征）。
- **API 合成指令流水线**：
  - Kimi K2.6：以「任务名称 + 目标物体 + 扰动描述」为种子，批量生成 **complex 干扰指令**（含同色诱饵、相似物体、否定词、多条件），每任务 300 条；同时生成 basic 指令同义改写。
  - DeepSeek V4 Pro：将指令与「动作语义文本」（由关键帧动作描述生成）配对，产出一批**对齐评分标注**，用于校准 LLM-judge（few-shot 示例）。
  - 指令难度分档：basic（单目标单动词语义明确）/ complex（多修饰、空间关系）/ distractor（出现语义干扰物）。每档各 800 条/任务。
- **Rollout 数据**：训练时每任务每步从环境中 rollout，记录 {obs 序列, 动作序列, 关键帧 RGBD, 任务成功标志, 专家路由 logits}。

### 3.2 方法设计（模块拆解、关键公式、超参数初值）
- **结构**：复现 DiTEA 式底座——DINOv2（冻结）+ SigLIP 视觉、语言 encoder 用 LLaMA 类骨干（7B 底座可降为 1.3–3B 行动头）、DiT 动作头换为 Action MoE（K=4 专家 + 1 共享专家，专家为同构 MLP），Task-Instruction Gate 用「指令提示 token」做软门控。
- **奖励函数**（GRPO 组内优势）：
  - `r = w_s·1[success] + w_j·J(x, a_text) + w_e·ρ + w_sm·ψ`
  - `J`：LLM-judge 对齐分（0–1，3 模型多数投票取中位数），`x` 为关键帧描述，`a_text` 为动作文本化描述。
  - `ρ`：路由熵正则 = −H(p_router)，鼓励门控明确选择专家（初值 w_e=0.05）。
  - `ψ`：平滑项（SmoothVLA 式 jerk/加速度惩罚，初值 w_sm=0.02）。
- **技能遗忘惩罚（核心新增）**：对每个任务 t，维护冻结的 SFT 期专家权重 W_t^ref；漂移项
  `L_drift = Σ_e (‖W_e^t − W_e^t_ref‖_F² / ‖W_e^t_ref‖_F²)`，以权重 λ_drift=0.1 作为 KL 之外的显式正则；同时用参考路由分布做 `KL(p_router ‖ p_router_ref)` 约束（β=0.1）。
- **GRPO**（沿 DeepSeek-R1 (arXiv:2501.12948)）：组内 G=8 个 rollout 共享同一任务，`A_i=(r_i−mean)/std`；裁剪代理目标 clip 0.2；对参考策略（SFT 底座）加 β·KL(π‖π_ref)，β=0.04。超参数初值表：

| 超参 | 初值 |
|---|---|
| 视觉编码器 | 冻结（前 500 步后解冻 SigLIP，lr×0.1） |
| 语言编码器 | 全参微调 |
| GRPO 组内 G | 8 |
| 训练步数 | 3000–6000（收敛判定：验证集 SR 不再提升 200 步） |
| lr | 1e-5（γ=0.95 余弦衰减） |
| 全局 batch | 64（8 task × 8 rollout，4×L40 梯度累积） |
| β (KL) | 0.04 |
| w_s / w_j / w_e / w_sm / λ_drift | 1.0 / 0.5 / 0.05 / 0.02 / 0.1 |

### 3.3 训练流程（优化器/学习率/批次/并行；RL 训练资源估算）
- **阶段 1 SFT（MoE 底座）**：AdamW，lr=2e-5，FSDP（shard 全部 7B 参数），全局 batch 128（梯度累积），20–40K 步，用时 3–5 GPU·天（4×L40）。监控门控负载均衡（每 500 步）。
- **阶段 2 GRPO 后训练**：rollout 在 CPU 并行 16 进程跑环境，GPU 仅训策略；每步 64 rollout，6000 步封顶，5–8 GPU·天。训练期间每 500 步在 20 条固定指令上测 SR 与三档通过率，早停。
- **资源估算**：总 8–13 GPU·天/模型（不含真机）；如需 3 seed，×3 但在 4 卡上串行分批跑（每 seed 一轮）。

### 3.4 评测流程
- **仿真评测**：SIMPLER WidowX 4 任务（含 OOD 变体，官方流程）、RLBench-OG 6 任务、robosuite 3 任务；每任务 50 episode；报告 mean±std（5 seed）。
- **指令评测**：自建三档指令集（每档 800 条/任务），人工「gold 对齐」标签抽 100 条校准 LLM-judge，报通过率 + judge-人工一致率（Cohen's κ）。
- **真机（可选加分）**：Franka 8 任务，复刻 DiTEA 协议（每任务 5 个演示位形 × 10 trials），2–3 周人工轮换。

## 4. 环境/数据集细节
- **SIMPLER**：WidowX 桌面操作仿真器（官方开源，PyBullet 后端），含 4 基任务 + 视觉变体；成功判据由官方回调给出；许可：开源研究用。
- **RLBench-OG**（arXiv:2508.05186）：OOD 变体，6 个难度任务；RLBench 本体为社区基准（软件，其 OOD 研究版以该 arXiv 引用）。
- **robosuite**（arXiv:2009.12293）：MuJoCo 后端模块化操作框架，用「Stack」「PickPlace」「NutAssembly」3 任务。
- **BridgeV2 / DROID**：经 Open X-Embodiment（arXiv:2310.08864）下载子集（开源，遵循各自许可）。
- 真机：Franka Panda + 抓爪，8 任务（参照 DiTEA 表 4 任务清单）。

## 5. 基线复现
| 基线 | 官方代码/出处 | 复现要点 |
|---|---|---|
| CogACT | DiTEA 论文描述的结构（作者开源；无独立 arXiv 收录，以 DiTEA 为准） | 无 MoE 的扩散 VLA 底座 |
| DiTEA | DOI:10.1609/aaai.v40i22.38902（作者未公开代码则按其论文结构重实现） | Action MoE + TIG，纯 SFT |
| OpenVLA | arXiv:2406.09246（官方开源） | 7B 自回归 VLA，LoRA 微调 |
| π0（微调） | arXiv:2410.24164（官方开源） | 流匹配 VLA，LoRA |
| VLA-R1 | arXiv:2510.01623 | GRPO+区域对齐奖励（若代码公开） |
| SmoothVLA | arXiv:2603.13925 | GRPO+平滑奖励 |
| Z-1 | arXiv:2606.31846 | 任务级 GRPO（RoboCasa 口径） |

- **预期指标表（SIMPLER 平均 SR，5 seed）**：OpenVLA 1.0%、CogACT 35.2%、DiTEA 40.8%、π0 微调 38%±2、SmoothVLA（其上改底座，预估）42%±2、**Ours 46–50%±2**；三档指令通过率：Ours 在 distractor 档比 DiTEA 高 ≥10 点。
- **统一口径**：所有模型用同一 500-step 推理配置；成功判据一律用环境官方回调；指令评测使用统一指令集（禁止各模型自带指令库）。

## 6. 实验矩阵
- **A 主实验**：Ours（GRPO 全套）vs 全部基线（表 5 口径）。
- **B 奖励消融**：B1 去 LLM-judge（仅 success）；B2 去遗忘惩罚（无 λ_drift/路由 KL）；B3 去平滑项；B4 去路由熵。
- **C 结构消融**：门控参与 RL vs 门控冻结；专家数 K∈{2,4,8}；共享专家有无。
- **D 泛化性**：对 RLBench-OG OOD 任务 zero-shot；新增 1 个未训练任务（如「新物体装配」）。
- **E 跨域迁移**：SIMPLER→robosuite 异构真值 / 视觉域（域随机化渲染）下的 SR 保持率。
- **F 稳定性**：训练期动作抖动（jerk 时序曲线）、推理时方差（50 seed 内 SR 标准差）。

## 7. 评测协议
- **SR**：episode 级二值成功 / episode 数（50 episode/任务）。
- **指令通过率**：LLM-judge 判定「指令语义被完成」的通过数 / 总数；以 100 条人工 gold 抽样校准，报 Cohen's κ。
- **遗忘率**：按「逐任务顺序训练」时序，记录每个旧任务在其后每 1000 步的 SR，定义回退率 = max(初始 SR − 后期 SR, 0)/初始 SR。
- **统计**：5 随机 seed（RL 训练 seed ∈ {0..4}，环境 seed 独立固定），报告 mean±std；显著性用配对 t-test（α=0.05，Bonferroni 校正）。

## 8. 算力与资源计划
- 4×L40 = 192GB：FSDP 全参 7B 底座可行（激活约 4×48GB 上限内，梯度累积压缩到 64 batch/步）。
- 分阶段 GPU·天：SFT 3–5；GRPO 5–8；消融（B×5 + C×4 + D/E）约 8–12；合计 **16–25 GPU·天**（含 3 seed 主实验）。
- 存储：数据 + 检查点 ≈ 300GB。
- **API 用量与成本**：LLM-judge 每步 64 rollout × 3 模型投票 × 6000 步 ≈ 115 万次调用；DeepSeek V4 Flash（主）约 $300–500，V4 Pro 校准与复杂档约 $150；Kimi K2.6 指令生成 + 投票约 $200；合计 **≤$1,000**。Flash 分流可再降 50%。

## 9. 里程碑与时间线（按周，单人+4 卡）
- W1–2：环境与数据管道（SIMPLER/RLBench/robosuite 安装、BridgeV2/DROID 子集清洗）。
- W3–4：复现 DiTEA 底座 SFT（对齐 40.8%）；建立指令集与 LLM-judge 管线（κ 校准）。
- W5–6：实现 GRPO + 奖励组装，小规模（4 任务、2000 步）验证 H1。
- W7–9：主实验 3 seed + 三档指令评测；B/C 消融。
- W10–11：D/E/F 泛化与稳定性实验；真机 Franka（可选）。
- W12：论文写作与图表；投稿 CoRL/ICRA。

## 10. 风险与备选方案（表）
| 风险 | 概率 | 影响 | 缓解/备选 |
|---|---|---|---|
| GRPO 后动作分布漂移、抖动 | 中 | 高 | 冻结视觉编码器、w_sm 平滑项、专家漂移正则；退化为只训门控+线性头 |
| LLM-judge 噪声 | 中 | 中 | 3 模型投票取中位数、阈值卡控、κ<0.7 则改人工标注档位 |
| 底座复现达不到 DiTEA 数字 | 中 | 中 | 直接改用官方 CogACT 检查点作为底座，只做 RL 部分 |
| 遗忘惩罚过强抑制新任务学习 | 低 | 中 | λ_drift 从 0.05 起搜 {0,0.05,0.1,0.2} |
| 真机安全/时间 | 中 | 低 | 真机仅作加分；仿真不达标则不报真机 |

## 11. 论文写作计划
- 目标会议：CoRL 2027（约 2026 年 6 月截稿，已过）→ 改为 **ICRA 2027（2026 年 9 月截稿）** 主投；次选 RSS 2027（2027 年 2 月截稿）。
- 差异化卖点：①首个「MoE 门控结构 × RLVR」融合；②显式技能遗忘惩罚（可量化的回退率指标）；③三档指令跟随可验证奖励（LLM-judge 统一协议）。
- 图表清单：①方法图（结构+奖励流）；②SIMPLER/RLBench 多任务 SR 表；③三档指令通过率柱状图；④逐任务遗忘回退曲线；⑤路由混淆矩阵；⑥收敛曲线（SR/熵/漂移）；⑦泛化保持率雷达图。
- 相关工作覆盖：VLA 基础模型（OpenVLA/π0/GR00T N1/RT-2/Diffusion Policy）、MoE-VLA（DiTEA/MoTVLA/InstructVLA/FedVLA/MoE-ACT/ManualVLA/AffordanceVLA）、RL 后训练（VLA-R1/Z-1/SmoothVLA/TacCoRL/RL Bootstrap）、RLVR 范式（DeepSeek-R1/Kimi K1.5）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）
- DiTEA: DOI:10.1609/aaai.v40i22.38902
- VLA-R1: arXiv:2510.01623
- Z-1: arXiv:2606.31846
- SmoothVLA: arXiv:2603.13925
- TacCoRL: arXiv:2606.11743
- RL Bootstrapping of OpenVLA-OFT: arXiv:2608.01013
- RLBench-OG: arXiv:2508.05186
- OpenVLA: arXiv:2406.09246
- π0: arXiv:2410.24164
- GR00T N1: arXiv:2503.14734
- RT-2: arXiv:2307.15818
- Diffusion Policy: arXiv:2303.04137
- MoTVLA: arXiv:2510.18337
- InstructVLA: arXiv:2507.17520
- FedVLA: arXiv:2508.02190
- MoE-ACT: arXiv:2603.15265
- ManualVLA: arXiv:2512.02013
- AffordanceVLA: arXiv:2606.06155
- Open X-Embodiment: arXiv:2310.08864
- robosuite: arXiv:2009.12293
- Imitation Is Not Enough: arXiv:2212.11419
- DeepSeek-R1: arXiv:2501.12948
- Kimi K1.5: arXiv:2501.12599
- Qwen2.5-VL: arXiv:2502.13923
