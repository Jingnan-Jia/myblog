# 实验设计书：StyleGrow —— LLM 自动发现风格参数与跨域风格解释器迁移

## 0. 摘要（3-5 句）
LCDSP（Machine Learning VII · 论文 79）证明「风格参数 SP」能把抽象战术指令翻译成可执行的行为风格，但 SP 与 reward shaping 全为人工设计、换环境必须重做，风格解释器 SI 也未做跨域验证。StyleGrow 让 LLM 自动为新环境生成「候选行为集 → 风格参数 SP + reward shaping 描述」，并用「行为区分度/可奖励性」筛选（复用 PSS 条件熵思想）；同时提出 SI 的域适配微调——冻结主体、只改 ASaB 缩放头，实现「足球 → 四足步态 / 导航」的指令-风格跨域迁移。全流程可脚本化、零手工 reward，总 GPU 预算 <5 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义
- 输入：一个新 RL 环境（状态空间、动作空间、仿真器）+ 一组抽象指令（如「踢得保守」「走快一点」）。
- 目标：自动产出 ①可用的 SP 集合（每维有语义含义、可调制 reward shaping）；②能把这些指令翻译成 SP 的风格解释器 SI；③该 SI 能从已有域迁移到新域。
- 约束：零手工 reward engineering；全部由 LLM + 可验证信号（行为区分度）驱动。

### 1.2 相关工作不足
- (Machine Learning VII · 论文 79 · Complex Instruction Following with Diverse Style Policies in Football Games)：SP + reward shaping 全人工（10 行为 10 SP）；SI（Qwen2.5-0.5B + ASaB）只吃合成指令，无跨域验证；无统一指令完成判据。
- MAPL (arXiv:2606.25398)：LLM 按语义维度对轨迹做偏好替代手工 reward，但未给出「可移植的指令→行为参数」接口。
- Eurekaverse (arXiv:2411.01775)：LLM 生成环境课程（代码级环境修改），验证「LLM 可做环境/任务生成」，但不生成行为风格参数。
- SayTap (arXiv:2306.07580)：语言→足式步态参数，思想同源，但参数由人定义、无自动发现。
- LLM-MARL (arXiv:2506.04251)：LLM 高层协调 + PPO，但无风格参数。
- 结论：**「LLM 自动发现 SP + 跨域迁移 SI」组合无先例**。

### 1.3 为什么是现在、为什么你的环境适合做
- 现在：DeepSeek V4 / Kimi K2.6 的代码与语义生成能力强到可以「读环境 API → 生成行为与 reward 代码」；PSS 提供了现成的「行为区分度」验证信号；LCDSP 开源复现链成熟。
- 环境：**纯仿真**——GRF 5v5（复现 LCDSP 域）、MuJoCo 四足（SayTap 式步态参数域）、MiniGrid/Habitat 导航域；三域均有脚本化评估，全流程可复现、可自动化。

## 2. 研究目标与可验证假设（2-4 条）
1. **H1（自动 SP 达到人工 SP 效果）**：LLM 自动发现的 SP 集，其风格区分度（SEU/SMUL/SELO 与行为聚类纯度）达到 LCDSP 人工 SP 的 ≥80%。→ 可观测：GRF 域内指标分化度对比。
2. **H2（零手工 reward 可行）**：LLM 生成的 reward shaping 描述可直接用于 DST 训练，风格达成率 ≥ 人工 SP 版本的 90%。→ 可观测：各 SP 组合下行为指标区分。
3. **H3（SI 跨域迁移）**：只在 ASaB 缩放头微调的 SI，在四足/导航域的指令→SP 映射 MAE 相比从头训练误差 ≤1.5×。→ 可观测：few-shot（每域 200 条）迁移 MAE。
4. **H4（LLM 生成的 reward 合法）**：生成 reward 通过静态检查 + 仿真安全约束（不产生越界/自毁行为）。→ 可观测：安全检查 100% 通过、仿真碰撞率不上升。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据/轨迹流水线
- **环境 API 快照**：为每域提供「环境类名、obs 字典、动作空间、内置指标函数」的静态描述，作为 LLM 生成上下文。
- **SP 候选生成**：DeepSeek V4 Pro 每域产出候选行为集（每行为含：名称、语义描述、建议 SP 维度与取值范围、reward shaping 代码草案），每域 10–15 个候选。
- **可奖励性/区分度筛选**：对候选做「条件熵校验」（复用 PSS：对每组 SP 组合预跑短 PPO，度量行为分布的条件熵下降量），筛选出最终 SP 集（每域 8–12 维，Float/Bool 混合）。
- **指令-标签对**：Kimi K2.6 生成每域 2000 条指令（含多义、否定、条件）；由 DeepSeek/Kimi 互审 + 规则约束产出 SP 标签；抽样 300 条人工复核。
- **轨迹数据**：DST 训练期采集 (s, ω) 轨迹，用于 SI 的偏好/回归辅助。

### 3.2 方法设计（模块拆解、关键公式、超参数初值）
- **SP 调制 reward**（LCDSP 式）：`R_t(s,a|ω) = R_base(s,a) + Σ_i ω_i · φ_i(s,a)`，φ_i 为 LLM 生成的 shaping 项（每项给一段可执行函数）。
- **多风格策略 DST**：策略 π(a|s,ω)，ω 从风格分布采样；用 Prioritized Style Sampling 按条件熵下降量优先采有区分度的 ω。PPO（arXiv:1707.06347），3 个 seed。
- **风格解释器 SI 跨域迁移**：
  - 结构：Qwen2.5-0.5B/1.5B 冻结骨干 + ASaB（每维自适应缩放 (γ,β)）回归头。
  - 域适配：冻结主干与 GRF 先验头参数，仅微调新域的 (γ,β) 层（few-shot 200 条）；对比全量微调。
- **行为区分度度量**（自动筛选用）：对 ω 采样组合，跑短 rollouts 得行为统计向量 b(ω)，定义 `D(ω_a,ω_b)=KL(p(b|ω_a)‖p(b|ω_b))`；选取使成对区分度均值最大且方差可控的 SP 子集。
- **超参数初值**：

| 超参 | 初值 |
|---|---|
| PPO lr | 3e-4（GRF）/ 1e-3（四足/导航） |
| PPO rollout | 4096（GRF）/ 8192（四足） |
| 训练步数 | GRF 5v5 3K episode 预跑筛选；正式训练 20K episode |
| DST 风格采样 | 每 episode 采样 1 个 ω，PSS 优先 |
| SI 学习率 | 3e-4（ASaB 头）/ 1e-5（若解冻 LoRA） |
| SI 训练量 | 2000 条/域，10 epoch |
| 迁移微调量 | 200 条/域（few-shot） |

### 3.3 训练流程（优化器/学习率/批次/并行；RL 训练资源估算）
- 策略侧：GRF 5v5（PPO 单卡 ~2 GPU·天）、四足（MuJoCo 单卡 1–2 GPU·天）、导航（MiniGrid 单卡 0.5 天）。并行 4 卡跑 4 个 seed/域。
- SI 侧：Qwen2.5-1.5B 微调 1 卡，2–3 小时/域。
- LLM 生成与判分全部走 API（不计 GPU）。总 GPU：**3–5 GPU·天**。
- 种子：策略 3 seed，SI 5 seed。

### 3.4 评测流程
- **GRF 域**：6 战术（积极进攻/全力进攻/均衡/反击/摆大巴/Tiki-Taka）在进球/控球/传球/间距/胜率指标上的分化（对照 LCDSP 图 4 口径）。
- **四足域**：步态风格（步频/步幅/摆腿幅度）指标分化；SayTap 参数空间对照。
- **导航域**：指令「走快/保守/绕远」对应的轨迹特征（速度均值/与障碍距离/路径长度）。
- **SI 评测**：指令→SP 回归 MAE、风格达成率（LLM-judge + 指标）；跨域 zero-shot/few-shot 迁移率。
- **安全**：LLM 生成 reward 的静态检查（越界访问、除零、单调性）+ 仿真 1000 rollouts 碰撞/自毁率。

## 4. 环境/数据集细节
- **GRF 5v5**：Google Research Football 环境（开源，Python API），scenario 为 LCDSP 使用的 5v5 定制场景；行为/SP 体系按论文 79 复现。
- **四足**：MuJoCo 开源四足模型（如 Ant / 四足 walker 参数空间），按 SayTap 步态参数定义域。
- **导航**：Gym-MiniGrid（开源）目标指令域；备选 Habitat 短程目标导航。
- 许可：GRF 为 Apache-2.0；MuJoCo 为 Apache-2.0；MiniGrid 为 BSD。

## 5. 基线复现
| 基线 | 出处 | 复现要点 |
|---|---|---|
| LCDSP（复现） | (Machine Learning VII · 论文 79) | 人工 SP 全套 |
| SayTap | arXiv:2306.07580 | 语言→步态参数，人工参数空间 |
| LLM-MARL | arXiv:2506.04251 | LLM 子目标 + PPO（在本项目域重放） |
| MAPL | arXiv:2606.25398 | LLM 偏好→多头奖励（替代本项目的 SP 调制） |
| 手写 reward 的 PPO | PPO (arXiv:1707.06347) | 无风格参数 |
- **预期指标表（GRF 域）**：LCDSP SEU 3.50、手写 PPO 无分化、LLM-MARL 有协调无风格、**Ours SEU 2.8–3.5**、SI MAE 0.70–0.80（人工 SP 0.671 基线）。跨域：SI few-shot 迁移 MAE 相比从头训练 ≤1.5×。
- **统一口径**：风格达成率统一用「行为统计向量的分类器 + LLM-judge 双盲一致率」。

## 6. 实验矩阵
- **A 主实验**：StyleGrow 全流水线 vs 各基线（三域）。
- **B SP 来源消融**：LLM 自动发现 vs 人工 SP vs LLM 随机（无筛选）。
- **C SI 迁移消融**：全冻结/只 ASaB/全量微调/零样本/few-shot(50,200,1000)。
- **D 跨域配对**：GRF→四足、GRF→导航、四足→导航（含双向）。
- **E reward 生成消融**：LLM 生成 shaping vs 纯零 shaping（仅 SP 条件输入）。
- **F 规模**：SP 维度 4/8/12、候选筛选阈值敏感性。

## 7. 评测协议
- **风格达成率**：域内指标分化度（成对 KL 或分类器 AUC）+ LLM-judge 一致率（人工 100 条抽样）。
- **SI MAE**：预测 SP 与 gold SP 的归一化绝对误差；5 seed mean±std。
- **迁移率**：few-shot 相对全量微调的 MAE 比值。
- **显著性**：配对 t-test（α=0.05，Bonferroni 校正）。

## 8. 算力与资源计划
- GPU：策略训练 3–5 GPU·天（4 卡并行 4 seed）；SI 微调小时级。总 **<5 GPU·天**。
- 存储：<50GB。
- **API 成本**：SP 生成/筛选/指令合成/互审 ≈ 5–8 万次调用；DeepSeek V4 Pro 约 $100–150，Kimi K2.6 约 $80；合计 **≤$250**。

## 9. 里程碑与时间线（按周，单人+4 卡）
- W1：环境安装 + LCDSP 复现（GRF 域基线）。
- W2–3：LLM SP 发现流水线 + PSS 筛选；与人工 SP 对比。
- W4：SI 主体 + GRF 域训练。
- W5–6：跨域迁移实验（四足/导航）。
- W7：reward 生成 + 安全检查。
- W8–9：消融与写作。
- W10：投稿（NeurIPS/ICML/AAAI 线）。

## 10. 风险与备选方案（表）
| 风险 | 概率 | 影响 | 缓解/备选 |
|---|---|---|---|
| LLM 生成 SP 不具区分性 | 中 | 高 | 用 PSS 条件熵校验筛选；失败则退回「半自动：LLM 提议 + 人确认」 |
| 跨域 SI 迁移失败 | 中 | 中 | 退化为「SP 自动发现」单点贡献单独成文 |
| 四足仿真真实度有限 | 中 | 低 | 用 MuJoCo 开源模型；明确仿真范围 |
| LLM 生成 reward 代码非法/不安全 | 低 | 中 | 静态检查 + 仿真碰撞/自毁护栏 + 白名单 API |
| LLM 指令标签噪声 | 中 | 中 | 双模型互审 + 300 条人工复核校准 |

## 11. 论文写作计划
- 目标会议：**NeurIPS 2027 / ICML 2027**（2027 年 1 月截稿）；备选 AAAI 2028（RL 方向）。方法+科学发现型。
- 差异化卖点：①首个「LLM 自动 SP 发现 + 可验证区分度筛选」；②SI 的跨域迁移协议（只改缩放头）；③零手工 reward 的端到端流水线脚本。
- 图表清单：①流水线图；②三域风格分化热力图/雷达图；③自动 vs 人工 SP 的 SEU 对比；④跨域迁移 MAE 矩阵；⑤安全审计结果表；⑥LLM 生成 reward 示例。
- 相关工作覆盖：LCDSP/SayTap、MAPL、Eurekaverse、LLM-MARL、STAIF、语言控制 RL（Voyager 技能库、Rethinking Agentic RL）。

## 12. 参考文献（只列真实核验过的 arXiv ID/DOI）
- LCDSP: (Machine Learning VII · 论文 79，AAAI 2026 收藏论文)
- SayTap: arXiv:2306.07580
- MAPL: arXiv:2606.25398
- Eurekaverse: arXiv:2411.01775
- LLM-MARL: arXiv:2506.04251
- STAIF: arXiv:2607.22649
- PPO: arXiv:1707.06347
- Voyager: arXiv:2305.16291
- Rethinking Agentic RL in LLMs: arXiv:2604.27859
- DeepSeek-R1: arXiv:2501.12948
- Kimi K1.5: arXiv:2501.12599
- Qwen2.5-VL: arXiv:2502.13923
