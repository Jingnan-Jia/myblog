# 实验设计书：Idea 8｜不确定度引导的课程式 RLVR 用于 GUI 语义对齐

> CurriGUI: Uncertainty-Guided Curriculum RLVR for GUI Semantic Alignment
> 资源假设：4× NVIDIA L40（192GB）；DeepSeek V4 Flash/Pro、Kimi K2.6 API。

## 0. 摘要
InfiGUI-G1 用 AEPO（多答案生成 + 自适应探索奖励 η=U/C）突破 RLVR 的语义对齐探索瓶颈，但均匀采样样本、缺课程顺序，且未利用"哪些样本该多花探索"的不确定度信号。本工作提出 CurriGUI：①以多答案分歧/自集成熵为不确定度，对高分歧难样本加大探索预算（不确定度重采样）；②课程式 RLVR——先易后难（空间对齐 → 组合语义 → 跨平台），每阶段动态推进；③把探索奖励 η=U/C 改为不确定度加权。底座 Qwen2.5-VL-3B/7B，数据含 ScreenSpot/Pro + WidgetCaptioning + 自采 Web 指令对。先 3B 起步验证（8 GPU·天）再扩 7B，总计 25-30 GPU·天。目标 grounding 准确率（尤其语义对齐类）刷新 SOTA。

## 1. 研究背景与动机

### 1.1 问题定义
GUI grounding（给定指令与截图，定位元素 bbox）从"坐标匹配"走向"语义对齐"（如"把最新一条消息标为已读"需理解组合语义）。RLVR 用可验证奖励（坐标命中）训练时，稀疏且语义模糊的样本难以探索。问题定义为：**如何让 RLVR 在 GUI 语义对齐任务上高效探索——对难样本投入更多探索预算，并按难度课程学习**。

### 1.2 相关工作不足
- 均匀采样无课程：InfiGUI-G1（Natural Language Processing III·论文 56·InfiGUI-G1）的 AEPO 用效率奖励 η=U/C 提升探索，但训练样本均匀采样，难易混排，无先易后难。
- 不确定度未用于预算分配：AER（Adaptive Exploration Reward）全局应用，未按样本不确定度加权。
- 朴素 RLVR 探索弱：GUI-R1（arXiv:2504.10458）统一动作空间 + GRPO，对语义模糊样本探索不足。

### 1.3 为什么是现在、为什么你的环境适合做
RLVR 的奖励工程（探索奖励、课程学习）正是 GUI agent 当前竞争点（GUI-Eyes、InfiGUI-G1、GUI-R1 合流）；课程学习与不确定度采样在 LLM RL 中已验证有效；ScreenSpot/Pro 与 WidgetCaptioning 公开可复现；4×L40 支持 3B/7B 全参 FSDP，先 3B 验证后扩 7B 符合算力现实。

## 2. 研究目标与可验证假设
1. **H1（不确定度重采样优于均匀采样）**：高不确定度样本加大探索预算后，语义对齐类准确率提升。
   - 成立时观测：语义对齐子集（组合指令类）准确率提升 ≥5 点；总准确率不降。
2. **H2（课程式 RLVR 优于混合式）**：先易后难分阶段训练收敛更快、最终更高。
   - 成立时观测：达到相同准确率所需 GPU·天减少 ≥30%；最终准确率更高。
3. **H3（不确定度加权 AER 优于全局 AER）**：η=U/C 按样本不确定度加权后探索更高效。
   - 成立时观测：难样本池的准确率提升 ≥8 点。
4. **H4（跨平台泛化）**：单平台训练后迁移到移动/Web/桌面三平台，3B 起步的曲线可预测地扩展到 7B。
   - 成立时观测：跨平台准确率差 ≤3 点；3B 与 7B 相对收益一致。

## 3. 总体方法设计

### 3.1 数据/轨迹采集流水线
- **训练数据**：ScreenSpot（2000+ 指令）/ScreenSpot-Pro（2400+ 难指令）+ WidgetCaptioning（描述-位置对）+ 自采 5K Web 指令对（从真实网页 DOM 生成"语义指令→元素"标注）。
- **难度标注（三阶段课程）**：Phase1 空间对齐（"点击右上角关闭按钮"）；Phase2 组合语义（"把第三个标签页重命名为会议纪要"）；Phase3 跨平台（同一语义指令在移动/Web/桌面变体）。
- **难样本扩充（LLM-as-generator）**：DeepSeek V4 Pro 基于现有指令生成语义更复杂的变体（加条件、隐含指代、多步），prompt：
  ```
  把指令 {I} 改写为 3 个更难版本：①加一个条件约束；②用隐含指代；③需要两步组合语义。
  保证元素仍可由规则/坐标可验证。
  ```
  生成 3K 条；Kimi K2.6 做难样本语义一致性筛（judge：指令-元素是否自洽）。
- **弱标注候选**：V4 Flash 批量生成候选 bbox（弱标签），配合规则过滤（DOM 文本匹配）留 2K 条。
- **数量**：基础 ~7K + 难例 3K + 弱标注 2K ≈ 12K 条。

### 3.2 系统/算法设计
- **不确定度估计**：对每样本采样 N=8 次（温度 0.7）得候选 bbox，不确定度 `U(x) = 1 − IoU(多数bbox聚类) 或自集成熵 H(p)`；高 U = 高分歧难语义。
- **重采样调度**：每 RL 轮，按 `π(x) ∝ (U(x)+ε)^γ` 采样训练批，γ 随轮数衰减（先难后收敛）；对比均匀采样。
- **课程调度**：三阶段串联，阶段阈值 = 该阶段验证集准确率 ≥85% 自动推进；阶段内也按 U 重采样。
- **不确定度加权 AER**：奖励 `R = R_align + η(x)·U(x)·R_explore`，其中 `η(x)=U(x)/C(x)`（C=上下文长度），使难样本探索奖励更大；`R_align` = 坐标 IoU 可验证命中（IoU≥0.8）。
- **GRPO 设置**：group=8，clip=0.2，rewards 为 grounding 命中 + AER 探索项；输出为 bbox 回归 + 指令改写。

### 3.3 训练流程
- **SFT 冷启**：Qwen2.5-VL-3B/7B 在基础数据上 SFT 1 epoch（3B：3 GPU·天；7B：8 GPU·天）。
- **RLVR**：先 3B 全参 FSDP 调参（8 GPU·天）→ 转 7B（15 GPU·天/阶段 × 3 阶段，但 7B 全程 3 阶段共 ~15-20 GPU·天）。lr=1e-6（RL）/1e-5（SFT），batch=64，adamw，warmup 3%。
- 合计：3B 调试 8 + 7B 训练 ~17 ≈ **25-30 GPU·天**。

### 3.4 推理与评测流程
- 评测：ScreenSpot-Pro 全量（grounding 准确率，分语义对齐/空间对齐子集）、GUIEnv 系 grounding 基准（若公开可复现）、MiniWob++（任务级可选，验证端到端）。
- 跨平台：移动（ScreenSpot 移动集）、Web（自采）、桌面（ScreenSpot-Pro 桌面集）。
- 每阶段结束在验证集早停评估，防过拟合。

## 4. 数据集/环境细节
- ScreenSpot/Pro：SeeClick（arXiv:2401.10935）公开；ScreenSpot-Pro 随 SeeClick 系发布（公开）。
- WidgetCaptioning：公开（GUI grounding 预训练数据）。
- 自采 5K Web 指令对：本团队从公开网页采样，无版权问题。
- MiniWob++：公开（OpenAI 发布，Apache-2.0）。
- 划分：训练 12K / 验证 1K / 测试用官方评测集。

## 5. 基线复现
| 基线 | 官方代码 | 复现要点 |
|---|---|---|
| InfiGUI-G1（NLP III·论文 56） | 若开源按官方 | AEPO + AER 全局效率奖励 |
| GUI-R1（arXiv:2504.10458） | GitHub: zhcsnoopy/GUI-R1 | 统一动作空间 + GRPO |
| 朴素 RLVR | 自实现 | GRPO 无探索奖励 |
| 随机采样 RLVR | 自实现 | 同 CurriGUI 但均匀采样 |
| SeeClick | GitHub: OS-Copilot/SeeClick | grounding SOTA 底座 |

统一口径：同底座（Qwen2.5-VL-3B/7B 均可）、同数据训练集、同官方评测集；报告总准确率 + 语义对齐子集。
预期表：ScreenSpot-Pro 语义对齐子集 SeeClick ~35%、InfiGUI-G1 ~48%、CurriGUI 目标 ≥52%；总准确率 ~50%。

## 6. 实验矩阵
- **A. 主实验**：CurriGUI vs 基线（总 + 子集准确率）。
- **B. 消融**：课程有无；不确定度重采样有无；AER 加权有无；γ 衰减策略；N（采样次数 4/8/16）。
- **C. 规模**：3B vs 7B；SFT-only vs RLVR。
- **D. 鲁棒性**：分辨率变化；指令歧义度（U 分布）；训练 seed 稳定性。
- **E. 泛化**：MiniWob++ 任务级；跨平台迁移（训练只含 Web）。

## 7. 评测协议
- Grounding 准确率 = IoU≥0.8 比例；子集按指令复杂度标注分列；MiniWob++ 用官方成功判定。
- 均值±方差：3 seed；配对 t 检验 p<0.05；训练早停用验证集。
- 成本上限：RLVR 每阶段 ≤8 轮、每轮采样 ≤512 样本；评测单 GPU 并行。

## 8. 算力与资源计划
- 4×L40：SFT（3B 3 + 7B 8）+ RLVR（3B 8 + 7B 15-20）≈ **30 GPU·天**（3-4 周）。
- 存储：数据 ~15GB、3B/7B 检查点 ~30GB。
- API 估算：V4 Pro 难例生成 3K ~2M token ¥60-150；V4 Flash 弱标注 2K ~1M ¥5-15；Kimi K2.6 难例 judge 1M ¥50-100。总计 ~¥120-270。

## 9. 里程碑与时间线（单人 + 4 卡）
- W1：数据整理 + 自采 5K + 难例生成/筛；基线复现（SeeClick/GUI-R1）。
- W2：SFT 冷启（3B）+ 首轮 RLVR 调参（固定 seed）。
- W3：不确定度模块 + 重采样；3B 全阶段训练 + 消融。
- W4-5：7B 训练（3 阶段）；MiniWob++ 评测。
- W6：跨平台 + 鲁棒性 + 写作。

## 10. 风险与备选方案
| 风险 | 概率 | 影响 | 缓解/备选 |
|---|---|---|---|
| RLVR 训练不稳定 | 中 | 高 | 3B 小规模定参；固定 seed + 早停；rewards 归一化 |
| 课程顺序难定 | 中 | 中 | 用验证集阈值自动推进；阈值扫描 |
| 语义对齐奖励稀疏 | 高 | 中 | 坐标规则化奖励 + AER 探索项兜底；弱标注过滤 |
| 难例生成质量参差 | 中 | 中 | Kimi judge 一致性筛 + 人工抽检 |

## 11. 论文写作计划
- 目标会议：CVPR 2027（高，GUI/多模态视觉）或 ICLR 2027（中）。
- 差异化卖点：首次把"不确定度驱动探索预算 + 课程式 RLVR"组合用于 GUI 语义对齐；明确"难样本优先探索"的机制收益；3B/7B 规模可预测扩展性。
- 图表清单：Fig.1 框架；Fig.2 不确定度分布与收益；Fig.3 课程推进曲线；Fig.4 语义对齐案例；Table1 主结果；Table2 消融；Table3 跨平台。
- 相关工作覆盖：InfiGUI-G1、GUI-Eyes、GUI-R1、SeeClick、SafeGround、Iterative Narrowing、RLVR。

## 12. 参考文献
- InfiGUI-G1（Natural Language Processing III·论文 56）
- GUI-Eyes, arXiv:2601.09770
- GUI-R1, arXiv:2504.10458
- SeeClick, arXiv:2401.10935
- SafeGround, arXiv:2602.02419
- Iterative Narrowing, arXiv:2411.13591
