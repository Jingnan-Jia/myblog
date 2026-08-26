# 实验设计书：编辑图（EditGraph）驱动的复杂指令图像编辑

## 0. 摘要

本实验设计把复杂图像编辑指令的规划显式建模为带一致性约束的「编辑图」：节点=子编辑（类型/区域/掩码），边=共享身份/风格/空间关系约束，由微调的 7B VLM planner 求解（图监督 + 指令跟随），交给现成编辑底模（InstructPix2Pix/MGIE 等）执行，并配"规划-执行-自校验"闭环（VQA 自检 + CLIP 相似度 + 身份保持）自动重做失败子编辑。全程 planner 免推理工具、执行零额外训练，配套全自动数据合成流水线（DeepSeek V4 Pro + Kimi K2.6）。目标是复杂指令编辑成功率相对现有方法提升 ≥10%，且冲突指令有可解释的约束消解。4×L40 约 5 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

输入原图 $I$ 与复杂指令 $\mathcal{C}$（含多个子任务，如"把杯子的颜色换成蓝色、把椅子移到桌子右边、保持整体暖色调"）。现有单步编辑模型处理不了；规划型方法把 $\mathcal{C}$ 拆为子指令序列但**不建模子任务间一致性**，多子编辑在共享区域冲突时无解。我们把问题形式化为编辑图 $G=(V,E)$：$V$=子编辑节点（含类型 type、区域掩码 $m$、目标描述 $d$），$E$=一致性边（share_identity 保身份、share_style 风格一致、spatial_rel 空间关系、exclude 互斥）。目标是求满足全部约束的可执行规划并验证执行结果。

### 1.2 相关工作不足

- 收藏论文 `X-Planner`（Computer Vision XI·73·Beyond Simple Edits: X-Planner for Complex Instruction-Based Image Editing）：用 CoT 拆子指令 + 自动掩码，但子任务一致性未显式建模，多子编辑冲突无解，也无"规划-执行-自校验"闭环。
- `RePlan`（arXiv:2512.16864）做区域规划，同样缺约束层面的显式表示。
- 单步编辑 `InstructPix2Pix`（DOI 10.1109/CVPR52729.2023.01764）、`MGIE`（arXiv:2309.17102，CVPR 2024）、`Emu Edit`（arXiv:2311.10089）、`MagicBrush`（arXiv:2306.10012，NeurIPS 2023，人工标注训练集）、`InstructEdit`（arXiv:2305.18047）均无规划/约束层。
- 复杂度可控基准 `Complex-Edit`（arXiv:2504.13143）确立"复杂指令"评测，但无一致性约束协议。

空白：**编辑图表示 + 约束求解 + 规划-执行-自校验闭环**在指令编辑上未被系统解决。

### 1.3 为什么是现在、为什么你的环境适合做

X-Planner 2026 年刚被 AAAI 录用，直接接力其"数据流水线自动化"资产做约束层增量，时效性强。本环境：7B planner 训练 4×L40 够用；执行/校验用现成模型并行推理；DeepSeek V4 Pro + Kimi K2.6 负责数据合成与歧义消解。数据靠 API、训练 ≤6 GPU·天，符合"小训练 + 大合成"打法。

## 2. 研究目标与可验证假设

- **H1**：编辑图约束显式建模降低冲突子编辑的失败率。*可观测结果*：含冲突指令（同一对象改颜色又换位置）的成功率相对 X-Planner 式序列规划 +20% 以上，且约束消解有解释。
- **H2**：自校验闭环优于开环。*可观测结果*：开环 vs 闭环，编辑成功率 +8–12%，重做次数与质量正相关。
- **H3**：图监督比纯序列 CoT 监督训练出的 planner 更准。*可观测结果*：planner 输出与 GT 子编辑集合的 F1、约束边 IoU 更高（+5 点以上）。
- **H4**：多底模集成优于单一底模。*可观测结果*：InstructPix2Pix + MGIE 投票/择优后，最终图像 CLIPScore + 身份保持同时优于任一单底模。

## 3. 总体方法设计

### 3.1 数据流水线（全自动合成）

**输入**：MagicBrush（arXiv:2306.10012）人工标注三元组（源图/指令/目标图）作种子 + COCO/SA-1B 子集。

**Step A — 复杂指令构造（DeepSeek V4 Pro + Kimi K2.6）**：
```
Compose a COMPLEX editing instruction combining 2-4 atomic sub-edits from {recolor, add_object, remove_object, reposition, style_change, texture_change, count_change} over a given scene description.
Ensure at least one pair shares an identity/style/spatial constraint. Ambiguities are OK but must be resolvable.
Scene: {caption}
```
对每条生成 3 个变体，保留约束类型可判定（用模板正则初筛）。

**Step B — 编辑图标注（Kimi K2.6）**：把指令解析为 JSON：
```
{"nodes":[{"id":1,"type":"recolor","mask_hint":"cup","target":"blue"}],
 "edges":[{"type":"share_identity","nodes":[1,2]},{"type":"spatial_rel","from":1,"to":3,"rel":"left_of"}],
 "conflicts":[{"nodes":[1,4],"resolution":"apply 1 first, then 4 on unaffected region"}]}
```

**Step C — 掩码生成**：对每个节点的 mask_hint 用 GroundingDINO（arXiv:2303.05499）出框 → SAM（arXiv:2304.02643）出 mask，IoU 质量过滤（相对 GT 掩码 IoU>0.7 才保留为训练样本；MagicBrush 有 GT 掩码可直接用）。

**Step D — 过滤与数量**：三档复杂度（2/3/4 节点）；约束类型均衡（identity/style/spatial/exclude 各 ≥20%）；目标 **12k 训练指令图 + 2k 验证**；人工抽检 200 例编辑图合格率 ≥85%。

### 3.2 模型/算法设计

- **Planner**：Qwen-VL-7B（或 DeepSeek-VL 等开源 VLM）LoRA。输入：指令 $\mathcal{C}$ + 原图（+ 可选掩码层提示）。输出：编辑图 JSON。loss = 序列交叉熵（指令跟随）+ 图结构辅助 loss：
  - 节点 loss：`类型分类头`（在 LLM 输出 token 上），$\mathcal{L}_{type}=-\log p(type_i)$；
  - 边 loss：`share_identity/share_style/spatial/exclude` 四类约束存在性二分类（多标签），$\mathcal{L}_{edge}=BCE$；
  - 掩码 loss：对每个节点，用输出 token 提示 SAM 生成掩码（SAM 冻结），IoU loss $\mathcal{L}_{mask}=1-\text{IoU}(\hat{m},m_{GT})$。
  总 loss：$\mathcal{L}=\mathcal{L}_{seq}+\lambda_t\mathcal{L}_{type}+\lambda_e\mathcal{L}_{edge}+\lambda_m\mathcal{L}_{mask}$，初值 $\lambda_t=\lambda_e=0.3,\lambda_m=0.5$。

- **约束求解器（推理期，非学习）**：把 planner 输出的图送约束传播：同一对象上的 color+position 冲突 → 按优先级（对象内：先改外观后位移；跨对象：保持空间关系的对象组不动）消解；share_identity 边 → 用同一身份 embedding 约束底模；exclude 边 → 错开区域执行。图着色对区域互斥（重叠 mask 的节点串行化）。

- **执行层**：底模候选 InstructPix2Pix / MGIE / Emu Edit（未开源则用 InstructPix2Pix 替代）。逐节点执行，输出中间图序列。多底模集成：每节点跑 2 底模，按 CLIPScore 择优。

- **自校验闭环**：执行后 (1) VQA（DeepSeek V4 Flash）逐节点确认"该编辑是否落实"；(2) 身份保持：face/CLIP 相似度与 $I$ 比对；(3) 若某节点失败 → 局部重做（最多 2 轮）。全部通过则输出。

### 3.3 训练流程

- Planner QLoRA（4-bit），4×L40，batch=8，grad-acc=4 → 128；lr=2e-4 cosine；总 12k 步（~3 轮 over 12k 样本）；max_seq_len 4096（图 token + JSON）。
- 掩码头：SAM 冻结，只训练 adapter（轻量）。
- 早停：验证集编辑图 F1。

### 3.4 推理与评测流程

推理：$I+\mathcal{C}$ → planner → 约束求解 → 逐节点执行（多底模并行 4 卡）→ 自校验 → 输出 $I'$。评测走统一评测框架（§7）。

## 4. 数据集细节

| 数据集 | 用途 | 划分 | 许可 |
|---|---|---|---|
| MagicBrush（arXiv:2306.10012） | 种子三元组 + 评测 | 官方 train/test | 学术 |
| Complex-Edit（arXiv:2504.13143） | 复杂指令评测 | 官方 test | 学术 |
| COCO（arXiv:1405.0312）+ SA-1B 子集 | 合成数据源 | 训练 6k 图 | 学术 |
| **EG-Bench（本工作）** | 复杂指令 + 编辑图 + 掩码评测集 | 1k 测试（分 3 档复杂度/4 类约束） | 随论文开源 |
| I2P（InstructPix2Pix 论文附带，DOI 10.1109/CVPR52729.2023.01764） | 附加评测 | 官方 | 学术 |

## 5. 基线复现

| 基线 | 官方代码/权重 | 复现要点 |
|---|---|---|
| InstructPix2Pix | https://github.com/timothybrooks/instruct-pix2pix | 官方权重直接推理 |
| MGIE | https://github.com/tsxuehu/MGIE | 官方权重 |
| MagicBrush-微调 InstructPix2Pix | 同上 | 用 MagicBrush 训练集微调 |
| Emu Edit | 官方权重（开源） | 若权重可用则评测 |
| InstructEdit | https://github.com/QianWangX/InstructEdit | 官方管线 |
| RePlan（arXiv:2512.16864） | 见论文官方仓库 | 区域规划管线 |
| X-Planner（CV XI·73） | 见论文（开源后） | 开源后复现，未开源则按其描述自建序列规划对照 |
| Ours | — | §3 全量 |

**预期指标表**（Complex-Edit 官方协议 + EG-Bench；数值以复现为准）：

| 方法 | 编辑成功率↑ | CLIPScore↑ | 身份保持(ID sim)↑ | 掩码 IoU↑ | 冲突指令成功率↑ |
|---|---|---|---|---|---|
| InstructPix2Pix | 低 | 基准 | 中 | — | 低 |
| MGIE | 中 | 略高 | 中 | — | 低 |
| RePlan | 中高 | 高 | 中 | 中 | 中 |
| X-Planner(自建) | 中高 | 高 | 中 | 中 | 中 |
| **Ours** | **+10 点以上** | **≥X-Planner** | **最高** | **最高** | **显著最高** |

统一口径：全部用相同评测集、相同 VQA judge 模板与 CLIP 版本（CLIP-L/14@336）。

## 6. 实验矩阵

- **A. 主实验**：全基线 + Ours，Complex-Edit / EG-Bench 全指标。
- **B. 消融**：B1 约束边有无（图→序列退化为 X-Planner 式）；B2 图监督 vs 仅序列监督；B3 自校验闭环有无、重做轮数；B4 约束求解器 vs 贪心串行；B5 多底模集成 vs 单底模；B6 数据量 4k/8k/12k。
- **C. 鲁棒性**：C1 指令带歧义/隐含约束；C2 密集场景（多人/多对象）；C3 长指令（>60 token）。
- **D. 泛化性**：D1 底模换 SDXL（arXiv:2307.01952）+ 编辑管线；D2 planner 换更小模型（Qwen-VL-4B）；D3 跨域（人物肖像 vs 静物）。

## 7. 评测协议

- **指标定义**：编辑成功率 = 所有节点经自校验确认通过的比例（VQA 二分类，DeepSeek V4 Flash judge，模板固定）；CLIPScore = CLIP-L/14@336 图文余弦；身份保持 = 编辑前后对象 region 的 face 相似度（有脸时）+ CLIP 区域余弦；掩码 IoU 相对 GT；LLM-as-judge 总体质量分（1–5，V4 Pro，去偏位置随机）。
- **均值±方差**：3 个评测种子影响仅底模采样随机性；报告 mean±std。
- **显著性**：成功率差用 McNemar 检验（p<0.05）；分数差用配对 Bootstrap。
- **人工验证子集**：200 例双人盲评（Win/Tie/Loss），报告与 judge 一致率。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 数据合成（API 为主）+ 掩码生成（SAM 批量） | 1 |
| P2 | planner LoRA 训练 | 3–4 |
| P3 | 执行/校验/集成推理（多底模并行） | 1–2 |
| P4 | 全量评测 + 人工子集 | 0.5 |
| **合计** | | **≈5.5–7.5** |

存储：数据与中间图 ~120GB。API：DeepSeek V4 Pro 指令合成 ≈ $40；Kimi K2.6 编辑图标注/歧义消解 ≈ $30；V4 Flash 自校验 ≈ $15；V4 Pro 最终 judge ≈ $10。总计 ≤ **$100**。

## 9. 里程碑与时间线

| 周 | 里程碑 |
|---|---|
| W1 | 数据流水线 A–D 跑通、EG-Bench 标注 |
| W2 | 底模基线复现（InstructPix2Pix/MGIE/MagicBrush-ft） |
| W3 | planner 训练 + B2 消融 |
| W4 | 约束求解器 + 闭环实现、A 主实验 |
| W5 | 全部消融 + 鲁棒性 + 泛化 |
| W6 | 统计、人工评测、论文初稿（CVPR 2027 截稿前 5 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 现成底模成为瓶颈（改不了复杂对象） | 高 | 高 | 多底模集成 + 限定测试复杂度 + 诚实报告失败案例；主贡献收敛到规划/约束/校验 |
| 编辑图标注质量不稳 | 中 | 高 | 模板正则初筛 + 约束类型均衡 + 抽检闭环 |
| 约束求解器规则过脆 | 中 | 中 | 规则可配置 + 失败回退到序列规划 |
| 自校验误判 | 中 | 中 | VQA judge 校准（用 MagicBrush 标注对抽校准） |
| 评测判分主观 | 中 | 中 | 多 judge + 人工盲评子集 |
| X-Planner 未开源导致对照弱 | 中 | 中 | 自建忠实序列规划对照，明确说明 |

## 11. 论文写作计划

- **目标**：CVPR 2027 主投；备选 ACM MM 2027。
- **差异化卖点**：(1) 首个"编辑图 + 约束求解"的复杂指令编辑；(2) 规划-执行-自校验闭环；(3) 全自动数据流水线 + 可复现 EG-Bench（复杂指令/冲突指令子集）。
- **图表清单**：图1 编辑图示例与求解流程；图2 方法总览；图3 冲突消解案例对比（vs X-Planner/RePlan）；图4 失败案例；表1 主实验；表2 消融；表3 鲁棒性；表4 泛化；表5 人工评测。
- **相关工作覆盖**：单步编辑（InstructPix2Pix DOI 10.1109/CVPR52729.2023.01764、MGIE arXiv:2309.17102、Emu Edit arXiv:2311.10089、MagicBrush arXiv:2306.10012、InstructEdit arXiv:2305.18047）、规划编辑（X-Planner CV XI·73、RePlan arXiv:2512.16864、Complex-Edit arXiv:2504.13143）、grounding（GroundingDINO arXiv:2303.05499、SAM arXiv:2304.02643）。

## 12. 参考文献

1. X-Planner（Computer Vision XI·73）· Beyond Simple Edits: X-Planner for Complex Instruction-Based Image Editing（收藏论文）
2. RePlan: Replanning-Enhanced Large Vision-Language Models for Comprehensive Instruction-Guided Image Editing. arXiv:2512.16864.
3. Brooks et al. InstructPix2Pix: Learning to Follow Image Editing Instructions. DOI 10.1109/CVPR52729.2023.01764（CVPR 2023）.
4. Fu et al. MGIE: Multi-modal Large Language Model Guided Image Editing. arXiv:2309.17102（CVPR 2024）.
5. Sheynin et al. Emu Edit: Precise Image Editing via Recognition and Generation Tasks. arXiv:2311.10089（EMNLP 2024）.
6. Zhang et al. MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing. arXiv:2306.10012（NeurIPS 2023）.
7. Wang et al. InstructEdit: Improving Automatic Masks for Diffusion-based Image Editing With User Instructions. arXiv:2305.18047.
8. Deng et al. Complex-Edit: Complex Image Editing With Holistic Instruction Tuning. arXiv:2504.13143.
9. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training. arXiv:2303.05499.
10. Kirillov et al. Segment Anything. arXiv:2304.02643（ICCV 2023）.
11. Podell et al. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952.
12. Lin et al. Microsoft COCO: Common Objects in Context. arXiv:1405.0312.
