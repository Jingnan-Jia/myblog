# 实验设计书：细粒度幻觉三元组偏好优化（属性-关系级幻觉缓解）

## 0. 摘要

本实验设计把 MLLM 幻觉缓解从「对象级」推进到「对象-属性-关系」三元组级：用 DeepSeek V4 Pro 把图文对自动分解为 `<主体, 谓词, 值>` 三元组并锚定视觉区域，构造「改值/增删谓词」的幻觉负样本，以三元组严重度加权的 DPO/GRPO 训练 7B MLLM，并配注意力掩码只调制相关视觉 token。配套发布 CHAIR 扩展的三元组级幻觉评测协议（TH-CHAIR）。目标是在 POPE、CHAIR、MMHal-Bench 全部不下降的前提下，把属性/关系幻觉显著压低（预期属性幻觉 -30% 以上），并提供可归因到具体三元组的失败分析。全程「API 合成数据为主 + 小规模 LoRA 训练」，4×L40 约 6 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

给定图像 $I$ 与 MLLM 生成的描述 $\hat{c}$，幻觉定义为描述中包含视觉上不成立的信息。现有评测与训练主要关注**对象级幻觉**（图里没有的物体被提到），而**属性幻觉**（颜色/材质/数量错）与**关系幻觉**（空间/动作关系错）缺乏系统信号。我们将其统一为**三元组级幻觉**：把真实描述 $c$ 分解为事实三元组集合 $T_{gt}=\{(s,p,o)\}$（$s$=主体, $p$=谓词, $o$=值，含"存在性"特例 $p=存在$），模型输出解析为 $T_{pred}$，幻觉即 $T_{pred}\setminus T_{gt}$。目标：训练使 $|T_{pred}\setminus T_{gt}|$ 最小，且不牺牲对象级/整体能力。

### 1.2 相关工作不足

- 收藏论文 `HD-DPO`（Machine Learning IX·66·Adaptive Hallucination Alleviation in Multimodal Large Language Models: From Strategic Data Selection to Severity-Guided Training）首次做严重度加权 DPO，但：(a) 严重度度量依赖外部 VLM 打分；(b) 幻觉停留在对象级，未触及属性/关系。
- `POPE`（arXiv:2305.10355，EMNLP 2023）只轮询对象存在性，测不到属性/关系。
- `CHAIR`（arXiv:1809.02156，EMNLP 2018）对象级 caption 幻觉指标。
- 数据侧 `LRV-Instruction`（arXiv:2306.14565，ICLR 2024）构造「合理-幻觉」指令对，但为句级，非三元组级。
- 对齐侧 `Silkie`（arXiv:2312.10665，AAAI 2024）、`LLaVA-RLHF`（arXiv:2309.14525，NeurIPS 2024，亦引入 MMHal-Bench）、`M3PO`（arXiv:2508.12458）、`MM-RLHF`（arXiv:2502.10391）均为对象级/句级偏好。

空白：**对象-属性-关系三元组粒度的评测与训练信号**无人系统化解决；`HD-DPO` 的严重度加权思路未与三元组粒度、区域锚定结合。

### 1.3 为什么是现在、为什么你的环境适合做

2024–2026 幻觉缓解赛道正从对象级走向细粒度；MME-SCI（arXiv:2508.13938）等新基准暴露 MLLM 在属性/关系上的系统性短板，正是三元组级方法的用武之地。本环境具备：(1) 4×L40 可快速 QLoRA 训练 7B；(2) DeepSeek V4 Pro 生成三元组/负样本，DeepSeek V4 Flash 批量打分，Kimi K2.6 处理长图多对象校验——**数据合成完全走 API，训练量小**，算力瓶颈低，可复现性强。

## 2. 研究目标与可验证假设

- **H1**：三元组级严重度加权 DPO 相比均匀加权 DPO，在属性/关系幻觉指标上显著更优。*成立时的可观测结果*：TH-CHAIR 属性子集 P@1 提升 ≥8 个点、CHAIRi 下降 ≥15%，且统计显著（p<0.05，见 §7）。
- **H2**：区域锚定（注意力掩码调制相关视觉 token）能进一步抑制局部幻觉且不伤害整体。*可观测结果*：加掩码 vs 不加掩码，TH-CHAIR 定位 F1 ≥ +5 点，MME/MMMU 保持不掉（Δ≤±1.0）。
- **H3**：MLLM 自动三元组分解 + 验证器过滤可在可接受成本内产出高质量训练信号。*可观测结果*：人工抽检 200 例三元组合格率 ≥85%，过滤后负样本与图像实际不符率 ≤10%。
- **H4**：三元组级方案在跨架构（LLaVA-1.5 与 Qwen2-VL-7B）上可迁移。*可观测结果*：两骨干下 TH-CHAIR 均有改善，且改善幅度同向。

## 3. 总体方法设计

### 3.1 数据流水线

**输入**：COCO train2017（~118k 图，含 80 类 mask/框）子集 + LLaVA-Instruct 指令数据（沿用 LLaVA 官方的 665K，仅取有明确对象属性的子集）。

**Step A — 三元组分解（DeepSeek V4 Pro，批量）**。Prompt 模板（英文生成，中文评测时另配模板）：
```
Task: extract atomic fact triples from the image+reference caption.
Given image I and caption C, output JSON array of {"subject","predicate","value","region_box"}.
Rules:
- predicate in {exists, has_color, has_material, count, spatial, action}.
- region_box = bounding box [x1,y1,x2,y2] (0-1 normalized) of the subject if locatable, else null.
- only facts strictly supported by the image; output [] if none.
Caption: {c}
```
对每张图生成 5–15 个三元组。用 GroundingDINO（arXiv:2303.05499）对 subject 文本做框定位，取与 GT 框 IoU>0.5 的为可信锚；无框者标记 region=null 并在训练中跳过掩码调制。

**Step B — 幻觉负样本构造（DeepSeek V4 Pro）**。对每个三元组按 4 类扰动各生成 1 个负样本：改值（颜色/材质换错）、数量增减、空间关系反转、谓词增删（凭空加一个对象或关系）。Prompt 模板：
```
Given triple (s,p,o), generate ONE plausible-but-false variation so the triple becomes hallucinated.
Output a full caption sentence containing the false triple. Do NOT change other triples.
```

**Step C — 严重度打分（DeepSeek V4 Flash 作为 judge）**。每对 (正样本, 负样本) 生成一个解释 + 1–5 严重度分：
```
Which error is more severe for the user: wrong object (exists), wrong attribute, wrong count, wrong spatial relation?
Score 1-5 (5=most severe: violates physics/anatomy or introduces non-existent object; 3=attribute error; 1=minor count off-by-one).
```
人工校验 200 例校准打分 prompt 后冻结模板。

**Step D — 过滤规则**：
1. 负样本与图像内容经 DeepSeek V4 Flash「图文一致性校验」得分 <3/5 则弃用（防负样本被模型一眼看穿）；
2. GroundingDINO 在负样本主体上定位不到框且该三元组是 exists 型 → 保留（这正是存在性幻觉负样本）；
3. 三元组合格率按 H3 抽检；
4. 数量目标：最终训练集 ≈ **60k 三元组偏好对**（COCO 15k 图 × 4 扰动），严重度分布强制均衡（1–5 各 ≈20%），验证集 2k 对。

### 3.2 模型/算法设计

- **骨干**：LLaVA-1.5-7B（Vicuna-7B + CLIP-ViT-L/336，arXiv:2310.03744）。LoRA rank=64, alpha=128, target 含 attention 的 q/k/v/o 与 MLP。
- **DPO 损失（按三元组严重度加权）**：
  $\mathcal{L}=-\mathbb{E}_{(x,y_w,y_l,w_t)}[\,w_t\cdot\log\sigma(\beta\,\Delta r(x,y_w,y_l))\,]$，
  $\Delta r=\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$。
  其中 $w_t=1+\alpha(s-3)/2$，$s\in\{1..5\}$ 为严重度，$\alpha=0.8$ 初值（消融 0/0.4/0.8/1.2）。
- **区域锚定（注意力掩码调制）**：训练时对负样本三元组 $t$ 的 subject 锚框 $b$，在 LLM 注意力 score 上加 soft mask：
  $A_{i,j}\leftarrow A_{i,j}+\lambda\,m_j$，$m_j=1$ 当第 $j$ 个视觉 token 中心落在 $b$ 内，$\lambda=-4$（抑制）；同时对正样本对应区域 $\lambda=+2$（增强）。实现为仅对视觉-文本注意力的微调钩子，避免改架构。
- **GRPO 变体（备选）**：若 DPO 调参不收敛，改 GRPO：组内 $\{y_w,y_l\}$ 收益 $r=w_t\cdot(\text{judge 偏好分})$，组归一化。

### 3.3 训练流程

- QLoRA 4-bit（bitsandbytes），4×L40，FSDP + offload；batch=8/卡，grad-acc=8 → 有效 batch 256；lr=2e-4（AdamW，cosine），warmup 3%；LoRA dropout 0.05；max_seq_len 2048。
- 步骤：Step1 用 LRV-Instruction 负样本 + 本三元组数据混合 SFT 1 epoch（稳住格式，~4k 步）；Step2 DPO 15k–25k 步（~1–1.5 epoch 于 60k 对），每 500 步在验证 2k 对上算 TH-CHAIR。
- 显存：7B QLoRA ≈ 20GB/卡，batch=8 下 4 卡无压力。

### 3.4 推理与评测流程

- 推理：LLaVA-1.5 pipeline，生成 temperature=0.2、top_p=0.9、max_new_tokens=512。
- 解析：输出 JSON 三元组 → 与 GT 对比；解析失败时用 DeepSeek V4 Flash 兜底结构化。
- 评测项：POPE（random/popular/adversarial）、CHAIRs/CHAIRi、MMHal-Bench、MME、MMMU、新增 TH-CHAIR（见 §4）。

## 4. 数据集细节

| 数据集 | 来源 | 用途 | 划分 | 许可 |
|---|---|---|---|---|
| COCO train2017/val2017（arXiv:1405.0312） | 官方下载 | 三元组分解 + 锚框 | train 15k 图 / val 用于 TH-CHAIR 构建 | CC-BY-4.0（图片版权归原作者） |
| LLaVA-Instruct-665K（arXiv:2304.08485） | LLaVA 官方 | SFT 混合 | 取对象属性相关子集 | 学术 |
| LRV-Instruction（arXiv:2306.14565） | 官方 | SFT 负样本 | 全量 | 学术 |
| POPE（arXiv:2305.10355） | 官方 | 对象级评测 | 官方固定划分 | 学术 |
| MMHal-Bench（出自 arXiv:2309.14525） | LLaVA-RLHF 仓库 | 评测 | 官方固定 | 学术 |
| **TH-CHAIR（本工作新增）** | COCO val + MLLM 生成 caption | 三元组级幻觉评测：800 图 × 2 模型 caption，标注 `<s,p,o>` GT 与 region | 与训练图无重叠 | 随论文开源 |

**TH-CHAIR 构建**：取 COCO val2017 随机 800 图（排除训练用 15k 图），DeepSeek V4 Pro 生成 GT 三元组 + 锚框，GroundingDINO 复核；指标：$TH$-$CHAIR_i = \frac{|\bigcup_pred T_{pred}\setminus T_{gt}|}{|\bigcup_pred T_{pred}|}$、$TH$-$CHAIR_s$（按句子算）、属性/关系子集 $TH$-$CHAIR_i^{attr/rel}$、三元组定位 F1（subject 框命中）。

## 5. 基线复现

| 基线 | 官方代码 | 复现步骤要点 |
|---|---|---|
| LLaVA-1.5-7B | https://github.com/haotian-liu/LLaVA | 官方权重直接评测（基线 anchor） |
| Silkie（arXiv:2312.10665） | 官方权重（huggingface: RLHF-V） | 加载权重评测，不重训 |
| LLaVA-RLHF（arXiv:2309.14525） | https://github.com/RLHF-V/LLaVA-RLHF | 官方权重评测 |
| HD-DPO（ML IX·66） | 见论文（代码公开） | 复现数据采样+DPO，或官方权重 |
| M3PO（arXiv:2508.12458） | 官方仓库 | 同上 |
| 均匀 DPO（自建） | — | 本流水线去严重度加权 |
| Ours（本方法） | — | §3 全量 |

**预期指标表**（初值参考公开报告，单位 %，越高越好除标注）：

| 方法 | POPE ACC | CHAIRi↓ | CHAIRs↓ | MMHal | TH-CHAIRi↓ | TH-CHAIRi(attr)↓ |
|---|---|---|---|---|---|---|
| LLaVA-1.5 | ~85.9 | ~45.5 | ~33.9 | ~2.3 | 基准 | 基准 |
| Silkie | ~86.5 | ~41 | ~29 | ~2.5 | 略降 | 略降 |
| LLaVA-RLHF | ~87 | ~40 | ~28 | ~2.6 | 略降 | 略降 |
| HD-DPO | ~87.5 | ~38 | ~26 | ~2.8 | 中幅降 | 中幅降 |
| 均匀 DPO | ~87.5 | ~37 | ~25 | ~2.8 | 中幅降 | 中幅降 |
| **Ours** | **≥87.5** | **≤32** | **≤21** | **≥3.0** | **大幅降** | **显著降** |

*注：绝对数值以复现为准；核心主张是相对改善与属性/关系子集的差异化优势。* 统一评测口径：全部模型用 vLLM 加载、同一 prompt 模板、同种子解码（见 §7）。

## 6. 实验矩阵

- **A. 主实验**：Ours 全量 vs 全部基线，全指标。
- **B. 消融**：B1 严重度加权 vs 均匀（α=0）；B2 区域锚定有无（λ=0）；B3 三元组来源（仅对象级 vs 全三元组）；B4 负样本扰动类型各剔除一类；B5 训练数据量 20k/40k/60k；B6 模型规模（7B vs 13B QLoRA）。
- **C. 鲁棒性**：C1 低幻觉密度数据集（LRV 合理样本混合比例 0/10/25%）；C2 长 caption（>50 token）子集；C3 跨域（TextVQA 风格自然图像）。
- **D. 泛化性**：D1 骨干换成 Qwen2-VL-7B（arXiv:2409.12191）LoRA 复跑主实验；D2 冷启动（无 SFT 直接 DPO）。

## 7. 评测协议

- **指标定义**：POPE ACC（3 种设置）；CHAIRs/i 按官方脚本；MMHal-Bench score 按官方（LLaVA-RLHF 脚本）；TH-CHAIR 见 §4；MME/MMMU 官方 eval。
- **均值±方差**：每配置固定 5 个种子（123/2024/7/42/999），报告 mean±std；解码用 seed 控制的采样。
- **显著性检验**：对主指标用配对 Bootstrap（n=1000）或 Wilcoxon signed-rank（p<0.05）；报告效应量与 CI。
- **API judge 稳定性**：TH-CHAIR 与严重度打分各抽 200 例用 DeepSeek V4 Pro 复打，报告一致率（Kappa≥0.7 视为通过）。

## 8. 算力与资源计划

| 阶段 | 内容 | 4×L40 GPU·天 |
|---|---|---|
| P1 | 三元组分解 + 锚框（并行推理） | 1–1.5 |
| P2 | 负样本合成 + 严重度打分（API 为主，本地少量推理） | ~0.5 |
| P3 | SFT（1 epoch） | ~1 |
| P4 | DPO（15–25k 步） | 3–4 |
| P5 | 全量评测（并行跑 8 模型 × 6 基准） | ~1.5 |
| **合计** | | **≈7–8.5** |

存储：COCO + LLaVA 数据 ~150GB；中间产物（三元组/锚框 json）<5GB。API 用量/成本估算：DeepSeek V4 Pro 三元组分解 15k 图（≈10k tokens/图）≈ $60；V4 Flash 打分/judge ≈ $15；Kimi K2.6 长图校验 ≈ $10。总计 API ≤ **$100**。

## 9. 里程碑与时间线（单人 + 4×L40）

| 周 | 里程碑 |
|---|---|
| W1 | 数据管线：COCO 下载、三元组分解脚本 + 抽检合格率 |
| W2 | 负样本构造 + 严重度打分模板定稿、TH-CHAIR 标注 |
| W3 | 基线复现（LLaVA-1.5/Silkie/LLaVA-RLHF）跑通全指标 |
| W4 | SFT + DPO 初版训练、B 消融（B1/B2） |
| W5 | 全消融 + 鲁棒性 + 泛化（D1） |
| W6 | 显著性与一致率统计、图表、论文初稿（CVPR 2027 截稿前 6 周） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| MLLM 三元组分解不稳定（错解/漏解） | 中 | 高 | 验证器过滤 + 人工抽检闭环；改用 GroundingDINO 预锚再让 LLM 补谓词 |
| 负样本严重度分布失衡 | 中 | 中 | 按严重度分桶采样（stratified batch） |
| 区域锚定导致训练不稳 | 低 | 高 | 先冻结主干只训掩码钩子 λ 从 0 升温 |
| TH-CHAIR 标注主观 | 中 | 中 | 双人标注 + 仲裁；只保留 Kappa>0.8 子集 |
| POPE 等对象级指标下降 | 中 | 高 | 加入对象级偏好对混合训练；防掉点 gate（Δ<1.0 才保留） |
| API 成本/限流 | 低 | 低 | 换 V4 Flash 批量 + 本地开源 judge 兜底 |

## 11. 论文写作计划

- **目标**：CVPR 2027 主投；若 W4 主实验不达预期，降级为 NeurIPS 2026 D&B 方向（改以 TH-CHAIR 评测协议为核心贡献）。
- **差异化卖点**：(1) 首个三元组粒度的幻觉训练+评测闭环；(2) 严重度加权 × 区域锚定的正交组合；(3) TH-CHAIR 开源协议（对象/属性/关系子指标），可复现。
- **图表清单**：图1 方法总览；图2 三元组分解示例+失败案例；图3 严重度-指标曲线；表1 主实验；表2 消融（B1–B6）；表3 鲁棒性；表4 跨骨干；表5 TH-CHAIR 与 CHAIR/POPE 相关性分析。
- **相关工作覆盖**：评测（POPE arXiv:2305.10355、MME arXiv:2306.13394、MME-SCI CV VIII·15）、数据（LRV-Instruction arXiv:2306.14565）、对齐（Silkie arXiv:2312.10665、LLaVA-RLHF arXiv:2309.14525、M3PO arXiv:2508.12458、MM-RLHF arXiv:2502.10391、HD-DPO ML IX·66）、grounding（GroundingDINO arXiv:2303.05499、Ferret arXiv:2310.07704）。

## 12. 参考文献

1. HD-DPO（Machine Learning IX·66）· Adaptive Hallucination Alleviation in Multimodal Large Language Models: From Strategic Data Selection to Severity-Guided Training（收藏论文）
2. Li et al. Visual Instruction Tuning（LLaVA）. arXiv:2304.08485.
3. Liu et al. Improved Baselines with Visual Instruction Tuning（LLaVA-1.5）. arXiv:2310.03744.
4. Li et al. Evaluating Object Hallucination in Large Vision-Language Models（POPE）. arXiv:2305.10355.
5. Rohrbach et al. Object Hallucination in Image Captioning（CHAIR）. arXiv:1809.02156.
6. Liu et al. Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning（LRV-Instruction）. arXiv:2306.14565.
7. Sun et al. Silkie: Preference Distillation for Large Visual Language Models. arXiv:2312.10665.
8. Sun et al. Alleviating Hallucination in Large Vision-Language Models through Hallucination-Aware Direct Preference Optimization（LLaVA-RLHF）. arXiv:2309.14525.
9. Kim et al. M3PO: Multi-Modal Model Preference Optimization. arXiv:2508.12458.
10. MM-RLHF: The Next Step Forward in Multimodal LLM Alignment. arXiv:2502.10391.
11. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training. arXiv:2303.05499.
12. Lin et al. Microsoft COCO: Common Objects in Context. arXiv:1405.0312.
13. Wang et al. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191.
14. MME-SCI（Computer Vision VIII·15）· MME-SCI: A Comprehensive and Challenging Science Benchmark for Multimodal Large Language Models（收藏论文，arXiv:2508.13938）
15. Peng et al. Ferret: Refer and Ground Anything Anywhere at Any Granularity. arXiv:2310.07704.
