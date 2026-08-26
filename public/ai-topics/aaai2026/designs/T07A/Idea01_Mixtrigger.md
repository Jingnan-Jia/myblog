# 实验设计书：Mixtrigger —— 跨模型可迁移的多目标混合后门（视觉触发 × 文本触发协同）

## 0. 摘要

本实验设计书把 AAAI 2026 收藏论文 MTAttack（Computer Vision X · 论文 1 · MTAttack: Multi-Target Backdoor Attacks Against Large Vision-Language Models，arXiv:2511.10098）从"单模态像素触发"升级为"视觉触发 + 文本触发双通道协同"的多目标后门框架。核心创新有三：① 双模态触发对（图像扰动 + 指令子串）共享同一代理类空间联合优化，实现"触发要么全在、要么全无"的一一映射；② 把 PSP/TPA 约束改造成"跨模型代理空间一致性"，用多个开源 LVLM 编码器平均梯度提升触发器跨模型迁移性，正面回应图像越狱/后门跨 VLM 迁移性差的已知痛点（arXiv:2407.15211）；③ 文本触发器由 LLM 合成"语义自然"的伪触发短语而非随机 token，规避困惑度检测。方法全部可在 4×L40（192GB）上用 LoRA 完成，总预算约 5 GPU·天，投稿目标 CCS 2026 / USENIX Security 2027。

## 1. 研究背景与动机

### 1.1 问题定义

威胁模型：gray-box——攻击者能访问受害 LVLM 的视觉编码器（CLIP/EVA 等公开权重）与文本 tokenizer，能向受害者微调流水线注入有限数量中毒样本（指令微调阶段），但无权修改微调脚本或优化器。攻击目标：植入 N 个双模态触发对 `{ (δ_i^v, δ_i^t) }`（视觉扰动 + 文本子串），建立与 N 个预设错误输出概念的一一映射，使干净输入输出正常、仅带视觉触发、仅带文本触发、双触发兼有四种情形都稳定触发对应目标。攻击者不控制模型部署后的任何流程。

关键攻击面：LVLM 的输入通道同时含图像与文本，单通道防御（视觉扰动检测 或 文本困惑度过滤）只能封堵一例；双通道协同可在任一侧被防御时仍通过另一侧激活，形成"模态冗余"。

### 1.2 相关工作不足

- **（Computer Vision X · 论文 1 · MTAttack）**：首个 LVLM 多目标后门，PSP+TPA 约束解决多触发干扰，但触发器为纯像素扰动、单模态，且跨模型迁移只做经验性验证，未系统设计迁移约束。
- **（Philosophy and Ethics of AI · 论文 18 · 6DAttack）**：把后门引入 6DoF 姿态，仍是单模态（3D 物体触发）连续参数控制，与文本通道无关。
- **VL-Trojan（arXiv:2402.13851）**：指令级文本后门，单模态。
- **Revisiting Backdoor Attacks against LVLMs from Domain Shift（arXiv:2406.18844）**：域偏移视角，仍以视觉触发器为主。
- **Failures to Find Transferable Image Jailbreaks Between VLMs（arXiv:2407.15211）**：实证显示图像越狱跨 VLM 迁移性差——说明"仅靠视觉通道"的触发器天然受编码器差异影响，需显式迁移约束。
- **Universal Jailbreak Backdoors from Poisoned Human Feedback（arXiv:2311.14455）**：RLHF 阶段文本触发后门，无视觉侧。
- **空白总结**：现有后门要么纯视觉、要么纯文本；没有工作把两通道联合优化并让触发器共享代理类空间，更没有为跨模型迁移设计的代理空间一致性约束。这是本文的核心差异化点。

### 1.3 为什么是现在、为什么你的环境适合做

- **时机**：MTAttack 刚建立多目标后门框架与 TCR 度量，把其"单模态"扩展为"双模态协同 + 迁移约束"是自然且低风险的下一跳；跨模型迁移痛点在 arXiv:2407.15211 被反复强调，2026 年正是评估迁移性攻击方法学的成熟期。
- **环境**：LLaVA-1.5-7B / Qwen2.5-VL-7B / MiniGPT-v2 均可在 4×L40 上 LoRA 微调（单卡即可推理 7B）；COCO/Flickr30K 可离线下载；DeepSeek V4 Flash 负责合成自然文本触发短语，Kimi K2.6 负责 LLM-as-judge 判定目标概念命中——本地算力与 API 分工正好匹配本设计。
- **可复现性**：所有组件（模型、数据集、触发器优化）均开源或可合法获取，无合规障碍。

## 2. 研究目标与可验证假设

1. **H1（双模态触发可达性）**：存在双模态触发对使"仅视觉 / 仅文本 / 双通道"三档都能以高概率激活对应目标概念。
   - 成立时可观测结果：三档 ASR 均 ≥85%，且双通道档 ASR ≥ 任一单通道档；TCR ≤ 10%。
2. **H2（迁移性约束有效）**：跨模型代理空间一致性正则使"在 LLaVA 学、在 Qwen2.5-VL/MiniGPT-v2 激活"的迁移 ASR 显著高于无该正则的 MTAttack 风格基线。
   - 成立时可观测结果：迁移 ASR 提升 ≥15 个绝对百分点，且干净任务 CIDEr 下降 ≤2%。
3. **H3（隐蔽性）**：LLM 合成的语义自然文本触发 + 受限像素扰动能通过困惑度过滤与 PSNR 阈值检测。
   - 成立时可观测结果：触发文本的困惑度与干净文本同分布（p≥0.1，两样本 KS 检验），PSNR ≥ 35dB。
4. **H4（对防御的鲁棒性）**：双通道触发对"仅视觉扰动检测""仅文本过滤器""两者皆有"三类防御均需同时被部署才能显著降低 ASR。
   - 成立时可观测结果：任一单侧防御下 ASR 仍 ≥50%；双侧防御下 ASR 下降 ≥50% 需同时检测两通道。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据/对抗样本流水线

1. **干净集**：COCO 2014 train2014（约 8.3 万张）与 Flickr30K 各抽 1.5 万张，按 MTAttack 口径构造视觉指令样本 `(image, instruction, response)`；用 LLM 批量生成通用指令模板 200 条（"Describe this image" / "What is shown?" 等中英双语）。
2. **目标概念集**：从 COCO 类别与 Common Object 语义中选 N∈{1,2,4} 组互斥目标概念 c_i（如 "a red Ferrari" / "a crashed airplane" / "a spaceship"），对应错误文本输出模板。
3. **文本触发合成（DeepSeek V4 Flash）**：对每个目标概念生成 50 条候选自然触发短语（如 "sync mode echo" / "the green ledger"），约束：① 长度 3–6 token；② 与领域词汇无关（避免误触发）；③ 困惑度低。过滤规则：perplexity ≤ 干净指令分布 p95；与安全/色情词表交集为空；不与目标概念语义相关（CLIP 文本编码器 cos < 0.3）。最终每概念保留 10 条作为文本触发候选池。
4. **像素触发器**：初始化为 16×16 贴片或全图 ℓ∞≤8/255 扰动，在干净集 D'_0（与微调集 D_0 不相交）上联合优化（见 3.2）。
5. **中毒样本生成**：对每张训练图，以概率组合四种情形——仅视觉触发、仅文本触发、双触发、干净——比例为 1:1:1:1，总中毒样本 ≈1 万条/概念，指令拼接文本触发到固定位置（system 段末尾）。
6. **数量**：训练集 1.2 万条/概念 × N；验证集 1000 条/概念（含干净与三档触发）。

### 3.2 攻击/算法设计

**阶段一：双模态触发联合优化（沿用并扩展 MTAttack 的 PSP/TPA）。**

对每个触发器 δ_i = (δ_i^v, δ_i^t)：

- **代理类空间划分（PSP 扩展）**：为每个触发建立代理类 c_i^proxy（unseen class），优化使中毒图像的特征与代理类对齐、远离干净图像类；文本侧把触发子串的 embedding 投影到同一代理类向量。约束：
  - `L_psp = max(0, cos(z_i^v, c_i^proxy) - cos(z_i^v, c_j^proxy))` 对 ∀j≠i，其中 z_i^v = g_φ(x + δ_i^v)。
  - 文本侧 `z_i^t = E_t(δ_i^t)`（token embedding 均值），与代理类做 softmax 分类损失。
- **触发原型锚定（TPA）**：维持触发原型 μ_i，惩罚 `‖g_φ(x+δ_i^v) - μ_i‖²` 与 `‖E_t(δ_i^t) - μ_i^t‖²`。
- **跨模型代理空间一致性（本文新增）**：在 K=3 个公开编码器 {CLIP ViT-L/14, SigLIP, EVA-CLIP} 上同时计算上述约束并取平均梯度：`∇ = (1/K)Σ_k ∇ L_k(δ_i)`。作用：把触发器优化导向"对编码器架构差异鲁棒"的解。
- **总优化目标**：`min_δ Σ_i [ L_gen(δ_i) + λ1 L_psp + λ2 L_tpa + λ3 L_transfer ]`，其中 L_gen 是带触发指令的生成 CE 损失（使用代理受害者头一次性近似），超参初值 λ1=1.0, λ2=0.5, λ3=1.0。
- **优化器**：Adam，lr=1e-2（像素）/ 1e-3（文本 embedding 梯度上升），各 200 步，batch 64。像素扰动每 10 步投影回 ℓ∞≤8/255 球。

**阶段二：后门植入（LoRA 视觉指令微调）。**

- 用阶段一的触发器生成中毒样本，与干净样本混合（比例 ≈5% 中毒，参照 MTAttack 口径），对 LLaVA-1.5-7B 做 LoRA（r=64, α=128, target: q_proj/v_proj/k_proj/o_proj + mlp）指令微调。
- 损失：标准 response CE 交叉熵。

### 3.3 训练/注入流程

- 框架：HuggingFace TRL + DeepSpeed ZeRO-3 + gradient checkpointing。
- 4×L40 并行，per-device batch=8，梯度累积 4 → 全局 batch=128。
- 学习率 2e-5，warmup 3%，余弦衰减，epoch=2（约 1 万混合样本/轮）。
- 耗时估算：单轮 ≈2–3 GPU·小时；触发器联合优化（3 编码器前向）≈1 GPU·天；总计 ≈2 GPU·天完成单模型全流程。

### 3.4 评测与防御对照流程

- **攻击侧**：对每个受害模型评测四档（干净 / 仅视觉 / 仅文本 / 双触发）× N 目标 → ASR 与 TCR 矩阵。
- **迁移侧**：在 Qwen2.5-VL-7B 与 MiniGPT-v2 上直接加载（不微调）→ 迁移 ASR。
- **防御对照**：实现三档防御——① 视觉扰动检测（Fourier 谱异常 + PSNR 阈值）；② 文本困惑度过滤器（perplexity > 阈值则拒绝）；③ 双侧组合；报告每档下 ASR 下降率，验证 H4。

## 4. 数据集/目标模型细节

| 项 | 细节 |
|---|---|
| 受害模型 | LLaVA-1.5-7B（植入 + 训练），Qwen2.5-VL-7B、MiniGPT-v2（迁移测试） |
| 干净数据集 | COCO 2014 train2014、Flickr30K（各 1.5 万张） |
| 目标概念 | N∈{1,2,4}，互斥概念组 3 套（交通工具/物品/场景类各 1 套） |
| 触发器 | 像素：16×16 贴片或全图扰动（ℓ∞≤8/255）；文本：3–6 token 自然短语 |
| 基准 | AdvBench 不做为多模态后门基准；评测复用 MTAttack 口径（Flickr30K/COCO 指令集） |
| 许可 | COCO（CC-BY 4.0）、Flickr30K（研究许可）、模型权重均开源 |

## 5. 基线复现

| 基线 | 官方代码/出处 | 复现步骤 | 预期指标（在 LLaVA-1.5-7B，N=4） |
|---|---|---|---|
| MTAttack | arXiv:2511.10098 补充材料 | 单视觉触发 + PSP/TPA，LoRA 微调 | ASR≈80–90%，TCR 低 |
| VL-Trojan | arXiv:2402.13851 | 指令级文本后门（单模态） | 文本侧 ASR 高、视觉侧无触发 |
| BadNets（适配） | 经典 | 固定贴片触发 | ASR 高但易被 PSNR 检测、TCR 高 |
| ReFool 风格（域偏移） | arXiv:2406.18844 | 域偏移扰动触发 | 迁移性中 |

统一口径：所有基线用同一 1 万条中毒/干净混合训练集、同一 LoRA 配置、同一评测脚本；ASR 均以 Kimi K2.6 judge 判定是否命中目标概念（双人评审一致性 ≥0.9 才采纳）。

## 6. 实验矩阵

- **A（主实验）**：Mixtrigger vs 3 基线 × N∈{1,2,4} × 四档触发 → ASR/TCR 全表。
- **B（消融）**：① 无文本触发（纯视觉，等价 MTAttack）；② 无视觉触发（纯文本）；③ 无跨模型一致性正则 λ3=0；④ 文本触发为随机 token vs LLM 合成——四组消融定位各组件贡献。
- **C（跨模型迁移）**：在 LLaVA 训练 → 分别在 Qwen2.5-VL-7B / MiniGPT-v2 / LLaVA-Next-7B 测迁移 ASR。
- **D（对防御鲁棒性）**：三档防御 × 四档触发 × ASR 下降率，绘制"防御-触发模态"矩阵。
- **E（干净性能）**：干净集 CIDEr 分数 + GPT-4o 评分，验证中毒不明显损伤正常能力。

## 7. 评测协议

- 主指标：ASR（含四档触发口径）、TCR（触发-目标错配率，沿用 MTAttack 定义）、迁移 ASR、干净 CIDEr。
- 隐蔽性：触发图像 PSNR/SSIM、Fourier 谱偏移；触发文本 perplexity（KS 检验 vs 干净分布）。
- 统计：每组实验固定 3 个随机种子（42/2024/2026），报告均值±标准差；对 ASR/TCR 做配对 t 检验（α=0.05）与 95% Bootstrap 置信区间。
- 判定器：Kimi K2.6 judge + 人工抽检 200 例（一致性 Cohen's κ ≥ 0.8）。

## 8. 算力与资源计划

- 触发器优化（3 编码器并行）≈1 GPU·天；LoRA 植入（LLaVA-1.5-7B，2 epoch）≈2 GPU·天；迁移评测 + 防御矩阵（3 模型 × 4 档 × N=4）≈1 GPU·天；数据合成与 judge 评测：DeepSeek V4 Flash ≈30 万 token、Kimi K2.6 ≈20 万 token，成本合计 < 15 美元。
- 存储：COCO/Flickr30K ≈ 40GB；模型权重 + LoRA 检查点 ≈ 60GB；合计 < 120GB，4×L40 本机 NVMe 足够。
- 总预算 ≈5 GPU·天（含 20% 重跑余量）。

## 9. 里程碑与时间线（按周，单人 + 4 卡）

| 周 | 任务 |
|---|---|
| W1 | 环境搭建、数据下载与指令构造；DeepSeek Flash 合成文本触发候选池并过滤；复现 MTAttack 基线（单视觉触发） |
| W2 | 实现双模态触发联合优化（PSP/TPA + 跨模型一致性正则）；完成 H1/H3 可行性验证 |
| W3 | LoRA 植入全流程（LLaVA-1.5-7B）；三档触发 ASR 主实验；基线全复现 |
| W4 | 跨模型迁移矩阵（C）、防御鲁棒性矩阵（D）；消融 B；指标整理与显著性检验 |
| W5 | 论文初稿（CCS 2026 版）；图 1–4 成型；buffer 周（复现校验、人工抽检） |

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| 双触发同时激活在长指令下概率下降 | 中 | 高 | 按触发组合做课程式训练（先单后双）；或把文本触发位置设为固定强位置（system 末尾） |
| 迁移 ASR 不达预期（跨模型一致性正则失效） | 中 | 高 | 增加编码器种类到 5 个；改用"触发-代理类"跨模型蒸馏；报告"迁移上限"（在目标模型做 100 步触发微调） |
| 文本触发被困惑度检测 | 低 | 中 | 用更大搜索空间（beam 采样 + perplexity 约束优化）；降级为"低频 token 组合"并报告检测代价 |
| Kimi judge 判定不一致 | 低 | 中 | 双 judge（DeepSeek Pro + Kimi）多数投票，分歧样本人工裁决 |

## 11. 论文写作计划

- **目标会议**：CCS 2026（截稿约 2026-04-25）、USENIX Security 2027（截稿约 2026-08）。若 CCS 未中，USENIX 2027 版补充更多防御对抗。
- **差异化卖点**：① 首个"双模态触发协同"多目标 LVLM 后门（视觉+文本冗余）；② 首个为跨模型迁移设计的代理空间一致性约束；③ 语义自然文本触发（对抗困惑度检测）。
- **图表清单**：图1 框架总览；图2 触发联合优化示意（代理空间可视化 t-SNE）；图3 跨模型迁移矩阵热力图；图4 防御-触发模态 ASR 矩阵；表1 主实验 ASR/TCR；表2 消融；表3 干净性能。
- **对防御的启示（必写）**：给出双通道触发下防御的评估下限——防御方必须同时检查视觉与文本通道（跨模态一致性检测），且评测基线应含"模态组合触发"；本文提供可复现的触发库作为未来防御评测用例。

## 12. 参考文献

- arXiv:2511.10098 —— MTAttack: Multi-Target Backdoor Attacks Against Large Vision-Language Models
- arXiv:2407.15211 —— Failures to Find Transferable Image Jailbreaks Between VLMs
- arXiv:2402.13851 —— VL-Trojan: Multimodal Instruction Backdoor Attacks against Autoregressive Visual Language Models
- arXiv:2406.18844 —— Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift
- arXiv:2311.14455 —— Universal Jailbreak Backdoors from Poisoned Human Feedback
- arXiv:2404.12916 —— Physical Backdoor Attack can Jeopardize Driving with Large Vision-Language Models
