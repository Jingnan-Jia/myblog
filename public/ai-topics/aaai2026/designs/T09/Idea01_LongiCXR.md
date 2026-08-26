# 实验设计书：纵向胸片时序感知报告生成（LongiCXR）

> 对应调研报告「六、研究 Idea」Idea 1｜纵向胸片时序感知报告生成。主题：T09 医疗AI与AI for Science。硬件：4×L40（192GB）。API：DeepSeek V4 Flash/Pro、Kimi K2.6。优先级：高。

## 0. 摘要（3-5 句）

本研究提出 LongiCXR，一个把"患者既往影像+既往报告"作为**显式疾病进展推理目标**的纵向胸片报告生成框架。与 PriorRG 只把先验当"条件输入"不同，我们先把进展监督信号（trend label + 证据句）由 LLM 从相邻报告对自动合成，再训练"当前发现 + 相对变化"双分支解码器，最后用带时态分量的 GRPO 临床奖励对齐。我们以 MI-CXR 纵向推理基准为主评测，辅以 RadGraph 实体/关系 F1 与医生盲评，验证"显式进展表示 + 时态奖励"是否显著优于以历史为条件的先验编码。项目将在 MIMIC-CXR/MIMIC-ABN 上以约 5–6 GPU·天完成全部训练与评测。

## 1. 研究背景与动机

### 1.1 问题定义

给定同一患者在不同时点的两张胸片（历史图 X_{t−1}、当前图 X_t）及对应的既往报告文本 H_{t−1}，任务是在当前图 X_t 上生成报告 R_t，其中报告**必须显式包含相对变化描述**（如"与 2025-01 相比，右上肺结节增大"），并对变化方向（improving / worsening / stable）做出正确判断。形式化：模型需联合建模 P(R_t | X_t, X_{t−1}, H_{t−1}) 与 P(Δ_t | R_t^{prev}, R_t^{cur})，其中 Δ_t 为进展标签。评价"变化描述是否与金标准一致"的纵向推理准确率是主目标。

### 1.2 相关工作不足

- **（Computer Vision VI·论文 44·PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation）DOI: 10.1609/aaai.v40i9.37657**：把既往影像/病史作为编码条件，但没有把"疾病进展"变成可监督的中间表示；训练目标仍是交叉熵+对比损失，无临床奖励。
- **（Machine Learning I·论文 21·Medical Vision–Language Pretraining with LLM-Guided Temporal Supervision）DOI: 10.1609/aaai.v40i24.39047（TAMM）**：用 LLM 从时序报告生成趋势标签注入预训练，但未做生成下游验证，也不含临床奖励。
- **MI-CXR [arXiv:2605.15574]**：首次建立多时点胸片纵向推理基准，证明现有模型纵向推理几乎随机水平——说明"纵向推理"是明确空白。
- **DobicVLM [arXiv:2607.18988]**：GRPO + 程序化临床奖励对齐报告，但奖励不含时态分量；**REVA-PO [arXiv:2607.10147]**：解决报告 RL 不稳定，是 GRPO 稳定化的技术参考。
- **SHOVIR [arXiv:2606.30201]**：证明 BLEU 与临床质量弱相关——本设计把纵向推理准确率与 RadGraph 作为主指标，规避评测塌缩。

### 1.3 为什么是现在、为什么你的环境适合做

- **时机**：TAMM 证明了"LLM 合成趋势标签"可行，PriorRG 证明了"先验条件"有效，MI-CXR 刚提供了公开纵向基准，DobicVLM 刚证明 GRPO+临床奖励有效——三者缺"纵向进展表示 + 时态奖励 + 纵向基准"的合流，2026 年正是补全窗口。
- **环境匹配**：7B VLM LoRA 微调 20k 样本仅需 4×L40 单机（192GB）数据并行 1–2 天；GRPO 2k 步×8 rollout 2–3 天；DeepSeek V4 Pro 批量离线合成进展标签，成本可控；无需大规模预训练。

## 2. 研究目标与可验证假设

1. **H1（进展表示有效性）**：显式 progress embedding + 双分支解码优于把既往报告仅作条件输入（PriorRG 式）。*成立时的可观测结果*：纵向推理准确率 +≥5pt，RadGraph 关系 F1 +≥2pt（同为 LoRA 基座对比）。
2. **H2（时态奖励有效性）**：在 GRPO 中加入 RadGraph 差分 + 进展一致率奖励，比不加时态分量的奖励更优。*成立时的可观测结果*：进展一致率 +≥4pt，BLEU-4 不显著下降（≥−0.5pt 内）。
3. **H3（数据合成质量）**：LLM 合成进展标签的质量（人工抽检通过率≥90%）与下游纵向推理准确率正相关。*成立时的可观测结果*：按合成质量分组（高/低置信）训练，高置信组纵向准确率高 ≥3pt。
4. **H4（泛化）**：纵向训练的模型在"未见患者的未来时点"上的变化描述仍可靠。*成立时的可观测结果*：按患者切分的 held-out 纵向子集上，进展一致率 ≥ 随机水平 15pt 以上。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

1. **纵向子集构建**：从 MIMIC-CXR v2.0 中筛选同一 patient_id、同一 view_position（PA 或 AP 单独成组）且 ≥2 次检查的序列；相邻检查间隔 0–180 天；约可得 4–6 万对（盘点后以实际为准）。MIMIC-ABN（异常子集）用于跨集泛化。
2. **进展监督合成（Stage A，DeepSeek V4 Pro 离线批量）**：
   - 输入 prompt 模板（中文注释版，运行时为英文）：把"历史报告全文 + 当前报告全文 + 检查日期差"交给 DeepSeek V4 Pro，要求输出 JSON：`{"findings_aligned":[...], "trend": "improving|worsening|stable|new_onset|resolved", "evidence_sentences":[...], "confidence":0-1}`。其中 findings_aligned 是把历史与当前报告中的 finding 逐条对齐后的差异说明。
   - **脱敏要求**：所有 prompt/输出中禁止出现患者 ID、出生日期、机构名；MIMIC 已统一脱敏，仍要求在输出侧做正则过滤（18 位 patient ID、日期、地名人名黑名单）。
   - **质量门控**：对合成结果抽检 5%（约 2000 条）由 2 名受训标注员（医学背景研究生）按 5 级评分；通过率（≥4 分）< 90% 时调 prompt 重跑该批次；每条保留 `confidence` 字段用于 H3 分组。
3. **报告变体增强（DeepSeek V4 Flash）**：对当前报告做 2 个同 finding 集改写变体，提高训练多样性（参考 Idea 12 的口径）。
4. **数量**：训练 20k 对（进展标签合成全覆盖），验证 2k，测试 2k（MI-CXR 对齐测试 1k + 自建 held-out 1k）。

### 3.2 方法设计（模块拆解）

1. **基座**：Qwen2.5-VL-7B（首选，视觉编码强）或 LLaVA-Med（对照），LoRA rank 32，作用在 q/k/v/o 与 MLP。
2. **双分支解码器**：基座正常解码当前报告 R_t；同时在 [EOT] 前插入 `<change>` 标记触发变化句解码分支。变化分支输入为：当前图 CLS 特征 ⊕ 历史图低秩特征 ⊕ 进展 embedding（由对齐后的历史报告摘要经小投影头得到）。损失为两分支交叉熵加权：`L = L_report + λ L_change`，λ=1.0。
3. **进展 embedding 结构**：`e_prog = Proj(H_{t-1})`，Proj 为 2 层 MLP（768→2048→hidden），把既往报告摘要（DeepSeek V4 Pro 生成的 3 句话摘要）映射进视觉编码器隐空间，与图像 token 拼接。
4. **GRPO 阶段（Stage C）**：策略为 LoRA 模型，参考奖励：
   - `r_clin = RadGraphF1(R_t, R_gt) 实体 F1 × 0.5 + 关系 F1 × 0.5`
   - `r_prog = 1{trend(R_t, H_{t-1}) == trend_gt}`（LLM 抽取变化句趋势，DeepSeek V4 Flash）
   - `r_lang = BLEU-4(R_t, R_gt)`（语言惩罚项，系数 0.1）
   - 总奖励 `r = 0.5·r_clin + 0.4·r_prog + 0.1·r_lang`
   - 稳定化技巧：沿用 REVA-PO 的 reward shaping（KL 锚定、advantage 中心化、梯度裁剪），Kimi K3 不做主实验。

### 3.3 训练流程

- **Stage B（LoRA 微调）**：AdamW，lr 2e-4，cosine，batch 8×4（4 卡）×梯度累积 4=128 等效；20k 样本 ×2 epoch ≈ 1–2 天；fp16/bf16，DeepSpeed ZeRO-2 或 FSDP。
- **Stage C（GRPO）**：2k 步，每步 8 rollout（每卡 2），参考模型为 Stage B 结果；KL 系数 0.05；≈2–3 天。
- **医疗数据注意**：训练只用已获授权的 MIMIC 数据；模型仅用于学术评测，不部署于临床。

### 3.4 评测流程

- **自动**：MI-CXR 纵向推理准确率（LLM 判定变化方向与事实）、RadGraph 实体/关系 F1、BLEU-4/ROUGE-L、进展一致率。
- **临床相关**：2 名放射科医生对 100 例"变化描述 vs 金标准变化"盲评（一致/不一致/不确定），计算 Cohen's κ 与一致率；LLM-as-judge（DeepSeek V4 Pro，双盲、顺序打乱、与医生 200 例校准）作为可扩展代理。

## 4. 数据集细节

### 4.1 来源、许可与伦理合规
- **MIMIC-CXR v2.0 [arXiv:1901.07042] / DOI: 10.1038/s41597-019-0322-0**：PhysioNet Credentialed Access（2–4 周审批）；必须完成 CITI 数据使用培训，注册 PhysioNet 并接受 DUA；禁止在论文中发布影像，只可发布聚合指标与少量脱敏示例。
- **MIMIC-ABN**：异常子集，随 MIMIC-CXR 同一授权下使用。
- **MI-CXR [arXiv:2605.15574]**：按其发布条款下载，用于测试集对齐。
- **伦理**：无患者直接接触；本研究为回顾性、纯学术；结果不得用于临床决策；论文中声明数据使用协议与 IRB 豁免情况（视机构要求）。

### 4.2 划分与预处理
- 划分按 patient 切分（防止同一患者跨集泄漏）；测试集仅用于最终评测。
- 预处理：图像归一化 + resize 到 512×512（Qwen2.5-VL 原生分辨率可放宽）；报告去尾（截到 IMPRESSION 结尾）；日期差作为额外特征。

## 5. 基线复现

| 基线 | 获取方式 | 复现要点 |
|---|---|---|
| R2Gen [arXiv:2010.16056] | 官方仓库 | Transformer 解码器；在纵向子集上重训 + 原版 |
| PriorRG [DOI:10.1609/aaai.v40i9.37657] | 无官方代码，按论文复现 | Stage1 时空对比预训练 + Stage2 从粗到细解码；若复现成本高则用"既往图低秩特征作为条件"的等效实现并注明 |
| CXR-LLaVA [DOI:10.1007/s00330-024-11339-6] | 公开权重 | 直接评测 |
| LLaVA-Med [arXiv:2306.00890] | 公开权重 | 直接评测 |
| DobicVLM [arXiv:2607.18988] | 无官方代码 | 复现其"GRPO+临床奖励"子模块作为奖励基线 |

- **统一口径**：所有模型在**同一纵向子集划分**上评测；报告统一后处理（大写/标点归一）；BLEU 用 4-gram 并带长度惩罚；统计 3 个随机种子均值±方差。
- **预期指标表**（对齐 MI-CXR 论文口径，占位后实测更新）：纵向推理准确率 R2Gen ~随机、PriorRG 式 +3–6pt、本方法 +8–12pt；RadGraph F1 本方法较 PriorRG +2pt。

## 6. 实验矩阵

- **A 主实验**：LongiCXR（Stage B+C） vs 全部基线，在 MI-CXR + 自建 held-out 上。
- **B 消融**：B1 去掉进展 embedding（退化到条件编码）；B2 去掉双分支（单分支输出变化句）；B3 去掉时态奖励 r_prog；B4 用 RadGraph 差分替代 LLM 趋势判定；B5 LoRA rank 8/32/64。
- **C 鲁棒性**：C1 不同间隔（≤30 天 / 30–90 / 90–180 天）；C2 对既往报告做部分遮蔽/加噪声后的变化描述鲁棒性；C3 合成标签置信度分组（H3）。
- **D 泛化**：D1 MIMIC-ABN 上零样本；D2 按患者切分的未来时点预测（H4）；D3 用 Kimi K2.6 作趋势判定的跨裁判一致性。

## 7. 评测协议

- **主指标**：MI-CXR 纵向推理准确率、RadGraph 实体/关系 F1（统计量定义见 3.4）。
- **次指标**：BLEU-4、ROUGE-L、进展一致率。
- **临床指标**：医生盲评一致率（%）、Cohen's κ；LLM-as-judge 与医生 Spearman 相关（校准后报告）。
- **显著性**：对主指标在 3 个种子下做配对 t 检验或 Mann-Whitney U，p<0.05 视为显著；报告均值±std；所有随机种子固定（seed 42/2024/2026）。
- 评测脚本、prompt 模板、LLM 裁判系统提示全部开源。

## 8. 算力与资源计划

| 阶段 | 资源 | GPU·天 |
|---|---|---|
| Stage A 进展标签合成 | DeepSeek V4 Pro（60–80k 次调用） | API，约 $200–400 |
| 报告变体增强 | DeepSeek V4 Flash（80–120k 次） | API，约 $30–60 |
| Stage B LoRA 微调 | 4×L40，ZeRO-2 | 1–2 |
| Stage C GRPO | 4×L40 | 2–3 |
| 评测（6 模型×纵向集） | 4×L40 | 1–2 |
| **合计** | **4×L40** | **≈5–7 GPU·天 + API ~$300–500** |

- 存储：MIMIC-CXR 影像约 300GB，纵向子集抽取后约 40GB，工作区预留 500GB。
- 监控：W&B 或本地 tensorboard；保存最优/最后一轮 checkpoint。

## 9. 里程碑与时间线（单人 + 4 卡）

| 周 | 里程碑 |
|---|---|
| W1–W2 | PhysioNet 申请等待；写数据集盘点脚本 + 纵向子集构建；搭 DeepSeek V4 Pro 合成流水线 |
| W3 | 拿到数据；合成进展标签（全量）+ 抽检门控；基线 R2Gen/LLaVA-Med 搭好 |
| W4 | Stage B LoRA 微调完成；双分支消融 B2 跑通 |
| W5 | Stage C GRPO 训练 + 稳定化调参（REVA-PO 技巧） |
| W6 | 主实验 A + 消融 B 全集；MI-CXR 评测对齐 |
| W7 | 鲁棒性 C、泛化 D；医生盲评 100 例 + LLM-as-judge 校准 |
| W8 | 写论文初稿（方法+实验）+ 复现包整理；内部评审 |

## 10. 风险与备选方案

| 风险 | 影响 | 备选方案 |
|---|---|---|
| MIMIC-CXR 申请被拒/延迟 | 高 | 改用已开源公开权重模型的第三方评测 + 申请期间先跑合成流水线；若长期未获批，用 CT-RATE 或合成胸片数据做概念验证（降低贡献） |
| 纵向子集样本不足 | 高 | 放宽间隔/视图约束；或引入 MIMIC-ABN 补样本；报告中如实披露样本量 |
| GRPO 训练不稳定 | 中 | 用 REVA-PO 的 reward shaping/KL 锚定；退化为 DPO（用合成的 good/bad 进展对）；仍不稳则只保留 Stage B + 奖励评测（降为 B4 对照） |
| 进展标签合成质量差 | 中 | 提高抽检比例至 10%；加"conflict 重生成"循环；H3 提供证据 |
| 评测塌缩争议 | 中 | 主指标锁定 MI-CXR 纵向准确率 + RadGraph + 医生盲评三层，不依赖 BLEU 作结论 |
| 数据合规 | 高 | 全程遵守 PhysioNet DUA；不发布影像；成果仅学术 |

## 11. 论文写作计划

- **目标会议**：NeurIPS 2026（主会，摘要截稿约 2026 年 5 月、全文 5 月下旬——以官网为准）；备选 MIDL 2026。
- **差异化卖点**：① 首个"显式进展表示 + 双分支变化解码 + 时态 GRPO 奖励"三位一体的纵向报告生成；② 首个把 LLM 合成进展标签的质量门控与下游表现挂钩（H3）；③ 全开源：数据集划分、合成标签、代码、评测协议。
- **图表清单**：图1 框架总览（数据流水线+双分支+GRPO）；图2 变化句生成样例（improving/worsening 对比）；图3 消融柱状图（主指标）；图4 医生盲评 vs LLM-as-judge 散点；表1 数据集统计；表2 主结果；表3 消融；表4 鲁棒性/泛化。
- **相关工作覆盖**：收藏论文（PriorRG、TAMM、CTInstruct）+ 外部（R2Gen、MI-CXR、DobicVLM、REVA-PO、SHOVIR、CARE-X、Harrison.Rad 1.5）。

## 12. 参考文献（真实核验）

1. 收藏：PriorRG DOI:10.1609/aaai.v40i9.37657；TAMM DOI:10.1609/aaai.v40i24.39047
2. Johnson et al., MIMIC-CXR. DOI:10.1038/s41597-019-0322-0；arXiv:1901.07042
3. MI-CXR：arXiv:2605.15574
4. DobicVLM：arXiv:2607.18988；REVA-PO：arXiv:2607.10147
5. R2Gen：arXiv:2010.16056
6. LLaVA-Med：arXiv:2306.00890；CXR-LLaVA：DOI:10.1007/s00330-024-11339-6
7. SHOVIR：arXiv:2606.30201；CARE-X：arXiv:2608.03890
8. Harrison.Rad 1.5：arXiv:2607.05880
