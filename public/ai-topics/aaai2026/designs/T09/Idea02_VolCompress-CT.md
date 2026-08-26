# 实验设计书：3D CT 医学 VLM 的自适应体素 token 压缩与区域对齐（VolCompress-CT）

> 对应调研报告「六、研究 Idea」Idea 2｜3D CT 医学 VLM 的自适应体素 token 压缩与区域对齐。主题：T09 医疗AI与AI for Science。硬件：4×L40（192GB）。API：DeepSeek V4 Flash/Pro、Kimi K2.6。优先级：高。

## 0. 摘要（3-5 句）

本研究提出 VolCompress-CT：在 3D CT 医学 VLM 中引入**解剖感知的分层 token 压缩**与**区域级跨模态对齐**，使模型在 2×/4× token 压缩下保留甚至提升体素级理解能力。核心是先用轻量分割先验（MedSAM 2D + 投影）给体素 token 打分，非兴趣区 token 合并、病灶区 token 保留，再用"区域 bbox/mask → 报告句"的区域对比损失做对齐；方法以插件形式挂到 CTInstruct/E3D-GPT 类架构上，不改动主干。我们将在 CT-RATE 报告生成、3D-RAD VQA、LIDC 诊断/分割上评测，主指标是压缩比下的指标保持率 + 病灶级 Grounding mIoU + 推理延迟，预算约 5–6 GPU·天。

## 1. 研究背景与动机

### 1.1 问题定义

给定一个 3D CT volume（通常 128–256 层），3D 医学 VLM 把体素切块后送入 LLM，产生数千至上万视觉 token。问题：(a) 大量 token 属于背景/空气/无关解剖区，导致显存与延迟成本高；(b) 现有"区域对齐"依赖影像级弱监督，病灶级（肺叶、结节、器官）语义未被显式对齐到文本。目标：设计**训练无关优先、可插件化**的 token 压缩 + 区域对齐方案，使得在 2×/4× 压缩下：(i) 报告生成/VQA 指标保持率 ≥95%；(ii) 病灶级 grounding mIoU 提升；(iii) 推理延迟与显存下降 ≥40%。

### 1.2 相关工作不足

- **（Computer Vision V·论文 5·Versatile Vision-Language Model for 3D Computed Tomography）DOI:10.1609/aaai.v40i8.37517（CTInstruct）**：多粒度预训练 + 统一指令微调，但体素 token 冗余、推理显存/延迟高，且区域对齐靠弱监督。
- **MedARC [arXiv:2607.26554]**：训练无关 token 冗余压缩，但未与区域监督结合，且未做病灶级对齐评测。
- **Generalist 3D CT 基础模型 [arXiv:2403.17834]**、**Merlin/CT-RATE [arXiv:2406.06512]**、**E3D-GPT [arXiv:2410.14200]**：3D CT VLM 主干，均存在 token 效率与区域对齐空白。
- **RadGenome-Chest CT [arXiv:2404.16754]**：提供报告句→分割 mask 的区域级监督，是区域对齐的理想数据。
- **MedSAM [arXiv:2304.12306]**：2D 通用分割先验，用于区域打分；**CT-GLIP [arXiv:2404.15272]** 提供 3D 语言-图像预训练参考。

### 1.3 为什么是现在、为什么你的环境适合做

- **时机**：CTInstruct 明确指出 token 效率是 3D VLM 核心瓶颈；MedARC 刚验证压缩可行性但缺区域对齐；RadGenome-Chest CT（2024）提供了区域级监督数据。三者合流，2026 年正是做"压缩 × 区域对齐"合一的窗口。
- **环境匹配**：7B LLM 冻结 + 3D 编码器 LoRA 微调 20k 体数据在 4×L40（FSDP/ZeRO-2）2–3 天可完成；训练无关压缩可先用单卡快速验证；推理延迟评测并行跑。

## 2. 研究目标与可验证假设

1. **H1（解剖感知压缩有效）**：基于分割先验的分层 token 合并，比随机/平均合并显著更好地保持指标。*成立时的可观测结果*：2× 压缩下 RadGraph F1 保持率 ≥95%，随机合并 ≤85%。
2. **H2（区域对齐提升 grounding）**：加入区域级跨模态对比损失后，病灶级 grounding mIoU 显著提升且不损害报告生成。*成立时的可观测结果*：Grounding mIoU +≥5pt，报告 BLEU-4/RadGraph F1 不降。
3. **H3（压缩可插拔）**：同一压缩模块在 CTInstruct 与 E3D-GPT 类架构上都能无损挂载。*成立时的可观测结果*：两种主干上 4× 压缩指标保持率均 ≥90%。
4. **H4（临床等效）**：放射科医生认为压缩前后生成的报告临床等效（等效接受率 ≥90%）。*成立时的可观测结果*：50 例盲评等效接受率 ≥90%。

## 3. 总体方法设计（详细到可复现）

### 3.1 数据流水线

1. **3D 采样**：体积重采样到统一 spacing（如 1.0×1.0×1.0mm），裁剪到 [256,256,128]，clip 窗位窗宽（胸部 40/400）。
2. **区域先验**：对关键层（每 8 层取一）跑 MedSAM-2D，获得器官/候选病灶 mask；把 mask 经坐标投影回 3D 空间，得到每体素"区域重要性"得分 s_v ∈[0,1]（病灶>器官>空气背景）。
3. **"区域—描述"对构建（DeepSeek V4 Flash 批量）**：把 RadGenome-Chest CT 的 grounded 报告句按 bounding box/mask 切分，输入 prompt：*"给定 CT 区域 mask 的类别与位置（如 '右上叶 5mm 结节'），把该句改写为区域级描述，输出 JSON {"region":"...","bbox":"...","desc":"..."}。"*；Kimi K2.6 做区域描述与 mask 的一致性校验（NLI 风格），不合格的丢弃。
4. **脱敏与过滤**：不含患者 ID；过滤掉 bbox 面积 <20 像素² 的碎片区域；只保留有 ground-truth 报告匹配的样本。
5. **数量**：CT-RATE 训练 20k 体数据（报告生成 + 区域对）；3D-RAD 全量做 VQA；LIDC-IDRI 的结节区（诊断/分割，约 1000 例）用于区域评测。

### 3.2 方法设计（模块拆解）

1. **主干**：基座 3D CT VLM——首选复现 CTInstruct 的 ResNet-ViT 混合编码器 + 7B LLM（Qwen2.5-7B）；对照 E3D-GPT 类（权重冻结 LLM，只训编码器头）。若无公开 checkpoint，用 Merlin/CT-RATE 训练的开源 3D VLM（如 Med3D-VLM 类）替代并注明。
2. **分层 token 压缩（训练无关版本，先做快速验证）**：
   - token 重要性打分：`w_i = λ·s_v(i) + (1−λ)·attn_i`，attn_i 为第一层 attention 对 CLS 的平均注意力；λ=0.6。
   - 分层合并：按空间近邻将 token 分组，组内按 w_i 加权平均（soft merging）；病灶区（w_i 高于阈值 θ）不合并、单独保留；KV 缓存友好。
   - 推理时动态 token 预算 B ∈ {2×,4×}，θ 自适应：`θ = argmin |#kept − B|`。
3. **区域级跨模态对比损失（Stage B）**：
   - 从 RadGenome 取 (mask, 句) 对；区域 token 经可学习投影头得到 e_reg，句子经冻结文本编码器得到 e_txt；
   - 对比损失：`L_cl = −log exp(sim(e_reg,e_txt)/τ) / Σ_j exp(sim(e_reg,e_txt_j)/τ)`，τ=0.07；负例为批内其他句子/其他区域。
   - 总损失 `L = L_caption + α·L_cl`，α=0.5。
4. **区域 Grounding 头**：在 LLM 顶部加轻量投影，输出每视觉 token 与文本实体的对齐分数（供 mIoU 评测）。

### 3.3 训练流程

- Stage A（训练无关压缩）：无训练，直接评测指标保持率——先验证压缩不塌指标再投入训练。
- Stage B（区域对齐）：冻结 7B LLM，LoRA rank 16 微调 3D 编码器 + 区域投影头；AdamW，lr 3e-4，cosine，batch 4×4×grad_accum 8=128；20k 体数据 ×1 epoch ≈ 2–3 天；bf16 + FSDP/ZeRO-2。
- **医疗数据注意**：CT-RATE 需按 Merlin 公开流程申请；RadGenome 按 HuggingFace/官方许可使用；LIDC 公开免申请但注明 TCIA 条款。

### 3.4 评测流程

- 报告生成：BLEU-4、ROUGE-L、RadGraph 实体/关系 F1；VQA：3D-RAD 准确率；诊断：LIDC 恶性分类 AUC；分割/grounding：病灶级 mIoU（对齐 DiagCoT 同口径）。
- 效率：每 volume 推理延迟（ms）与峰值显存（GB），在 4×L40 单卡测，报告 2×/4× 压缩 vs 无压缩。
- 临床：放射科医生对 50 例压缩前后报告做等效性盲评（临床等效/非等效/不确定）。

## 4. 数据集细节

### 4.1 来源、许可与伦理合规
- **CT-RATE（Merlin [arXiv:2406.06512]）**：按官方流程申请（同意研究用途 + 非商业）；25,692 非增强胸部 CT + 结构化报告。
- **RadGenome-Chest CT [arXiv:2404.16754]**：在 CT-RATE 基础上生成，含区域 mask + grounded 报告 + 130 万 VQA；按发布许可（研究用途）使用。
- **3D-RAD [arXiv:2506.11147]**：3D 放射 VQA 数据集，按发布条款使用（含时序分析题）。
- **LIDC-IDRI**：TCIA 公开数据集；结节级标注；CT 影像与结节 mask 均公开，标注来源协议需在论文中致谢。
- **伦理**：均为回顾性公开数据；不发布影像；区域对齐研究不涉及个体识别；报告数据使用条款。

### 4.2 划分与预处理
- 按患者切分 train/val/test（CT-RATE 官方划分优先，否则按患者 8:1:1）。
- 预处理：重采样 + 窗位窗宽 + 归一化；结节 bbox 从 LIDC 标注直接取；区域 mask 下采样到编码器粒度。

## 5. 基线复现

| 基线 | 获取方式 | 复现要点 |
|---|---|---|
| CTInstruct [DOI:10.1609/aaai.v40i8.37517] | 无公开权重，按论文复现或引用官方数字 | 混合 ResNet-ViT + 多粒度预训练；标注"官方数字 vs 复现" |
| Merlin/CT-RATE [arXiv:2406.06512] | 有开源权重（若可用） | 直接评测 |
| E3D-GPT [arXiv:2410.14200] | 无公开权重 | 复现其"专家基础模型增强"主干或引用数字 |
| MedARC [arXiv:2607.26554] | 训练无关压缩基线 | 用其压缩策略在同样主干上复现 |
| 无压缩主干 | 本方法主干本身 | 指标保持率的 100% 基准 |

- **统一口径**：同一主干（能复现的）、同一数据划分、同一窗位窗宽与采样；延迟在同一卡（L40）上 3 次平均；压缩率用"视觉 token 数比"定义。
- **预期指标表**：无压缩 RadGraph F1=baseline；MedARC 2×≈−2pt；本方法 2×≈−1pt 且 grounding +5pt（占位，实测更新）。

## 6. 实验矩阵

- **A 主实验**：VolCompress-CT（Stage A+B）vs 基线，在报告生成/VQA/诊断/分割上 2×、4× 两档压缩。
- **B 消融**：B1 去掉区域先验打分（退化为 VGS 式注意力打分）；B2 去掉区域对比损失；B3 不同 λ（0.2/0.4/0.6/0.8）；B4 合并策略（mean pooling vs soft merging vs top-k 丢弃）；B5 不同压缩率（1.5×/2×/4×/8×）。
- **C 鲁棒性**：C1 掩膜噪声（对区域先验加 ±2 像素腐蚀/膨胀）；C2 不同窗位窗宽；C3 病灶大小分层（≤5mm/5–10mm/≥10mm）。
- **D 泛化**：D1 压缩模块挂到 E3D-GPT 类主干（H3）；D2 跨数据（在 CT-RATE 训练，3D-RAD/LIDC 评测）；D3 动态 token 预算下的延迟-质量曲线。

## 7. 评测协议

- **主指标**：压缩保持率 =（压缩模型指标 / 无压缩指标），阈值 ≥95%（2×）/≥90%（4×）；病灶级 Grounding mIoU；延迟/显存下降率。
- **次指标**：BLEU-4、ROUGE-L、RadGraph F1、3D-RAD 准确率、LIDC 恶性 AUC、VQA 闭合题准确率。
- **临床指标**：50 例等效性盲评接受率 + Cohen's κ。
- **显著性**：3 种子均值±std；指标保持率用 bootstrap 95%CI；mIoU 用配对检验 p<0.05。
- 评测代码与 prompt 模板开源。

## 8. 算力与资源计划

| 阶段 | 资源 | GPU·天 |
|---|---|---|
| Stage A 训练无关压缩验证 | 4×L40 推理并行 | 1 |
| 区域描述合成（DeepSeek V4 Flash，20k 区域对） | API | ~$40–80 |
| Stage B 区域对齐微调 | 4×L40 FSDP | 2–3 |
| 推理延迟/显存评测 | 4×L40 | 1–2 |
| **合计** | **4×L40** | **≈5–6 GPU·天 + API ~$100** |

- 存储：CT-RATE 体积数据约 1–2TB（可按需抽子集）；RadGenome mask 约 100GB；预留 3TB。
- 关键：先跑通 Stage A 的"训练无关压缩"再投入 Stage B 训练，避免压缩与对齐的收益被架构改动吃掉。

## 9. 里程碑与时间线（单人 + 4 卡）

| 周 | 里程碑 |
|---|---|
| W1 | 申请 CT-RATE/下载 LIDC；搭主干（Merlin 或 CTInstruct 复现）并跑通推理 |
| W2 | Stage A 训练无关压缩（打分+分层合并）在 2× 验证保持率 |
| W3 | RadGenome 区域对合成 + Kimi 一致性校验；区域对比损失代码 |
| W4 | Stage B 微调（20k×1 epoch） |
| W5 | 全量评测（报告/VQA/LIDC/mIoU）+ 延迟显存曲线 |
| W6 | 消融 B、鲁棒性 C、泛化 D；医生 50 例等效盲评 |
| W7 | 补实验 + 图表；复现包整理 |
| W8 | 论文初稿 + 内部评审 |

## 10. 风险与备选方案

| 风险 | 影响 | 备选方案 |
|---|---|---|
| CTInspect 无公开权重难复现 | 高 | 用 Merlin/开源 3D VLM 作主干并声明差异；压缩模块本身主干无关 |
| CT-RATE 申请延迟 | 中 | 先 LIDC + 3D-RAD 完成区域对齐验证，CT-RATE 到位后补报告生成实验 |
| 压缩塌指标 | 高 | Stage A 快速验证前置：若 2× 保持率 <90%，改用更保守的"top-k 保留+近邻合并"并调 θ/λ |
| 区域先验质量差 | 中 | 用 atlas 模板替代 MedSAM；报告 mIoU 上限受先验影响（C1 量化） |
| 区域对齐收益不显著 | 中 | 放大 RadGenome 区域对占比；α 调参；退化为"压缩 + 报告级对齐"仍可发表 |
| 数据许可 | 高 | 全部用研究许可数据集；不发布影像；论文声明各数据集条款 |

## 11. 论文写作计划

- **目标会议**：CVPR 2027（效率）或 NeurIPS 2026（效率/系统）；备选 MICCAI 2027。
- **差异化卖点**：① 首个"解剖感知分层 token 压缩 + 区域级对比对齐"插件，与主干解耦；② 2×/4× 压缩下的"保持率 + grounding 增益"联合指标；③ 全开源：压缩模块、区域对、评测脚本。
- **图表清单**：图1 框架（打分→分层合并→区域对比）；图2 压缩可视化（token 保留/合并分布）；图3 延迟-质量曲线；图4 病灶级 grounding 对比；表1 数据统计；表2 主实验；表3 消融；表4 鲁棒性/泛化。
- **相关工作覆盖**：收藏论文（CTInstruct、3DTeethSAM、TAE）+ 外部（Merlin、CT-GLIP、E3D-GPT、3D-RAD、MedARC、RadGenome、Generalist 3D CT、MedSAM、Jolia）。

## 12. 参考文献（真实核验）

1. 收藏：CTInstruct DOI:10.1609/aaai.v40i8.37517；3DTeethSAM DOI:10.1609/aaai.v40i9.37702；TAE（AAAI Emerging Trends in AI·论文 106·Can Large Language Models Grasp 3D Medical Anatomy Shapes?，收藏论文，未核验 DOI，正文按「Session·论文N」格式引用）
2. Merlin/CT-RATE：arXiv:2406.06512；Generalist 3D CT：arXiv:2403.17834；CT-GLIP：arXiv:2404.15272
3. E3D-GPT：arXiv:2410.14200；3D-RAD：arXiv:2506.11147；MedARC：arXiv:2607.26554；Jolia：arXiv:2606.24570
4. RadGenome-Chest CT：arXiv:2404.16754
5. MedSAM：arXiv:2304.12306
6. DiagCoT：arXiv:2509.06409
