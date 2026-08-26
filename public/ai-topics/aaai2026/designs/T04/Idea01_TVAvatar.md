# 实验设计书：Idea 1 TVAvatar — 文本指令驱动的身份保持单图 3D 人头头像重建与编辑

> 对应调研报告 Idea 1。资源假设：4×L40（48GB/卡，共 192GB）；DeepSeek V4 Flash/Pro、Kimi K2.6 API。

## 0. 摘要

TVAvatar 将"单图全头 3D 重建"与"文本指令编辑（发型/表情/配饰/风格）"统一到一个扩散-高斯框架，文本经 tokenizer 注入扩散注意力、身份经 ArcFace 特征注入 cross-attention，实现"改属性不改脸"。本设计书给出可复现的两阶段训练协议（Stage1 多视图重建监督 30K 步，Stage2 联合编辑 LoRA 10K 步）、数据流水线（多视图数字人视频 + 单图重投影 + LLM 生成文本指令对）、评测协议（重建 PSNR/SSIM/LPIPS + 身份保持 ArcFace cosine + 编辑 CLIP 对齐 + 人类评测）。总预算约 9 GPU·天，主风险是"编辑与重建共用潜在空间的冲突"，以渐进式 LoRA 解耦注意力头缓解。

## 1. 研究背景与动机

### 1.1 问题定义

输入：单张真实人像图 $I\in\mathbb{R}^{H\times W\times3}$ + 文本指令 $t$（如"换成短发并戴上墨镜"）。输出：一个可渲染、可驱动、身份保持的 3D 高斯头像 $\mathcal{G}$，使得 (a) 重建分支 $\mathcal{G}$ 从新视角渲染与 $I$ 保持身份一致；(b) 编辑分支 $\mathcal{G}'$ 满足 $t$ 描述的属性变化且身份余弦相似度不塌陷。

正式目标：$\min_{\theta} \mathcal{L}_\text{rec}(\hat I_{\pi},I_\pi) + \lambda_\text{edit} \mathcal{L}_\text{edit}(\hat I'_{\pi}, t) + \lambda_\text{id}\mathcal{L}_\text{id}(\mathcal{G}')$，其中 $\pi$ 为视角。

### 1.2 相关工作不足

- 重建型（快但无语言控制）：`SEGA`（arXiv:2504.14373）、`Any3DAvatar`（arXiv:2604.13856）、`OMEGA-Avatar`（arXiv:2602.11693）均无文本编辑能力；收藏论文（Computer Vision III · 论文 14 · High-Quality Full-Head 3D Avatar Generation from Any Single Portrait Image）只做重建、依赖私有捕获数据与逐主体优化，语言条件缺失。
- 编辑型（有语言但丢身份/需优化）：`HeadSculpt`（arXiv:2306.03038）文本驱动雕刻但需逐主体 SDS 优化、不保身份；`3D Gaussian Blendshapes`（arXiv:2404.19398）、`DiffusionAvatars`（arXiv:2311.18635）专注表情驱动而非语义编辑。
- 两套管线割裂导致"重建完再编辑"出现身份漂移，本文是首个统一框架。

### 1.3 为什么是现在、为什么你的环境适合做

- 时机：单图前馈头像（SEGA/Any3DAvatar 类）2025–2026 刚成熟，构成可复用的"重建主干"；LLM API（DeepSeek V4 Pro）可低成本批量生产图文-编辑指令监督对，这是私有人工标注之外的关键数据杠杆。
- 环境适配：主干 ~1B 全参 + LoRA 在 4×L40（FSDP + 梯度检查点）可承受；无需外部算力。数据（FFHQ + 开源多视图人头视频/合成）全部公开可得；编辑监督由 API 生成，规避人工标注瓶颈。

## 2. 研究目标与可验证假设

1. **H1（统一框架优于级联）**：单阶段联合训练的重建-编辑模型，编辑后身份保持（ArcFace cosine）比"重建→再编辑"两段式高 ≥5 个点。
   - 成立时观测：测试集上两段式平均 id-cos=0.72，联合式 ≥0.77。
2. **H2（正交条件解耦）**：文本与身份作为正交条件（文本注入 U-Net attention，身份注入 cross-attention）可实现"属性可控、身份稳定"。
   - 成立时观测：消融掉身份注入后编辑属性照常改变但 id-cos 骤降 >0.15；消融掉文本注入后属性成功率骤降 >20 个点。
3. **H3（LLM 评审监督有用）**：Kimi K2.6 作为 LLM 评审器提供的编辑质量弱监督，可使 CLIP 对齐分数 +3 个点。
   - 成立时观测：加 LLM 弱监督后 CLIP 对齐从 0.24 → 0.27，且人类评测满意率 +8%。
4. **H4（一致性损失防漂移）**：重建-编辑一致性损失（编辑渲染与文本 CLIP 对齐 + 身份约束）在 LoRA 高秩下依然有效。
   - 成立时观测：λ_cons 消融实验（0/0.1/0.5/1.0）中 λ=0.5 时 id-cos 最高且 LPIPS 不劣化。

## 3. 总体方法设计

### 3.1 数据流水线（含 API 合成文本/语义条件的 prompt 思路）

数据来源与构造（详 §4）：

1. **多视图 pair 构造**：对每段数字人视频（227 序列、96 视角、21,792 帧，若无法获得则用开源多视图人头数据），以随机参考帧 $I_{ref}$ 为"单图输入"，随机采样 5–8 帧其他视角为重建监督。构造 N=180K 个 (输入图, 目标多视图) 对。
2. **单图重投影增强**：对 FFHQ 单图用现成 3DMM 拟合（FLAME DOI:10.1145/3130800.3130813）渲染伪多视图，补入 60K 对，提升跨域泛化。
3. **编辑指令对合成（DeepSeek V4 Pro）**：对每个重建对，调用 DeepSeek V4 Pro，prompt 模板（中文注释见下）：

```
你是 3D 头像数据标注师。给定主体描述 D（属性+身份词）与一组属性 A，
为 D 生成 8 条编辑指令，要求：
1) 每条指令只改变一个属性维度（发型/发色/眼镜/表情/风格/配饰）；
2) 指令中必须含相对空间或程度限定词（如"更卷曲""颜色更深"）；
3) 附带编辑强度 ε∈[0,1] 与目标属性名。
输出 JSON：{"edit": 指令文本, "attr": 属性名, "strength": ε}
```

产出 180K×8≈1.44M 条候选指令；用规则去重 + Kimi K2.6 质量过滤后保留 500K 条。对其中 20K 条做**真值编辑对**：由图形学工具/手工 PBRT 渲染编辑后的同一主体多视角图，构成编辑监督；其余 480K 条作为文本条件（仅弱监督，配合 LLM 评审分数）。

4. **Hard-negative 构造**：用 DeepSeek V4 Pro 生成"属性对容易张冠李戴"的困难负例（如把"换发型"写成"换发色"）共 10K 条，用于 LLM 评审器训练与过滤。

### 3.2 模型/算法设计

**主干 = 多视图扩散（EpiDiff 式外极注意力，arXiv:2312.06725）+ 前馈 3DGS 解码器（MVSplat 式，arXiv:2403.14627）。**

1. **扩散主干**（~1B，DiT/U-Net 混合）：
   - 条件：参考图 $I_{ref}$（CLIP/ViT 编码）+ 相机位姿（外极注意力）+ 文本 $t$（T5 编码，经 cross-attention 注入）+ 身份 $f_{id}$（ArcFace, arXiv:1801.07698，经独立 cross-attention 头注入）。
   - 输出：4 个目标视角的潜在图 $\{z_\pi\}$。
2. **前馈 3DGS 解码器**（~100M）：将多视角潜在解码为每组高斯 $\{G_k^j=(p_k^j, s_k^j, r_k^j, o_k^j, c_k^j)\}$，按 MVSplat 的可微 splatting 渲染。
3. **身份与文本正交化**：
   - 文本：注入 U-Net 自注意力（key/value 拼接）。
   - 身份：注入独立 cross-attention 头，输出经 zero-init 门控 $g$（初始 0），避免初期扰动。
   - LoRA 解耦：Stage2 中文本相关注意力头 LoRA rank 64、身份头 LoRA rank 32，互不共享，缓解"编辑与重建共用潜在空间的冲突"。

**关键损失**：
- 重建：$\mathcal{L}_\text{rec} = \lambda_1\mathcal{L}_\text{mse}(\hat I_\pi,I_\pi) + \lambda_2\mathcal{L}_\text{lpips} + \lambda_3\mathcal{L}_\text{ssim}$
- 编辑：$\mathcal{L}_\text{edit} = \text{CLIPAlign}(R(\mathcal{G}'), t) + \lambda_\text{llm}\mathcal{S}_\text{Kimi}(R(\mathcal{G}'), t)$，$\mathcal{S}_\text{Kimi}$ 为 Kimi 评审分数（1–5，缩放后当弱监督）。
- 身份：$\mathcal{L}_\text{id} = 1 - \cos(f_{id}(R(\mathcal{G}')), f_{id}(I_{ref}))$
- 一致性总损失：$\mathcal{L} = \mathcal{L}_\text{rec} + \lambda_\text{edit}\mathcal{L}_\text{edit} + \lambda_\text{id}\mathcal{L}_\text{id}$

**超参数初值**：λ1=1.0, λ2=1.0, λ3=0.2, λ_edit=0.5, λ_id=1.0, λ_llm=0.1；分辨率 512；LoRA rank 64/32。

### 3.3 训练流程

- **Stage1（重建，30K 步）**：FSDP 全参训练扩散主干 + 3DGS 解码器，global batch 64（多视图 crop），lr 3e-4 cosine decay 到 3e-5，warmup 1K 步，AdamW(β=(0.9,0.95))，梯度裁剪 1.0，梯度检查点。4×L40 FSDP shard + offload，约 6 GPU·天。
- **Stage2（联合编辑 LoRA，10K 步）**：冻结主干，仅训 LoRA + 门控 + 身份注入头；lr 1e-4；编辑对 batch 32 + 重建对 batch 32 混合。约 3 GPU·天。
- **并行**：FSDP + DeepSpeed ZeRO-3，数据并行 4 卡；序列/图 batch 内切分。

### 3.4 推理与评测流程

单图 + 文本 → 一次前馈得到高斯头像（重建）→ 在编辑分支上加 LoRA 编辑 → 渲染多视角。评测管线：加载 checkpoint → 遍历测试集（FFHQ 真人 2K 张 × 4 条编辑指令）→ 渲染前视图/侧视图/45° 视图 → 计算指标 → Kimi 批量评分 → 汇总表。推理单卡 L40 约 0.5–1 秒/主体。

## 4. 数据集细节

| 数据集 | 用途 | 来源/许可 | 数量 | 预处理 |
|---|---|---|---|---|
| FFHQ | 单图输入/评测 | 公开，研究用途（非商用注意核对） | 70K 图 | 人脸对齐 512²，ArcFace 特征预提取 |
| 多视图数字人视频 | 重建多视图监督 | 公开多视图人头数据/合成（论文14 私有数据不可得，用开源替代） | 180K 对 | 每对采样参考帧 + 5–8 目标帧 |
| 伪多视图（FLAME 渲染） | 泛化增强 | 由 FFHQ+3DMM 自产 | 60K 对 | 3DMM 拟合→渲染→扰动光照/相机 |
| 编辑真值对 | 编辑监督 | 自产（PBRT 渲染编辑后主体） | 20K 对 | 同一主体前后渲染多视角 |
| 文本指令对 | 文本条件 | DeepSeek V4 Pro 生成 + Kimi 过滤 | 500K 条 | JSON 结构化，去重 |

划分：训练 90% / 验证 5% / 测试 5%（按身份不重叠划分）。许可注意：FFHQ 非商用授权，若投稿需核对；数字人数据选用开源许可（如开源 scan/视频数据集）。

## 5. 基线复现

| 基线 | 官方代码 | 复现要点 |
|---|---|---|
| SEGA（arXiv:2504.14373） | GitHub 公开仓库（官方 release） | 单图前馈，直接跑官方 checkpoint 推理 |
| Any3DAvatar（arXiv:2604.13856） | 官方仓库（若未开源则用 SEGA 替代） | 同 SEGA |
| HeadSculpt（arXiv:2306.03038） | 官方仓库 | 逐主体 SDS 优化，每主体 5–10 分钟 |
| GaussianHead（arXiv:2312.01632） | 官方仓库 | 需 FLAME 绑定与多视图输入 |
| 两段式基线（重建+编辑串联） | 本工作自建 | TVAvatar Stage1 + 独立编辑模型 |

统一评测口径：同一评测集（FFHQ 2K 张 × 4 指令），同一渲染相机矩阵，同一指标实现库（lpips 官方、torchmetrics 的 SSIM/PSNR），ArcFace 用官方预训练权重。预期指标表（重建侧）：TVAvatar 目标 PSNR≥29.5 / SSIM≥0.93 / LPIPS≤0.055，与 SEGA 持平或更优；编辑侧 id-cos≥0.77。

## 6. 实验矩阵

- **A（主实验）**：TVAvatar 全模型 vs 全基线（重建指标 + 编辑指标 + 人类评测）。
- **B（消融）**：①身份条件有无；②文本注入位置（U-Net 自注意 vs cross-attention vs 拼接 latent）；③LoRA 解耦（共享 vs 分离）；④一致性损失权重 λ_cons∈{0,0.1,0.5,1.0}；⑤LLM 评审弱监督有无。
- **C（鲁棒性）**：真实自拍图（强光照/遮挡/配饰）子集 500 张；极端姿态（±60°）；跨域（合成图）。
- **D（泛化性）**：到未见过身份的编辑指令泛化；编辑强度 ε∈{0.3,0.7,1.0} 连续性。

## 7. 评测协议

- 指标：重建 PSNR/SSIM/LPIPS（官方实现）；身份 ArcFace cosine（编辑前后）；编辑属性 CLIP 对齐分数（CLIP ViT-B/32，指令文本 vs 编辑渲染）；人类评测（5 人 × 双盲 × 200 样本，满意度/身份保持/属性正确三问）。
- 统计：每实验 3 个随机种子（42/2024/2026），报告均值±方差；主对比做配对 t 检验（p<0.05）与 95% CI；人类评测报 Fleiss' κ。

## 8. 算力与资源计划

| 阶段 | 卡·天 | 存储 | 说明 |
|---|---|---|---|
| Stage1 重建 | 6 | 数据 ~0.6TB、ckpt ~20GB/份 | FSDP 4 卡 |
| Stage2 编辑 LoRA | 3 | LoRA ~200MB/份 | 冻结主干 |
| 推理评测 | 1 | — | 单卡批量 |
| API 用量 | — | — | DeepSeek V4 Pro：~600K 次调用（指令合成，约 $60–90）；Kimi K2.6：~200K 次（评审/过滤，约 $20–40）；DeepSeek V4 Flash：~100K 次（粗筛） |

合计 **≈10 GPU·天**，API 成本 <$150。

## 9. 里程碑与时间线（单人 + 4 卡）

| 周 | 任务 |
|---|---|
| W1 | 数据管线：FFHQ 对齐、多视图对构造、FLAME 伪渲染；基线 SEGA 复现跑通 |
| W2 | Stage1 重建训练启动 + 调参；DeepSeek 指令合成脚本上线 |
| W3 | Stage1 收敛验证（重建指标达基线）；编辑对构建完成 |
| W4 | Stage2 联合编辑训练；消融 ①② 跑起 |
| W5 | 消融 ③④⑤ + 鲁棒性子集；人类评测准备 |
| W6 | 论文写作（方法+主实验+消融）、图表制作、投稿版定稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解 |
|---|---|---|
| 文本-属性绑定不精确（"张冠李戴"） | 高 | LLM 评审器 hard-negative 过滤 + 双模型一致性 |
| 身份漂移 | 高 | 身份约束损失 + 多视图重建监督 + zero-init 门控 |
| 编辑与重建共用潜在空间冲突 | 中高 | 渐进式 LoRA 解耦注意力头；分头 rank 不同 |
| 4×L40 内存不足 | 低 | 主干 1B→500M；梯度检查点；ZeRO offload |
| 开源多视图数据域差 | 中 | 伪多视图增强 + 少量真实数据微调 |
| 数据集许可风险 | 中 | 投稿前核对 FFHQ 非商用条款，必要时改用开源人脸视频 |

## 11. 论文写作计划

- 目标：CVPR 2027（截稿 2026-11 前后）/ ICCV 2027（备份）。差异化卖点："统一重建+编辑 + 正交条件 + LLM 评审监督"的完整闭环，工业可直接用单图+一句话出可编辑头像。
- 图表清单：方法图（双分支条件注入）；重建-编辑可视化对比（与 SEGA/HeadSculpt）；消融表；编辑强度连续性曲线；用户研究散点图（身份保持 vs 属性正确）。
- 相关工作覆盖：收藏论文（Computer Vision III · 论文 14）+ SEGA/Any3DAvatar/OMEGA-Avatar/HeadSculpt/GaussianHead/3D Gaussian Blendshapes/DiffusionAvatars/EpiDiff/MVSplat/ArcFace。

## 12. 参考文献

- Computer Vision III · 论文 14 · High-Quality Full-Head 3D Avatar Generation from Any Single Portrait Image（AAAI 2026）
- SEGA: arXiv:2504.14373；Any3DAvatar: arXiv:2604.13856；OMEGA-Avatar: arXiv:2602.11693
- HeadSculpt: arXiv:2306.03038；GaussianHead: arXiv:2312.01632；3D Gaussian Blendshapes: arXiv:2404.19398；DiffusionAvatars: arXiv:2311.18635
- EpiDiff: arXiv:2312.06725；MVSplat: arXiv:2403.14627；3D Gaussian Splatting: arXiv:2308.04079
- ArcFace: arXiv:1801.07698；FLAME: DOI:10.1145/3130800.3130813；CLIP: arXiv:2103.00020
