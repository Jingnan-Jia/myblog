# 实验设计书：跨视频功能步对齐网络（Cross-Video Functional Step Alignment, FSA-Net）

> 主题：T02 视频理解与视频多模态大模型 ｜ 优先级：高 ｜ 目标会议：CVPR 2027

## 0. 摘要

CrossVid（Computer Vision V·论文38）最难的 FSA（功能步对齐）任务人类 85.2% vs 最强模型 13.4%，是跨视频推理的最大方法缺口，而现有时间定位方法只做单视频内对齐（Moment-DETR、TimeChat、GroundVTS（arXiv:2604.02093））。本工作提出 **FSA-Net**：把"跨视频功能等价步骤对齐"设为核心训练目标。利用 YouCook2（同食谱不同实例）与 Assembly101（多视角同一程序）构造跨视频步骤对齐训练集，设计**时间-功能双嵌入对齐目标**（步骤对比学习 + 区间级伪标签），输出规范化为 [start,end] 区间并做双视频步骤序列推理。两阶段训练：SFT（对比损失 + 区间回归）→ GRPO（区间 IoU + 功能语义等价 LLM-judge 奖励）。预期贡献：(1) 首个面向跨视频功能步对齐的训练方法；(2) 从 13.4% 基线出发的显著提升；(3) 跨视频步骤对齐训练集构造协议。

## 1. 研究背景与动机

### 1.1 问题定义

给定两个视频 V_A、V_B（如两个不同人做同一道菜），模型需 (a) 各自定位功能等价步骤的时间区间，(b) 建立步骤间的对应关系（步骤 i 在 V_A ↔ 步骤 j 在 V_B），(c) 输出对齐结果（含区间 [start,end]）。其难点在于"功能等价"是语义概念而非逐像素相似。

### 1.2 相关工作不足

- **跨视频推理只评测无方法**：CrossVid（Computer Vision V·论文38）FSA 人类 85.2% vs 模型 13.4%，未给任何方法。
- **单视频时间定位不可直接迁移**：Moment-DETR、TimeChat、LLaVA-MR（arXiv:2411.14505）、GroundVTS（arXiv:2604.02093）均在单视频内对齐 query 与 moment，无跨视频"功能等价"监督；Chrono（arXiv:2406.18113）用蓝图但限于单视频时间感知。
- **可用数据未利用**：YouCook2、Assembly101 含步骤级标注（步骤名+时间戳），但从未被用于跨视频对齐训练。
- **空白**：无"时间-功能双嵌入 + 跨视频步骤对齐 + 区间规范化"的训练目标。

### 1.3 为什么是现在、为什么你的环境

- **时机**：CrossVid 2025 底发布即成为 CVR 标准；FSA 断崖任务是最大、最清晰的得分空间。
- **环境契合**：SFT 5 + GRPO 6 = ≈11 GPU·天，4×L40 足够；伪标签/LLM-judge 全部 API 离线完成。

## 2. 研究目标与可验证假设

- **H1（FSA 可训练）**：跨视频步骤对齐训练显著提升 FSA 准确率（相对基线 +50%+）。
  *成立时的可观测结果*：CrossVid FSA 从 ~13.4%（模型基线）提升到 ≥25%（相对 +85%）。
- **H2（双嵌入必要性）**：时间-功能双嵌入对齐优于单嵌入（纯时间或纯语义）。
  *成立时的可观测结果*：双嵌入消融版 FSA 最高。
- **H3（伪标签有效）**：跨视频步骤伪标签（LLM-judge 过滤后）显著贡献训练。
  *成立时的可观测结果*：去除伪标签后 FSA 显著下降。
- **H4（跨任务泛化）**：学到的对齐能力迁移到单视频时刻检索与步骤排序。
  *成立时的可观测结果*：QVHighlights mAP@avg、CrossVid PSS 同时提升。

## 3. 总体方法设计

### 3.1 数据流水线

1. **训练语料**：YouCook2（同食谱不同实例对）、Assembly101（多视角同一程序对）、Charades-STA（伪标签对）。
2. **步骤语义标注（DeepSeek V4 Pro）**：对 YouCook2/Assembly101 步骤名生成统一动作-宾语结构（verb+object+attributes），作为功能等价判定的语义锚。
3. **伪标签构造**：同食谱不同视频对，用步骤名语义相似度（CLIP 文本-帧 + 句向量）初始匹配，DeepSeek V4 Pro 生成"等价/不等价"判断，Kimi K2.6 交叉验证，置信度 <0.7 剔除。预期 **~8K 对齐对**。
4. **区间伪标签**：由源数据集步骤时间戳直接继承；抽检 5% 人工校准。

### 3.2 模型/算法设计

- **基座**：Qwen2.5-VL-7B（LoRA rank 64）。
- **输出格式（严格）**：`<alignment>{V_A:[{step_id,s,e,desc}], V_B:[...], mapping:[[a_i,b_j],...]}</alignment>`。
- **时间-功能双嵌入**：
  - 时间嵌入：视频片段序数位置编码（区间 [s,e] → 相对时间特征）；
  - 功能嵌入：步骤语义（LLM 生成的 verb+object 结构向量）；
  - 对齐目标：对比损失 `L = −Σ log(exp(sim(e_ai, e_bj)/τ) / Σ_k exp(sim(e_ai, e_bk)/τ))` + 区间回归 MSE。
- **SFT loss**：`L = L_nll(序列) + λ1·L_contra + λ2·L_iou`，λ1=0.3、λ2=0.5。
- **GRPO 奖励**：`r = tIoU(pred, GT) + 0.4·1[功能等价 LLM-judge 判定一致] − 0.1·格式违规惩罚`。
- **课程**：同食谱对 → 跨程序对（不同菜但步骤结构相似）→ 混合。
- **超参初值**：lr 1e-5（SFT）/1e-6（GRPO）；batch 16；τ=0.07。

### 3.3 训练流程
- Stage A SFT 5 GPU·天；Stage B GRPO 6 GPU·天（8 rollout/组，3 epoch）；2 卡训练+2 卡 rollout。

### 3.4 推理与评测流程
- 推理：输出 alignment JSON → 解析；temperature=0。
- 评测：CrossVid FSA/PSS、QVHighlights（mAP@avg/R1@0.5）、Charades-STA。

## 4. 数据集细节

### 4.1 数据集清单与来源/许可
| 数据集 | 用途 | 来源/许可 |
|---|---|---|
| YouCook2 | 训练（步骤标注）| 公开（research）|
| Assembly101 | 训练（多视角程序）| 公开（research）|
| Charades-STA | 训练/评测 | 公开 |
| CrossVid（arXiv:2511.12263）| 评测（FSA/PSS）| 公开 |
| QVHighlights | 评测 | 公开（research）|

### 4.2 划分与数量
- 训练：8K 对齐对（YouCook2 3K、Assembly101 3K、Charades-STA 2K 伪标签）。
- 评测：CrossVid 官方 FSA/PSS 集、QVHighlights 测试、Charades-STA 测试。

### 4.3 预处理与格式
- 每视频 32 帧、224×224；区间秒→归一化 [0,1]；
- JSONL：`{pair_id, vA, vB, steps_A, steps_B, mapping, pseudo_conf}`。

## 5. 基线复现

### 5.1 基线列表
| 基线 | 引用 | 官方代码 |
|---|---|---|
| CrossVid 报告模型（22 个）| Computer Vision V·论文38 | 官方评测脚本 |
| GroundVTS | arXiv:2604.02093 | 论文开源则复现 |
| Chrono | arXiv:2406.18113 | 论文开源则复现 |
| LLaVA-MR | arXiv:2411.14505 | 论文开源则复现 |
| Qwen2.5-VL-7B | 官方 | github.com/QwenLM/Qwen2.5-VL |

### 5.2 复现步骤与预期指标表
统一 32 帧、temperature=0。预期主表：

| 方法 | CrossVid FSA | CrossVid PSS | QVHighlights mAP@avg | R1@0.5 |
|---|---|---|---|---|
| Gemini-2.5-Pro（官方）| 13.4 | 官方值 | — | — |
| Qwen2.5-VL-7B（本地）| ~10 | 官方值 | 基准 | 基准 |
| GroundVTS | — | — | 单视频 mIoU+7.7（官方）| — |
| **FSA-Net-7B** | ≥25 | 显著提升 | ≥基准+2 | ≥基准+2 |

### 5.3 统一评测口径
所有方法同 prompt 模板、同帧采样、同 alignment 解析器；FSA 以官方 metric 为准。

## 6. 实验矩阵

- **A（主实验）**：完整 FSA-Net（双嵌入 SFT + GRPO）。
- **B1（嵌入消融）**：仅时间 / 仅功能 / 双嵌入。
- **B2（伪标签消融）**：无伪标签 / 置信度≥0.7 / ≥0.85。
- **B3（奖励消融）**：仅 tIoU / +LLM-judge 等价 / +格式惩罚。
- **B4（课程消融）**：无课程 / 同食谱→跨程序 / 混合。
- **C（鲁棒性）**：视频对数量 {2,3}、不同实例差异度、噪声帧。
- **D（泛化性）**：跨到 QVHighlights 单视频、Charades-STA、CrossVid PSS。

## 7. 评测协议

- 指标：FSA accuracy、PSS accuracy、mAP@avg、R1@0.5、对齐映射准确率、步骤召回。
- 3 种子 mean±std；bootstrap p<0.05；LLM-judge 一致性报告。

## 8. 算力与资源计划（4×L40）

- 阶段 GPU·天：SFT 5 + GRPO 6 + 评测 2 = **≈11 GPU·天**。
- 存储：模型 + 视频缓存 350GB。
- API：DeepSeek V4 Pro 步骤语义标注/等价判断 ≈ 400 万 token；Kimi K2.6 等价 LLM-judge ≈ 200 万 token；成本 ≈ **$200-450**。

## 9. 里程碑与时间线（周，单人+4卡）

| 周 | 任务 |
|---|---|
| 1 | YouCook2/Assembly101 获取与步骤标注解析；对构造 |
| 2 | 伪标签生成 + 双 LLM 过滤 + 抽检 |
| 3 | 双嵌入 + 区间回归实现；基座复现 |
| 4 | Stage A SFT + 单视频/跨视频对照 |
| 5 | Stage B GRPO + 奖励调参 |
| 6 | 消融 + 泛化实验 |
| 7 | 论文初稿 |
| 8 | 投稿 CVPR 2027（deadline ~2026-11）|

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| 跨视频步骤伪标签噪声大 | 中高 | 中高 | 双 LLM 过滤 + 置信度阈值 + 5% 人工抽检；课程从同食谱对开始 |
| FSA 任务难度过大难收敛 | 中 | 高 | 先 PSS/单视频定位热身；分步课程；必要时降低目标为"对齐召回" |
| LLM-judge 功能等价判定不稳 | 中 | 中 | Kimi K2.6 固定 prompt + 温度 0 + 三重采样投票 |
| 区间回归与序列输出冲突 | 中 | 中 | 输出格式严格 schema + 解析校验 + 格式惩罚奖励 |

## 11. 论文写作计划

- **目标会议/截稿**：CVPR 2027。
- **差异化卖点**：攻克 CrossVid 最难关卡的第一个训练方法；时间-功能双嵌入；可复现的跨视频步骤对齐训练集协议。
- **图表清单**：Fig.1 FSA 任务与框架；Fig.2 双嵌入示意；Fig.3 对齐案例可视化；Tab.1 主表（含人类上界）；Tab.2 消融；Tab.3 泛化。
- **相关工作覆盖**：CVR（CrossVid/All-Angles）、单视频定位（GroundVTS/Chrono/LLaVA-MR）、程序理解（Assembly101/YouCook2）。

## 12. 参考文献（真实核验）

- Computer Vision V·论文38·CrossVid（arXiv:2511.12263）
- GroundVTS: arXiv:2604.02093
- Chrono: arXiv:2406.18113
- LLaVA-MR: arXiv:2411.14505
- All-Angles Bench: arXiv:2504.15280
- Computer Vision V·论文77·EgoCross（arXiv:2508.10729）
