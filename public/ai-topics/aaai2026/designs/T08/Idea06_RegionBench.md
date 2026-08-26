# 实验设计书：Idea 6 区域级视觉文档 RAG 评测（RegionBench）

## 0. 摘要
本项目构建 RegionBench：带"区域级证据（bbox + 语义 region）"标注的多语言视觉文档问答基准，量化页级 vs 区域级检索的精度与视觉 token 效率差异。标注管线 = VLM 定位证据区域 + DeepSeek 生成问题与理由 + 人工抽检；评测协议包含双粒度（页 vs 区域）检索对比、视觉 token 压缩率与区域级检索到生成的可追溯性。在论文/财报/手册/表格四类双栏文档上，对照 RegionRAG、ColPali、OCR+BGE、CLIP 稠密检索。约 3–5 GPU·天，投稿 CVPR 2027 / ACM MM 2027（若进度快可 ECCV 2026）。

## 1. 研究背景与动机
### 1.1 问题定义
多模态 RAG 以整篇文档/整页为检索单元会注入大量无关视觉内容，稀释注意力。RegionRAG 证明"语义区域级"检索能提精度、省 token，但社区没有标准基准能复现/对比"证据到底在页内哪里"。问题：构建一个区域级证据标注 + 双粒度评测的视觉文档 RAG 基准，量化区域级检索的收益。

### 1.2 相关工作不足
- RegionRAG（CV V · 论文 85 · RegionRAG: Region-level Retrieval-Augmented Generation for Visual Document Understanding）提出区域级检索但无配套基准、无区域级证据标注。
- Double-Bench（NLP IV · 论文 30 · Are We on the Right Way to Assess Document Retrieval-Augmented Generation?）证据粒度止于"页"。
- OmniDocBench（arXiv:2412.07626）偏文档解析（bbox 类标注）不评 RAG 链路；PureDocBench（arXiv:2605.07492）批评其标注错误。
- ColPali（arXiv:2407.01449）页级多向量，无法验证区域级证据；HPC-ColPali（arXiv:2506.21601）只做量化加速。
- "Lost in OCR Translation?"（arXiv:2505.05666）对比视觉 vs OCR RAG 系统，未做区域级。

### 1.3 为什么是现在、为什么你的环境适合做
- 证据：视觉 RAG 评测 2025–2026 正热（OmniDocBench/PureDocBench 迭代）；RegionRAG 是 AAAI 2026 新鲜工作，区域级评测是明确空白；多语言双栏文档（财报/论文）是现实刚需。
- 环境：Qwen2.5-VL-7B LoRA 微调在 4×L40 上 2–3 GPU·天；VLM 定位 + DeepSeek 校验全 API；资源匹配。

## 2. 研究目标与可验证假设
- H1（区域级更优）：区域级检索在证据命中率上显著高于页级检索，且视觉 token 用量更低。成立时观测：区域级 Recall@1 提升 ≥ +5%，视觉 token 减少 ≥ 20%。
- H2（标注管线可靠）：VLM 定位 + DeepSeek 校验的区域级证据标注与人工标注一致率 ≥85%。成立时观测：500 条小规模标定后，人工抽检 300 条的 bbox IoU ≥0.5 比例 ≥85%。
- H3（区域级提升生成）：区域级检索注入的生成准确率高于页级，尤其在表格/财务文档。成立时观测：QA 准确率提升 ≥ +3 点（表格类）。
- H4（多语言稳健）：跨 4 语言区域级证据标注质量一致。成立时观测：各语言证据命中率差 ≤5%。

## 3. 总体方法设计
### 3.1 语料/数据流水线
1. 文档源：论文（arXiv open-access）、财报（SEC 10-K）、产品手册、财务报表表格——四类，每类 500 页。
2. 区域标注（Qwen2.5-VL-7B）：prompt `Localize the regions (bounding boxes) in this page that support answering: <question>` → bbox + 语义 region（把相关 patch 聚合成完整区域，仿 RegionRAG 动态聚 region）。
3. 问题生成（DeepSeek V4 Pro）：基于标注区域生成 QA 与理由（prompt：`Given the region content, generate a question whose answer is fully supported by this region, plus the supporting rationale`）。
4. 校验（Kimi K2.6）：判定"区域确实支持该答案"，过滤低置信。
5. 人工抽检：300 条（每类 75）人工核对 bbox 与证据；500 条小规模先标定定位器准确率（≥85% 才扩量）。
6. 多语言：DeepSeek/Kimi 把问题合成中/英/法/西/阿变体（每文档 1 种额外语言）。

### 3.2 方法设计
- 区域检索器：Qwen2.5-VL-7B 视觉编码 + 区域池化，LoRA 微调做区域-query 匹配（对比学习）；页级检索器对照 ColPali（arXiv:2407.01449）。
- 双粒度协议：同一查询下，页级检索 top-k 页 vs 区域级检索 top-k 区域；评估 (a) 证据命中率（证据区域在 top-k），(b) 视觉 token 压缩率（区域 token/整页 token）。
- 生成评测：检索区域/页 → Qwen2.5-7B（text）或多模态生成器 → 答案准确率；报告可追溯性（答案引用的证据区域是否命中标注）。
- 超参数初值：region 聚成 min 3 patches、top-k 区域=8、top-k 页=4、LoRA r=16 α=32。

### 3.3 训练流程
- 区域检索器 LoRA 微调 Qwen2.5-VL-7B：batch 8、lr 1e-4、对比 loss（正=证据区域、负=同页无关区域），2–3 GPU·天。
- 数据标注全 API（Qwen-VL 定位 + DeepSeek 生成 + Kimi 校验）。
- 可选：页级检索器用 ColPali 官方 checkpoint 冻结。

### 3.4 评测流程
- 检索评测：区域级 Recall@1/@5、证据命中率、视觉 token 压缩率。
- 生成评测：QA 准确率（每类文档分别报）；区域级可追溯率。
- 消融：粒度选择（页/区域/混合）、top-k、跨语言。

## 4. 数据集细节
| 数据集 | 来源 | 许可 | 用途 |
|---|---|---|---|
| 论文 | arXiv OA（按 lic 筛选） | 各论文许可 | 区域标注 + 评测 |
| 财报 SEC 10-K | SEC EDGAR | Public domain | 表格/财务区域 |
| 产品手册 | 自建（开源手册） | 各原许可 | 手册类 |
| DocVQA / ViDoRe | 官方 | 各原许可 | 复用 + 对照 |
| Double-Bench 子集 | arXiv:2508.03644 官方 | 开源 | 页级对照 |
| 自建 RegionBench 标注 | 本项目 | 自建（CC-BY 发布） | 主评测 |
- 预处理：PDF→页面渲染（300dpi）→VLM patch；bbox 归一化；JSONL（page_id, bbox, region_id, question, answer, rationale）。

## 5. 基线复现
| 基线 | 官方代码 | 预期证据命中率（区域级 Recall@1，粗估） |
|---|---|---|
| RegionRAG | arXiv:2510.27261 作者仓库（若可获取） | ~70–80% |
| ColPali（页级） | github.com/illuin-tech/colpali | ~55–65% |
| OCR + BGE dense | PaddleOCR + bge-m3 | ~45–55% |
| CLIP 稠密（页级） | open_clip | ~40–50% |
| RegionBench 区域检索器（本项目） | 自建 | ~75–85% |
- 复现步骤：ColPali 官方推理（ViDoRe 验证其页级 Recall）；OCR+BGE 用 PaddleOCR 提取文本 + bge-m3 索引；RegionRAG 无官方代码则按论文实现混合监督 + 动态聚 region。
- 统一口径：同一生成器、同一 top-k、同一评测脚本；页级基线也按"证据页命中"口径计算，保证可比。

## 6. 实验矩阵
- A（主实验）：区域级 vs 页级检索 × 4 类文档 × 4 语言（主为英文，其余 3 语言各 1 子集）。
- B（消融）：粒度选择（页/区域/混合 top-k）；区域数阈值；LoRA rank；是否用 Kimi 校验过滤。
- C（鲁棒性）：低分辨率页面（150dpi）；倾斜/扫描噪声；表格密集文档单独报告。
- D（泛化性）：跨 VLM 定位器（Qwen2.5-VL vs 其他）；跨生成器（text-only 7B vs 多模态）；在 DocVQA/ViDoRe 上测 zero-shot 迁移。

## 7. 评测协议
- 检索指标：区域级 Recall@1/@5、证据命中率；页级 Recall@1（对照口径）。
- 效率指标：视觉 token 压缩率（区域 token / 整页 token）、延迟。
- 生成指标：QA 准确率（每类文档）、区域级可追溯率。
- 标注质量：人工抽检 300 条 bbox IoU≥0.5 比例、judge 一致性 κ。
- 统计：3 种子；均值±std；配对 bootstrap p<0.05。

## 8. 算力与资源计划
- 4×L40：Qwen2.5-VL-7B LoRA 2–3 GPU·天；生成评测 1 GPU·天；消融 1–2 GPU·天；合计 4–6 GPU·天。
- 存储：PDF 渲染 + patch 缓存 <200GB。
- API：DeepSeek 问题/理由生成 ≈ 150–300 美元；Kimi 校验 + 多语言合成 ≈ 100–200 美元；合计 ≤ 500 美元。

## 9. 里程碑与时间线（按周，单人 + 4 卡）
| 周 | 任务 |
|---|---|
| 1 | 文档收集 + 渲染；500 条定位器标定（≥85% 达标） |
| 2 | 区域标注管线 + 问题生成 + 校验；人工抽检 300 条 |
| 3 | 区域检索器 LoRA 训练；基线复现 |
| 4 | 主实验 A（4 类 × 4 语言） |
| 5 | 消融 B + 鲁棒性 C + 泛化 D |
| 6 | 统计 + 论文初稿 + 数据集开源 |

## 10. 风险与备选方案
| 风险 | 等级 | 对策 |
|---|---|---|
| 区域标注成本与 VLM 定位噪声 | 高 | 先 500 条标定定位准确率（≥85%）再扩量；失败样例计入报告 |
| RegionRAG 无官方代码 | 中 | 按论文自实现并开源；或以 ColPali 页级为对照主基线 |
| 多语言标注质量不稳 | 中 | 每语言人工抽检 50 条；跨语言一致性检查 |
| 多模态生成评测昂贵 | 中 | 生成用 text-only（OCR 文本）+ 多模态二选一，控制 API |

## 11. 论文写作计划
- 目标：CVPR 2027 / ACM MM 2027；若进度快（2026 年内完成）可投 ECCV 2026。
- 差异化卖点：首个区域级视觉文档 RAG 基准；双粒度评测协议 + 视觉 token 压缩率指标；多语言四类文档。
- 图表清单：图1 标注管线与区域可视化；图2 区域 vs 页级检索对比；图3 视觉 token 压缩曲线；表1 数据集统计；表2 主结果；表3 消融；表4 跨语言/跨域。
- 相关工作覆盖：RegionRAG（CV V · 论文 85）、ColPali（arXiv:2407.01449）、Double-Bench（NLP IV · 论文 30）、OmniDocBench（arXiv:2412.07626）、HPC-ColPali（arXiv:2506.21601）。

## 12. 参考文献
- RegionRAG: arXiv:2510.27261（CV V · 论文 85）
- Double-Bench: arXiv:2508.03644（NLP IV · 论文 30）
- ColPali: arXiv:2407.01449
- OmniDocBench: arXiv:2412.07626
- PureDocBench: arXiv:2605.07492
- HPC-ColPali: arXiv:2506.21601
- Lost in OCR Translation?: arXiv:2505.05666
