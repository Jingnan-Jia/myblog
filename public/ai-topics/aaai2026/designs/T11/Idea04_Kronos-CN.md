# 实验设计书：Kronos-CN：面向 A 股/港股/可转债的多市场 K 线基础模型与合成数据闭环

> 所属主题：T11 金融AI与决策 ｜ 文件：Idea04_Kronos-CN.md
> 关联收藏论文：Machine Learning VII · 论文 47（Kronos）

## 0. 摘要

Kronos-CN 以 Kronos 为蓝本，预训练一个面向中国市场的 K 线（OHLCV）时间序列基础模型，语料覆盖 A 股、港股与可转债，并显式处理涨跌停、停牌、T+1、前/后复权与退市样本等中国制度特征。Kronos 原版的空白有三：语料集中于欧美、合成 K 线只报分布保真度而未做下游闭环、零样本评测缺新兴市场。Kronos-CN 据此设计三个贡献：(1) 制度感知的 tokenizer（分位数 + 涨跌停/一字板标记）与防前视复权数据管线；(2) 合成数据闭环评测：用 Kronos-CN 生成合成 K 线 → 在数据稀疏的小票/新板块微调 → 比较回测 RankIC/换手率是否改善；(3) 显式处理幸存者偏差（保留退市样本）。预训练规模 1-3B 全参（4×L40 可承担），7B 用 QLoRA 版本作为扩展验证。

## 1. 研究背景与动机

### 1.1 问题定义

输入：多市场 K 线序列 `x_t = (open, high, low, close, volume, amount, 涨跌停状态, 停牌标记)`，时间粒度日线（分钟线作为扩展验证）。任务：
1. **价格预测**：给定过去 L 根 K 线，预测未来 h∈{1,5,10} 期的收益方向/幅度（RankIC 主指标）；
2. **波动率预测**：预测未来 h 期已实现波动率（MAE 主指标）；
3. **合成 K 线生成**：自回归生成条件 K 线序列，并把"生成数据→小样本市场微调→回测收益提升"作为闭环验证。

### 1.2 相关工作不足

- **Kronos（AAAI Machine Learning VII · 论文 47 · Kronos: A Foundation Model for the Language of Financial Markets）**：语料以 45 家欧美交易所为主，未覆盖 A 股制度（涨跌停/停牌/T+1/复权/退市）；合成生成只报分布保真度未做下游闭环；120 亿条不可复现、无蒸馏路线。
- **FinCast（arXiv:2508.19609）/ LENS（arXiv:2408.10111）/ DELPHYNE（arXiv:2506.06288）**：同样以欧美价格自回归为主，缺中国市场细粒度处理与制度感知。
- **Chronos（arXiv:2403.07815）/ Moirai（arXiv:2402.02592）/ TimeGPT（arXiv:2310.03589）**：通用 TSFM，无金融制度建模，零样本在 A 股表现弱。
- **FINSABER（arXiv:2505.07078）**：指出多数 LLM 投资策略收益被幸存者偏差/数据窥探夸大——正说明退市样本与防前视复权必须显式处理。

### 1.3 为什么是现在、为什么你的环境适合做

中国全 A 股 + 可转债的日线数据通过 tushare/baostock/akshare 公开可得（等级权限内）；Kronos 已开源 tokenizer 与训练框架可复用；社区对"合成数据闭环验证"有明确空白。环境优势：(1) 1-3B 全参预训练在 4×L40（192GB）FSDP 下 2-4 周可完成，属于可复现规模；(2) tushare 权限（积分制）与 baostock 免费接口可拼出全 A 股日线；(3) 可与 Idea 5（新闻融合）、Idea 11（时间线自监督）直接复用权重。

## 2. 研究目标与可验证假设

| # | 假设 | 成立时的可观测结果 |
|---|------|---------------------|
| H1 | 制度感知 tokenizer 提升中国市场预测 | 加涨跌停/停牌标记的 Kronos-CN 在 A 股 RankIC 上高于无标记版 ≥3%（相对） |
| H2 | 中国市场语料预训练优于欧美模型零样本 | Kronos-CN 在 A 股/可转债零样本 RankIC > Kronos 原版零样本 |
| H3 | 合成 K 线闭环可提升小样本市场预测 | 在数据稀疏的小票/新板块上，用 Kronos-CN 合成数据增强后微调的 RankIC 增量 > 0 且显著 |
| H4 | 防前视复权与退市样本处理影响评测公平性 | 用前复权（含未来分红信息）vs 无前视复权做对照，后者 RankIC 显著低于前者（量化污染幅度） |

第一验证实验：1-3B 小模型在单行业（如银行）上做 1 周预训练 pilot，对比有无制度标记的 RankIC（H1）。

## 3. 总体方法设计

### 3.1 数据流水线

- **数据源**：tushare（pro_bar，日线/分钟线，需积分）、baostock（日线，免费）、akshare（公告/公司行动，免费）；港股用 akshare 港股接口；可转债用 tushare/交易所转债行情；
- **清洗**：
  - 复权：**双版本**——前复权（因子含未来信息）与"无前视后复权（后复权=每日只用当日及之前数据计算，不引入未来分红因子）"。训练与评测以无前视版本为准，前复权版本仅用于对照实验量化污染；
  - 停牌剔除：停牌日标记为特殊 token（不合并删除，保留停牌 token），防止时间错位；
  - 退市样本保留（含退市日），显式打上退市标记 token，避免幸存者偏差；
  - 涨跌停：涨停/跌停/一字板各一个标记 token，按交易所规则（主板 ±10%、创业板/科创板 ±20%、ST ±5%、可转债 ±20% 或临停）计算；
- **tokenizer**：Kronos 式分位数 tokenizer（按全语料统计量分箱）+ 制度标记 token；每根 K 线输出固定 5 个 token（O/H/L/C/V）+ 可选制度 token；
- **数量**：全 A 股（约 5,000 只）× 2020-2026 日线（约 1,500 交易日）+ 港股（约 500 只主要成分）+ 可转债（约 300 只）≈ 8,700 万根 K 线（约 6 亿 token）；分钟线做 1 个月 pilot 验证可行性后再定；
- **API 用途（辅助）**：DeepSeek V4 Flash 生成"市场制度说明 → 数据校验文档"；Pro 做回测报告总结与异常检测；**模型输出不参与训练语料**。

### 3.2 方法设计

- **模型**：Kronos 结构（token 化输入 → causal transformer 自回归 next-token）；1-3B（layers 12-24，dim 768-2048）；
- **目标**：`L = L_CE(next-token)`，预测下一根 K 线的 5-token 组；附加制度标记的辅助分类损失（涨停/跌停）可叠加；
- **下游 head**：价格预测用 token 序列解码出预测收益分布；波动率预测用同一序列取最后隐状态回归；
- **合成数据闭环**：条件生成（给定股票风格 token + 市场状态 token）→ 过滤（分布矩/ACF 保真度 top-k）→ 与真实小样本拼接微调；
- **防前视清单**：无前视复权；交易信号只用 ≤t 收盘；涨跌停日不可成交（无法按收盘价成交）；T+1 买入限制；停牌不可交易。

### 3.3 训练流程

- 优化器 AdamW（β=(0.9,0.98)），warmup 2,000 步，峰值 lr 1e-3（cosine 衰减），batch 0.5-1M token，fp16 混合精度；
- 4×L40 FSDP（shard=True），sequence packing；1-3B 全参 2-4 GPU·周；7B QLoRA 版（r=32）作为扩展；
- 数据集按 90/5/5 切分（训练/验证/评测，**按时间切分**，评测期最后 12 个月 2025-07 至 2026-06 不参与训练）；
- 分钟线 pilot：先用 200 只股票 × 3 个月分钟线验证可行性与显存，再决定是否加入主训练。

### 3.4 回测与评测流程

- **回测引擎**：事件驱动（开卷），严格约束：T+1、交易成本（佣金+印花税+滑点，双边 30-50bp）、涨跌停不可成交、停牌跳过、成交量上限（成交比例 ≤ 日成交额 10%）；
- **评测期**：2025-07 至 2026-06（窗口外，与训练零重叠）；
- **防泄漏**：股票池 = 评测期初存续 + 退市样本；复权用无前视版本；任何因子在 t 日只能使用 ≤t 数据；
- **合成闭环协议**：在数据稀疏集（如创业板小盘 100 只、可转债）上，对照组=仅真实数据微调，实验组=真实+合成数据微调，比较 RankIC/ICIR/回测 Sharpe（成本后）；
- 报告均带 mean±std（3 种子）。

## 4. 数据集细节

| 项 | 说明 |
|---|---|
| 名称 | Kronos-CN v1（自建语料，公开数据源） |
| 来源 | tushare / baostock / akshare（A 股、港股、可转债） |
| 许可 | 数据源各自许可（tushare 积分条款 / baostock 免费 / akshare MIT）；项目仅发布 tokenizer 配置与训练配置，原始行情不重分发 |
| 划分 | 时间切分 90/5/5；评测期 2025-07 至 2026-06 |
| 规模 | ~8,700 万根 K 线 → ~6 亿 token |
| 预处理 | 无前视复权、停牌/涨跌停/退市标记、异常值（0 价/负价）剔除、MinHash 去重行业重复股 |
| 质量门 | 与 Wind/东方财富抽查 500 只股票价格一致率 >99.9% |

## 5. 基线复现

| 基线 | 类型 | 复现方式/官方地址 | 预期指标 |
|---|---|---|---|
| Kronos 原版（零样本） | TSFM | https://github.com/shiyu-coder/Kronos（权重或按 2508.02739 微调/零样本） | RankIC / vol MAE |
| Chronos | 通用 TSFM | https://github.com/amazon-science/chronos-forecasting | 同上 |
| Moirai | 通用 TSFM | https://github.com/SalesforceAIResearch/uni2ts | 同上 |
| TimeGPT-1 | 通用 TSFM | https://github.com/Nixtla/nixtla（API/库） | 同上 |
| FinCast / LENS | 金融 TSFM | arXiv:2508.19609 / arXiv:2408.10111（以官方仓库为准） | 同上 |
| 经典基线 | LGBM / ARIMA / GARCH | 自实现 | 同上 |
| Kronos-CN（无制度标记） | 消融 | 本设计去标记版 | 同上 |

**统一口径**：同一评测期（2025-07 至 2026-06）、同一股票池、同一 RankIC 计分器、同一无前视复权输入；零样本 vs 微调分开报告。

## 6. 实验矩阵

- **A 主实验**：Kronos-CN（1-3B）vs 全部基线的零样本 + 微调 RankIC / vol MAE；
- **B tokenizer 消融**：有无涨跌停/停牌/退市标记、分位数粒度（k∈{32,64,128}）；
- **C 复权消融**：无前视后复权 vs 前复权（量化污染幅度，H4）；
- **D 合成数据闭环**：小票/可转债上"仅真实 vs 真实+合成"微调对比（H3）；
- **E 规模扩展**：1B vs 3B vs 7B-QLoRA 的指标-算力曲线；
- **F 泛化**：跨市场（A→港股）零样本迁移、可转债专项；
- **G 稳健性**：评测期滚动（3 个半年）、交易成本敏感性（20/50/100bp）、种子。

## 7. 评测协议

- **RankIC**：截面 Spearman（预测收益 vs 真实收益），按日平均 + ICIR；显著性与方差按日序列 t 检验；
- **vol MAE**：预测 vs 已实现波动率的 MAE（相对值）；
- **生成保真度**：分布矩（均值/方差/偏度/峰度）、自相关 ACF(1-5)、收益分布 KS 检验；
- **闭环收益**：合成增强微调后的 RankIC 增量（配对 t 检验）；
- **回测**：成本后年化收益、Sharpe、Calmar、最大回撤、换手率、胜率；**所有回测在窗口外评测期**；
- **无前视**：见 3.4 清单，逐条写进论文附录。

## 8. 算力与资源计划

| 阶段 | 资源 | 量 |
|---|---|---|
| 数据清洗与 tokenizer | CPU | 1-2 人·周 |
| 1-3B 全参预训练 | 4×L40 FSDP | 2-4 GPU·周（56-112 GPU·天） |
| 7B QLoRA 扩展 | 4×L40 | 1-2 GPU·周 |
| 下游微调 + 合成闭环 | 4×L40 | 3-5 GPU·天/实验 |
| 回测与评测 | CPU + 4×L40（并行） | 2-3 GPU·天 |
| 存储 | 原始行情 + token 化 + checkpoint | 2-3TB（SSD+机械混合） |
| 总计 | 4×L40 | 约 70-140 GPU·天 |

**API 成本**：本 idea 以训练为主，API 辅助（Flash 文档/校验、Pro 回测总结）约 1,000 万 token，估算 $50-200。

## 9. 里程碑与时间线（单人 + 4×L40，12 周）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | 数据接入 + 复权/停牌/涨跌停清洗 | 语料 v1 + 质量报告 |
| W2 | tokenizer 实现 + 分钟线 pilot | tokenizer + pilot 报告（决定是否加分钟线） |
| W3 | 单行业 pilot 预训练（银行） | H1 pilot（制度标记对照） |
| W4 | 1B 全量预训练启动 | 1B checkpoint（周级日志） |
| W5-W6 | 1B 预训练完成 + 3B 启动 | 1B 评测初表 |
| W7 | 3B 预训练 + 下游 head | 3B checkpoint |
| W8 | 零样本主实验 A + 基线复现 | A 表 |
| W9 | tokenizer/复权消融 B/C | B/C 表 |
| W10 | 合成数据闭环 D | D 表（H3 检验） |
| W11 | 规模扩展 + 泛化 E/F | E/F 表 |
| W12 | 回测审计 + 论文初稿 | 提交稿 |

## 10. 风险与备选方案

| 风险 | 等级 | 缓解/备选 |
|---|---|---|
| 中国市场制度导致 TSFM 收益不如预期（涨跌停截断分布） | 高 | 1-3B 小模型 + 单行业 pilot 先行；若 RankIC 无增量，弱化为"合成数据生成 + 波动率预测"主线 |
| 复权数据引入前视偏差 | 高 | 无前视后复权为主版本，前复权仅作对照量化污染；写进回测协议 |
| tushare 权限/配额限制 | 中 | baostock/akshare 兜底；分批下载 + 本地缓存 |
| 训练成本超支 | 中 | 1B 先行验证再升 3B；7B 仅 QLoRA；用序列长度裁剪控制 token 预算 |
| 幸存者偏差 | 中 | 退市样本保留；股票池按评测期初存续定义 |
| 分钟线扩展不可行 | 低 | 定位日线为主，分钟线作附录 |

## 11. 论文写作计划

- **目标会议**：NeurIPS 2026（9 月截稿，以官方公告为准）/ ICML 2027（基础模型方向）。
- **差异化卖点**：(1) 首个制度感知（涨跌停/停牌/复权/退市）的中国 K 线 TSFM；(2) 首个合成 K 线"下游闭环"验证（不只报保真度）；(3) 显式量化复权前视污染幅度。
- **图表清单**：图1 数据管线（复权/标记）；图2 tokenizer 示意图；图3 预训练损失与评测曲线；图4 合成闭环流程与增量；图5 复权对照污染图；表1 语料统计；表2 主实验；表3 消融；表4 回测审计；表5 泛化。
- **相关工作覆盖**：Kronos（2508.02739）、FinCast（2508.19609）、LENS（2408.10111）、DELPHYNE（2506.06288）、Chronos（2403.07815）、Moirai（2402.02592）、TimeGPT（2310.03589）、FINSABER（2505.07078）、Advancing Financial Engineering（2507.18577）。

## 12. 参考文献

1. Shi, Y., Fu, Z., Chen, S., Zhao, B., Xu, W., Zhang, C., Li, J. *Kronos: A Foundation Model for the Language of Financial Markets*. arXiv:2508.02739.
2. *FinCast: A Financial Time Series Foundation Model*. arXiv:2508.19609.
3. *LENS: A Foundation Model for Financial Time Series*. arXiv:2408.10111.
4. *DELPHYNE: ... Financial Time Series Foundation Model*. arXiv:2506.06288.
5. Ansari, A., et al. *Chronos: Learning the Language of Time Series*. arXiv:2403.07815.
6. Woo, G., et al. *UniTS / Moirai: A Time Series Foundation Model*. arXiv:2402.02592.
7. Garza, A., & Mergenthaler-Canseco, M. *TimeGPT-1*. arXiv:2310.03589.
8. *FINSABER: Twenty-Year Regime-Switching Backtesting of LLM Investment Strategies*. arXiv:2505.07078.
9. *Advancing Financial Engineering with Foundation Models*. arXiv:2507.18577.
10. *Timer-S1: Billion-Parameter Serial Scaling for Time Series*. arXiv:2603.04791.
11. *The Alpha Illusion*. arXiv:2605.16895.
