# Citation Benchmark v2 设计文档

**目标**：重新设计引用质量基准，消除 v1 的循环论证缺陷，使 SurveyFlow 的实验结论"立得住"，可直接用于 BESC demo paper 的改写。

**状态**：设计稿，待审批后进入代码实现。

**对齐基准**：本文档所有"可复用"标注均已对照 `evaluation_citation/` 现有代码核实（截至 2026-06-15）。

---

## 0. 一句话概括

把基准测量的核心对象从 **"引用是否存在"（对检索系统恒真，循环论证）** 转为 **"引用是否真的支持它所挂的那句断言"（ALCE 风格的句子级归因，对检索系统不恒真）**。存在性指标保留，但降级为预处理闸门 + 失败诊断。

---

## 1. 背景：v1 为什么被审查打穿

v1 基准（`evaluation_citation/`）报告三个指标：existence / identifier precision / faithfulness，对比 SurveyFlow vs LLM_ONLY，结论是"存在率 34% → 94.4%"。

审查指出的三条硬伤，按严重度：

| # | 缺陷 | 根因 | v2 对策 |
|---|------|------|---------|
| 1 | **存在性循环论证（最致命）** | SurveyFlow 只引用自己下载并嵌入的论文，引用 by construction 必然存在；"存在率 94.4%"近乎同义反复 | 主指标改为句子级归因（§4.A）；存在性降级（§4.B） |
| 2 | **基线是稻草人** | LLM_ONLY = 同模型去检索、凭记忆引用，结论退化为"RAG vs 无 RAG"（领域常识） | 加 Naive-RAG arm，做成消融阶梯（§3） |
| 3 | **统计不严谨** | 每领域 1 主题、无重复、无方差/CI、把 9/11 报成 81.8%（虚假精度） | ≥10 主题、3 seed、报计数+Wilson CI、配对检验（§5） |
| 4 | **忠实度评判无校准** | 单一 LLM judge，无人工标注、无 κ | 接通现有 `human_validation/` 脚手架，跑真实 κ（§6） |
| 5 | **arXiv 单源混淆** | 非 CS 领域召回不足（bio/HCI 仅 2–3 篇），混淆"召回质量"与"领域代表性" | 报语料统计；可选第二检索源（§7） |
| 6 | **确定性≠可复现** | 控制流确定，但 LM 采样 + arXiv 时变使输出不可复现 | 用 3-seed 实测输出方差，措辞切开两概念（§8） |

---

## 2. 核心重构：存在性 → 归因质量

### 2.1 为什么存在性是循环的

SurveyFlow 的 `writer_node._generate_references` 从真实下载论文的 metadata 装配引用表（已核实：`nodes/writer_node.py:634`）。因此"引用解析到真实论文"几乎恒成立——这不是方法的功劳，是数据流的必然。审查的原话是对的：真正有信息量的反而是"为什么不是 100%"。

### 2.2 新主指标问的问题

借用论文已引用却没真正使用的 ALCE（`gao2023alce`）的核心问题：

> 对正文里每一句带引用的断言 s，它引用的论文内容**是否真的支持 s**？

这个指标对检索系统**不恒真**：即便每条引用都真实存在，LM 仍可能
- 把引用挂在源文档并不支持的句子上（**幻觉式归因**），
- 一句话堆一串无关引用充数（**过度引用**），
- 做事实陈述却完全不引用（**裸断言**）。

SurveyFlow 仍应胜出，但赢得有信息量：证明的是 **grounding 真的起作用**，而非"我下载的论文确实存在"。

---

## 3. 对比系统：消融阶梯（回应审查 #2）

所有 arm 共享同一 generator model、同一语料、同一 chunking，把差异**收敛到 grounding 策略**：

| 系统 | 检索策略 | 角色 | 实现来源 |
|------|----------|------|----------|
| `LLM_ONLY` | 无，凭参数记忆引用 ~10 篇 | 消融下界（**降级**，不再是主基线） | 现有 `generators/`，复用 |
| `NAIVE_RAG` | 全语料平铺检索一次，top-k 拼接进上下文 | **诚实真基线**：同模型/同语料/同 chunking，仅去掉结构化 | **新增**，在 writer 上去掉 cluster/section 映射 |
| `SURVEYFLOW` | cluster-scoped + section-mapped 检索 | 本文方法 | 现有 pipeline，复用 |

结论从"RAG vs 无 RAG"（常识）升级为 **"结构化 grounding > 平铺 grounding > 无 grounding"**，贡献被干净隔离。

> **范围决策**：AutoSurvey/STORM 实证对比理想但工程量大，且引用机制不同，强行入表反而稀释对照。保留在 related work 讨论，不进结果表。（Tier 3 可选）

---

## 4. 三轴指标（精确定义）

记号：survey 切分为带引用的断言句集合 S；对句子 s，其引用集为 C(s)；每条引用 c 关联其源证据 E(c) = {论文摘要} ∪ {该论文实际被检索到的 chunks}。

### 4.A 归因质量（主指标，修复循环性）

**A1. Citation Recall（断言被支持率）**
对每个 s：判定 E(C(s)) 整体是否**蕴含/支持** s。判定器输出 {SUPPORTED:1.0, PARTIAL:0.5, UNSUPPORTED:0.0}。

```
CitationRecall(system, topic) = mean over s in S of support_score(s)
```

**A2. Citation Precision（引用不冗余率）**
对每个 s 及其每条引用 c ∈ C(s)：去掉 c 后剩余引用是否仍支持 s。若去掉 c 不改变"支持"判定，则 c 对该句**无贡献**（冗余）。

```
CitationPrecision = (有贡献的引用数) / (总引用挂载数)
```

**A3. 裸断言率（Unsupported-Claim Rate，诊断）**
做事实陈述但 C(s) = ∅ 的句子占比。综述不应有裸断言，越低越好。

```
BareAssertionRate = |{s : is_factual(s) and C(s)=∅}| / |{s : is_factual(s)}|
```

**判定器**：默认 LLM judge（rubric 见 §4.D），人工校准其可靠度（§6）。可选 NLI 模型（如 DeBERTa-MNLI）做交叉验证。

### 4.B 存在性 / 可解析性（降级为闸门 + 诊断，回应 #1）

仍然计算，但**重新定位为"引用装配流程正确性"**，并**分解失败原因**（审查明确要的）：

```
对每条不可解析的引用，归类到：
  - METADATA_PARSE_ERROR   (作者/标题/年份抽取失败)
  - ID_EXTRACTION_FAIL     (arXiv id / DOI 抽取失败)
  - DEDUP_COLLISION        (去重把不同论文并掉)
  - GENUINELY_FABRICATED   (生成器凭空造，仅 LLM_ONLY 期望出现)
```

报每个系统的失败分布。SurveyFlow 的 5.6% 失败若集中在前三类，正好说明"不是幻觉，是工程 bug"——把审查 #1 从软肋变成亮点。LLM_ONLY 的失败集中在 GENUINELY_FABRICATED，保留其对照价值。

**复用**：现有 `verifiers/arxiv_verify.py` + `identifier_check.py`（已核实：title_similarity = SequenceMatcher ratio + token Jaccard 混合，阈值 `EXISTENCE_SIM_THRESHOLD=0.85` / `IDENTIFIER_SIM_THRESHOLD=0.80`）。仅需在解析失败处加分类标签。

### 4.C 主题相关性（收紧为 claim-relative）

v1 的 faithfulness 实测的是"引用论文是否与 survey 主题相关"。保留，但改为**相对于该句断言**的相关性，避免与 4.A 重叠。

**复用**：现有 `judges/faithfulness_judge.py`（VERDICT_SCORE = {RELEVANT:1.0, PARTIAL:0.5, IRRELEVANT:0.0}），改 prompt 把判定对象从"topic"换成"claim s"。

### 4.D 判定 rubric（A1/A2 共用）

```
输入：断言句 s（含上下文）、引用集 C(s)、每条引用的 E(c)（标题+摘要+检索到的 chunk）
任务：E(C(s)) 是否支持 s 中的事实主张？
输出 JSON：{ "verdict": "SUPPORTED|PARTIAL|UNSUPPORTED",
            "score": 1.0|0.5|0.0,
            "rationale": "...",
            "per_citation_contribution": { "c1": true/false, ... } }
判定原则：
  - SUPPORTED：s 的核心主张可由 E(C(s)) 直接推出
  - PARTIAL：部分主张被支持，或仅主题相关但不直接支持具体论断
  - UNSUPPORTED：E(C(s)) 与 s 无支持关系（含主题相关但不支持论断的情形）
```

**judge 模型**：沿用 `JUDGE_MODEL=deepseek-v4-pro`，opposite-family 回退到 `glm-5.1`，`JUDGE_TEMPERATURE=0.0`（已核实 `configs.py`）。⚠️ 模型名真实性见 §10。

---

## 5. 统计协议（修复 #3）

| 项 | v1 | v2 |
|----|----|----|
| 主题网格 | 5（每领域 1） | **≥10（每领域 2）** |
| 重复 | 无 | **每 (系统,主题) 跑 3 seed** |
| 报告粒度 | 百分比到 0.1% | **分子/分母计数为主**（如 `9/11`），百分比辅助 |
| 比例区间 | 无 | **Wilson score CI**（小样本比正态近似稳） |
| 系统间检验 | 无 | **Wilcoxon signed-rank**（跨主题配对，SurveyFlow vs 每个基线） |
| 差值区间 | 无 | **配对 bootstrap** 给差值 CI + 报效应量 |

> 只有 10+ 主题 + 配对检验 + 效应量，摘要里"large and stable gap"这类措辞才站得住。否则改为保守措辞。

**3-seed 的双重用途**：既刻画 LM 采样方差（统计），又为 §8 可复现性提供实测数据。

---

## 6. 人工校准（修复 #4，性价比最高）

`evaluation/human_validation/annotate.py` + `cohen_kappa.py` **已存在但从未跑出真实 κ 值**（已核实：目录无任何输出文件）。接上真数据即可：

1. **分层抽样 100 条** (句子, 引用) 对，覆盖各系统 × 各主题。
2. **2 名标注者**按 §4.D rubric 独立打 {1.0, 0.5, 0.0}。
3. 报 **标注者间 Cohen's κ**（标注本身可靠性）。
4. 报 **人工 vs LLM judge 一致性**（κ 或 Spearman ρ）。

即便 judge 一致性不高，也**量化**了它的可靠度，而非像 v1 仅在 Limitations 口头承认。工程量小（脚手架现成），直接堵住审查最硬的一条。

---

## 7. 混淆控制（回应 #5）

**必做（低成本）**：每个主题报语料统计——下载论文数、年份跨度、chunk 总数、可引用论文数。这直接解释 bio/HCI 主题为何只召回 2–3 篇（arXiv 覆盖偏差，非方法缺陷），把混淆变量摊开给读者看。

**选做（高成本，Tier 3）**：retrieval 加 OpenAlex 或 Semantic Scholar 第二源，拓宽非 CS 领域覆盖。`s2_client.py` 已存在，可复用。

---

## 8. 可复现性的诚实测量（回应 #6）

不再口头声称 "reproduces"。用 §5 的 3-seed 重跑：

```
OutputVariance(topic) = std over seeds of [CitationRecall, Existence, ...]
ReferenceSetJaccard(topic) = mean pairwise Jaccard of 引用集 over seeds
```

论文措辞明确切开两个概念：
- **控制流确定性**（control-flow determinism）：节点序列每次相同——这是真的。
- **输出可复现性**（output reproducibility）：受 LM 采样 + arXiv 时变影响——用方差/Jaccard 量化，不夸大。

---

## 9. 组件映射：复用 vs 新建

| 组件 | 状态 | 来源 / 说明 |
|------|------|------------|
| 五段流水线 generate→parse→verify→judge→aggregate | ✅ 复用骨架 | `run_benchmark.py` |
| `citation_parser.py`（解析引用表） | ✅ 复用 | `parse_survey_md`, `ParsedReference` |
| `verifiers/arxiv_verify.py` + `identifier_check.py` | ✅ 复用 + 加失败分类标签 | title_similarity 逻辑不变 |
| `judges/faithfulness_judge.py` | ✅ 复用 + 改 prompt（topic→claim） | VERDICT_SCORE 不变 |
| judge 回退机制 | ✅ 复用 | `_pick_judge_model` opposite-family |
| `human_validation/{annotate,cohen_kappa}.py` | ⚠️ 脚手架存在，**接真数据** | 从未产出 κ |
| **句子切分 + 引用挂载抽取** | 🆕 新建 | 从 survey 正文切出 S 与 C(s)（v1 只解析参考文献表，不解析正文内引用） |
| **A1/A2/A3 归因指标** | 🆕 新建 | §4.A |
| **NAIVE_RAG generator** | 🆕 新建 | §3，writer 去 cluster/section |
| **统计模块**（Wilson CI / Wilcoxon / bootstrap） | 🆕 新建 | §5 |
| **语料统计采集** | 🆕 新建 | §7 必做项 |
| **3-seed runner + 方差/Jaccard** | 🆕 新建 | §8 |
| `aggregate.py` + LaTeX 表生成 | 🔧 改造 | 输出新指标列、计数、CI |

> **最大的新工作量是"正文内引用抽取"**：v1 的 parser 只处理参考文献列表（`Reference: ...` 行），而 A1/A2 需要把正文切成句子并识别每句挂了哪些引用 marker。这是 v2 的技术核心。

---

## 10. 目录结构与产物

```
evaluation_citation_v2/
├── configs.py              # 10+ topics, 3 systems, 3 seeds, 阈值
├── run_benchmark.py        # 编排：generate→parse→attribute→verify→judge→aggregate
├── generators/
│   ├── surveyflow_runner.py    # 复用
│   ├── llm_only_runner.py      # 复用
│   └── naive_rag_runner.py     # 🆕
├── parsers/
│   ├── reference_parser.py     # 复用 citation_parser
│   └── claim_extractor.py      # 🆕 正文句子切分 + 引用挂载
├── verifiers/                  # 复用 + 失败分类
├── judges/
│   ├── attribution_judge.py    # 🆕 A1/A2 (rubric §4.D)
│   └── relevance_judge.py      # 复用 faithfulness_judge 改造
├── stats/
│   ├── intervals.py            # 🆕 Wilson CI
│   ├── tests.py                # 🆕 Wilcoxon + bootstrap
│   └── reproducibility.py      # 🆕 方差 / Jaccard
├── human_validation/           # 接现有脚手架
└── results/
    ├── <system>/<topic>__seed<k>__{generate,parse,attribute,verify,judge}.json
    ├── corpus_stats.json       # §7
    ├── reproducibility.json    # §8
    ├── human_kappa.json        # §6
    ├── summary.json
    └── tables/{main,per_system,attribution,reproducibility}.tex
```

每段持久化到磁盘（沿用 v1 可断点续跑设计）。

---

## 11. 分层实施计划

| Tier | 内容 | 工作量 | 单独能否立住论文 |
|------|------|--------|------------------|
| **Tier 1（核心，必做）** | 正文引用抽取 + A1/A2/A3 归因指标；存在性失败分析；计数+Wilson CI；接通人工 κ | 中 | ✅ 正面回应 #1/#3/#4 三条最硬批评 |
| **Tier 2（强化）** | NAIVE_RAG arm；主题扩到 10；3-seed 方差 | 中高 | 让 #2/#6 也干净 |
| **Tier 3（锦上添花）** | 第二检索源消除 arXiv 偏差；AutoSurvey/STORM 实证对比 | 高 | 回应 #5 + 补强 #2 |

**Tier 1 单独就能把论文从"被打穿"变成"立得住"。**

---

## 12. 未决事项（需用户拍板）

1. ⚠️ **模型名真实性（卡脖子）**：`glm-5.1` / `deepseek-v4-pro` 必须是真实调用的模型。`.env` 与 `configs.py` 里都写死了这两个名字，但我无法确认它们是真实发布版本。**若是占位名，重跑 benchmark 也救不了真实性质疑。**
2. **实施层级**：先做 Tier 1，还是 Tier 1+2？（截止窗口已过，这是为下一个投稿窗口/扩展版准备）
3. **NLI 交叉验证**：A1/A2 是否要加 NLI 模型做 judge 之外的第二判定器？（增强可信度，但增加依赖）
4. **主题网格**：扩到 10 个时，新增 5 个主题是否仍用"每领域 2 个"原则？是否需要避开 arXiv 覆盖差的领域？
