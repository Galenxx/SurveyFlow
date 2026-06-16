# Cohen's κ 校准标注指南

## 任务
对抽样的 30 条 cited claim，判断引文是否支持所述内容。

## 标注类别

| 类别 | 含义 | 分数 |
|------|------|------|
| **SUPPORTED** | 引文完全支持该句的核心事实主张 | 1.0 |
| **PARTIAL** | 引文部分支持（支持部分主张但非全部） | 0.5 |
| **UNSUPPORTED** | 引文完全不支持该句内容 | 0.0 |

## 判断标准
1. 只看"引文原文是否能验证该句"，不考虑常识是否正确
2. PARTIAL = 引用确实讨论了相关主题，但具体数字/方法/结论不完全匹配
3. 如果句子有多个引文，只要有一个支持即可（per-citation contribution 已单独记录）

## 操作步骤
1. 打开 `kappa_sample.json`
2. 对每条记录，阅读 `text` 字段（被引句）
3. 根据 `llm_rationale` 中描述的证据来源，判断支持程度
4. 在 `human_verdict` 填入: SUPPORTED / PARTIAL / UNSUPPORTED
5. 可选在 `human_note` 写备注

## 统计方法
标注完成后运行 `python -m evaluation_citation.kappa_compute` 计算：
- Cohen's κ（三分类 ordinal weighted）
- 混淆矩阵
- 逐系统 κ

预期耗时：~30 分钟（每条 ~1 分钟）
