"""
Cohen's κ 校准：从 attribute 产物中分层抽样 30 条 cited claim。
产出 kappa_sample.json（供人工标注）+ kappa_sample_guide.md（标注指南）。

抽样策略：10 条/系统，分层保证 verdict 分布近似原始比例。
固定 seed 保证可复现。
"""

import json
import random
from pathlib import Path
from collections import defaultdict

RESULTS = Path(__file__).parent / "results"
SYSTEMS = ["SURVEYFLOW", "NAIVE_RAG", "LLM_ONLY"]
SAMPLE_PER_SYSTEM = 10
RANDOM_SEED = 42


def collect_all_claims():
    """从所有 attribute JSON 中收集全部 per_sentence 记录"""
    claims = []
    for system_dir in ["surveyflow", "naive_rag", "llm_only"]:
        sys_label = system_dir.upper()
        pattern = f"{system_dir}/*__attribute.json"
        for fpath in sorted(RESULTS.glob(pattern)):
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            system = data.get("system", sys_label)
            topic_id = data.get("topic_id", "")
            seed = data.get("seed", 0)
            for sent in data.get("per_sentence", []):
                if sent.get("n_cites", 0) == 0:
                    continue  # 跳过 bare assertion
                claims.append({
                    "system": system,
                    "topic_id": topic_id,
                    "seed": seed,
                    "sent_id": sent["sent_id"],
                    "section": sent.get("section_title", ""),
                    "text": sent["text"],
                    "n_cites": sent["n_cites"],
                    "llm_verdict": sent["judge"]["verdict"],
                    "llm_score": sent["judge"]["score"],
                    "llm_rationale": sent["judge"]["rationale"],
                    "source_file": str(fpath.name),
                })
    return claims


def stratified_sample(claims, n_per_system):
    """分层抽样：每系统 n 条，verdict 分布按比例"""
    random.seed(RANDOM_SEED)
    sampled = []

    for sys in SYSTEMS:
        pool = [c for c in claims if c["system"] == sys]
        # 按 verdict 分组
        by_verdict = defaultdict(list)
        for c in pool:
            by_verdict[c["llm_verdict"]].append(c)

        # 按比例分配名额
        total = len(pool)
        allocation = {}
        remaining = n_per_system
        for v in sorted(by_verdict.keys()):
            count = len(by_verdict[v])
            alloc = max(1, round(count / total * n_per_system))
            allocation[v] = min(alloc, count, remaining)
            remaining -= allocation[v]
            if remaining <= 0:
                break

        # 如果还有余额，补到最大组
        if remaining > 0:
            largest = max(by_verdict.keys(), key=lambda v: len(by_verdict[v]))
            allocation[largest] = allocation.get(largest, 0) + remaining

        # 抽样
        for v, n in allocation.items():
            picked = random.sample(by_verdict[v], min(n, len(by_verdict[v])))
            sampled.extend(picked)

    return sampled


def main():
    claims = collect_all_claims()
    print(f"收集到 {len(claims)} 条 cited claims（跨 3 系统 × 10 topic × 3 seed）")

    for sys in SYSTEMS:
        n = sum(1 for c in claims if c["system"] == sys)
        print(f"  {sys}: {n} 条")

    sample = stratified_sample(claims, SAMPLE_PER_SYSTEM)
    print(f"\n抽样 {len(sample)} 条（每系统 {SAMPLE_PER_SYSTEM} 条）")

    # 添加人工标注字段
    for i, item in enumerate(sample):
        item["sample_id"] = i + 1
        item["human_verdict"] = ""  # 待标注: SUPPORTED / PARTIAL / UNSUPPORTED
        item["human_note"] = ""     # 可选：标注备注

    # 保存
    out_path = RESULTS / "kappa_sample.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")

    # 生成标注指南
    guide = """# Cohen's κ 校准标注指南

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
"""
    guide_path = RESULTS / "kappa_sample_guide.md"
    guide_path.write_text(guide, encoding="utf-8")
    print(f"wrote {guide_path}")

    # 打印 verdict 分布
    from collections import Counter
    dist = Counter(c["llm_verdict"] for c in sample)
    print(f"\n抽样 verdict 分布: {dict(dist)}")


if __name__ == "__main__":
    main()
