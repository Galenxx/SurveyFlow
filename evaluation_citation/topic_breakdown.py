"""
Per-topic breakdown 可视化 + LaTeX 表
从 summary_topic.json 生成：
  1. 按 topic 的三系统对比 LaTeX 表
  2. CSV 方便后续画图
"""

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

def main():
    with open(RESULTS / "summary_topic.json", encoding="utf-8") as f:
        rows = json.load(f)

    # 按 topic 分组
    topics = sorted(set(r["topic_id"] for r in rows))
    systems = ["SURVEYFLOW", "NAIVE_RAG", "LLM_ONLY"]

    # 构建 lookup: (system, topic) -> row
    lookup = {(r["system"], r["topic_id"]): r for r in rows}

    # ===== 1. LaTeX 表：per-topic recall/precision/relevance =====
    metrics = [
        ("citation_recall", "Recall"),
        ("citation_precision", "Precision"),
        ("relevance_rate", "Relevance"),
        ("existence_resolved_rate", "Existence"),
        ("bare_assertion_rate", "Bare-Assert."),
    ]

    # 短标签映射
    topic_labels = {}
    for r in rows:
        tid = r["topic_id"]
        if tid not in topic_labels:
            # 截断到 25 字符
            label = r.get("topic_label", tid)
            if len(label) > 30:
                label = label[:27] + "..."
            topic_labels[tid] = label

    # 生成主 breakdown 表（recall + relevance，最关键的两个）
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Per-topic citation recall and relevance rate (\%). Bold = best per topic.}")
    lines.append(r"\label{tab:topic-breakdown}")
    lines.append(r"\begin{tabular}{l|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c|}{Citation Recall} & \multicolumn{3}{c}{Relevance Rate} \\")
    lines.append(r"Topic & SF & RAG & LM & SF & RAG & LM \\")
    lines.append(r"\midrule")

    for tid in topics:
        label = topic_labels[tid]
        vals_recall = []
        vals_rel = []
        for sys in systems:
            r = lookup.get((sys, tid))
            if r:
                vals_recall.append(r.get("citation_recall") or 0)
                vals_rel.append(r.get("relevance_rate") or 0)
            else:
                vals_recall.append(0)
                vals_rel.append(0)

        # 标粗最佳
        def fmt_best(vals):
            best = max(vals)
            cells = []
            for v in vals:
                s = f"{v*100:.1f}"
                if v == best and best > 0:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            return cells

        rc = fmt_best(vals_recall)
        rl = fmt_best(vals_rel)
        # bare_assertion_rate 越低越好，这里不用
        lines.append(f"{label} & {rc[0]} & {rc[1]} & {rc[2]} & {rl[0]} & {rl[1]} & {rl[2]} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = RESULTS / "table_topic_breakdown.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {tex_path}")

    # ===== 2. CSV 全指标 =====
    import csv
    csv_path = RESULTS / "topic_breakdown.csv"
    fieldnames = [
        "system", "topic_id", "topic_label",
        "citation_recall", "citation_precision",
        "bare_assertion_rate", "relevance_rate",
        "existence_resolved_rate", "n_references",
        "n_genuinely_fabricated",
        "seed_std_recall", "seed_std_precision",
    ]
    def sf(v):
        """None → 0.0"""
        return v if v is not None else 0.0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "system": r["system"],
                "topic_id": r["topic_id"],
                "topic_label": r.get("topic_label", ""),
                "citation_recall": f"{sf(r.get('citation_recall')):.4f}",
                "citation_precision": f"{sf(r.get('citation_precision')):.4f}",
                "bare_assertion_rate": f"{sf(r.get('bare_assertion_rate')):.4f}",
                "relevance_rate": f"{sf(r.get('relevance_rate')):.4f}",
                "existence_resolved_rate": f"{sf(r.get('existence_resolved_rate')):.4f}",
                "n_references": r.get("n_references", 0) or 0,
                "n_genuinely_fabricated": r.get("n_genuinely_fabricated", 0) or 0,
                "seed_std_recall": f"{sf((r.get('seed_var_recall') or {}).get('std')):.4f}",
                "seed_std_precision": f"{sf((r.get('seed_var_precision') or {}).get('std')):.4f}",
            })
    print(f"wrote {csv_path}")

    # ===== 3. 终端打印摘要 =====
    print("\n===== Per-Topic Breakdown (Recall / Relevance) =====")
    print(f"{'Topic':<28} {'SF_Rec':>7} {'RAG_Rec':>8} {'LM_Rec':>7} | {'SF_Rel':>7} {'RAG_Rel':>8} {'LM_Rel':>7}")
    print("-" * 88)
    for tid in topics:
        label = topic_labels[tid][:27]
        sfd = lookup.get(("SURVEYFLOW", tid), {})
        rag = lookup.get(("NAIVE_RAG", tid), {})
        lm = lookup.get(("LLM_ONLY", tid), {})
        print(f"{label:<28} {sf(sfd.get('citation_recall'))*100:>6.1f}% {sf(rag.get('citation_recall'))*100:>7.1f}% {sf(lm.get('citation_recall'))*100:>6.1f}% | "
              f"{sf(sfd.get('relevance_rate'))*100:>6.1f}% {sf(rag.get('relevance_rate'))*100:>7.1f}% {sf(lm.get('relevance_rate'))*100:>6.1f}%")


if __name__ == "__main__":
    main()
