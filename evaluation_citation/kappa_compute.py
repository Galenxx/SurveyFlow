"""
Cohen's κ 计算：对比人工标注 vs LLM judge 的归因判断一致性。
用法：完成 kappa_sample.json 中 human_verdict 标注后运行
  python -m evaluation_citation.kappa_compute
"""
import json
from pathlib import Path
from collections import Counter

SAMPLE_PATH = Path(__file__).parent / "results" / "kappa_sample.json"
OUTPUT_PATH = Path(__file__).parent / "results" / "kappa_result.json"

# Ordinal 编码：SUPPORTED=2, PARTIAL=1, UNSUPPORTED=0
LABEL_MAP = {"SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 0}


def _linear_weights(n_cat=3):
    """线性加权矩阵 (Cohen's weighted κ)"""
    import numpy as np
    w = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            w[i, j] = 1 - abs(i - j) / (n_cat - 1)
    return w


def cohens_weighted_kappa(y_true, y_pred, n_cat=3):
    """手动计算 linear-weighted Cohen's κ（不依赖 sklearn）"""
    import numpy as np
    # 混淆矩阵
    cm = np.zeros((n_cat, n_cat), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    n = cm.sum()
    if n == 0:
        return 0.0, cm

    # 期望矩阵
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / n

    # 加权
    w = _linear_weights(n_cat)
    po = (w * cm).sum() / n
    pe = (w * expected).sum() / n

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)
    return kappa, cm


def main():
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # 检查标注完成度
    annotated = [s for s in samples if s.get("human_verdict")]
    n_total = len(samples)
    n_done = len(annotated)
    print(f"已标注: {n_done}/{n_total}")

    if n_done < 20:
        print("标注不足 20 条，κ 估计不可靠。请继续标注。")
        return

    # 编码
    y_llm = []
    y_human = []
    for s in annotated:
        llm_v = s["llm_verdict"].upper()
        hum_v = s["human_verdict"].upper().strip()
        if hum_v not in LABEL_MAP:
            print(f"  [WARN] 无法识别 human_verdict='{hum_v}'，跳过 (sample_id={s['sample_id']})")
            continue
        y_llm.append(LABEL_MAP.get(llm_v, 0))
        y_human.append(LABEL_MAP[hum_v])

    if len(y_llm) < 10:
        print("有效标注对不足 10，无法计算。")
        return

    import numpy as np
    kappa, cm = cohens_weighted_kappa(y_human, y_llm)

    # 按系统分组
    per_system = {}
    system_groups = {}
    for s in annotated:
        hum_v = s.get("human_verdict", "").upper().strip()
        if hum_v not in LABEL_MAP:
            continue
        sys = s["system"]
        if sys not in system_groups:
            system_groups[sys] = {"human": [], "llm": []}
        system_groups[sys]["human"].append(LABEL_MAP[hum_v])
        system_groups[sys]["llm"].append(LABEL_MAP.get(s["llm_verdict"].upper(), 0))

    for sys, grp in system_groups.items():
        if len(grp["human"]) >= 5:
            k, _ = cohens_weighted_kappa(grp["human"], grp["llm"])
            per_system[sys] = {"kappa": round(k, 4), "n": len(grp["human"])}
        else:
            per_system[sys] = {"kappa": None, "n": len(grp["human"])}

    # 简单 agreement rate
    exact_agree = sum(1 for h, l in zip(y_human, y_llm) if h == l)
    agreement_rate = exact_agree / len(y_human)

    # 输出
    result = {
        "n_annotated": n_done,
        "n_valid_pairs": len(y_llm),
        "weighted_kappa": round(kappa, 4),
        "exact_agreement_rate": round(agreement_rate, 4),
        "confusion_matrix": {
            "rows": "human (UNSUPPORTED=0, PARTIAL=1, SUPPORTED=2)",
            "cols": "llm (UNSUPPORTED=0, PARTIAL=1, SUPPORTED=2)",
            "matrix": cm.tolist()
        },
        "per_system_kappa": per_system,
        "interpretation": _interpret(kappa)
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nweighted κ = {kappa:.4f} ({_interpret(kappa)})")
    print(f"exact agreement = {agreement_rate:.1%}")
    print(f"\n混淆矩阵 (行=人工, 列=LLM):")
    labels = ["UNSUP", "PART", "SUP"]
    print(f"{'':>8} {labels[0]:>6} {labels[1]:>6} {labels[2]:>6}")
    for i, row_label in enumerate(labels):
        print(f"{row_label:>8} {int(cm[i,0]):>6} {int(cm[i,1]):>6} {int(cm[i,2]):>6}")

    print(f"\n逐系统 κ:")
    for sys, info in per_system.items():
        k_str = f"{info['kappa']:.4f}" if info['kappa'] is not None else "N/A"
        print(f"  {sys}: κ={k_str} (n={info['n']})")

    print(f"\nwrote {OUTPUT_PATH}")


def _interpret(kappa):
    """Landis & Koch 解释"""
    if kappa < 0:
        return "poor (< 0)"
    elif kappa < 0.21:
        return "slight (0.00-0.20)"
    elif kappa < 0.41:
        return "fair (0.21-0.40)"
    elif kappa < 0.61:
        return "moderate (0.41-0.60)"
    elif kappa < 0.81:
        return "substantial (0.61-0.80)"
    else:
        return "almost perfect (0.81-1.00)"


if __name__ == "__main__":
    main()
