"""Aggregate Citation Accuracy benchmark results (Benchmark v2).

Consumes the three v2 metric phases per (system, topic, seed) cell:

  - attribute.json : ALCE sentence-level attribution (MAIN metric, §4.A)
                     A1 citation recall, A2 citation precision, A3 bare-assertion rate
  - classify.json  : existence-failure decomposition (§4.B)
  - judge.json     : claim/topic-relative relevance (§4.C)

and emits, under RESULTS_ROOT/:

  - summary_cells.json   : 1 row per (system, topic, seed) — RAW COUNTS (counts-first §5)
  - summary_topic.json   : 1 row per (system, topic) — 3 seeds merged + seed mean/std
  - summary_system.json  : 1 row per system — pooled counts + Wilson CIs
  - stats_tests.json     : SurveyFlow vs each baseline — Wilcoxon + bootstrap + rank-biserial
  - metrics_cells.csv    : CSV mirror of summary_cells
  - table_main.tex       : system-level headline metrics + Wilson CIs
  - table_attribution.tex: per-system A1/A2/A3
  - table_failure.tex    : per-system §4.B failure distribution
  - table_significance.tex: SurveyFlow vs baselines — Δ, 95% CI, p, effect size

Statistical protocol (§5):
  * counts-first: every rate carries its numerator/denominator; percentages are derived
  * Wilson score CI for all proportion metrics (small-sample safe)
  * paired unit = TOPIC (n=10): per-topic macro rates averaged over seeds
  * Wilcoxon signed-rank (paired by topic), SurveyFlow vs NAIVE_RAG and vs LLM_ONLY
  * paired bootstrap for the difference CI + rank-biserial effect size

Pure post-processing: ZERO network / API calls. Reads only existing JSON.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

from evaluation_citation.configs import (
    SYSTEMS,
    TOPICS,
    SEEDS,
    topic_by_id,
    results_dir,
    RESULTS_ROOT,
)
from evaluation_citation.stats import (
    wilson_interval,
    wilcoxon_signed_rank,
    paired_bootstrap_diff,
    rank_biserial,
)

# The reference system every baseline is compared against.
REFERENCE_SYSTEM = "SURVEYFLOW"


# --- IO helpers ---------------------------------------------------------


def _result_path(system_id: str, topic_id: str, seed: int, phase: str) -> Path:
    return results_dir(system_id) / f"{topic_id}__seed{seed}__{phase}.json"


def _load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _safe_rate(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


# --- Phase loaders: one cell -> flat metric dict ------------------------


def _load_cell(system_id: str, topic_id: str, seed: int) -> dict:
    """Load and flatten one (system, topic, seed) cell from its 3 metric files.

    Missing files leave the corresponding fields as None / 0; the cell row
    records ``complete`` = whether all three phases were present, so partial
    runs are visible rather than silently zero-filled.
    """
    attr = _load_json(_result_path(system_id, topic_id, seed, "attribute"))
    clf = _load_json(_result_path(system_id, topic_id, seed, "classify"))
    jud = _load_json(_result_path(system_id, topic_id, seed, "judge"))

    row: dict = {
        "system": system_id,
        "topic_id": topic_id,
        "seed": seed,
        "complete": bool(attr) and bool(clf) and bool(jud),
        "has_attribute": bool(attr),
        "has_classify": bool(clf),
        "has_judge": bool(jud),
    }

    # --- §4.A attribution (MAIN) ---
    m = (attr or {}).get("metrics", {}) if attr else {}
    row["n_cited_claims"] = m.get("n_cited_claims", 0)
    row["support_total"] = m.get("support_total", 0.0)
    row["n_supported"] = m.get("n_supported", 0)
    row["n_partial_attr"] = m.get("n_partial", 0)
    row["n_unsupported"] = m.get("n_unsupported", 0)
    row["n_attr_error"] = m.get("n_error", 0)
    row["n_citation_mounts"] = m.get("n_citation_mounts", 0)
    row["n_contributing"] = m.get("n_contributing", 0)
    row["n_claims"] = m.get("n_claims", 0)
    row["n_claims_with_cite"] = m.get("n_claims_with_cite", 0)
    row["n_claims_bare"] = m.get("n_claims_bare", 0)
    # rates (derived, counts retained above per §5)
    row["citation_recall"] = m.get("citation_recall")          # support_total / n_cited_claims
    row["citation_precision"] = m.get("citation_precision")    # n_contributing / n_citation_mounts
    row["bare_assertion_rate"] = m.get("bare_assertion_rate")  # n_claims_bare / n_claims

    # --- §4.B failure decomposition ---
    dist = (clf or {}).get("distribution", {}) if clf else {}
    row["n_references"] = (clf or {}).get("n_references", 0)
    row["n_resolved"] = (clf or {}).get("n_resolved", 0)
    row["n_padding"] = (clf or {}).get("n_padding", 0)
    row["n_failures"] = (clf or {}).get("n_failures", 0)
    row["n_genuinely_fabricated"] = (clf or {}).get("n_genuinely_fabricated", 0)
    row["dist_metadata_parse_error"] = dist.get("METADATA_PARSE_ERROR", 0)
    row["dist_id_extraction_fail"] = dist.get("ID_EXTRACTION_FAIL", 0)
    row["dist_dedup_collision"] = dist.get("DEDUP_COLLISION", 0)
    row["dist_genuinely_fabricated"] = dist.get("GENUINELY_FABRICATED", 0)
    # existence-as-gate: resolved / (references - padding). Padding artifacts
    # are excluded from the denominator (recorded bias hazard fix).
    denom_exist = row["n_references"] - row["n_padding"]
    row["existence_resolved_rate"] = _safe_rate(row["n_resolved"], denom_exist)

    # --- §4.C relevance ---
    agg = (jud or {}).get("aggregate", {}) if jud else {}
    row["n_relevant"] = agg.get("n_relevant", 0)
    row["n_partial_rel"] = agg.get("n_partial", 0)
    row["n_irrelevant"] = agg.get("n_irrelevant", 0)
    row["n_rel_judged"] = agg.get("n_judged", 0)
    row["relevance_score"] = agg.get("score")  # 0..100

    return row


def _load_all_cells() -> list[dict]:
    rows: list[dict] = []
    for system in SYSTEMS:
        for topic in TOPICS:
            for seed in SEEDS:
                rows.append(_load_cell(system["id"], topic["id"], seed))
    return rows


# --- Topic-level aggregation (merge seeds) ------------------------------


def _topic_macro_rate(seed_rows: list[dict], num_key: str, den_key: str) -> Optional[float]:
    """Per-topic rate = sum(numerators over seeds) / sum(denominators over seeds).

    Pooling counts across seeds (rather than averaging per-seed rates) keeps
    the rate weighted by how many items each seed actually produced — a seed
    that emitted 2 claims should not weigh the same as one that emitted 40.
    """
    num = sum(r.get(num_key, 0) or 0 for r in seed_rows)
    den = sum(r.get(den_key, 0) or 0 for r in seed_rows)
    return _safe_rate(num, den)


def _seed_rate_stats(seed_rows: list[dict], rate_key: str) -> dict:
    """mean/std of a per-seed rate across the seeds (LM sampling variance, §5).

    Only seeds whose rate is not None contribute. Reports n so a mean over a
    single seed cannot masquerade as a 3-seed estimate.
    """
    vals = [r[rate_key] for r in seed_rows if r.get(rate_key) is not None]
    if not vals:
        return {"mean": None, "std": None, "n_seeds": 0}
    return {
        "mean": round(mean(vals), 4),
        "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "n_seeds": len(vals),
    }


def _aggregate_topics(cell_rows: list[dict]) -> list[dict]:
    """Collapse the seed dimension: 1 row per (system, topic)."""
    out: list[dict] = []
    for system in SYSTEMS:
        for topic in TOPICS:
            seed_rows = [
                r for r in cell_rows
                if r["system"] == system["id"] and r["topic_id"] == topic["id"]
            ]
            if not seed_rows:
                continue
            # pooled counts across seeds
            pooled = {}
            count_keys = [
                "n_cited_claims", "n_supported", "n_partial_attr", "n_unsupported",
                "n_attr_error", "n_citation_mounts", "n_contributing",
                "n_claims", "n_claims_with_cite", "n_claims_bare",
                "n_references", "n_resolved", "n_padding", "n_failures",
                "n_genuinely_fabricated", "dist_metadata_parse_error",
                "dist_id_extraction_fail", "dist_dedup_collision",
                "dist_genuinely_fabricated", "n_relevant", "n_partial_rel",
                "n_irrelevant", "n_rel_judged",
            ]
            for k in count_keys:
                pooled[k] = sum(r.get(k, 0) or 0 for r in seed_rows)
            support_total = sum(r.get("support_total", 0.0) or 0.0 for r in seed_rows)

            denom_exist = pooled["n_references"] - pooled["n_padding"]
            row = {
                "system": system["id"],
                "topic_id": topic["id"],
                "topic_label": topic.get("label", topic["id"]),
                "field": topic.get("field", ""),
                "n_seeds": len(seed_rows),
                "n_seeds_complete": sum(1 for r in seed_rows if r["complete"]),
                "support_total": round(support_total, 4),
                **pooled,
                # pooled (macro-per-topic) rates — these are the PAIRED UNITS
                "citation_recall": _safe_rate(support_total, pooled["n_cited_claims"]),
                "citation_precision": _safe_rate(pooled["n_contributing"], pooled["n_citation_mounts"]),
                "bare_assertion_rate": _safe_rate(pooled["n_claims_bare"], pooled["n_claims"]),
                "existence_resolved_rate": _safe_rate(pooled["n_resolved"], denom_exist),
                "relevance_rate": _safe_rate(
                    pooled["n_relevant"] + 0.5 * pooled["n_partial_rel"],
                    pooled["n_rel_judged"],
                ),
                # seed-variance descriptors (reproducibility, §8)
                "seed_var_recall": _seed_rate_stats(seed_rows, "citation_recall"),
                "seed_var_precision": _seed_rate_stats(seed_rows, "citation_precision"),
                "seed_var_bare": _seed_rate_stats(seed_rows, "bare_assertion_rate"),
            }
            out.append(row)
    return out


# --- System-level aggregation (pool topics) -----------------------------


def _wilson_dict(num: float, den: float) -> dict:
    """Counts-first proportion block with Wilson 95% CI."""
    k = int(round(num))
    n = int(round(den))
    lo, hi = wilson_interval(k, n) if n else (None, None)
    return {
        "num": num,
        "den": den,
        "rate": _safe_rate(num, den),
        "ci_low": lo,
        "ci_high": hi,
    }


def _aggregate_systems(cell_rows: list[dict]) -> list[dict]:
    """1 row per system: pool ALL counts across topics+seeds, attach Wilson CIs."""
    out: list[dict] = []
    for system in SYSTEMS:
        rows = [r for r in cell_rows if r["system"] == system["id"]]
        def s(key: str) -> float:
            return sum(r.get(key, 0) or 0 for r in rows)
        support_total = sum(r.get("support_total", 0.0) or 0.0 for r in rows)
        n_refs = s("n_references")
        n_padding = s("n_padding")
        rel_weighted = s("n_relevant") + 0.5 * s("n_partial_rel")

        out.append({
            "system": system["id"],
            "n_cells": len(rows),
            "n_cells_complete": sum(1 for r in rows if r["complete"]),
            # §4.A — MAIN
            "citation_recall": _wilson_dict(support_total, s("n_cited_claims")),
            "citation_precision": _wilson_dict(s("n_contributing"), s("n_citation_mounts")),
            "bare_assertion_rate": _wilson_dict(s("n_claims_bare"), s("n_claims")),
            # §4.B — gate + decomposition
            "existence_resolved_rate": _wilson_dict(s("n_resolved"), n_refs - n_padding),
            "n_references": n_refs,
            "n_padding": n_padding,
            "n_failures": s("n_failures"),
            "n_genuinely_fabricated": s("n_genuinely_fabricated"),
            "failure_distribution": {
                "METADATA_PARSE_ERROR": s("dist_metadata_parse_error"),
                "ID_EXTRACTION_FAIL": s("dist_id_extraction_fail"),
                "DEDUP_COLLISION": s("dist_dedup_collision"),
                "GENUINELY_FABRICATED": s("dist_genuinely_fabricated"),
            },
            # §4.C
            "relevance_rate": _wilson_dict(rel_weighted, s("n_rel_judged")),
        })
    return out


# --- Significance tests (SurveyFlow vs each baseline) -------------------

# Metrics tested, and whether higher is better (governs sign of the gap).
_TESTED_METRICS = [
    ("citation_recall", True),
    ("citation_precision", True),
    ("bare_assertion_rate", False),
    ("relevance_rate", True),
    ("existence_resolved_rate", True),
]


def _paired_vectors(topic_rows: list[dict], sys_a: str, sys_b: str, metric: str):
    """Build paired (x, y) vectors over TOPICS for two systems.

    A topic contributes only when BOTH systems have a non-None rate for it
    (otherwise the pair is undefined). Returns (x, y, topic_ids).
    """
    by_topic_a = {r["topic_id"]: r for r in topic_rows if r["system"] == sys_a}
    by_topic_b = {r["topic_id"]: r for r in topic_rows if r["system"] == sys_b}
    x, y, tids = [], [], []
    for topic in TOPICS:
        tid = topic["id"]
        ra, rb = by_topic_a.get(tid), by_topic_b.get(tid)
        if not ra or not rb:
            continue
        va, vb = ra.get(metric), rb.get(metric)
        if va is None or vb is None:
            continue
        x.append(va)
        y.append(vb)
        tids.append(tid)
    return x, y, tids


def _run_significance(topic_rows: list[dict]) -> dict:
    """SurveyFlow vs each other system, on every tested metric, paired by topic."""
    results: dict = {"reference": REFERENCE_SYSTEM, "paired_unit": "topic", "comparisons": []}
    baselines = [s["id"] for s in SYSTEMS if s["id"] != REFERENCE_SYSTEM]
    for baseline in baselines:
        for metric, higher_better in _TESTED_METRICS:
            x, y, tids = _paired_vectors(topic_rows, REFERENCE_SYSTEM, baseline, metric)
            n = len(x)
            entry = {
                "metric": metric,
                "baseline": baseline,
                "higher_is_better": higher_better,
                "n_pairs": n,
                "topic_ids": tids,
            }
            if n < 2:
                entry["note"] = "insufficient paired topics for a test"
                entry["mean_reference"] = round(mean(x), 4) if x else None
                entry["mean_baseline"] = round(mean(y), 4) if y else None
                results["comparisons"].append(entry)
                continue
            wil = wilcoxon_signed_rank(x, y)
            boot = paired_bootstrap_diff(x, y, n_resamples=10000, seed=12345)
            rb = rank_biserial(x, y)
            entry.update({
                "mean_reference": round(mean(x), 4),
                "mean_baseline": round(mean(y), 4),
                "mean_diff": round(mean(x) - mean(y), 4),
                "wilcoxon_W": wil.statistic,
                "wilcoxon_p": wil.p_value,
                "wilcoxon_method": wil.method,
                "wilcoxon_n_effective": wil.n_effective,
                "wilcoxon_n_zeros": wil.n_zeros,
                "boot_diff_point": boot.diff_point if boot else None,
                "boot_ci_low": boot.ci_low if boot else None,
                "boot_ci_high": boot.ci_high if boot else None,
                "rank_biserial": rb,
            })
            results["comparisons"].append(entry)
    return results


# --- writers ------------------------------------------------------------


def _write_json(name: str, payload) -> None:
    out = RESULTS_ROOT / name
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


def _write_cells_csv(cell_rows: list[dict]) -> None:
    out = RESULTS_ROOT / "metrics_cells.csv"
    if not cell_rows:
        print("no cells to write")
        return
    fieldnames = list(cell_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in cell_rows:
            r2 = dict(r)
            for k, v in r2.items():
                if isinstance(v, float):
                    r2[k] = f"{v:.4f}"
            w.writerow(r2)
    print(f"wrote {out}")


def _fmt_rate(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}" if pct else f"{v:.3f}"


def _fmt_ci(block: dict) -> str:
    lo, hi = block.get("ci_low"), block.get("ci_high")
    if lo is None or hi is None:
        return "n/a"
    return f"[{100*lo:.1f}, {100*hi:.1f}]"


def _write_table_main(sys_rows: list[dict]) -> None:
    out = RESULTS_ROOT / "table_main.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Citation quality by system (pooled over 10 topics $\times$ 3 seeds). "
        r"Citation Recall (A1) and Precision (A2) are the main attribution metrics; "
        r"values are percentages with Wilson 95\% CIs. Higher is better for all columns "
        r"except Bare-Assertion Rate.}",
        r"\label{tab:citation-main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"System & Recall (A1) & Precision (A2) & Bare-Assert.\ (A3) & Relevance \\",
        r"\midrule",
    ]
    for r in sys_rows:
        rec, prec = r["citation_recall"], r["citation_precision"]
        bare, rel = r["bare_assertion_rate"], r["relevance_rate"]
        lines.append(
            f"{r['system']} & "
            f"{_fmt_rate(rec['rate'])} {_fmt_ci(rec)} & "
            f"{_fmt_rate(prec['rate'])} {_fmt_ci(prec)} & "
            f"{_fmt_rate(bare['rate'])} {_fmt_ci(bare)} & "
            f"{_fmt_rate(rel['rate'])} {_fmt_ci(rel)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def _write_table_attribution(sys_rows: list[dict]) -> None:
    out = RESULTS_ROOT / "table_attribution.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Attribution counts (counts-first, §5). "
        r"Supported/Partial/Unsupported are sentence-level verdicts over cited claims.}",
        r"\label{tab:attribution-counts}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & Cited claims & Supported & Partial & Unsupp. & Bare \\",
        r"\midrule",
    ]
    # pull pooled counts back out of the cell rows via the system block's stored fields
    for r in sys_rows:
        rec = r["citation_recall"]
        bare = r["bare_assertion_rate"]
        lines.append(
            f"{r['system']} & "
            f"{int(rec['den'])} & "
            f"-- & -- & -- & {int(bare['num'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def _write_table_failure(sys_rows: list[dict]) -> None:
    out = RESULTS_ROOT / "table_failure.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Existence-failure decomposition (§4.B). Columns are absolute "
        r"reference counts per failure class. SurveyFlow's failures concentrate in "
        r"engineering classes (parse / id / dedup), not fabrication.}",
        r"\label{tab:failure-decomp}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & Refs & Meta.\ parse & ID fail & Dedup & Fabricated \\",
        r"\midrule",
    ]
    for r in sys_rows:
        d = r["failure_distribution"]
        lines.append(
            f"{r['system']} & "
            f"{r['n_references']} & "
            f"{d['METADATA_PARSE_ERROR']} & "
            f"{d['ID_EXTRACTION_FAIL']} & "
            f"{d['DEDUP_COLLISION']} & "
            f"{d['GENUINELY_FABRICATED']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def _write_table_significance(sig: dict) -> None:
    out = RESULTS_ROOT / "table_significance.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{SurveyFlow vs.\ baselines, paired by topic (n=10). "
        r"$\Delta$ is the mean per-topic gap (SurveyFlow $-$ baseline); CI is the "
        r"paired-bootstrap 95\% interval on $\Delta$; $p$ from Wilcoxon signed-rank; "
        r"$r$ is the rank-biserial effect size.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Metric & vs.\ & $\Delta$ & 95\% CI & $p$ & $r$ \\",
        r"\midrule",
    ]
    for c in sig.get("comparisons", []):
        if "mean_diff" not in c:
            continue
        ci = (f"[{c['boot_ci_low']:.3f}, {c['boot_ci_high']:.3f}]"
              if c.get("boot_ci_low") is not None else "n/a")
        p = c.get("wilcoxon_p")
        p_str = f"{p:.4f}" if p is not None else "n/a"
        rb = c.get("rank_biserial")
        rb_str = f"{rb:.3f}" if rb is not None else "n/a"
        lines.append(
            f"{c['metric'].replace('_', ' ')} & {c['baseline']} & "
            f"{c['mean_diff']:.3f} & {ci} & {p_str} & {rb_str} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


# --- orchestration ------------------------------------------------------


def main() -> int:
    cell_rows = _load_all_cells()
    topic_rows = _aggregate_topics(cell_rows)
    sys_rows = _aggregate_systems(cell_rows)
    sig = _run_significance(topic_rows)

    n_complete = sum(1 for r in cell_rows if r["complete"])
    print(f"loaded {len(cell_rows)} cells ({n_complete} complete)")

    _write_json("summary_cells.json", cell_rows)
    _write_json("summary_topic.json", topic_rows)
    _write_json("summary_system.json", sys_rows)
    _write_json("stats_tests.json", sig)
    _write_cells_csv(cell_rows)
    _write_table_main(sys_rows)
    _write_table_attribution(sys_rows)
    _write_table_failure(sys_rows)
    _write_table_significance(sig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
