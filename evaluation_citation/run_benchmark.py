"""Citation Accuracy benchmark runner.

Five phases, each independently resumable:

  1. generate  - produce one survey per (system, topic) cell
  2. parse     - extract Reference: lines from each survey.md
  3. verify    - resolve each reference via arXiv + S2; classify
                 FOUND / MISMATCH / MISSING_ID / AMBIGUOUS / NOT_FOUND
  4. judge     - LLM-as-judge topic relevance per reference
  5. aggregate - merge into per-cell metrics + LaTeX table

Run the full pipeline:

    cd survey_agent
    python -m evaluation_citation.run_benchmark

Run a single phase:

    python -m evaluation_citation.run_benchmark --phase verify
    python -m evaluation_citation.run_benchmark --phase judge
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Load the project's .env BEFORE importing core.llm_client. The
# evaluation_citation package's __init__ also tries to load it, but
# importing it directly here makes the order explicit and works even
# if the package is imported by a parent that has not pulled in
# __init__.py yet.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

from evaluation_citation.configs import (
    CITATIONS_PER_SURVEY,
    CITATION_COUNT_WARN_FRACTION,
    EXISTENCE_SIM_THRESHOLD,
    IDENTIFIER_SIM_THRESHOLD,
    SEEDS,
    TOPICS,
    SYSTEMS,
    results_dir,
    topic_by_id,
)
from evaluation_citation.citation_parser import parse_survey_md
from evaluation_citation.attribution_parser import parse_attribution
from evaluation_citation.verifiers import arxiv_verify, semantic_scholar_verify, identifier_check
from evaluation_citation.judges import faithfulness_judge
from evaluation_citation.judges import attribution_judge
from evaluation_citation.failure_classifier import classify_verify_payload


# --- per-(system, topic, seed) result file naming -----------------------
#
# Every phase writes ``<results>/<system>/<topic>__seed<k>__<phase>.json``.
# The seed dimension lets the benchmark run each cell multiple times to
# report variance and run paired significance tests; the per-system
# subdirectory (via ``results_dir``) scales to any number of systems
# without the old hard-coded SURVEYFLOW/LLM_ONLY split.


def _result_path(system_id: str, topic_id: str, seed: int, phase: str) -> Path:
    return results_dir(system_id) / f"{topic_id}__seed{seed}__{phase}.json"


# --- Phase 1: generate --------------------------------------------------


def _phase_generate(system_id: str, topic_id: str, seed: int) -> dict:
    """Run the configured generator for one (system, topic, seed) cell.

    Generation guards (Task #5):
      * The generated survey.md is persisted by each runner to a FIXED,
        benchmark-owned location (never the random survey_id dir), so the
        artifact can never be silently lost.
      * The realized reference count and corpus stats are recorded in the
        generate.json. When the realized count falls below
        ``CITATION_COUNT_WARN_FRACTION * requested`` we attach a loud
        ``warning`` field (we do NOT abort — a thin result is itself a
        datum, but it must never pass silently as it used to).
    """
    from evaluation_citation.generators.surveyflow_runner import run_surveyflow_detailed
    from evaluation_citation.generators.naive_rag_runner import run_naive_rag_detailed
    from evaluation_citation.generators.llm_only_runner import run_llm_only_detailed

    topic = topic_by_id(topic_id)
    if system_id == "SURVEYFLOW":
        # Ensure the topic's corpus is ingested once and cached, then run only
        # the writing half off that shared corpus (skips query/download/splitter).
        from evaluation_citation.corpus_cache import ensure_corpus
        ensure_corpus(topic_id, paper_count=CITATIONS_PER_SURVEY)
        info = run_surveyflow_detailed(
            topic["topic"], paper_count=CITATIONS_PER_SURVEY, topic_id=topic_id,
            seed=seed,
        )
    elif system_id == "NAIVE_RAG":
        # Reuse the same per-topic corpus SURVEYFLOW uses: ingest once, then
        # write off the shared chunks so both systems compete on identical
        # evidence (cleaner ablation) without re-downloading/re-embedding.
        from evaluation_citation.corpus_cache import (
            ensure_corpus, load_state_from_corpus,
        )
        ensure_corpus(topic_id, paper_count=CITATIONS_PER_SURVEY)
        corpus_state = load_state_from_corpus(
            topic_id, paper_count=CITATIONS_PER_SURVEY
        )
        info = run_naive_rag_detailed(
            topic["topic"], paper_count=CITATIONS_PER_SURVEY, seed=seed,
            corpus_state=corpus_state,
        )
    elif system_id == "LLM_ONLY":
        info = run_llm_only_detailed(topic["topic"], paper_count=CITATIONS_PER_SURVEY, seed=seed)
    else:
        raise ValueError(f"unknown system: {system_id}")

    n_realized = info.get("n_references_realized", 0)
    out = {
        "system": system_id,
        "topic_id": topic_id,
        "seed": seed,
        "md_path": info.get("md_path", ""),
        "survey_id": info.get("survey_id", ""),
        "n_references_requested": CITATIONS_PER_SURVEY,
        "n_references_realized": n_realized,
        "corpus_stats": info.get("corpus_stats", {}),
    }
    warn_floor = CITATION_COUNT_WARN_FRACTION * CITATIONS_PER_SURVEY
    if n_realized < warn_floor:
        out["warning"] = (
            f"LOW CITATION COUNT: realized {n_realized} references "
            f"< {warn_floor:.0f} (= {CITATION_COUNT_WARN_FRACTION:.0%} of "
            f"requested {CITATIONS_PER_SURVEY}). Inspect corpus_stats to see "
            f"whether the corpus was thin or the writer dropped sections."
        )
        print(f"    [WARN] {system_id}/{topic_id}/seed{seed}: {out['warning']}")

    out_path = _result_path(system_id, topic_id, seed, "generate")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# --- Phase 2: parse -----------------------------------------------------


def _phase_parse(system_id: str, topic_id: str, seed: int) -> dict:
    topic = topic_by_id(topic_id)
    gen = _result_path(system_id, topic_id, seed, "generate")
    if not gen.exists():
        raise FileNotFoundError(f"missing generate result: {gen}")
    gen_data = json.loads(gen.read_text(encoding="utf-8"))
    md_path = gen_data["md_path"]
    parsed = parse_survey_md(md_path, topic_hint=topic["topic"])
    out = _result_path(system_id, topic_id, seed, "parse")
    out.write_text(json.dumps(parsed.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return parsed.to_dict()


# --- Phase 3: verify ----------------------------------------------------


# Title-similarity floor for the arXiv abstract back-fill. The back-fill only
# wants to recover the abstract of a paper we have ALREADY resolved, from a
# near-duplicate arXiv record whose title the survey wrote imprecisely (e.g.
# survey "PET-SQL: ... Two-stage Framework" vs arXiv "PET-SQL: ... Two-Round
# Refinement"). Calibrated empirically on that real case: the CORRECT arXiv
# record scored title_sim=0.805 while the nearest WRONG candidate scored only
# 0.357 — a wide gap. 0.78 sits comfortably between them, recovering genuine
# near-duplicates while staying far above any wrong-paper match. Combined with
# the non-empty-abstract requirement this makes evidence poisoning very
# unlikely; every back-fill is also recorded (abstract_backfilled) for audit.
_ABSTRACT_BACKFILL_SIM = 0.78


def _backfill_abstract_from_arxiv(best_hit: dict) -> bool:
    """If ``best_hit`` resolved to a real paper but carries no abstract, try
    to recover one from arXiv via a FUZZY title search, gated by a strict
    title-similarity floor.

    Mutates ``best_hit`` in place (sets ``abstract`` and records the source)
    and returns True iff an abstract was back-filled. Makes at most one extra
    arXiv call and only when the abstract is genuinely missing, so it is cheap
    and deterministic. Never raises — a failed back-fill just leaves the
    abstract empty (the sentence stays NOT_VERIFIABLE downstream).
    """
    if not best_hit:
        return False
    if (best_hit.get("abstract") or "").strip():
        return False  # already have an abstract; nothing to do
    title = (best_hit.get("title") or "").strip()
    if not title:
        return False

    try:
        # Relaxed search: search_by_title wraps the title in ti:"..." (exact
        # phrase) — exactly what fails when the survey recorded the title
        # slightly wrong (the very case we are recovering). search_by_title_
        # relaxed keys on the title's distinctive tokens (acronyms, domain
        # terms) with relevance ranking, locating the real record even when
        # the surrounding wording differs. The similarity floor below still
        # guards against grabbing a different paper.
        hits = arxiv_verify.search_by_title_relaxed(title, max_results=5)
    except Exception:
        return False

    best_sim = 0.0
    best_abs = ""
    best_src_id = None
    for h in hits or []:
        abs_text = (h.get("abstract") or "").strip()
        if not abs_text:
            continue
        sim = identifier_check.title_similarity(title, h.get("title", ""))
        if sim > best_sim:
            best_sim = sim
            best_abs = abs_text
            best_src_id = h.get("arxiv_id")

    if best_abs and best_sim >= _ABSTRACT_BACKFILL_SIM:
        best_hit["abstract"] = best_abs
        best_hit["abstract_backfilled"] = {
            "source": "arxiv:title_search",
            "title_sim": round(best_sim, 4),
            "arxiv_id": best_src_id,
        }
        return True
    return False


def _verify_one_reference(topic: str, ref: dict) -> dict:
    """Resolve one reference and decide on existence + identifier."""
    # 1. arXiv first
    arxiv_result = arxiv_verify.resolve_reference(
        arxiv_id=ref.get("arxiv_id"),
        title=ref.get("title"),
    )
    candidates = list(arxiv_result["hits"])
    method = arxiv_result["method"]
    if not arxiv_result["found"]:
        # 2. Semantic Scholar as fallback
        s2_result = semantic_scholar_verify.resolve_reference(
            arxiv_id=ref.get("arxiv_id"),
            doi=ref.get("doi"),
            title=ref.get("title"),
        )
        if s2_result["found"]:
            candidates = s2_result["hits"]
            method = "s2:" + s2_result["method"]

    identification = identifier_check.identify_reference(
        survey_title=ref.get("title"),
        survey_authors=ref.get("authors"),
        survey_year=ref.get("year"),
        survey_arxiv_id=ref.get("arxiv_id"),
        survey_doi=ref.get("doi"),
        candidates=candidates,
        id_sim_threshold=IDENTIFIER_SIM_THRESHOLD,
        title_sim_threshold=EXISTENCE_SIM_THRESHOLD,
    )

    # Abstract backfill: the reference may have resolved to a real paper
    # (best_hit set) whose record happens to carry no abstract — typically
    # an S2 title-search hit on a record S2 stores without an abstract and
    # without an ArXiv id. The cited paper usually DOES have a full abstract
    # on arXiv; we recover it via a relaxed arXiv title search guarded by a
    # high similarity threshold so we never graft a different paper's text.
    _backfill_abstract_from_arxiv(identification.get("best_hit"))

    return {
        "ref_id": ref.get("ref_id"),
        "raw": ref.get("raw"),
        "parsed": {
            "title": ref.get("title"),
            "authors": ref.get("authors"),
            "year": ref.get("year"),
            "arxiv_id": ref.get("arxiv_id"),
            "doi": ref.get("doi"),
        },
        "existence": "FOUND" if candidates else "NOT_FOUND",
        "resolve_method": method,
        "candidates": [
            {
                "title": c.get("title"),
                "arxiv_id": c.get("arxiv_id"),
                "doi": c.get("doi"),
                "year": c.get("year"),
                "authors": c.get("authors"),
            }
            for c in candidates
        ],
        "identification": identification,
    }


def _phase_verify(system_id: str, topic_id: str, seed: int) -> dict:
    topic = topic_by_id(topic_id)
    parsed_path = _result_path(system_id, topic_id, seed, "parse")
    if not parsed_path.exists():
        raise FileNotFoundError(f"missing parse result: {parsed_path}")
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    refs = parsed.get("references", [])
    results = []
    for ref in refs:
        results.append(_verify_one_reference(topic["topic"], ref))
        # be polite
        time.sleep(0.3)
    out = _result_path(system_id, topic_id, seed, "verify")
    out.write_text(json.dumps({
        "system": system_id,
        "topic_id": topic_id,
        "seed": seed,
        "topic": topic["topic"],
        "n_references": len(refs),
        "per_reference": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"n_references": len(refs), "n_results": len(results)}


# --- Phase 3b: attribute (ALCE sentence-level attribution — main metric) ---


def _evidence_map_from_verify(verify_data: dict) -> dict:
    """Build {ref_id -> {title, abstract}} from a verify.json payload.

    The evidence E(c) the attribution judge needs (cited paper title +
    abstract) was already fetched during the verify phase and stored on each
    reference's ``identification.best_hit``. We reuse it verbatim so the
    attribute phase makes NO new network calls — it is deterministic given
    the verify output and trivially re-runnable.
    """
    evidence: dict[int, dict] = {}
    for r in verify_data.get("per_reference", []):
        ref_id = r.get("ref_id")
        if ref_id is None:
            continue
        ident = r.get("identification") or {}
        best = ident.get("best_hit") or {}
        # Fall back to the survey-side parsed title when the reference did
        # not resolve to a real hit (best is empty) so the judge still sees
        # *something*; an empty abstract is handled gracefully downstream.
        parsed = r.get("parsed") or {}
        abstract = (best.get("abstract") or "").strip()
        evidence[ref_id] = {
            "title": best.get("title") or parsed.get("title") or "",
            "abstract": abstract,
            "resolved": bool(best),
            # An attribution judgment is only meaningful when we actually
            # hold the cited paper's evidence text. ``best_hit`` sometimes
            # resolves a real paper (title matches) but carries no abstract
            # (arXiv/S2 omitted it, or the MISSING_ID path never fetched
            # one). Judging such a sentence against an EMPTY abstract forces
            # a spurious UNSUPPORTED that has nothing to do with the writer:
            # it is NOT-VERIFIABLE, not a genuine attribution failure. We
            # flag it here so the attribute phase can exclude those sentences
            # from the recall/precision denominator (ALCE assumes evidence is
            # visible) and report them separately.
            "verifiable": bool(abstract),
        }
    return evidence


def _phase_attribute(system_id: str, topic_id: str, seed: int) -> dict:
    """ALCE-style sentence-level attribution (Task #8, the v2 main metric).

    Pipeline for one (system, topic, seed) cell:
      1. parse the survey body into (claim sentence -> in-text citations),
         each citation resolved to a reference id (attribution_parser).
      2. pull each cited reference's evidence (title + abstract) from the
         already-completed verify phase — no new network calls.
      3. for every claim sentence carrying >=1 resolved citation, ask the
         attribution judge whether the cited evidence SUPPORTS the sentence.
      4. aggregate into Citation Recall / Precision (A1/A2) and carry the
         parser's BareAssertionRate (A3) for one-stop reporting.

    Depends on BOTH generate.json (for md_path) and verify.json (for
    evidence abstracts), so it runs after verify and before judge.
    """
    topic = topic_by_id(topic_id)

    gen_path = _result_path(system_id, topic_id, seed, "generate")
    if not gen_path.exists():
        raise FileNotFoundError(f"missing generate result: {gen_path}")
    md_path = json.loads(gen_path.read_text(encoding="utf-8")).get("md_path", "")

    verify_path = _result_path(system_id, topic_id, seed, "verify")
    if not verify_path.exists():
        raise FileNotFoundError(f"missing verify result: {verify_path}")
    verify_data = json.loads(verify_path.read_text(encoding="utf-8"))
    evidence_by_ref = _evidence_map_from_verify(verify_data)

    doc = parse_attribution(md_path, topic_hint=topic["topic"])

    sentence_judgments: list[dict] = []
    per_sentence_out: list[dict] = []
    n_not_verifiable = 0
    for s in doc.sentences:
        if not s.is_candidate_claim:
            continue
        # Only judge sentences whose citations resolved to a reference we
        # have evidence for; unresolved in-text cites cannot be attribution-
        # checked (they are captured separately by the verify phase).
        #
        # Crucially we also require the resolved reference to carry a non-
        # empty abstract: a citation that resolved to a real paper but whose
        # abstract we never obtained is NOT-VERIFIABLE, not unsupported.
        # Feeding its empty abstract to the judge would manufacture a
        # spurious UNSUPPORTED that penalizes the system for OUR missing
        # evidence, systematically deflating recall/precision for every
        # system. Such citations are dropped from the judgeable set; if a
        # claim has NO verifiable citation left, it is counted separately as
        # ``not_verifiable`` and excluded from the attribution denominator
        # (consistent with ALCE, which assumes the cited evidence is visible).
        labels_to_evidence: dict[str, dict] = {}
        label_to_ref: dict[str, int] = {}
        n_cites_resolved = 0
        for i, c in enumerate(s.cites):
            if c.resolved_ref_id is None:
                continue
            ev = evidence_by_ref.get(c.resolved_ref_id)
            if not ev:
                continue
            n_cites_resolved += 1
            if not ev.get("verifiable"):
                # resolved to a real paper but no abstract to judge against
                continue
            label = f"c{i}"
            labels_to_evidence[label] = {"title": ev["title"], "abstract": ev["abstract"]}
            label_to_ref[label] = c.resolved_ref_id

        if not labels_to_evidence:
            # No judgeable citation on this claim. Distinguish two cases for
            # honest reporting:
            #   * the claim had >=1 citation that resolved to a real paper
            #     but we lack its abstract  -> NOT_VERIFIABLE (excluded from
            #     the attribution denominator, reported separately).
            #   * the claim's citations were all unresolved / had no evidence
            #     -> nothing to do here (captured by verify/classify).
            if n_cites_resolved > 0:
                n_not_verifiable += 1
                per_sentence_out.append({
                    "sent_id": s.sent_id,
                    "section_title": s.section_title,
                    "text": s.text,
                    "n_cites": 0,
                    "label_to_ref": {},
                    "judge": {
                        "verdict": "NOT_VERIFIABLE",
                        "score": None,
                        "rationale": (
                            "all citations on this claim resolved to a real "
                            "paper but no abstract was available to judge "
                            "attribution; excluded from recall/precision"
                        ),
                        "per_citation_contribution": {},
                    },
                })
            continue

        j = attribution_judge.judge_sentence_attribution(
            sentence=s.text,
            cite_labels_to_evidence=labels_to_evidence,
        )
        sentence_judgments.append({"n_cites": len(labels_to_evidence), "judge": j})
        per_sentence_out.append({
            "sent_id": s.sent_id,
            "section_title": s.section_title,
            "text": s.text,
            "n_cites": len(labels_to_evidence),
            "label_to_ref": label_to_ref,
            "judge": j,
        })

    metrics = attribution_judge.attribution_metrics(sentence_judgments)
    metrics["n_not_verifiable"] = n_not_verifiable
    # A3 BareAssertionRate from the parser's whole-doc claim accounting.
    bare_rate = (doc.n_claims_bare / doc.n_claims) if doc.n_claims else None
    metrics["n_claims"] = doc.n_claims
    metrics["n_claims_with_cite"] = doc.n_claims_with_cite
    metrics["n_claims_bare"] = doc.n_claims_bare
    metrics["bare_assertion_rate"] = round(bare_rate, 4) if bare_rate is not None else None

    out = _result_path(system_id, topic_id, seed, "attribute")
    out.write_text(json.dumps({
        "system": system_id,
        "topic_id": topic_id,
        "seed": seed,
        "topic": topic["topic"],
        "metrics": metrics,
        "per_sentence": per_sentence_out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


# --- Phase 3c: classify (existence-failure decomposition — §4.B) --------


def _phase_classify(system_id: str, topic_id: str, seed: int) -> dict:
    """Decompose every reference's resolution outcome into a §4.B failure
    class (or RESOLVED). Pure post-processing over the verify.json payload:
    ZERO network calls — all signal (parsed metadata + identification
    verdict + candidate list) was captured during verify.

    Downgrades the v1 "existence rate" to a pipeline-correctness gate and
    reports the failure distribution per system, so SurveyFlow's residual
    failures can be shown to be engineering bugs (parse/id/dedup) rather
    than hallucination, while LLM_ONLY's concentrate in fabrication.
    """
    verify_path = _result_path(system_id, topic_id, seed, "verify")
    if not verify_path.exists():
        raise FileNotFoundError(f"missing verify result: {verify_path}")
    verify_data = json.loads(verify_path.read_text(encoding="utf-8"))

    result = classify_verify_payload(verify_data)
    out = _result_path(system_id, topic_id, seed, "classify")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "n_references": result["n_references"],
        "n_resolved": result["n_resolved"],
        "n_failures": result["n_failures"],
        "n_genuinely_fabricated": result["n_genuinely_fabricated"],
        "distribution": result["distribution"],
    }


# --- Phase 4: judge -----------------------------------------------------


def _phase_judge(system_id: str, topic_id: str, seed: int) -> dict:
    topic = topic_by_id(topic_id)
    verify_path = _result_path(system_id, topic_id, seed, "verify")
    if not verify_path.exists():
        raise FileNotFoundError(f"missing verify result: {verify_path}")
    v = json.loads(verify_path.read_text(encoding="utf-8"))
    per_ref = v.get("per_reference", [])

    # Only judge references that resolved to a real paper.
    # Unresolved references get a default IRRELEVANT score.
    judgments: list[dict] = []
    per_ref_out: list[dict] = []
    for r in per_ref:
        ident = r.get("identification") or {}
        verdict = ident.get("verdict", "NOT_FOUND")
        best = ident.get("best_hit") or {}
        if verdict in ("MATCH", "MISSING_ID") and best:
            j = faithfulness_judge.judge_reference(
                topic=topic["topic"],
                paper_title=best.get("title", ""),
                paper_abstract=best.get("abstract", ""),
            )
        else:
            j = {
                "verdict": "IRRELEVANT",
                "score": 0.0,
                "rationale": (
                    f"reference did not resolve (id_verdict={verdict}); "
                    f"defaulting to IRRELEVANT for faithfulness"
                ),
            }
        judgments.append(j)
        per_ref_out.append({
            "ref_id": r.get("ref_id"),
            "identification_verdict": verdict,
            "title_sim": ident.get("title_sim"),
            "judge": j,
        })

    agg = faithfulness_judge.faithfulness_score(judgments)
    out = _result_path(system_id, topic_id, seed, "judge")
    out.write_text(json.dumps({
        "system": system_id,
        "topic_id": topic_id,
        "seed": seed,
        "topic": topic["topic"],
        "n_references": len(per_ref),
        "per_reference": per_ref_out,
        "aggregate": agg,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return agg


# --- main loop ---------------------------------------------------------


def _all_cells() -> list[tuple[str, str, int]]:
    """Every (system, topic, seed) cell of the grid.

    With 3 systems x 10 topics x 3 seeds this is 90 cells per phase. The
    seed is the innermost dimension so that re-runs of the same
    (system, topic) land in adjacent files.
    """
    return [
        (s["id"], t["id"], seed)
        for s in SYSTEMS
        for t in TOPICS
        for seed in SEEDS
    ]


def _dispatch(phase: str, system_id: str, topic_id: str, seed: int) -> dict:
    if phase == "generate":
        return _phase_generate(system_id, topic_id, seed)
    if phase == "parse":
        return _phase_parse(system_id, topic_id, seed)
    if phase == "verify":
        return _phase_verify(system_id, topic_id, seed)
    if phase == "attribute":
        return _phase_attribute(system_id, topic_id, seed)
    if phase == "classify":
        return _phase_classify(system_id, topic_id, seed)
    if phase == "judge":
        return _phase_judge(system_id, topic_id, seed)
    raise ValueError(f"unknown phase: {phase}")


def _run_phase(phase: str, only: Optional[list[tuple[str, str, int]]] = None) -> None:
    cells = only or _all_cells()
    print(f"== phase={phase} on {len(cells)} cells ==")
    for system_id, topic_id, seed in cells:
        try:
            t0 = time.time()
            _dispatch(phase, system_id, topic_id, seed)
            dt = time.time() - t0
            print(f"  [OK] {system_id} / {topic_id} / seed{seed}  ({dt:.1f}s)")
        except Exception as e:
            print(f"  [FAIL] {system_id} / {topic_id} / seed{seed}: {e}")
            # do not raise - continue with other cells


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Citation Accuracy benchmark runner")
    p.add_argument(
        "--phase",
        choices=["generate", "parse", "verify", "attribute", "classify", "judge", "all"],
        default="all",
    )
    p.add_argument("--system", default=None, help="limit to one system id")
    p.add_argument("--topic", default=None, help="limit to one topic id")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="limit to one seed (must be one of configs.SEEDS)",
    )
    args = p.parse_args(argv)

    only: Optional[list[tuple[str, str, int]]] = None
    if args.system or args.topic or args.seed is not None:
        only = [
            (s["id"], t["id"], seed)
            for s in SYSTEMS
            for t in TOPICS
            for seed in SEEDS
            if (args.system is None or s["id"] == args.system)
            and (args.topic is None or t["id"] == args.topic)
            and (args.seed is None or seed == args.seed)
        ]
        if not only:
            print(
                f"No cells match --system={args.system} --topic={args.topic} "
                f"--seed={args.seed}. Valid seeds: {SEEDS}"
            )
            return 1

    if args.phase == "all":
        for ph in ("generate", "parse", "verify", "attribute", "classify", "judge"):
            _run_phase(ph, only=only)
    else:
        _run_phase(args.phase, only=only)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
