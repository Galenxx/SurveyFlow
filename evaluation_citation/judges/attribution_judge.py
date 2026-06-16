"""Sentence-level attribution judge (ALCE-style) — Task #8 core metric.

Given the (sentence -> cited papers) mapping produced by
``attribution_parser`` plus each cited paper's evidence E(c) (title +
abstract, optionally retrieved chunks), this judge answers the
*non-circular* question (§4.A / §4.D):

    "Does the cited evidence actually SUPPORT this sentence's factual
     claim?" — not v1's circular "does the reference exist?".

Three metrics come out of it (per system, per topic):

  * A1. Citation Recall   — mean support score over cited claims, where
        support is {SUPPORTED:1.0, PARTIAL:0.5, UNSUPPORTED:0.0}.
  * A2. Citation Precision — fraction of citation *mounts* that actually
        contribute to supporting their sentence (a citation that can be
        dropped without changing the support verdict is redundant).
  * A3. BareAssertionRate — fraction of factual claims with NO citation
        (computed upstream by the parser; surfaced here for one-stop
        reporting).

The judge reuses the project ``judge_json`` helper for prompt-contract
parity (same retry/JSON-extraction behaviour as the GQE benchmark and
the v1 faithfulness judge) and the same opposite-family model-selection
+ graceful-degradation logic as ``faithfulness_judge``.

This module performs NO network calls of its own beyond the judge LLM —
the evidence (abstracts) is passed in by the caller, which sources it
from the already-completed verify phase. Keeping evidence-fetching out
of here means the judge is deterministic given its inputs and trivially
unit-testable with synthetic evidence.
"""

from __future__ import annotations

from typing import Optional

from evaluation.judges.llm_judge import judge_json
from evaluation_citation.configs import (
    JUDGE_MODEL,
    JUDGE_FALLBACK_MODEL,
)


# Support rubric (§4.D). Mirrors faithfulness_judge's VERDICT_SCORE shape
# so aggregation code can treat both uniformly.
SUPPORT_SCORE = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNSUPPORTED": 0.0}


_SYSTEM = """You are a meticulous academic fact-checker assessing citation attribution.

You are given ONE sentence from a survey paper that makes a factual claim,
together with the cited evidence (each cited paper's title and abstract).
Your job is to decide whether the cited evidence SUPPORTS the sentence's
factual claim — exactly the question a careful reviewer asks when checking
that a citation is not just topically related but actually backs the
statement it is attached to.

Output a SINGLE JSON object with exactly these keys:
  "verdict": one of "SUPPORTED", "PARTIAL", "UNSUPPORTED"
  "rationale": one-sentence justification grounded in the evidence
  "per_citation_contribution": an object mapping each citation label
       (exactly as given in the EVIDENCE list, e.g. "c0", "c1") to
       true/false — true iff THAT citation's evidence contributes to
       supporting the claim (i.e. removing it would weaken support).

Definitions:
- SUPPORTED   : the sentence's core factual claim can be directly inferred
                from the cited evidence.
- PARTIAL     : only part of the claim is supported, OR the evidence is
                merely topically related but does not directly establish
                the specific assertion.
- UNSUPPORTED : the cited evidence does not support the claim (includes the
                case where it is on-topic but says nothing about the
                specific assertion).

Judge ONLY against the supplied evidence. Do not use outside knowledge to
fill gaps the evidence does not cover. Do not output any prose before or
after the JSON."""


def _evidence_block(cite_labels_to_evidence: dict[str, dict]) -> str:
    """Render the evidence list the judge sees, one entry per citation."""
    lines: list[str] = []
    for label, ev in cite_labels_to_evidence.items():
        title = (ev.get("title") or "(unknown title)").strip()
        abstract = (ev.get("abstract") or "").strip()
        if not abstract:
            abstract = "(no abstract available)"
        if len(abstract) > 1500:
            abstract = abstract[:1500] + "..."
        lines.append(f"[{label}] {title}\nAbstract: {abstract}")
    return "\n\n".join(lines)


def _user_prompt(sentence: str, cite_labels_to_evidence: dict[str, dict]) -> str:
    return (
        f"SENTENCE (makes a factual claim):\n{sentence.strip()}\n\n"
        f"EVIDENCE (the cited papers):\n{_evidence_block(cite_labels_to_evidence)}\n\n"
        f"Does the evidence support the sentence's factual claim? "
        f"Output JSON only."
    )


def _pick_judge_model(preferred: str) -> Optional[str]:
    """Pick an available judge model (opposite-family preferred), mirroring
    ``faithfulness_judge._pick_judge_model`` so both judges degrade the same
    way when only one provider key is configured."""
    from core import llm_client as _lc

    # The provider clients are CLASS attributes populated lazily by
    # ``LLMClient.__init__``. If nobody has instantiated the client yet
    # (e.g. the attribution judge runs before any generation in-process),
    # both are still None and we'd wrongly report "no judge model". Force a
    # one-time, idempotent init (singleton) so the env-derived clients exist
    # before we inspect them.
    _lc.LLMClient()

    preferred_provider = "deepseek" if preferred.startswith("deepseek") else "zhipu"
    if (
        (preferred_provider == "deepseek" and _lc.LLMClient._deepseek_client is not None)
        or (preferred_provider == "zhipu" and _lc.LLMClient._zhipu_client is not None)
    ):
        return preferred
    fallback_provider = "zhipu" if preferred_provider == "deepseek" else "deepseek"
    if (
        (fallback_provider == "deepseek" and _lc.LLMClient._deepseek_client is not None)
        or (fallback_provider == "zhipu" and _lc.LLMClient._zhipu_client is not None)
    ):
        return JUDGE_FALLBACK_MODEL
    return None


def judge_sentence_attribution(
    *,
    sentence: str,
    cite_labels_to_evidence: dict[str, dict],
    model: Optional[str] = None,
) -> dict:
    """Judge whether the cited evidence supports one sentence's claim.

    Args:
        sentence: the claim sentence text.
        cite_labels_to_evidence: {label -> {title, abstract}} for each
            in-text citation on the sentence. Labels are arbitrary stable
            keys (e.g. "c0", "c1") the caller assigns; the judge echoes
            them back in ``per_citation_contribution``.

    Returns a dict::

        {
          "verdict": "SUPPORTED|PARTIAL|UNSUPPORTED|ERROR",
          "score": 1.0|0.5|0.0,
          "rationale": str,
          "per_citation_contribution": {label: bool, ...},
        }

    Never raises — returns verdict='ERROR', score 0.0 on failure so a
    single bad cell never crashes the benchmark.
    """
    if not cite_labels_to_evidence:
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "rationale": "no evidence supplied for this sentence",
            "per_citation_contribution": {},
        }

    chosen = model or _pick_judge_model(JUDGE_MODEL)
    if chosen is None:
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "rationale": (
                "no judge model available; set DEEPSEEK_API_KEY or "
                "ZHIPUAI_API_KEY"
            ),
            "per_citation_contribution": {},
        }

    try:
        out = judge_json(
            _SYSTEM,
            _user_prompt(sentence, cite_labels_to_evidence),
            model=chosen,
            max_tokens=1024,
        )
    except Exception as e:
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "rationale": f"judge failed: {e}",
            "per_citation_contribution": {},
        }

    verdict = str(out.get("verdict", "")).upper()
    if verdict not in SUPPORT_SCORE:
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "rationale": f"unrecognized verdict: {verdict}",
            "per_citation_contribution": {},
        }

    # Normalize per-citation contributions to {label: bool} over exactly the
    # labels we asked about (judge may omit or hallucinate keys).
    raw_contrib = out.get("per_citation_contribution") or {}
    contrib: dict[str, bool] = {}
    for label in cite_labels_to_evidence:
        v = raw_contrib.get(label)
        if isinstance(v, bool):
            contrib[label] = v
        elif isinstance(v, str):
            contrib[label] = v.strip().lower() in ("true", "yes", "1")
        else:
            # If the judge said SUPPORTED/PARTIAL but didn't rate this
            # citation, default to contributing; if UNSUPPORTED, default
            # to non-contributing. Conservative w.r.t. precision.
            contrib[label] = verdict in ("SUPPORTED", "PARTIAL")

    return {
        "verdict": verdict,
        "score": SUPPORT_SCORE[verdict],
        "rationale": out.get("rationale", ""),
        "per_citation_contribution": contrib,
    }


def attribution_metrics(sentence_judgments: list[dict]) -> dict:
    """Aggregate per-sentence attribution judgments into the A1/A2 metrics.

    Each element of ``sentence_judgments`` is expected to look like::

        {
          "n_cites": int,            # citation mounts on this sentence
          "judge": { "verdict", "score", "per_citation_contribution" },
        }

    Returns counts + rates (counts-first per §5, so CIs can be computed by
    the stats layer)::

        {
          "n_cited_claims": int,        # sentences with >=1 citation
          "support_total": float,       # sum of support scores
          "citation_recall": float|None,# support_total / n_cited_claims
          "n_supported": int, "n_partial": int, "n_unsupported": int,
          "n_error": int,
          "n_citation_mounts": int,     # total citations across cited claims
          "n_contributing": int,        # citations that contributed
          "citation_precision": float|None,  # n_contributing / n_citation_mounts
        }
    """
    cited = [j for j in sentence_judgments if j.get("n_cites", 0) > 0]
    n_cited = len(cited)

    n_supported = sum(1 for j in cited if j["judge"].get("verdict") == "SUPPORTED")
    n_partial = sum(1 for j in cited if j["judge"].get("verdict") == "PARTIAL")
    n_unsupported = sum(1 for j in cited if j["judge"].get("verdict") == "UNSUPPORTED")
    n_error = sum(1 for j in cited if j["judge"].get("verdict") == "ERROR")
    support_total = sum(j["judge"].get("score", 0.0) for j in cited)

    # Citation precision is computed on the SAME support scale as recall so
    # the two metrics are directly comparable (per the project decision that
    # a PARTIAL claim's citations count as 0.5, not 0):
    #   * SUPPORTED sentence: each citation the judge marked as contributing
    #     counts 1.0; non-contributing mounts count 0 (preserves the judge's
    #     citation-level discrimination on multi-cite sentences).
    #   * PARTIAL sentence: EVERY mount counts 0.5, mirroring the 0.5 the
    #     sentence contributes to recall. The per-citation boolean is not used
    #     here because the judge marks PARTIAL mounts False ("does not fully
    #     support the specific claim"), which — if taken as a hard 0 — made
    #     precision incomparable to recall and systematically deflated it.
    #   * UNSUPPORTED / ERROR sentence: every mount counts 0.
    # ``n_contributing`` is kept as the integer count of judge-True mounts for
    # auditing; the precision numerator is the weighted ``contributing_weight``.
    n_mounts = 0
    n_contributing = 0
    contributing_weight = 0.0
    for j in cited:
        judge = j["judge"]
        verdict = judge.get("verdict")
        contrib = judge.get("per_citation_contribution") or {}
        n_cites = j.get("n_cites", 0)
        n_mounts += n_cites
        n_contributing += sum(1 for v in contrib.values() if v)
        if verdict == "SUPPORTED":
            contributing_weight += sum(1.0 for v in contrib.values() if v)
        elif verdict == "PARTIAL":
            contributing_weight += 0.5 * n_cites

    return {
        "n_cited_claims": n_cited,
        "support_total": round(support_total, 4),
        "citation_recall": round(support_total / n_cited, 4) if n_cited else None,
        "n_supported": n_supported,
        "n_partial": n_partial,
        "n_unsupported": n_unsupported,
        "n_error": n_error,
        "n_citation_mounts": n_mounts,
        "n_contributing": n_contributing,
        "contributing_weight": round(contributing_weight, 4),
        "citation_precision": round(contributing_weight / n_mounts, 4) if n_mounts else None,
    }
