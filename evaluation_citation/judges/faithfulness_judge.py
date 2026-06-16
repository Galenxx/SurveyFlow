"""Faithfulness (topic relevance) judge.

For each reference, ask an LLM judge whether the paper (identified by
title + abstract) is RELEVANT / PARTIAL / IRRELEVANT to the survey's
topic. This catches the "real but wrong paper" failure mode of
pure-LLM citation.

Reuses the existing project LLMClient + judge_json helper for prompt
contract parity with the GQE benchmark.
"""

from __future__ import annotations
import json
from typing import Optional

from core.llm_client import LLMClient
from evaluation.judges.llm_judge import judge_json
from evaluation_citation.configs import (
    JUDGE_MODEL,
    JUDGE_FALLBACK_MODEL,
    JUDGE_TEMPERATURE,
)


VERDICT_SCORE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}

_SYSTEM = """You are an expert academic reviewer.
You will be shown a survey topic and a candidate paper (title + abstract).
Decide whether the paper is on-topic for the survey.

Output a single JSON object with exactly these keys:
  "verdict": one of "RELEVANT", "PARTIAL", "IRRELEVANT"
  "rationale": one-sentence justification

Definitions:
- RELEVANT   : the paper's main contribution is clearly within the
               survey's topic; the survey could legitimately cite it
               as a representative work in that area.
- PARTIAL    : the paper touches the topic tangentially - e.g. the
               topic is a sub-problem, an application, or a related
               technique, but the paper is not a primary reference.
- IRRELEVANT : the paper is on a different problem, domain, or method;
               citing it in the survey would be a mistake.

Do not output any prose before or after the JSON."""


def _user_prompt(topic: str, paper_title: str, paper_abstract: str) -> str:
    abstract = (paper_abstract or "").strip()
    if not abstract:
        abstract = "(no abstract available)"
    if len(abstract) > 2000:
        abstract = abstract[:2000] + "..."
    return (
        f"Survey topic: {topic}\n\n"
        f"Paper title: {paper_title}\n\n"
        f"Paper abstract: {abstract}\n\n"
        f"Output JSON verdict only."
    )


def _pick_judge_model(preferred: str) -> Optional[str]:
    """Pick an available judge model.

    The preferred judge (opposite-family from the generator) is tried
    first. If the corresponding client cannot be initialized (e.g. the
    API key is not set), fall back to the secondary model. This keeps
    the benchmark runnable in environments where only one provider's
    key is configured.

    Returns the model name to use, or None if no provider is available.
    """
    from core.llm_client import LLMClient
    from core import llm_client as _lc

    # The provider clients are CLASS attributes populated lazily by
    # ``LLMClient.__init__``. If nobody has instantiated the client yet
    # (e.g. the judge phase runs in a fresh process with no prior
    # generation), both are still None and we'd wrongly report "no judge
    # model". Force a one-time, idempotent init (singleton) so the
    # env-derived clients exist before we inspect them. Mirrors the same
    # fix in attribution_judge._pick_judge_model.
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


def judge_reference(
    *,
    topic: str,
    paper_title: str,
    paper_abstract: str,
    model: Optional[str] = None,
) -> dict:
    """Return a dict with verdict, score (0/0.5/1), and rationale.

    Never raises - returns a dict with verdict='ERROR' on failure.
    """
    chosen = model or _pick_judge_model(JUDGE_MODEL)
    if chosen is None:
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "rationale": (
                "no judge model available; set DEEPSEEK_API_KEY or "
                "OPENAI_API_KEY environment variable"
            ),
        }
    try:
        out = judge_json(
            _SYSTEM,
            _user_prompt(topic, paper_title, paper_abstract),
            model=chosen,
        )
        verdict = str(out.get("verdict", "")).upper()
        if verdict not in VERDICT_SCORE:
            return {
                "verdict": "ERROR",
                "score": 0.0,
                "rationale": f"unrecognized verdict: {verdict}",
            }
        return {
            "verdict": verdict,
            "score": VERDICT_SCORE[verdict],
            "rationale": out.get("rationale", ""),
        }
    except Exception as e:  # network / parse failure
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "rationale": f"judge failed: {e}",
        }


def faithfulness_score(judgments: list[dict]) -> dict:
    """Aggregate per-reference judgments into a 0-100 score.

    `judgments` is a list of dicts each with a 'score' in {0, 0.5, 1}.
    ERROR verdicts count as 0 but are reported separately.
    """
    n = len(judgments)
    n_err = sum(1 for j in judgments if j.get("verdict") == "ERROR")
    n_rel = sum(1 for j in judgments if j.get("verdict") == "RELEVANT")
    n_par = sum(1 for j in judgments if j.get("verdict") == "PARTIAL")
    n_irr = sum(1 for j in judgments if j.get("verdict") == "IRRELEVANT")
    if n == 0:
        return {
            "score": 0.0,
            "n_relevant": 0,
            "n_partial": 0,
            "n_irrelevant": 0,
            "n_error": 0,
            "n_judged": 0,
        }
    total = sum(j.get("score", 0.0) for j in judgments)
    score = 100.0 * total / n
    return {
        "score": round(score, 2),
        "n_relevant": n_rel,
        "n_partial": n_par,
        "n_irrelevant": n_irr,
        "n_error": n_err,
        "n_judged": n,
    }
