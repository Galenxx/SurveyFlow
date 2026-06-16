# Citation Accuracy benchmark

Evaluates and compares the **citation accuracy** of surveys produced
by two systems on the same five topics:

| System        | Description                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------- |
| `SURVEYFLOW`  | Full SurveyFlow pipeline (retrieval-augmented, real PDF corpus)                               |
| `LLM_ONLY`    | Same generator LLM, no retrieval, parametric knowledge only — asked to cite ~10 real papers   |

For each `(system, topic)` cell we measure three orthogonal properties
of every cited reference, then aggregate:

| Metric                | Question it answers                                                       | Range    |
| --------------------- | ------------------------------------------------------------------------- | -------- |
| **Existence Rate**    | Does the reference resolve to a real paper at all?                        | 0..1     |
| **Identifier Precision** | When it does resolve, does the arXiv ID/DOI really point at the paper the survey meant? | 0..1 |
| **Faithfulness Score** | LLM-judge: is the cited paper actually about the survey's topic?          | 0..100   |

This complements the existing GQE gap-report benchmark under
`survey_agent/evaluation/` by focusing on a different failure mode
(incorrect/fabricated citations) with a different but compatible
metric structure.

## What the benchmark controls

Both systems are asked to produce **exactly 10 references** per survey.
This caps the per-cell cost and keeps the two systems directly
comparable (without the cap, SurveyFlow would tend to produce
substantially more references than the LLM-only baseline).

The five topics were picked to cover five different research fields,
so the result is not dominated by any one domain:

| Topic ID                | Field                                    |
| ----------------------- | ---------------------------------------- |
| `code_review_llm`       | Software Engineering / NLP               |
| `diffusion_3d`          | Computer Vision / Graphics               |
| `gnn_drug`              | Computational Biology / ML              |
| `short_video_misinfo`   | Social Computing / Multimedia            |
| `affective_remote_work` | HCI / Organizational Behavior            |

## Pipeline

The runner is split into 5 resumable phases; each cell's output is
persisted to disk as soon as it finishes, so a crash mid-run can pick
up from the next phase.

```
[1] generate    per (system, topic) - produce survey.md
[2] parse       per (system, topic) - extract "Reference:" lines
[3] verify      per (system, topic) - arXiv + Semantic Scholar lookup
[4] judge       per (system, topic) - LLM topic-relevance per reference
[5] aggregate   all cells        - summary.json, citation_accuracy.csv, *.tex
```

### Existence verification (phase 3)

For every parsed reference:

1. If the line contains an arXiv ID (`arXiv:NNNN.NNNNN` or
   `arXiv:cs.LG/0703001`), look it up via the arXiv API. This is the
   authoritative source.
2. If arXiv fails, try the DOI (Crossref-backed) via Semantic Scholar.
3. If DOI also fails, search the title via Semantic Scholar.

A reference is `FOUND` iff at least one candidate is returned by any
of these calls. Ambiguous title searches (multiple plausible matches,
top-1 within 0.05 of top-2) are recorded as `AMBIGUOUS` and treated
as `IRRELEVANT` downstream for fairness.

### Identifier precision (phase 3)

For every `FOUND` reference, `verifiers/identifier_check.py` decides:

- `MATCH`         the survey's arXiv ID/DOI maps to the resolved paper
                  AND title similarity ≥ `IDENTIFIER_SIM_THRESHOLD`
- `MISMATCH`      the survey gave an arXiv ID/DOI that resolves to a
                  different paper (or to nothing)
- `MISSING_ID`    the survey had no ID at all and we identified the
                  paper by title similarity
- `AMBIGUOUS`     the title match is not strong enough to be sure

### Faithfulness (phase 4)

For every reference that resolved to a real paper, an LLM judge (the
opposite-family model from the generator) reads the survey topic plus
the paper's title and abstract and returns one of three verdicts:

- `RELEVANT`     score 1.0
- `PARTIAL`      score 0.5
- `IRRELEVANT`   score 0.0

References that did not resolve default to `IRRELEVANT` (it is the
most conservative choice; a hallucinated reference is by definition
not on-topic).

The faithfulness score for a cell is `100 * mean(scores)`.

## How to run

```bash
cd survey_agent

# 1. (optional) inspect the topic grid
python -c "from evaluation_citation.configs import TOPICS, SYSTEMS; \
  [print(t['id'], '|', t['field']) for t in TOPICS]; \
  [print(s['id']) for s in SYSTEMS]"

# 2. full pipeline (5 systems x 5 topics = 10 surveys, then parse/verify/judge)
python -m evaluation_citation.run_benchmark

# 3. or: run phases individually (resumable)
python -m evaluation_citation.run_benchmark --phase generate
python -m evaluation_citation.run_benchmark --phase parse
python -m evaluation_citation.run_benchmark --phase verify
python -m evaluation_citation.run_benchmark --phase judge

# 4. restrict to a single cell for debugging
python -m evaluation_citation.run_benchmark --phase generate \
  --system LLM_ONLY --topic code_review_llm

# 5. aggregate
python -m evaluation_citation.aggregate
```

## Files written under `evaluation_citation/results/`

| File                                      | Contents                                          |
| ----------------------------------------- | ------------------------------------------------- |
| `surveyflow/{topic}__generate.json`       | MD path produced by SurveyFlow                    |
| `surveyflow/{topic}__parse.json`         | Parsed references (title, authors, year, arXiv)   |
| `surveyflow/{topic}__verify.json`         | API resolution + identification verdict per ref   |
| `surveyflow/{topic}__judge.json`          | LLM topic-relevance per ref + cell aggregate     |
| `llm_only/{topic}__*.json`                | Same four files for the LLM-only baseline         |
| `summary.json`                            | All per-cell metrics (also used by the CSV)       |
| `citation_accuracy.csv`                  | One row per (system, topic) - paper-friendly      |
| `table_main.tex`                         | LaTeX: per-cell breakdown                         |
| `table_per_system.tex`                   | LaTeX: per-system mean over topics                |

## Cost & runtime (estimate)

- Phase 1 (`generate`): 5 SurveyFlow runs (~15-37 min each, dominated
  by PDF download + parse) + 5 LLM_ONLY runs (~1-2 min each on glm-5.1).
  Total: ~80-200 min.
- Phase 3 (`verify`): 2 systems x 5 topics x 10 refs = 100 API calls
  to arXiv + S2, ~5-10 min total.
- Phase 4 (`judge`): 100 LLM judge calls, ~10-20 min total.
- Phases 2 and 5 are local and run in seconds.

## Limitations (acknowledged in the paper)

- 5 topics is small; we picked 5 deliberately different fields
  rather than 5 within-field variants.
- LLM-as-judge is approximate. The GQE benchmark's Cohen-κ
  human-validation subset can be reused here as a sanity check;
  we do not duplicate that work in this README.
- SurveyFlow's reference list is produced by the
  `Writer`-stage citation injection from real PDFs; the
  `LLM_ONLY` baseline cannot inspect those PDFs and must rely on
  parametric memory. This is the source of the expected gap.
- The "MISSING_ID" verdict is generous: it accepts a title-only
  reference that resolves to a unique real paper. A stricter
  benchmark would treat it as `NOT_FOUND`.
