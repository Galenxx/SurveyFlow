"""
Outline Agent prompts for survey outline generation.
"""

OUTLINE_SYSTEM_PROMPT = """Generate the outline of the survey paper following the format: [[1, '1 Introduction'], [1, '2 Perturbations'], [2, '2.1 Derivations'], ...].

Rules:
- First element: hierarchy level (1 to 2)
- Second element: section name
- Level 1: Abstract, Introduction, cluster sections, Future Directions, Conclusion, References
- Sections 1 (Abstract), 2 (Introduction), 6 (Future Directions), 7 (Conclusion), and 8 (References)
  are TOP-LEVEL ONLY — do NOT generate any level-2 subsections under them.
  They appear as single [1, 'N Section Title'] entries followed immediately by their prose content.
- Level 2: sub-topics under each cluster section. The number of level-2 subsections
  per cluster is determined by the per-cluster budget supplied in the user prompt.
  Follow the budget EXACTLY — do not exceed it, and do not produce fewer.
- DO NOT generate any level-3 subsections. The outline is two levels deep only.
- Subsection titles MUST be derived from the ACTUAL TOPICS of the papers in each cluster.
  DO NOT use generic abstract terms. Use concrete technical terms that reflect what
  the papers actually discuss (e.g., "Skill Extraction from Job Descriptions",
  "LLM-based Resume Ranking", "Gender Bias in Hiring LLMs" — NOT "Topological
  Structure Assimilation", "Cognitive Reasoning Paradigms", "Dual-Signal Decoupling").
- Each subsection title should contain at least one keyword from the papers in that cluster.
- Do not include colons in titles."""

OUTLINE_USER_PROMPT_TEMPLATE = """## Survey Title
{survey_title}

## Cluster Papers Descriptions
{cluster_with_claims}

## Per-Cluster L2 Budget (HARD CONSTRAINT)
Each cluster section (numbered 3, 4, 5, ...) must contain EXACTLY the number of
level-2 subsections listed below. Do NOT exceed the budget, and do not produce fewer.

{cluster_budget}

The budget is derived from the number of papers in each cluster (1 L2 per paper,
capped at 3, with a floor of 1). This keeps each subsection substantive and avoids
splitting a small number of papers into many fine-grained sub-sub-topics.

## First Level Sections (given)
{first_level_sections}

## Task
1. Generate level-2 subsections ONLY under the cluster sections (numbered 3, 4, 5, ...).
   Do NOT generate any subsections under sections 1 (Abstract), 2 (Introduction),
   6 (Future Directions), 7 (Conclusion), or 8 (References).
2. The number of level-2 subsections under each cluster must EQUAL the budget above.
3. DO NOT generate any level-3 subsections. The outline is two levels deep only.
4. Subsection names MUST reflect the ACTUAL TECHNICAL TOPICS discussed in the papers of that cluster.
   Look at the paper titles and descriptions for each cluster — the subsection names should
   mention concrete concepts from those papers (e.g., "Graph-based Job Matching",
   "LLM-driven Candidate Assessment", "Gender Bias Auditing", "Skill Extraction").
   DO NOT invent generic terms like "Topological Structure Assimilation" or
   "Cognitive Reasoning Paradigms" unless those exact concepts appear in the papers.
5. Follow the exact format: [[level, 'title'], ...]"""

OUTLINE_DEFAULT_FIRST_LEVEL_TEMPLATE = """[[1, '1 Abstract'], [1, '2 Introduction'], {cluster_sections}, [1, '{n_plus_3} Future Directions'], [1, '{n_plus_4} Conclusion'], [1, '{n_plus_5} References']]"""
