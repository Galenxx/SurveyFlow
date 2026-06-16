"""
Section Agent prompts for section title generation.
"""

SECTION_SYSTEM_PROMPT_TEMPLATE = """You are an academic paper writing expert. Based on paper descriptions from each category, generate a concise section title (within 10 words) for each category of the survey paper.

Survey topic: {survey_title}

IMPORTANT: Section titles must contain keywords from the ACTUAL PAPERS in each category.
Do NOT generate generic abstract terms. The title should reflect what the papers actually discuss.
For example, if papers are about job recommendation with LLMs, use terms like "LLM-driven Job Recommendation"
or "LLM Applications in Career Systems" — NOT "Topological Graph Learning" or "Cognitive Architectures"."""

SECTION_USER_PROMPT_TEMPLATE = """Category information:
{cluster_info}

Requirements:
1. Each title must contain concrete technical keywords from the papers listed in that category
2. Include at least one keyword from the survey topic
3. Avoid content overlap between titles
4. Use the ACTUAL paper topics as inspiration — look at the paper titles and descriptions
5. Titles should be specific and descriptive (e.g., "LLM-based Job Recommendation Methods",
   "AI-driven Candidate Assessment", "Bias and Fairness in Hiring Systems")

Output format (JSON list):
["Section Title 1", "Section Title 2", "Section Title 3"]"""

SECTION_REFINEMENT_PROMPT = """Here is a set of section titles:
{section_titles}

Please refine these section titles ensuring:
1. All cluster names are coherent and consistent with each other
2. Each name is clear, concise, and accurately reflects the corresponding papers
3. Remove overlapping information between cluster names
4. Each cluster name should be within 10 words and include at least one keyword
   from the ACTUAL PAPERS in that cluster (not generic ML terms)
5. Verify that the title actually matches the paper topics described

Response with a list of refined section titles in the following format without any other irrelevant information:
["Refined Title 1", "Refined Title 2", "Refined Title 3"]"""
