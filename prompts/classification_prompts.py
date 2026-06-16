"""
Classifier Agent prompts for paper classification.
"""

CLASSIFICATION_SYSTEM_PROMPT = """You are an academic paper classification expert. Based on paper descriptions, you decide the optimal number of categories and assign each paper to a category."""

CLASSIFICATION_USER_PROMPT_TEMPLATE = """You are an academic paper classification expert. Please decide how many categories to divide these papers into and assign each paper to a category.

## Survey Topic
{survey_title}

## Paper List
{paper_list}

## Task
1. Carefully read all paper descriptions and understand the core topics and methods of each paper
2. Decide the number of categories (recommended 2-5; too few causes content overlap, too many causes sparse papers per category)
3. Provide a brief definition for each category (1-2 sentences)
4. Assign category numbers to each paper

## Output Format (strict JSON, no other text):
{{
    "num_classes": 3,
    "class_definitions": {{
        "0": "definition of category 0",
        "1": "definition of category 1",
        "2": "definition of category 2"
    }},
    "paper_classifications": {{
        "0": ["paper title 1", "paper title 3", ...],
        "1": ["paper title 2", "paper title 5", ...],
        "2": ["paper title 4", ...]
    }},
    "reasoning": "classification rationale"
}}"""
