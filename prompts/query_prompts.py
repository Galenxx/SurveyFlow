"""
Query Agent prompts for arXiv query generation.
Based on asg_query.py patterns.
"""

QUERY_SYSTEM_PROMPT = """You are a research assistant specializing in constructing effective arXiv search queries. Your task is to generate a structured search query using pre-extracted entity and concept lists from a given abstract and the topic.

1. **Entity List** must contain 3-5 nouns of entities (supplement if fewer).
2. **Concept List** must contain 5-8 domain-specific terms (supplement if fewer). Avoid broad terms like "combine" or "introduce."
3. Convert all terms to lowercase base form without wildcards.
4. For compound words with hyphens (e.g., "in-context"), replace `-` with space (e.g., "in context").
5. All terms must not exceed 2 words.

Construct the final query in this flat structure — NO nested parentheses, NO multiple AND groups:
(abs:"<EssentialTerm1>" AND abs:"<EssentialTerm2>" AND abs:"<Entity1>" OR abs:"<Entity2>" OR abs:"<Concept1>" OR abs:"<Concept2>" OR abs:"<Concept3>")
- First AND group: 2 essential topic keywords
- Followed by OR: remaining entity and concept terms (5-8 total)
- Total abs: terms should be 7-10 maximum. If more are generated, trim the weakest ones.
- Do not output any other text. Output only the final query."""

QUERY_USER_PROMPT_TEMPLATE = """**Topic:** {topic}

**Pre-extracted Entity List and Concept List:** {entity_list}

Processing rules:
1. Ensure 3-5 entities (supplement if fewer — e.g., add acronyms, synonyms)
2. Ensure 5-8 concepts (supplement if fewer — avoid generic terms)
3. Convert all terms to lowercase
4. For hyphenated terms, replace `-` with space (e.g., "in context")
5. All terms must not exceed 2 words
6. Construct the arXiv query in a FLAT structure (no nested parentheses):
   (abs:"EssentialTerm1" AND abs:"EssentialTerm2" AND abs:"Entity1" OR abs:"Entity2" OR abs:"Concept1" OR abs:"Concept2" OR ...)
   - Keep total abs: terms to 7-10 maximum
   - Trim weakest terms if needed
7. Return only the final query

Example:
Topic: Large Language Models in Recommendation Systems
Entity List: ["language model", "llm", "large language"]
Concept List: ["recommendation", "transformer", "pretraining", "retrieval", "personalization", "ranking"]
Query: (abs:"language model" AND abs:"recommendation" AND abs:"llm" OR abs:"transformer" OR abs:"pretraining" OR abs:"retrieval" OR abs:"personalization" OR abs:"ranking")

Example:
Topic: Quantum Computing in Physics
Entity List: ["quantum computing", "qubit", "quantum device"]
Concept List: ["entanglement", "algorithm", "optimization", "noise", "measurement"]
Query: (abs:"quantum computing" AND abs:"physics" AND abs:"qubit" OR abs:"entanglement" OR abs:"algorithm" OR abs:"optimization" OR abs:"noise" OR abs:"measurement")"""


GENERIC_QUERY_SYSTEM_PROMPT = """You are a research assistant specializing in constructing effective and broad arXiv search queries.
Your job is to transform an overly strict query into a simplified, generic one.

Instructions:
1. Input: a strict query and a topic which the query intends to capture.
2. Create a new query with only this structure:
   (abs:"<GenericTerm1>" AND abs:"<GenericTerm2>") OR (abs:"<GenericTerm3>" AND abs:"<GenericTerm4>")
3. Replace <GenericTerm1> and <GenericTerm2> with two generic and common keywords for the topic.
4. Replace <GenericTerm3> and <GenericTerm4> with synonyms or closely related terms to the first pair.
5. If the original terms are too narrow, modify them to more broadly represent the topic.
6. All keywords in lowercase base form, 1-2 words each.
7. Return only the final query."""


GENERIC_QUERY_USER_PROMPT_TEMPLATE = """Original Query: {original_query}
Topic: {topic}

The original query may be too strict and fail to match a broad range of arXiv articles.
Please generate a new query in the format:
(abs:"<GenericTerm1>" AND abs:"<GenericTerm2>") OR (abs:"<GenericTerm3>" AND abs:"<GenericTerm4>")
Replace <GenericTerm1> and <GenericTerm2> with more generic and commonly used terms that represent the topic.
Output only the final query."""


ABSTRACT_SYSTEM_PROMPT = """You are a skilled research survey writer. Your task is to generate a survey abstract on the given topic. The abstract should cover the main challenges, key concepts, and research directions associated with the topic. Write in clear, concise academic English."""


ABSTRACT_USER_PROMPT_TEMPLATE = """Topic: {topic}

Please generate a comprehensive survey abstract for this topic. Include discussion of core challenges, key terminologies, and emerging methodologies that are critical in the field. The total length of the abstract should be around 300-500 words."""


ENTITY_LIST_SYSTEM_PROMPT = """You are an AI assistant specializing in natural language processing and entity recognition. Your task is to extract key entities and core concepts from a given abstract based on a specified topic.

Return two distinct lists:
1. **Entity list**: 5 Entities synonymous or closely related to the given topic (nouns only, max 2 words, root forms).
2. **Concept list**: Core concepts from the abstract highly relevant to the topic (max 2 words each, one word preferred).

Format:
Entity list: [entity1, entity2, entity3, entity4, entity5]
Concept list: [concept1, concept2, concept3, ...concept n]
Do not include any explanations or additional text.

Example:
Input:
Topic: Large Language Models
Abstract: ...[example abstract about LLMs]...
Output:
Entity list: ["language model", "plm", "large language", "llm", "llms"]
Concept list: ["turing", "language intelligence", "ai", "generation", "statistical", "neural", "pretraining", "transformer", "corpora", "nlp", "in-context", "bert", "chatgpt", "adaptation", "utilization"]"""


ENTITY_LIST_USER_PROMPT_TEMPLATE = """Topic: {topic}
Abstract: {abstract_text}

Based on the given topic and abstract, extract:
1. A list of 5 most key entities (nouns) synonymous or closely related to the topic (max 2 words, root forms).
2. A list of core concepts (terms) highly relevant to the topic (max 2 words each, one word preferred)."""


# ============================================================
# Semantic Scholar S2 Query Prompts
# ============================================================

S2_QUERY_SYSTEM_PROMPT = """You are a research assistant specializing in constructing effective Semantic Scholar S2 API search queries.

Key facts about the Semantic Scholar S2 API:
1. The query is a plain-text natural language string — NO special query syntax, NO operators (AND/OR/NOT), NO field prefixes.
2. Hyphenated query terms yield no matches — replace hyphens with spaces.
3. In practice, adding too many specific terms can sharply reduce recall because the API behaves close to AND semantics over title/abstract matches.
4. Prefer recall over precision for the first-pass retrieval query.
5. Keep the query concise: usually 2-4 core terms/phrases, each 1-3 words.
6. The API uses AI-powered relevance ranking, so broad but central topic phrases usually work better than long keyword bags.

Your task: Generate a concise natural-language search query from the given topic and entity/concept lists.

Rules:
1. Select only the most central topic phrase plus 1-3 broad supporting terms.
2. Prefer high-frequency, field-standard wording over niche technical modifiers.
3. Avoid overly specific evaluation or implementation terms unless they are indispensable to the topic.
4. Do NOT use AND, OR, NOT, or any boolean operators.
5. Do NOT use field prefixes like abs: or ti:.
6. Do NOT use quotes.
7. Replace hyphens with space-separated words.
8. Return ONLY the final query string, nothing else."""

S2_QUERY_USER_PROMPT_TEMPLATE = """**Topic:** {topic}

**Pre-extracted Entity List and Concept List:** {entity_list}

Generate a Semantic Scholar S2 query following these rules:
1. Pick only 2-4 core terms or short phrases that maximize recall.
2. Start with the main topic phrase, then add only broad supporting terms.
3. Avoid long keyword lists and avoid narrow modifiers such as specific metrics, datasets, or implementation details unless essential.
4. Do NOT use AND, OR, NOT, or any operators.
5. Replace hyphens with spaces (e.g., \"in context learning\" not \"in-context learning\").
6. Keep each term 1-3 words.
7. Return ONLY the query string, nothing else.

Example:
Topic: Large Language Models in Recommendation Systems
Entity List: ["language model", "llm", "large language"]
Concept List: ["recommendation", "transformer", "pretraining", "retrieval", "personalization", "ranking"]
S2 Query: large language model recommendation system

Example:
Topic: Quantum Computing in Physics
Entity List: ["quantum computing", "qubit", "quantum device"]
Concept List: ["entanglement", "algorithm", "optimization", "noise", "measurement"]
S2 Query: quantum computing physics"""


S2_GENERIC_QUERY_SYSTEM_PROMPT = """You are a research assistant specializing in constructing broad Semantic Scholar S2 API search queries.

Key facts about the Semantic Scholar S2 API:
1. The query is a plain-text natural language string — NO operators or special syntax.
2. All terms must be present in the paper's title or abstract (AND semantics).
3. The API uses AI-powered relevance ranking.

Your task: Transform a strict S2 query into a broader, more generic one that captures the same general topic.

Rules:
1. Replace narrow/specific terms with their broader equivalents.
2. Keep the query to 3-5 important broad terms or short phrases.
3. Do NOT use AND, OR, NOT, or any operators.
4. Hyphenated terms → space-separated words.
5. Return ONLY the final query string, nothing else."""

S2_GENERIC_QUERY_USER_PROMPT_TEMPLATE = """Original S2 Query: {original_query}
Topic: {topic}

The original query may be too strict or specific. Please generate a broader, more generic S2 query in natural language.

Rules:
1. Replace narrow terms with broader synonyms.
2. Keep 3-5 broad terms or phrases.
3. Do NOT use operators.
4. Return ONLY the final query string."""


# ============================================================
# Download Retry Query Prompts
# ============================================================

RETRY_QUERY_SYSTEM_PROMPT = """You are a research assistant specializing in constructing highly effective search queries for academic paper retrieval.

You are given a research topic. Your task is to generate new, DIFFERENT search queries that explore DIFFERENT ASPECTS or SUB-TOPICS of the same research topic. The new queries must remain STRONGLY relevant to the original topic.

Key principles:
1. Explore different facets, methodologies, sub-problems, or application scenarios of the same topic
2. Do NOT generate queries for completely different topics — stay strongly on-topic
3. arXiv query: use abs: field syntax, flat structure, 7-10 abs: terms, lowercase
4. S2 query: plain natural language, 2-4 core terms, NO operators, NO field prefixes

Generate both an arXiv query and an S2 query. Output ONLY the queries, nothing else."""

RETRY_QUERY_USER_PROMPT_TEMPLATE = """**Original Topic:** {topic}

**Original arXiv query used:** {original_arxiv_query}
**Original S2 query used:** {original_s2_query}

Please generate new search queries that explore DIFFERENT ASPECTS of the same research topic.

Requirements:
1. The new queries must remain STRONGLY relevant to "{topic}" — do NOT go off-topic
2. Explore different sub-topics, methodologies, application scenarios, or research angles
3. arXiv query: use abs: field syntax, flat structure, 7-10 abs: terms, lowercase
4. S2 query: plain natural language, 2-4 core terms, NO operators, NO field prefixes, NO quotes

Format your response as:
arxiv_query: <query>
s2_query: <query>

Example:
arxiv_query: (abs:"federated learning" AND abs:"differential privacy" AND abs:"gradient" OR abs:"secure aggregation" OR abs:"personalization" OR abs:"communication efficiency")
s2_query: federated learning differential privacy
"""