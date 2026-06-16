"""Minimal diagnostic script for QueryNode LLM calls."""
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_client import LLMClient
from prompts.query_prompts import (
    ABSTRACT_SYSTEM_PROMPT,
    ABSTRACT_USER_PROMPT_TEMPLATE,
    ENTITY_LIST_SYSTEM_PROMPT,
    ENTITY_LIST_USER_PROMPT_TEMPLATE,
    QUERY_SYSTEM_PROMPT,
    QUERY_USER_PROMPT_TEMPLATE,
)
from nodes.query_node import _extract_query


def make_print_callback(label: str):
    """Return a streaming callback that prints tokens to stdout."""
    def on_chunk(token: str):
        print(token, end="", flush=True)
    return on_chunk


def main() -> None:
    load_dotenv()
    topic = "retrieval augmented generation for scientific surveys"
    llm = LLMClient()

    print("=== STEP 1: abstract ===")
    abstract = llm.generate(
        system_prompt=ABSTRACT_SYSTEM_PROMPT,
        user_prompt=ABSTRACT_USER_PROMPT_TEMPLATE.format(topic=topic),
        max_tokens=2048,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=make_print_callback("[abstract] "),
    )
    print()
    print(repr(abstract[:1000]))
    print()

    print("=== STEP 2: entity list ===")
    entity_list = llm.generate(
        system_prompt=ENTITY_LIST_SYSTEM_PROMPT,
        user_prompt=ENTITY_LIST_USER_PROMPT_TEMPLATE.format(topic=topic, abstract_text=abstract),
        max_tokens=2048,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=make_print_callback("[entity] "),
    )
    print()
    print(repr(entity_list[:1000]))
    print()

    print("=== STEP 3: raw query response ===")
    query_response = llm.generate(
        system_prompt=QUERY_SYSTEM_PROMPT,
        user_prompt=QUERY_USER_PROMPT_TEMPLATE.format(topic=topic, entity_list=entity_list),
        max_tokens=2048,
        temperature=0.5,
        model="deepseek-v4-pro",
        on_chunk=make_print_callback("[query] "),
    )
    print()
    print(repr(query_response[:1000]))
    print()

    print("=== STEP 4: extracted query ===")
    extracted = _extract_query(query_response)
    print(repr(extracted))


if __name__ == "__main__":
    main()
