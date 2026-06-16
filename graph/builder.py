"""
LangGraph StateGraph builder.
Assembles all nodes into a directed workflow graph.
"""
from langgraph.graph import StateGraph, END

from core.state import SurveyState, GapType
from graph.nodes import (
    query_node, download_node, splitter_node, gap_node,
    retriever_node, classifier_node, section_node, outline_node, writer_node,
)
from graph.edges import should_run_gap


def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(SurveyState)

    builder.add_node("query_node", query_node)
    builder.add_node("download_node", download_node)
    builder.add_node("splitter_node", splitter_node)
    builder.add_node("gap_node", gap_node)
    builder.add_node("retriever_node", retriever_node)
    builder.add_node("classifier_node", classifier_node)
    builder.add_node("section_node", section_node)
    builder.add_node("outline_node", outline_node)
    builder.add_node("writer_node", writer_node)

    builder.add_edge("__start__", "query_node")
    builder.add_edge("query_node", "download_node")
    builder.add_edge("download_node", "splitter_node")

    builder.add_conditional_edges(
        "splitter_node",
        should_run_gap,
        {
            "run_gap": "gap_node",
            "skip_gap": "retriever_node",
        }
    )

    builder.add_edge("gap_node", "retriever_node")
    builder.add_edge("retriever_node", "classifier_node")
    builder.add_edge("classifier_node", "section_node")
    builder.add_edge("section_node", "outline_node")
    builder.add_edge("outline_node", "writer_node")
    builder.add_edge("writer_node", END)

    return builder.compile()
