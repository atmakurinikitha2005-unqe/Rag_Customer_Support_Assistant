from __future__ import annotations

from typing import TypedDict, List

from langgraph.graph import END, StateGraph


class AssistantState(TypedDict):
    question: str
    context: List[str]
    answer: str


def _answer_node(state: AssistantState) -> AssistantState:
    question = state["question"]
    context = state.get("context", [])

    if not context:
        state["answer"] = (
            "I could not find relevant content in the uploaded PDF. "
            "Please upload a document or ask a question related to it."
        )
        return state

    joined_context = "\n\n".join(context[:4])
    state["answer"] = (
        "Based on the uploaded document, here are the most relevant points:\n\n"
        f"{joined_context}\n\n"
        f"Question: {question}\n\n"
        "Note: This starter version retrieves relevant PDF text locally. "
        "You can connect Groq in app/main.py for full LLM-style answers."
    )
    return state


def build_graph():
    """Build a minimal LangGraph graph so `from app.graph.builder import build_graph` works."""
    graph = StateGraph(AssistantState)
    graph.add_node("answer", _answer_node)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()
