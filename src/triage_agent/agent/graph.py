"""LangGraph triage workflow.

The workflow is an explicit state graph — not one giant prompt:

    START
      -> fetch_repository_context
      -> fetch_item                    (issue or PR; fatal errors skip to summary)
      -> classify_request              (LLM, structured output)
      -> gather_additional_context
      -> search_duplicates             (issues only)
      -> assess_priority_and_severity
      -> determine_actions
      -> safety_check                  (policy + confidence gating)
      -> human_approval                (only when approval is required; interrupt)
      -> execute_actions
      -> verify_actions
      -> generate_summary
      -> END
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.nodes import TriageNodes
from triage_agent.agent.state import ObjectType, TriageState


def _after_fetch(state: TriageState) -> Literal["classify_request", "generate_summary"]:
    return "generate_summary" if state.error else "classify_request"


def _after_classify(
    state: TriageState,
) -> Literal["gather_additional_context", "assess_priority_and_severity", "generate_summary"]:
    if state.error:
        return "generate_summary"
    if state.object_type is ObjectType.PULL_REQUEST:
        # PR triage does not run the issue-duplicate pipeline.
        return "assess_priority_and_severity"
    return "gather_additional_context"


def _after_safety(
    state: TriageState,
) -> Literal["human_approval", "execute_actions"]:
    return "human_approval" if state.pending_approval else "execute_actions"


def build_graph(
    deps: TriageDeps,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[TriageState]:
    """Compile the triage graph with the given dependencies."""
    nodes = TriageNodes(deps)
    graph: StateGraph[TriageState] = StateGraph(TriageState)

    graph.add_node("fetch_repository_context", nodes.fetch_repository_context)
    graph.add_node("fetch_item", nodes.fetch_item)
    graph.add_node("classify_request", nodes.classify_request)
    graph.add_node("gather_additional_context", nodes.gather_additional_context)
    graph.add_node("search_duplicates", nodes.search_duplicates)
    graph.add_node("assess_priority_and_severity", nodes.assess_priority_and_severity)
    graph.add_node("determine_actions", nodes.determine_actions)
    graph.add_node("safety_check", nodes.safety_check)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("execute_actions", nodes.execute_actions)
    graph.add_node("verify_actions", nodes.verify_actions)
    graph.add_node("generate_summary", nodes.generate_summary)

    graph.add_edge(START, "fetch_repository_context")
    graph.add_edge("fetch_repository_context", "fetch_item")
    graph.add_conditional_edges("fetch_item", _after_fetch)
    graph.add_conditional_edges("classify_request", _after_classify)
    graph.add_edge("gather_additional_context", "search_duplicates")
    graph.add_edge("search_duplicates", "assess_priority_and_severity")
    graph.add_edge("assess_priority_and_severity", "determine_actions")
    graph.add_edge("determine_actions", "safety_check")
    graph.add_conditional_edges("safety_check", _after_safety)
    graph.add_edge("human_approval", "execute_actions")
    graph.add_edge("execute_actions", "verify_actions")
    graph.add_edge("verify_actions", "generate_summary")
    graph.add_edge("generate_summary", END)

    return graph.compile(checkpointer=checkpointer)
