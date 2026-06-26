"""
langgraph_workflow.py
---------------------
Valet Living LangGraph Multi-Agent Workflow with LangFuse Tracing
- Defines a multi-agent workflow using LangGraph
- Orchestrates SQL Agent, RAG Agent, and Recommendation Agent
- Uses LangGraph to define the flow between agents
- Entry point for the full multi-agent pipeline

Flow:
START → SQL Agent → RAG Agent → Recommendation Agent → END
"""

from typing import TypedDict
from pathlib import Path
import sys

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langfuse import observe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Add parent directory to path so agents can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from sql_agent import sql_agent
from groq_rag_agent import ask as rag_agent
from recommendation_agent import recommendation_agent


# ── Agent State — shared across all nodes ──
class AgentState(TypedDict):
    question:       str   # original user question
    sql_result:     str   # output from SQL agent
    rag_result:     str   # output from RAG agent
    recommendation: str   # final recommendation from recommendation agent


# ── Node 1: SQL Agent ──
def run_sql_agent(state: AgentState) -> AgentState:
    """Query Azure SQL gold tables for structured analytics."""
    print("\n[SQL Agent] Querying Azure SQL...")
    result = sql_agent(state["question"])
    print(f"[SQL Agent] Done.\n")
    return {**state, "sql_result": result}


# ── Node 2: RAG Agent ──
def run_rag_agent(state: AgentState) -> AgentState:
    """Search FAISS vector DB for relevant complaint context."""
    print("\n[RAG Agent] Searching complaint history...")
    result = rag_agent(state["question"])
    print(f"[RAG Agent] Done.\n")
    return {**state, "rag_result": result}


# ── Node 3: Recommendation Agent ──
def run_recommendation_agent(state: AgentState) -> AgentState:
    """Generate strategic recommendations from SQL + RAG results."""
    print("\n[Recommendation Agent] Generating recommendations...")
    result = recommendation_agent(
        question=state["question"],
        sql_result=state["sql_result"],
        rag_result=state["rag_result"]
    )
    print(f"[Recommendation Agent] Done.\n")
    return {**state, "recommendation": result}


# ── Build LangGraph workflow ──
def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow connecting all 3 agents.
    
    Flow: SQL Agent → RAG Agent → Recommendation Agent → END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("sql_agent",            run_sql_agent)
    workflow.add_node("rag_agent",            run_rag_agent)
    workflow.add_node("recommendation_agent", run_recommendation_agent)

    # Define edges (flow between nodes)
    workflow.set_entry_point("sql_agent")
    workflow.add_edge("sql_agent",            "rag_agent")
    workflow.add_edge("rag_agent",            "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)

    return workflow.compile()

@observe(name="valet_living_pipeline")  # traces entire pipeline in LangFuse
def run_pipeline(question: str) -> dict:
    """
    Run the full multi-agent pipeline for a given question.
    
    Args:
        question: Natural language question about Valet Living data
    Returns:
        Final state dict with sql_result, rag_result, recommendation
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    app = build_workflow()

    final_state = app.invoke({
        "question":       question,
        "sql_result":     "",
        "rag_result":     "",
        "recommendation": ""
    })

    return final_state


def main():
    """Test the full LangGraph pipeline with sample questions."""

    questions = [
        "Which properties are most at risk of churn and how can we improve retention?",
        "What are the most common service complaints and how should we address them?",
    ]

    for question in questions:
        result = run_pipeline(question)

        print("\n" + "="*60)
        print("SQL FINDINGS:")
        print(result["sql_result"])

        print("\nRAG CONTEXT:")
        print(result["rag_result"])

        print("\nFINAL RECOMMENDATION:")
        print(result["recommendation"])
        print("="*60 + "\n")


if __name__ == "__main__":
    main()