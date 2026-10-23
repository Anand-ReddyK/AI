"""
Centralized Supervisor Multi-Agent Architecture with Metric Tracking.
"""

import os
import sys
from pathlib import Path

import time
from typing import Annotated, Dict, Any, List, Literal, Sequence, TypedDict
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.llm_provider import get_chat_model, extract_text_from_response
from utils.metrics import ExecutionMetrics
from agents.agentic_rag.rag_tools import RAGToolKit


# ===============================================================
# 1. TOOLS & DEFINITION
# ===============================================================

rag_toolkit = RAGToolKit()

@tool
def search_knowledge_base(query: str) -> str:
    """Web search"""
    print(f"\n   ⚙️  [Tool Invoked: search_knowledge_base] Query: '{query}'")
    results = rag_toolkit.search_web(query=query)
    print(f"   📥 [Tool Finished: search_knowledge_base] Retrieved {len(results)} result(s).")
    
    formatted = []
    for r in results:
        src = r.get("payload", {}).get("source", "Unknown")
        txt = r.get("payload", {}).get("text", "")[:1200]
        formatted.append(f"[{src}]: {txt}")
    
    return "\n------\n".join(formatted) if formatted else "No relevant information found."


@tool
def calculate(expression: str) -> str:
    """Safely evaluates basic arithmetic expressions (e.g., '120 * 0.15 + 45')."""
    print(f"\n   ⚙️  [Tool Invoked: calculate] Expression: '{expression}'")
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        print(f"   ❌ [Tool calculate Error] Unsupported characters in expression.")
        return "Error: Unsupported characters in mathematical expression."
    try:
        result = eval(expression, {"__builtins__": None}, {})
        print(f"   ✅ [Tool Finished: calculate] Computed Result: {result}")
        return f"Calculation Result: {result}"
    except Exception as e:
        print(f"   ❌ [Tool calculate Error] {e}")
        return f"Calculation Error: {e}"


# ===============================================================
# 2. WORKER AGENTS
# ===============================================================

# Base models
worker_llm = get_chat_model(temperature=0.0)
supervisor_llm = get_chat_model(temperature=0.0)

# Specialized sub-agents (each has isolated scratchpad)

researcher_agent = create_react_agent(
    model=worker_llm,
    tools=[search_knowledge_base],
    prompt=(
        "You are a Research Worker. Your job is to search internal documents"
        "and external sources for factual data. Be precise and synthesize findings into a "
        "concise summary with citations. DO NOT do complex math; only collect verified facts."
    )
)


analyst_agent = create_react_agent(
    model=worker_llm,
    tools=[calculate],
    prompt=(
        "You are a specialized Data & Quantitative Worker. Your job is to process "
        "numerical data, calculate percentages, differences, or metrics using the calculator tool. Return the final computed results clearly."
    )
)


# ===============================================================
# 3. SUPERVISOR STATE & ROUTING SCHEMA
# ===============================================================

class SupervisorRouter(BaseModel):
    """The structured decision schema produced by the supervisor."""
    next_worker: Literal["research", "analyst", "FINISH"] = Field(
        description="The next specialized worker to delegate to, or 'FINISH' if user query is fully answered."
    )
    instructions: str = Field(
        description="Specific directions and context for the assigned worker."
    )

class MultiAgentState(TypedDict):
    """High-level shared state seen by the Supervisor."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_worker: str
    instructions: str
    metrics: ExecutionMetrics


# ===============================================================
# 4. GRAPH NODES
# ===============================================================

def print_worker_trajectory(agent_name: str, messages: list):
    """Prints the internal ReAct thoughts, tool calls, and observations of a worker."""
    print(f"\n--- 🔬 Internal ReAct Trace: {agent_name.upper()} ---")
    # Skip index 0 (the initial instructions)
    for i, msg in enumerate(messages[1:], start=1):
        if isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print(f"   [Step {i}] 🤖 LLM Decided to Call Tool: '{tc['name']}' | Args: {tc['args']}")
            else:
                txt = extract_text_from_response(msg).replace('\n', ' ')
                print(f"   [Step {i}] 🧠 LLM Final Synthesis: {txt[:140]}...")
        elif msg.__class__.__name__ == "ToolMessage":
            obs_preview = str(msg.content)[:120].replace("\n", " ")
            print(f"   [Step {i}] 📥 Tool Observation: {obs_preview}...")
    print(f"--- 🔬 End of Trace: {agent_name.upper()} ---\n")


def supervisor_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Suporvisor Node: Reads the conversation history and decides whether to 
    delegate to a specialized worker or conclude the task.
    """
    messages = state["messages"]
    metrics = state["metrics"]

    system_prompt = SystemMessage(
        content=(
            "You are the Central Supervisor managing two specialized workers:\n"
            "1. 'research': Retrives factual and documatary information.\n"
            "2. 'analyst': Performs arthimetic calculations, statistics, and metric comparisions.\n\n"
            "CRITICAL WORKFLOW RULES:\n"
            "1. Review the conversation history carefully.\n"
            "2. If an assigned worker has ALREADY returned the required calculation or facts, "
            "DO NOT delegate to them again. You MUST select 'FINISH'.\n"
            "3. When you select 'FINISH', put the complete final answer for the user in 'instructions'.\n"
            "4. Only delegate to 'analyst' if calculations are still missing.\n"
            "5. Only delegate to 'research' if facts or documents are still missing."
        )
    )
    
    # Enforce structured routing output
    structured_llm = supervisor_llm.with_structured_output(SupervisorRouter)
    decision: SupervisorRouter = structured_llm.invoke([system_prompt] + list(messages))

    # We estimate 1 call and count tokens for the supervisor step
    # Note: When using with_structured_output, LangChain encapsulates response
    metrics.llm_calls += 1

    print(f"\n[Supervisor] Decision: {decision.next_worker} | Instructions: {decision.instructions}")

    update: Dict[str, Any] = {
        "next_worker": decision.next_worker,
        "instructions": decision.instructions
    }

    # When finishing, append the final answer to state messages
    if decision.next_worker == "FINISH":
        update["messages"] = [AIMessage(content=decision.instructions)]


    return update


def researcher_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Executes Researcher worker in an isolated state scope.
    Only the synthesized result is pushed back to supervisor messages.
    """

    metrics = state["metrics"]
    instructions = state["instructions"]

    print(f"[Researcher] 🔍 Working on: {instructions}")
    # Local worker call with isolated state
    worker_result = researcher_agent.invoke({
        "messages":  [HumanMessage(content=instructions)]
    })

    # Track metrics from the worker's internal interactions
    for msg in worker_result["messages"]:
        if isinstance(msg, AIMessage):
            metrics.record_call(msg)
    
    # 🔍 Print internal ReAct trajectory
    print_worker_trajectory("Researcher", worker_result["messages"])

    final_report = extract_text_from_response(worker_result["messages"][-1])
    print(f"[Reseacrcher] ✅ Report: {final_report[:120]}...")

    # Return as HumanMessage so Gemini treats it as an incoming report to the supervisor
    return {
        "messages": [HumanMessage(content=f"RESEARCH REPORT:\n{final_report}", name="researcher")]
    }


def analyst_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Executes Analyst worker in an isolated state.
    Only the synthesized calculation is pushed back to supervisor messages.
    """
    metrics = state["metrics"]
    instructions = state["instructions"]

    print(f"[Analyst] 📊 Working on: {instructions}")
    worker_result = analyst_agent.invoke({
        "messages": [HumanMessage(content=instructions)]
    })

    for msg in worker_result["messages"]:
        if isinstance(msg, AIMessage):
            metrics.record_call(msg)
    
    print_worker_trajectory("Analyst", worker_result["messages"])

    final_result = extract_text_from_response(worker_result["messages"][-1])
    print(f"[Analyst] ✅ Result: {final_result}")

    # Return as HumanMessage so Gemini treats it as an incoming report to the supervisor
    return {
        "messages": [HumanMessage(content=f"ANALYST RESULT:\n{final_result}", name="analyst")]
    }



# ===============================================================
# 5. GRAPH COMPILATION & ROUTING
# ===============================================================

def route_next(state: MultiAgentState) -> str:
    """Conditional routing based on supervisor decision."""
    target = state["next_worker"]
    if target in ["research", "analyst"]:
        return target
    
    return END

workflow = StateGraph(MultiAgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)

# Edges: START -> Supervisor -> Worker -> Back to Supervisor
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "research": "researcher",
        "analyst": "analyst",
        END: END
    }
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyst", "supervisor")

app = workflow.compile()


# ==============================================================================
# 6. EXECUTION
# ==============================================================================

def run_supervisor_workflow(user_query: str):
    """Entry point to run and measure the supervisor multi-agent system."""
    metrics = ExecutionMetrics()
    print(f"\n🚀 Running Supervisor Multi-Agent System\nQuery: '{user_query}'")

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "next_worker": "supervisor",
        "instructions": "",
        "metrics": metrics,
    }

    final_state = app.invoke(initial_state)
    metrics.finalize()

    # Final answer
    print("\n" + "=" * 60)
    print("🏁 FINAL SYNTHESIZED ANSWER:")
    print("=" * 60)
    print(final_state["messages"][-1].content)
    
    # Print the quantitative scorecard
    metrics.print_summary("Centralized Supervisor (6.1)")
    return final_state, metrics


if __name__ == "__main__":
    # final_state, metrics = run_supervisor_workflow(
    #     "What is the current population of tokyo? "
    # )
        
    # final_state, metrics = run_supervisor_workflow(
    #     "What is 23% of 347383298 + 34% of 32833291?"
    # )

    final_state, metrics = run_supervisor_workflow(
        "Subtract the current population of Tokyo from Hyderabad's"
    )
        