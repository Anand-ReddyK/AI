"""
Decentralized Swarm with Dynamic Agentic Handoffs.
- Peer-to-peer delegation via LangGraph Command(goto=...)
- Handoff tools as first-class routing primitives
- Cycle / infinite ping-pong mitigation (MAX_HANDOFFS)
- Performace metric tracking
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Dict, Any, Literal

# Ensure Project Root is in sys.path for imports 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.errors import ParentCommand

from utils.llm_provider import get_chat_model, extract_text_from_response
from utils.metrics import ExecutionMetrics
from agents.multi_agent_system.supervisor_pattern import search_web, calculate, print_worker_trajectory


MAX_HANDOFFS = 6

# ==========================================================================
# 2. HANDOFF TOOLS (ROUTING SIGNALS)
# ==========================================================================

@tool
def transfer_to_analyst(
        instructions: str,
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
    """
    Transfer control to the Analyst Agent when numerical data needs computation,
    precentages, arithmetic, or quantitative operations.
    """
    print(f"\n🤝 [HANDOFF TRIGGERED] Transferring directly to -> ANALYST")
    print(f"   Instructions: '{instructions}'")
    return Command(
        graph=Command.PARENT,  # 👈 Tells LangGraph to jump out of create_react_agent
        goto="analyst",
        update={
            "messages": [
                ToolMessage(content=f"Handoff executed: forward to Ananlyst.", tool_call_id=tool_call_id),
                HumanMessage(content=f"[Handoff from Researcher]: {instructions}")
            ],
            "handoff_count": 1 # Incremented in state reducer
        }
    )


@tool
def transfer_to_researcher(
        instructions: str,
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
    """
    Transfer control to Researcher Agent when extrnal afcts, documents,
    or web information need to be retrived before calculations can proceed.
    """
    print(f"\n🤝 [HANDOFF TRIGGERED] Transferring directly to -> RESEARCHER")
    print(f"   Instructions: '{instructions}'")
    return Command(
        graph=Command.PARENT,  # 👈 Tells LangGraph to jump out of create_react_agent
        goto="researcher",
        update={
            "messages": [
                ToolMessage(content=f"Handoff executed: forward to Researcher.", tool_call_id=tool_call_id),
                HumanMessage(content=f"[Handoff from Analyst]: {instructions}")
            ],
            "handoff_count": 1
        }
    )


# ======================================================================================
# 2. STATE DEFINITION
# ======================================================================================

def reduce_handoffs(current: int, increment: int) -> int:
    """Tracks cumulative peer-to-peer handoffs to prevent infinite cycles."""
    return current + increment

class SwarmState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    metrics: ExecutionMetrics
    handoff_count: Annotated[int, reduce_handoffs]
    

# ======================================================================================
# 4. SWARM AGENT NODES
# ======================================================================================

# Global model
model = get_chat_model(temperature=0.0)

# Bind respective toolsets
researcher_tools = [transfer_to_analyst, search_web]
analyst_tools = [transfer_to_researcher, calculate]

researcher_agent = create_react_agent(
    model=model,
    tools=researcher_tools,
    prompt=(
            "You are a specialized Research Agent in a peer-to-peer swarm.\n"
            "1. Use 'search_web' tool to find factual data.\n"
            "2. If calculations, precentages, or math operations are required on the found data, "
            "DO NOT attempt the math yourself. Call 'transfer_to_analyst' with the exact instructions.\n"
            "3. If all factual information is ready and no futrther math is needed, provide the final answer."
        )
)
analyst_agent = create_react_agent(
    model=model,
    tools=analyst_tools,
    prompt=(
        "You are a Quantitative Analyst Agent in a peer-to-peer swarm.\n"
        "1. Use 'calculate' to compute arithmetic expressions, percentages, and metrics.\n"
        "2. If you lack the base numbers or facts needed for the calculation, call 'transfer_to_researcher'.\n"
        "3. Once calculations are complete, synthesize and present the final answer clearly."
    )
)

def run_agent_with_metrics(agent, agent_name: str, state: SwarmState):
    handoffs = state.get("handoff_count", 0)
    print(f"\n[{agent_name} Node] 🔬 Active (Handoff count: {handoffs})")

    collected_messages = []
    try:
        # Stream updates so tokens & calls are captured in real-time
        for chunk in agent.stream({"messages": state["messages"]}, stream_mode="updates"):
            for node, update in chunk.items():
                if "messages" in update:
                    for m in update["messages"]:
                        collected_messages.append(m)
                        if isinstance(m, AIMessage):
                            state["metrics"].record_call(m)

        # Normal completion (no handoff)
        print_worker_trajectory(agent_name, [state["messages"][0]] + collected_messages)
        return {"messages": collected_messages}

    except ParentCommand as pc:
        # Handoff triggered! Print the trace up to the handoff, then return the Command
        print_worker_trajectory(agent_name, [state["messages"][0]] + collected_messages)
        return pc.args[0]


def researcher_node(state: SwarmState):
    return run_agent_with_metrics(researcher_agent, "Researcher", state)


def analyst_node(state: SwarmState):
    return run_agent_with_metrics(analyst_agent, "Analyst", state)



# def researcher_node(state: SwarmState) -> Command[Literal["analyst", "__end__"]] | Dict[str, Any]:
#     """
#     Specialized Researcher Node:
#     Searches web for facts. If math is required, directly hands off to Analyst.
#     """
#     handoffs = state.get("handoff_count", 0)
#     print(f"\n[Researcher Node] 🔬 Active (Handoff count: {handoffs})")

#     n = len(state["messages"])
#     print(f"LENGTH OF MESSAGES SENT TO AGENT: {n}")
#     prev_len = max(n - 1, 0)

#     # Only pass the messages to the agent
#     result = researcher_agent.invoke({"messages": state["messages"]})

#     # Record metrics
#     for msg in result["messages"][prev_len:]:
#         if isinstance(msg, AIMessage):
#             state["metrics"].record_call(msg)
    
#     # 🔍 Print internal ReAct trajectory
#     print_worker_trajectory("Researcher", result["messages"][prev_len:])
    
#     return {"messages": result["messages"]}


# def analyst_node(state: SwarmState) -> Command[Literal["researcher", "__end__"]] | Dict[str, Any]:
#     """
#     Specialized Analyst Node:
#     Computes math. If missing factual data, directly hands off to Researcher.
#     """
#     handoffs = state.get("handoff_count", 0)
#     print(f"\n[Analyst Node] 📊 Active (Handoff count: {handoffs})")

#     n = len(state["messages"])
#     print(f"LENGTH OF MESSAGES SENT TO AGENT: {n}")
#     prev_len = max(n - 1, 0)
    
#     result = analyst_agent.invoke({"messages": state["messages"]})
    
#     for msg in result['messages'][prev_len:]:
#         if isinstance(msg, AIMessage):
#             state["metrics"].record_call(msg)
    

#     # 🔍 Print internal ReAct trajectory
#     print_worker_trajectory("Analyst", result["messages"][prev_len:])

#     return {"messages": result["messages"]}



# ======================================================================================
# 5. GRAPH COMPILATION (DECENTRALIZED TOPOLOGY)
# ======================================================================================

workflow = StateGraph(SwarmState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)

# START goes to researcher. No static edge between researcher and analyst.
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", END)
workflow.add_edge("analyst", END)

app = workflow.compile()


# ======================================================================================
# 6. EXECUTION (TESTING THE PEER-TO-PEER HANDOFF SWARM)
# ======================================================================================

def run_swarm_workflow(user_query: str):
    metrics = ExecutionMetrics()
    print(f"\n🚀 Running Decentralized Swarm Multi-Agent System\nQuery: '{user_query}'")

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "metrics": metrics,
        "handoff_count": 0
    }

    # Run the workflow
    final_state = app.invoke(initial_state)
    metrics.finalize()
    
    # Print results
    print("=" * 80)
    print(f"\n[FINAL ANSWER]\n")
    print("=" * 80)
    print(extract_text_from_response(final_state["messages"][-1]))
    

    metrics.print_summary("Decentralized Swarm (6.2)")

    return final_state, metrics


if __name__ == "__main__":
    run_swarm_workflow(
        "Subtract the current population of Tokyo from Hyderabad's"
    )

    # run_swarm_workflow(
    #     "What is 23% of 347383298 + 34% of 32833291?"
    # )