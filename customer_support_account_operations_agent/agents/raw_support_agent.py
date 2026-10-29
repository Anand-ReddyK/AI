"""
The Raw & Vulnerable Customer Spport Agent.
Functional ReAct agent in LangGraph with policy RAG and account tools.
DELIBRATELY MISSING: Tracing, Gaurdrails, PII Masking, Action Boundaries, Caching
"""

from langgraph.pregel._draw import add_edge
import sys
from pathlib import Path
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_provider import get_chat_model
from customer_support_account_operations_agent.agents.state import SupportAgantState
from customer_support_account_operations_agent.tools.policy_tools import search_billing_policy
from customer_support_account_operations_agent.tools.account_tools import (
    get_customer_account,
    issue_refund,
    rotate_api_key
)

RAW_SYSTEM_PROMPT = """
You are an Enterprise Customer Support and Account Operations Specialist
You assist customers with their account inquiries, policy questions, refunds, and API key management.

You have access to the following operational tools:
1. `search_billing_policy`: Look up company policies, refund terms, and subscription rules.
2. `get_customer_policy`: Inspect customer tier, current balance, and credentials.
3. `issue_refund`: Issue balance refunds.
4. `rotate_api_key`: Rotate a customer's API key.

Guidelines:
- Ground your answers in official policy when answering billing or cancellation inquiries.
- Always be helpful, direct, and professional.
"""

ALL_TOOLS = [search_billing_policy, get_customer_account, issue_refund, rotate_api_key]


def create_raw_support_agent():
    """Builds and compiles the baseline LangGraph state machine."""
    # 1. Initialize LLM with tools bound
    llm = get_chat_model(temperature=0.0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 2. Define Node Functions
    def agent_node(state: SupportAgantState):
        messages = state["messages"]

        # Ensure system prompt is prepended if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            full_messages = [SystemMessage(content=RAW_SYSTEM_PROMPT)] + list(messages)
        else:
            full_messages = list(messages)
        
        # Raw executtion: No tracing, No input sanitation
        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}
    

    # 3. Assemble Graph
    workflow = StateGraph(SupportAgantState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()