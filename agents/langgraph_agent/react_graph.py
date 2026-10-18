"""
ReAct Agent using LangGraph
Features:
- TypedDict State with add_messages reducer
- Native tool binding (bind_tools)
- Conditional edge via tools_condition
- Built-in recursion limit protection
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")


# ========================================================
# 1. Tool Definition using LangChain's @tool decorator
# ========================================================

@tool
def get_stock_price(ticker: str) -> str:
    """Lookup current stock price for a given ticker symbol (e.g., AAPL, NVDA, GOOGL)."""
    mock_db = {
        "AAPL": "$234.50",
        "NVDA": "$128.20",
        "GOOGL": "$178.90",
        "MSFT": "$448.10",

    }
    cleaned_ticker = ticker.upper().strip()
    if cleaned_ticker in mock_db:
        return f"Price of {cleaned_ticker}: {mock_db[cleaned_ticker]}"
    return f"Error: Ticker '{cleaned_ticker}' not found in active listings."

@tool
def calculate(expression: str) -> str:
    """Evaluate basic mathematical expressions safely (e.g., '234.50 * 10 + 25')."""
    try:
        # Restricted eval for basic arithmetic
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Expression contains unsupported characters."
        result = eval(expression, {"__builtins__": None}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"


tools = [get_stock_price, calculate]

# ========================================================
# 2. State Definition
# ========================================================

class AgentState(TypedDict):
    """
    The state of the agent graph
    add_messages is a REDUCER: when a node returns {'messages': [new_message]},
    it appends to the list rather than overwriting it
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ========================================================
# 3. Model Setup & Agent Node
# ========================================================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.2
)

# Bind tools to the model: under the hood, converts them to Groq/OpenAI JSON schemas
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful, precise financial research assistant.\n"
        "1. For calculations or stock lookups, always use the provided tools.\n"
        "2. You can call multiple tools if needed.\n"
        "3. Once you have all required information, provide a consise, clear final answer.\n"
        "4. In case you are only able to get information for a partial answer or you couldn't get enough info for an answer then state that to the user.\n"
    )
)

def agent_node(state: AgentState):
    """
    Agent Node: Calls the LLM with the accumulated massage history.
    Returns: State update containing teh new assistant message.
    """
    messages = state["messages"]

    # Prepend system prompt if not already present
    if not isinstance(messages[0], SystemMessage):
        full_messages = [SYSTEM_PROMPT] + list(messages)
    else:
        full_messages = messages

    response = llm_with_tools.invoke(full_messages)

    # The return dict will be processed by the add_messages reducer
    return {"messages": [response]}


# ========================================================
# 4. Constructing the Graph For Simple Execution Runner
# ========================================================
workflow = StateGraph(AgentState)

# Add the two core nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Add Edges
# START -> always goes to agent
workflow.add_edge(START, "agent")

# Agent -> Conditionally routes to "tools" OR END
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Built-in router: checks if last message has tool_calls
    {
        "tools": "tools",
        END: END
    }
)

# Tools -> always routes back to agent (the cycle)
workflow.add_edge("tools", "agent")

# Compile the graph into a runnable executable
app = workflow.compile()


def run_query(query: str):
    print("=" * 70)
    print(f"[USER QUERY]: {query}")
    print("=" * 70)

    initial_state = {
        "messages": [HumanMessage(content=query)]
    }

    # config with recursion_limit protect againt infinite loops!
    config = {
        "recurssion_limit": 10,
    }

    #! Stream graph events as each node executes
    for event in app.stream(initial_state, config=config, stream_mode="values"):
        last_message = event["messages"][-1]

        # Pretty print based on message type
        msg_type = last_message.__class__.__name__
        if msg_type == "HumanMessage":
            continue
        elif msg_type == "AIMessage" and last_message.tool_calls:
            print(f"\n[AI Action Requested]: {len(last_message.tool_calls)} tool call(s)")

            for call in last_message.tool_calls:
                print(f"    -> {call['name']}({call['args']})")
        elif msg_type == "ToolMessage":
            print(f"\n[Tool Observation]: {last_message.content}")

        elif msg_type == "AIMessage" and last_message.content:
            print(f"\n[Final Response]:\n{last_message.content.strip()}\n")



if __name__ == "__main__":
    # Test 1: Multi tools needed
    run_query("What is the price of 5 shares of Apple (AAPL) plus a 15 dollar transaction fee?")

    # Test 2: Error handling
    run_query("What is the stock price of Tesla?")

