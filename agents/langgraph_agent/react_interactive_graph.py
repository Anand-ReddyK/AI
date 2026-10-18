"""
ReAct Agent using LangGraph
Features:
- TypedDict State with add_messages reducer
- Native tool binding (bind_tools)
- Conditional edge via custom_router
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

from agents.langgraph_agent.react_graph import get_stock_price, calculate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")


# ========================================================
# 1. Tool Definition using LangChain's @tool decorator
# ========================================================
@tool
def buy_shares(ticker: str, quantity: int) -> str:
    """
    Execute a stock purchase.
    High-impact action requiring Human confirmation.
    """

    return f"SUCCESS: Bought {quantity} shares of {ticker.upper()}"

@tool
def ask_human(question: str) -> str:
    """
    Use this tool when agent needs clarification, missing parameters, or human
    confirmation.
    """
    # Calling interrupt() directly inside the tool pauses execution
    answer = interrupt({
        "question": question
    })

    return f"User answer: {answer}"


tools = [get_stock_price, calculate, buy_shares, ask_human]

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
        "5. At any point you need to get more info from the user or have a conversation with the user, clarification, missing parameters, or human confirmation"
        "you SHOULD use ask_human tool, if you provide a direct response then it will be considered as your final answer (REMEMBER THAT)"
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
# 4. Human Reviw Node using interrupt()
# ========================================================
def human_review_node(state: AgentState):
    """
    Pauses the graph execution and requests human input.
    When resumend with a Command(resume=...), execution continues right here.
    """

    last_message = state["messages"][-1]
    pending_tool_calls = last_message.tool_calls

    # Pause graph and yeild control back to the caller with prompt info
    user_decision = interrupt({
        "prompt": "Approval Required!",
        "pending_actions": pending_tool_calls,
        "instructions": "Type 'yes' to proceed, or provide specific feedback/choice."
    })

    # Once resumend, process the human's decision:
    if str(user_decision).lower().strip() == "yes":
        # Approved: Route to tool node to execute
        return Command(goto="tools")
    else:
        # Rejected / Human provided a different opinion:
        # Inject human response as a tool observation so agent can adapt!
        rejection_messages = []
        for call in pending_tool_calls:
            rejection_messages.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=f"Action declined by human. Feedback/Decision: {user_decision}"
                )
            )

        # Route back to agent node with the rejection feedback
        return Command(goto="agent", update={"messages": rejection_messages})


# ========================================================
# 5. Custom Conditional Edge Function
# ========================================================
def custom_router(state: AgentState) -> str:
    """
    Inspects tool calls and routes accordingly:
    - If buy_shares is called -> 'human_review'
    - If other tools are called -> 'tools'
    - If no tools are called -> END
    """
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"

    # Check if any tool call is sensitive
    for call in last_message.tool_calls:
        if call["name"] == "buy_shares":
            return "human_review"

    return "tools"

# ========================================================
# 6. Constructing the Graph For Interactive Execution Runner
# ========================================================
workflow = StateGraph(AgentState)

# Add the two core nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("human_review", human_review_node)

# Add Edges
# START -> always goes to agent
workflow.add_edge(START, "agent")

# Agent -> Conditionally routes to "tools" OR END
workflow.add_conditional_edges(
    "agent",
    custom_router,  # Built-in router: checks if last message has tool_calls
    {
        "tools": "tools",
        "human_review": "human_review",
        "end": END
    }
)

# Tools -> always routes back to agent (the cycle)
workflow.add_edge("tools", "agent")


# Checkpointer is REQUIRED for HITL (tracks state across pauses/resumes)
checkpointer = MemorySaver()

# Compile the graph into a runnable executable
app = workflow.compile(checkpointer=checkpointer)


def run_interactive_session(query: str, thread_id: str = "session-101"):
    print("=" * 70)
    print(f"[USER QUERY]: {query}")
    print("=" * 70)

    initial_state = {
        "messages": [HumanMessage(content=query)]
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        # Run the graph until it reaches END Node or hits an INTERRUPT
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                # Guard against non-dict events (like '__interrupt__' tuples)
                if not isinstance(state_update, dict):
                    continue

                messages = state_update.get("messages", [])
                
                for msg in messages:
                    # 1. Output from Agent Node
                    if node_name == "agent":
                        # Extract reasoning if the model emitted it
                        reasoning = (
                            msg.additional_kwargs.get("reasoning_content")
                            or getattr(msg, "reasoning", None)
                        )
                        if reasoning:
                            print(f"\n[Thought]:\n{reasoning.strip()}")

                        # Extract tool calls requested
                        if msg.tool_calls:
                            print(f"\n[Action]: Calling {len(msg.tool_calls)} tool(s)...")
                            for call in msg.tool_calls:
                                print(f"   -> {call['name']}({call['args']})")

                    # 2. Output from Tools Node
                    elif node_name == "tools":
                        print(f"[Observation]: {msg.content}")

        # Check if the graph is paused
        snapshot = app.get_state(config)
        if not snapshot.next:
            # No next node -> Graph completed successfuly
            print("\n[Final Answer]:", snapshot.values["messages"][-1].content)
            break

        # Graph is paued at an interrupt
        interrupt_data = snapshot.tasks[0].interrupts[0].value
        print("\n----- AGENT ASKS -----")
        if "question" in interrupt_data:
            print(interrupt_data["question"])
        else:
            print(interrupt_data.get("prompt", "Approval Required!"))
            for action in interrupt_data.get("pending_actions", []):
                print(f"  -> Action: {action['name']}({action['args']})")
            print(f"Instructions: {interrupt_data.get('instructions', '')}")
        
        # Get human reply from CLI/UI
        user_response = input("Your Reply => ")

        # Prepare to resume on the next loop iteration
        initial_state = Command(resume=user_response)


if __name__ == "__main__":
    # Test 1: Multi tools needed
    # run_interactive_session("What is the price of 5 shares of Apple (AAPL) plus a 15 dollar transaction fee?")

    # # Test 2: Error handling
    # run_interactive_session("What is the stock price of Tesla?")

    # # Test 3: Incomplete query (needs to use ask_human tool)
    # run_interactive_session("I want to know the price of a stock.")

    # Test 4: Incomplete query and custom router check
    run_interactive_session("I want to buy some stocks.")

