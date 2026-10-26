"""
Autonomous LangGraph Agent with External MCP Tools.
Connects dynamically to production_server.py over stdio JSON-RPC.
"""

import sys
import asyncio
from pathlib import Path

# Ensure Project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from utils.llm_provider import get_chat_model, extract_text_from_response
from agents.multi_agent_system.supervisor_pattern import print_worker_trajectory


async def run_mcp_agent(query: str):
    server_script = PROJECT_ROOT / "mcp_servers" / "production_server.py"
    
    print("=" * 60)
    print("🚀 Initializing LangGraph Agent with External MCP Client")
    print(f"Connecting to: {server_script.name}")
    print("=" * 60)

    # 1. Configure the MultiSrverMCPClient to spawn the server over stdio
    client = MultiServerMCPClient({
        "rag_and_math_server": {
            "command": sys.executable,
            "args": [str(server_script)],
            "transport": "stdio"
        }
    })

    # 2. Dynamically fetch and adapt remote tools to LangChain tools
    print("\n🔍 Discovering remote tools over JSON-RPC (tools/list)...")
    tools = await client.get_tools()
    # print(tools)

    print(f"✅ Loaded {len(tools)} remote tools into LangChain runtime:")
    for t in tools:
        print(f"   - {t.name}: {t.description.strip().splitlines()[0]}")
    
    # 3. Crete the LangGraph agent
    model = get_chat_model(temperature=0.0)
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""
        You are a helpful research assistant connected to extranl MCP server.
        1. Use 'query_knowledge_base' to look up verified facts.
        2. Use 'calculate' to perform numerical computations.
        3. Synthesize your final answer clearly.
        """
    )

    # 4. Execute the query
    print(f"\n🧠 Running Agent Query: '{query}'\n")
    state = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

    # 5. Print the ReAct trajectory
    print_worker_trajectory("MCP Agent", state["messages"])

    # 6. Final Answer
    print("=" * 60)
    print("🏁 FINAL SYNTHESIZED ANSWER:")
    print("=" * 60)
    print(extract_text_from_response(state["messages"][-1]))


if __name__ == "__main__":
    test_query = "What is the population difference between Tokyo and Hyderabad?"
    asyncio.run(run_mcp_agent(test_query))