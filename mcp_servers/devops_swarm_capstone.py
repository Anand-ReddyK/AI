"""
Milestone 6.6 Capestone: Automated DevOps & Security Triage Swarm.
Uses all 3 MCP Primitives:
    - Fetches Prompt Runbook from MCP Server
    - Injects Live Telemetry Resource from MCP Server
    - Equips Auditor Agent with dynamic MCP Tools
    - Multi-Agent Consensus: Author -> Security Critic -> SRE Lead
"""

import sys
import asyncio
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from utils.llm_provider import get_chat_model, extract_text_from_response
from utils.metrics import ExecutionMetrics
from agents.multi_agent_system.supervisor_pattern import print_worker_trajectory

MAX_REVISIONS = 2

# ==============================================================================
# 1. STATE & SCHEMAS
# ==============================================================================

class CriticOutput(BaseModel):
    """Structured audit verdict from the Security Critic."""
    status: Literal["PASSED", "REVISE"] = Field(
        description="'PASSED' if all security violations and cost waste calculations are mathematically accurate. 'REVISE' if numbers don't match or violations were missed."
    )
    critique: str = Field(description="Actionable verification feedback.")
    confidence_score: float = Field(description="Confidence rating from 0.0 to 1.0.")


class DevOpsState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    telemetry_resource: str
    audit_findings: str
    critic_verdict: str
    critic_feedback: str
    confidence_score: float
    remediation_plan: str
    revision_count: int
    metrics: ExecutionMetrics


# ==================================================================================
# 2. PROMPTS
# ==================================================================================

CRITIC_PROMPT = """
You are a Strict Cloud Security & Financial Compliance Auditor.
Inspect the findings from the Infrastucture Auditor.
1. Check if idle instances has their waste computed correctly via 'calculate_waste'.
2. Cheeck if all exposed ports were evaluated via 'check_port_exposure'.
3. If any calculation is missing or hallucinated, reject with status='REVISE' and a 'critique' explaining the errors.
4. If all cost waste and port violations are strictly verified, issue status='PASSED'.

IMPORTANT:
Provide all three fields for every response
"""

SRE_LEAD_PROMPT = """
You are the Pinciple SRE Incident Commander.
Your job is to synthesize the verified findings and produce an executive Incident Remediation Report.
Include 1. Executive Summary, 2. Financial Waste Breakdown, 3. Immediate Bash/CLI Fix Commands, 4. Security Rating.
"""


async def build_devops_swarm():
    # ==================================================================================
    # 3. CONFIG  (MODEL, AGENT, & MCP CLIENT)
    # ==================================================================================
    server_script = PROJECT_ROOT / "mcp_servers" / "devops_server.py"

    print("🔌 [MCP Setup] Spawning devops_server.py over stdio...")
    mcp_client = MultiServerMCPClient({
        "devops": {
            "command": sys.executable,
            "args": [str(server_script)],
            "transport": "stdio"
        }
    })

    # Fetch Tools from MCP
    tools = await mcp_client.get_tools()
    print(f"✅ Loaded {len(tools)} remote tools into LangChain runtime:")
    for t in tools:
        print(f"   - {t.name}: {t.description.strip().splitlines()[0]}")
    
    model = get_chat_model(temperature=0.0)

    # Auditor ReAct Agent equipped with external MCP tools
    auditor_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""
        You are a Cloud Infrastructure Auditor.
        Analyze the telemetry data provided in context.
        **STRICT INSTRUCTIONS**:
        1. You have full permission to execute MCP Tools.
        2. Do NOT guess financial figures. Use 'calculate_waste' for all cost calculations.
        3. Do NOT assume security compliance. Use 'check_port_exposure' for all port verifications.
        4. Present a thorough breakdown of all findings.
        """
    )


    # ==================================================================================
    # 4. NODES
    # ==================================================================================


    # NODE 1: INFRASTRUCTURE AUDITOR (MCP TOOLS)

    async def auditor_node(state: DevOpsState):
        rev = state.get("revision_count", 0)
        print(f"\n🔍 [Auditor Node] Active (Revision attempt: {rev})")

        task = state["messages"][-1].content
        telemetry = state["telemetry_resource"]

        prompt = f"""
        Official Runbook Task:
        {task}

        Live Infrastructure Telemetry:
        {telemetry}
        """

        if state.get("critic_feedback"):
            prompt += f"\nPrevious Auditor Critique: {state['critic_feedback']}\nPlease re-verify and fix the flagged items."
        
        result = await auditor_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                state["metrics"].record_call(msg)
        
        print_worker_trajectory("Auditor (MCP Tools)", result["messages"])

        findings = extract_text_from_response(result["messages"][-1])
        return {"audit_findings": findings}


    # NODE 2: SECURITY CRITIC

    async def critic_node(state: DevOpsState):
        print("\n🧐 [Security Critic Node] Auditing findings against compliance policy...")
        critic_llm = model.with_structured_output(CriticOutput, include_raw=True)

        prompt = f"""
        Telemetry Provided:
        {state["telemetry_resource"]}

        Auditor Findings to Validate:
        {state["audit_findings"]}

        Audit all cost calculations and port security violations.
        """

        response = await critic_llm.ainvoke([
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=prompt)
        ])
        state["metrics"].record_call(response["raw"])
        verdict: CriticOutput = response["parsed"]

        print(f"   [Verdict] Status: {verdict.status} | Confidence: {verdict.confidence_score * 100:.0f}%")
        print(f"   [Critique] {verdict.critique}")

        return {
            "critic_verdict": verdict.status,
            "critic_feedback": verdict.critique,
            "confidence_score": verdict.confidence_score,
            "revision_count": state.get("revision_count", 0) + (1 if verdict.status == "REVISE" else 0)
        }
    

    # NODE 3: SRE LEAD (Final Plan)

    async def sre_lead_node(state: DevOpsState):
        print("\n🛠️ [SRE Lead Node] Drafting finalized incident remediation plan...")

        prompt = f"""
        Incident Task: {state["messages"][0].content}

        Verified Audit Finidings:
        {state["audit_findings"]}

        Compliance Status: {state['critic_verdict']} (Confidence: {state['confidence_score'] * 100:.0f}%)
        """

        response = await model.ainvoke([
            SystemMessage(content=SRE_LEAD_PROMPT),
            HumanMessage(content=prompt)
        ])
        state["metrics"].record_call(response)
        plan = extract_text_from_response(response)

        return {
            "remediation_plan": plan,
            "messages": [AIMessage(content=plan)]
        }
    

    # =================================================================================
    # 5. ROUTER (Consensus Loop Gaurd)
    # =================================================================================
    def router_decision(state: DevOpsState) -> Literal["auditor", "sre_lead"]:
        verdict = state.get("critic_verdict")
        revisions = state.get("revision_count", 0)

        if verdict == "REVISE" and revisions < MAX_REVISIONS:
            print(f"🔄 [Routing] Consensus not reached. Sending back to Auditor ({revisions}/{MAX_REVISIONS}).")
            return "auditor"
        
        if verdict == "REVISE":
            print(f"⚠️ [Circuit Breaker] Max revisions ({MAX_REVISIONS}) reached. Proceeding to SRE Lead.")
        else:
            print("✅ [Routing] Consensus achieved. Proceeding to SRE Lead.")
        
        return "sre_lead"
    
    # Graph
    workflow = StateGraph(DevOpsState)

    workflow.add_node("auditor", auditor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("sre_lead", sre_lead_node)


    workflow.add_edge(START, "auditor")
    workflow.add_edge("auditor", "critic")
    workflow.add_conditional_edges("critic", router_decision)
    workflow.add_edge("sre_lead", END)

    app = workflow.compile()

    return app, mcp_client


# =======================================================================================
# 6. EXECUTION HARNESS
# =======================================================================================

async def run_devops_capstone(incident_id: str, environment: str):
    metrics = ExecutionMetrics()
    print(f"\n🚨 [Alert Received] Triage triggered for {incident_id} [{environment}]")

    app, mcp_client = await build_devops_swarm()

    # STEP A: Fetch the standardized SRE Prompt Runbook from MCP
    print("\n📜 [MCP Pre-Flight] Fetching Prompt: 'incident_triage_runbook'...")
    prompt_msgs = await mcp_client.get_prompt(
        "devops",
        "incident_triage_runbook",
        arguments={"incident_id": incident_id, "environment": environment}
    )

    initial_instruction = prompt_msgs[0].content
    print(f"    [Runbook Loaded]:\n     {initial_instruction}")

    # STEP B: Read the Live Telementry Resource from MCP 
    print("\n📂 [MCP Pre-Flight] Reading Resource: 'cloud://fleet/telemetry'...")
    resources = await mcp_client.get_resources("devops", uris=["cloud://fleet/telemetry"])
    telemetry_data = resources[0].as_string()
    print(f"    [Telemetry Loaded]: {len(telemetry_data)} characters")

    # STEP C: Initialize Graph State
    inital_state = {
        "messages": [HumanMessage(content=initial_instruction)],
        "telemetry_resource": telemetry_data,
        "audit_findings": "",
        "critic_verdict": "",
        "critic_feedback": "",
        "confidence_score": 0.0,
        "revision_count": 0,
        "remediation_plan": "",
        "metrics": metrics
    }

    final_state = await app.ainvoke(inital_state)
    metrics.finalize()

    print("\n" + "=" * 80)
    print("🏁 FINAL INCIDENT REMEDIATION REPORT (CAPSTONE 6.6)")
    print("=" * 80)
    print(final_state["remediation_plan"])

    metrics.print_summary("Devops Cloud Security & Cost Triage Swarm (6.6)")


if __name__ == "__main__":
    asyncio.run(run_devops_capstone("alert-prod-2024-01", "production"))


    
    