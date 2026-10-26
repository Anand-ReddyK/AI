"""
DevOps & Clous Security MCP Server
Exposes Infrastructure Telemetry (Resource), SRE Runbooks (Prompts), and Cost/Audit Calculators (Tools).
"""

import sys

try:
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:
    # pyrefly: ignore[missing-import]
    from mcp.server.mcpserver import MCPServer

server = MCPServer("devops-infra-server")

# ===========================================================================
# 1. RESOURCE (Passive Telemetry Snapshot)
# ===========================================================================

@server.resource("cloud://fleet/telemetry")
def get_fleet_telemetry() -> str:
    """Returns the live health, resource usage, and cost telemetry of cloud instances."""
    return """
    ACTIVE CLUSTER TELEMETRY SNAPSHOT:
    - instance_id: vm-prod-auth-01 | cpu_load: 96% | open_ports: [80, 443] | rate: $1.80/hr | status: RUNNING
    - instance_id: vm-prod-db-replica | cpu_load: 2% | open_ports: [22, 5432] | rate: $4.50/hr | status: IDLE (14 days)
    - instance_id: vm-dev-test-sandbox | cpu_load: 1% | open_ports: [22, 8080, 27017] | rate: $2.20/hr | status: IDLE (21 days)
    FIREWALL POLICY: Port 22 and Database ports (5432, 27017) MUST NOT be exposed to 0.0.0.0/0
    """


# ==============================================================================
# 2. PROMPT (Centralized SRE Runbook Template)
# ==============================================================================

@server.prompt("incident_triage_runbook")
def incident_runbook(incident_id: str, environment: str) -> str:
    """Renders the official SRE Incident Triage Runbook for an alert."""
    return  f"""
    PRIORITY INCIDENT TRIAGE RUNBOOK - Alert ID: {incident_id} | [{environment.upper()}]
    Instructions for Autonomous Triage Swarm:
    1. Audit the live fleet telemetry provided in context for security violations.
    2. Use 'calculate_waste' to compute the exact dollar cost of idle instances (24 hours/day).
    3. Use 'check_port_exposure' to flag high-risk ports.
    4. Fact-Checker must audit and approve the calculations.
    5. SRE Lead must produce a final remediation command list.
    """

# ==============================================================================
# 3. TOOLS (Dynamic Calculations & Policy Verifications)
# ==============================================================================

@server.tool()
def calculate_waste(hourly_rate: float, idle_days: float) -> str:
    """Calculate the wasted dollar spend for an idle cloud instance."""
    total_hours = idle_days * 24
    wasted = hourly_rate * total_hours
    return f"${wasted:.2f} (over {idle_days:.0f} days at ${hourly_rate}/hr)"

@server.tool()
def check_port_exposure(port: int) -> str:
    """Evaluartes whether an open network port violates security compliance rules."""
    high_risk = {
        22: "CRITICAL: SSH (Port 22) exposed to public internet. Violation of CIS Benchmark 5.2",
        27017: "CRITICAL: MangoDB (Port 27017) exposed without VPC isolation.",
        5432: "WARNING: PostgreSQL (Port 5432) open. Potential data exfiltration risk."
    }

    return high_risk.get(port, f"Port {port} complies with standard perimeter security policy.")


if __name__ == "__main__":
    server.run(transport="stdio")