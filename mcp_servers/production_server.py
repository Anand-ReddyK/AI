"""
Production MCP Server using the Official Python SDK (MCPServer)
Expose Tools, Resources, and Prompts over standard stdio transport.
"""

from mcp.server.mcpserver import MCPServer

# Initialize the server
server = MCPServer("production-rag-tools")

# ==============================================================================
# 1. TOOLS (Model-Controlled: LLM decids to call these)
# ==============================================================================

@server.tool()
def calculate(expression: str) -> str:
    """
    Evaluvate a Mathematical arthimetic expression
    Example expression: 23 * 45', '(14000000 - 11000000) / 100'.
    """
    allowed_chars = set("0123456789+-*/(). %")

    if not set(expression).issubset(allowed_chars):
        return "Error: Expression contains disallowed characters."

    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)

    except Exception as e:
        return f"Error evaluating expression: {e}"


@server.tool()
def query_knowledge_base(topic: str, top_k: int = 3) -> str:
    """
    Retrive domain-specific facts and documentation from the knowledge base.
    """

    # Mock knowledge base (can be hooked to Qdrant or hybrid retriver)
    knowledge = {
        "tokyo": "Tokyo Metropolis population is approximately 14,178,000 (prefecture) and 37,036,000 (metro).",
        "hyderabad": "Hyderabad urban agglomeration population is approximately 11,069,000 to 11,564,000.",
    }
    key = topic.lower().strip()
    
    for k, v in knowledge.items():
        if k in key:
            return f"[Match Found]: {v}"
    
    return f"No direct records found for topic: '{topic}'"


# ==============================================================================
# 2. RESOURCES (Application-Controlled: Passive data read via URI)
# ==============================================================================

@server.resource("docs://system/stats")
def get_system_stats():
    """Returns the operational status,  version, and index staistics."""
    return (
        "System: Production MCP Server v2.0\n"
        "Active Indexes: Qdrant Hybrid (BM25 + DENSE)\n"
        "Staus: Healthy, stdio active"
    )


# ==============================================================================
# 3. PROMPTS (User-Controlled: Reusable prompot workflows)
# ==============================================================================

@server.prompt("compare_cities")
def city_comparison_prompt(city_a: str, city_b: str) -> str:
    """Generates a standard prompt template to compare two cities."""
    return (
        f"Please analyze and compare {city_a} and {city_b}.\n"
        f"1. Use 'query_knowledge_base' to fetch verified facts about population, climate, and key features for each city.\n"
        f"2. Use 'calculate' if there is a need to compute any mathematical operations regarding the data retrieved.\n"
        f"3. Sunthesize a clean mardown summary."
    )


if __name__ == "__main__":
    # Runs server listening on stdio
    server.run(transport="stdio")