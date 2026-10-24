"""
A rae MCP Client testing the JSON-RPC 2.0 handshake over stdio.
"""

import sys
import json
import subprocess
from pathlib import Path


class NativeMCPClient:
    def __init__(self, server_script_path: str):
        self.server_path = server_script_path
        self.proc = None
        self._req_id = 0
    

    def start(self):
        """Spawns the MCP server as a subprocess with piped stdio."""
        print(f"🚀 [Client] Launching MCP server: {self.server_path}")
        self.proc = subprocess.Popen(
            [sys.executable, self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr, # Pass stderr directly to terminal for debugging
            text=True,
            bufsize=1
        )
    

    def send_request(self, method: str, params: dict = None) -> dict:
        """Sends a JSON-RPC request and waits for the single-line response."""
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method
        }

        if params is not None:
            payload["params"] = params

        print(f"📧 [Client] Sending Request {self._req_id} -> {method}")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

        # Read response from server's stdout
        response_line = self.proc.stdout.readline()
        print(f"📧 [Server] Response {self._req_id} <- {response_line}")
        return json.loads(response_line.strip())
    

    def send_notification(self, method: str, params: dict = None):
        """Sends a JSON-RPC notification (no 'id', expects no response)."""
        payload = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params is not None:
            payload["params"] = params
        
        raw_msg = json.dumps(payload)

        print(f"📢 [Client Notification] {raw_msg}")
        self.proc.stdin.write(raw_msg + "\n")
        self.proc.stdin.flush()

    
    # ------------------------------------------------------------------------------
    # TOOLS
    # ------------------------------------------------------------------------------
    def list_tools(self) -> list:
        res = self.send_request("tools/list")
        return res.get("result", {}).get("tools", [])
    
    def call_tool(self, name: str, arguments: dict) -> str:
        res = self.send_request("tools/call", {"name": name, "arguments": arguments})
        return res.get("result", {}).get("content", [{}])[0].get("text", "")
    
    # ------------------------------------------------------------------------------
    # RESOURCES (Passive Content Read)
    # ------------------------------------------------------------------------------
    def list_resources(self) -> list:
        res = self.send_request("resources/list")
        return res.get("result", {}).get("resources", [])
    
    def read_resources(self, uri: str) -> str:
        res = self.send_request("resources/read", {"uri": uri})
        contents = res.get("result", {}).get("contents", [{}])
        return contents[0].get("text", "")
    
    # ------------------------------------------------------------------------------
    # PROMPTS (Reusable Workflows)
    # ------------------------------------------------------------------------------
    def list_prompts(self) -> list:
        res = self.send_request("prompts/list")
        return res.get("result", {}).get("prompts", [])
    
    def get_prompts(self, name: str, arguments: dict) -> str:
        res = self.send_request("prompts/get", {"name": name, "arguments": arguments})
        messages = res.get("result", {}).get("messages", [])

        if messages:
            return messages[0].get("content", {}).get("text", "")
        
        return ""    

    def close(self):
        """Cleanly terminate the server subprocess."""
        if self.proc:
            print("\n🛑 [Client] Shutting down server...")
            self.proc.terminate()
            self.proc.wait()


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    server_path = str(current_dir / "production_server.py")

    client = NativeMCPClient(server_path)
    client.start()

    try:
        # Step 1: Handshake (initialize)
        init_res = client.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        })

        # Step 2: Confirm handshake
        client.send_notification("notification/initialized")

        # =========================================================
        # 1. TEST TOOLS (Model-Controlled)
        # =========================================================
        print("\n" + "=" * 60)
        print("🛠️ 1. TESTING TOOLS")
        print("=" * 60)

        tools = client.list_tools()
        
        print(f"\nDiscovered {len(tools)} tools:")
        for t in tools:
            print(f"  - Tool: {t['name']} | Schema: {list(t['inputSchema']['properties'].keys())}")
        

        tool_result = client.call_tool("query_knowledge_base", {"topic": "tokyo", "top_k": 2})
        print(f"\nTool Execution Result: {tool_result}")


        # =========================================================
        # 2. TEST RESOURCES (Application-Controlled passive read)
        # =========================================================
        print("\n" + "=" * 60)
        print("📂 2. TESTING RESOURCES")
        print("=" * 60)
        
        resources = client.list_resources()
        print(f"\nDiscovered {len(resources)} resources")
        for r in resources:
            print(f"  - Resource: {r['name']} | URI: {r['uri']}")

        if resources:
            target_uri = resources[0]["uri"]
            content = client.read_resources(target_uri)
            print(f"\n   Content of '{target_uri}':\n{content}")
        

        # =========================================================
        # 3. TEST PROMPTS (User-Controlled workflows)
        # =========================================================
        print("\n" + "=" * 60)
        print("💬 3. TESTING PROMPTS")
        print("=" * 60)

        prompts = client.list_prompts()
        print(f"\nDiscovered {len(prompts)} prompt templates:")
        for p in prompts:
            arg_names = [a["name"] for a in p.get("arguments", [])]
            print(f"  - Prompt Template: {p['name']} | Arguments: {arg_names}")
        
        prompt_text = client.get_prompts("compare_cities", {"city_a": "Tokyo", "city_b": "Hyderabad"})
        print(f"\n   Rendered Prompt:\n{prompt_text}")



    finally:

        client.close()
