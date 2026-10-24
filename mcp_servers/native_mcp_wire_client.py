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
    

    def close(self):
        """Cleanly terminate the server subprocess."""
        if self.proc:
            print("\n🛑 [Client] Shutting down server...")
            self.proc.terminate()
            self.proc.wait()


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    server_path = str(current_dir / "raw_math_server.py")

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

        # Step 3: Discover available tools
        tools_res = client.send_request("tools/list")

        tools = tools_res.get("result", {}).get("tools", [])
        print(f"\nDiscovered {len(tools)} tools:")
        for t in tools:
            print(f"  - {t['name']}: {t['description']} (Schema: {t['inputSchema']['properties']})")
        

        # Step 4: Execute a tool call
        call_res = client.send_request("tools/call", {
            "name": "multiply",
            "arguments": {"a": 12, "b": 15}
        })
        output = call_res["result"]["content"][0]["text"]
        print(f"\n Tool Execution Output: 12 * 15 = {output}")

    finally:

        client.close()
