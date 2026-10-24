"""
A minimal, dependecy-free MCP server implementing JSON-RPC 2.0 over stdio.
"""

import sys
import json

def send_response(data: dict):
    """Write JSON-RPC message to stdout followed by a newline and flush."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def log_debug(msg: str):
    """Log to stderr so it does not interfere with the JSON stream on stdout."""
    sys.stderr.write(f"[Server Debug] {msg}\n")
    sys.stderr.flush()


def main():
    log_debug("Raw Math MCP Server started, awaiting requests on stdin....")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log_debug(f"Malformed JSON: {e}")
            continue

        method = req.get("method")
        req_id = req.get("id")

        # -------------------------------------------------
        # 1. Lifecycle Handshake: initialize
        # -------------------------------------------------
        if method == "initialize":
            log_debug("Handling 'initialize'")
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "raw-math-server",
                        "version": "1.0.0"
                    }
                }
            })
        

        # -------------------------------------------------------
        # 2. Lifecycle Notification: notification/initialized
        # -------------------------------------------------------
        elif method == "notification/initialized":
            log_debug("Client confirmed initialized. Ready to serve requests.")
            # Notifications DO NOT send a response back in JSON-RPC
        

        # -------------------------------------------------------
        # 3. Tool Discovery: tools/list
        # -------------------------------------------------------
        elif method == "tools/list":
            log_debug("Handling 'tools/list'")
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "add",
                            "description": "Add two integers.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "a": {
                                        "type": "integer",
                                        "description": "First integer."
                                    },
                                    "b": {
                                        "type": "integer",
                                        "description": "Second integer."
                                    }
                                },
                                "required": ["a", "b"]
                            }
                        },
                        {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "a": {
                                        "type": "integer",
                                        "description": "First integer."
                                    },
                                    "b": {
                                        "type": "integer",
                                        "description": "Second integer."
                                    }
                                },
                                "required": ["a", "b"]
                            }
                        }
                    ]
                }
            })
        
        # ---------------------------------------------------------------
        # 4. Tool Execution: tools/call
        # ---------------------------------------------------------------
        elif method == "tools/call":
            params = req.get("params", {})
            tool_name = params.get("name")
            args = params.get("arguments", {})

            log_debug(f"Executing tool '{tool_name}' with args: {args}")

            if tool_name == "add":
                res = args.get("a", 0) + args.get("b", 0)
                content = [{"type": "text", "text": str(res)}]
            elif tool_name == "multiply":
                res = args.get("a", 0) * args.get("b", 0)
                content = [{"type": "text", "text": str(res)}]
            else:
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found."
                    }
                })
                continue

            # MCP tool results are returned as a list of content blocks
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": content
                }
            })


if __name__ == "__main__":
    main()


