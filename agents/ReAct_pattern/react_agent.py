"""
ReAct Agent for performing reasoning and action steps.
Features:
- Explicit Thought -> Action -> Observation tracking
- Cycle Detection (prevents infinite thrashing loops)
- Observation truncation (prevents context blowout)
- Defensive recovery intervention
"""

import os
import sys
import json
from pathlib import Path
import hashlib
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-use Tool Registry
from agents.native_tool_calling.sample_tools import registry

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "qwen/qwen3.8-27b"
MODEL = "openai/gpt-oss-20b"

class CycleDetector:
    """
    Tracks tool calls across turns.
    Detects when an agent gets stuck calling identical tools with identical arguments.
    """
    def __init__(self, threshold: int = 2):
        self.call_history: List[str] = []
        self.threshold = threshold


    def _hash_call(self, tool_name: str, arguments: str) -> str:
        # Normalize JSON arguments to avoid false negatives due to whitespace
        try:
            parsed = json.loads(arguments)
            normalized = json.dumps(parsed, sort_keys=True)
        except Exception:
            normalized = arguments
        raw_signature = f"{tool_name}:{normalized}"
        return hashlib.md5(raw_signature.encode()).hexdigest()

    def record_and_check(self, tool_name: str, arguments: str) -> bool:
        """Returns True if thrashing is detected (same call >= threshold)."""
        call_hash = self._hash_call(tool_name, arguments)
        repeat_count = self.call_history.count(call_hash)
        if repeat_count >= self.threshold:
            return True
        
        self.call_history.append(call_hash)
        return False


def truncate_observation(observation: str, max_chars: int = 1500) -> str:
    """
    Prevents large tool payloads (e.g. big JSON tables or scraped HTML)
    from blowing out the context window.

    Naive Truncation
    """
    if len(observation) > max_chars:
        return observation[:max_chars] + f"\n...[Truncated: {len(observation) - max_chars} chars omitted]"
    return observation


def run_react_agent(user_query: str, max_turns: int = 6) -> str:
    """
    Executes a guarded ReAct loop:
    1. Directs model to reason (Thought) before acting.
    2. Detects infinite thrashing loops via CycleDetector.
    3. Injects steering prompts if the agent is stuck.
    4. Truncates large observations to prevent context blowout.
    """

    system_prompt = (
        "You are an autonomous research assistant operating under the ReAct framework.\n\n"
        "Rules:\n"
        "1. For each step, formulate a concise 'Thought' explaining what you know and what action is required.\n"
        "2. If real-time info or calculation is needed, call the appropriate tool.\n"
        "3. Carefully evaluate tool observations against your original goal.\n"
        "4. If a tool fails or repeats, DO NOT repeat the exact same call. Try a different parameter or explain the limitation.\n"
        "5. Once you have all necessary facts, provide a complete, verified final answer.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    tool_schema = registry.get_tool_schemas()
    cycle_detector = CycleDetector(threshold=2)

    print("=" * 70)
    print(f"[User Query]: {user_query}")
    print("=" * 70)

    for turn in range(1, max_turns + 1):
        print(f"\n--->[Turn {turn}/{max_turns}] Invoking Model.....")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_schema,
            tool_choice="auto",
            temperature=0.2
        )

        choice = response.choices[0]
        message = choice.message
        messages.append(message)

        # 1. Display the Thought / Reasoning
        thought = message.reasoning if hasattr(message, "reasoning") and message.reasoning else "No explicit reasoning provided."
        print(f"[Thought]:\n{thought.strip()}\n")

        # 2. Case A: Model decides to call tools (Action)
        if message.tool_calls:
            print(f"[*] Action Requested: {len(message.tool_calls)}\n")

            for idx, tool_call in enumerate(message.tool_calls, start=1):
                call_id = tool_call.id
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments

                print(f"     Action {idx}: {func_name}({func_args})")

                # Defensive check: Cycle / Thrashing Detection
                if cycle_detector.record_and_check(func_name, func_args):
                    warning_observation = (
                        f"SYSTEM INTERVANTION: You have called '{func_name}' with identical arguments multiple times "
                        f"without progress. Stop repeating this action. Use an alternative tool, change arguments, "
                        f"or explain to the user why the requested data is unavailable."
                    )

                    print(f"    [!] Cycle Detected for {func_name}! Injecting intervention.")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": warning_observation
                    })
                    continue

                # Execute normally through 5.1 registry
                raw_observation = registry.execute_tool(func_name, func_args)
                bounded_observation = truncate_observation(raw_observation, max_chars=1500)

                print(f"    Observation: {bounded_observation}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": bounded_observation
                })

            # ReAct continue to the next turn so model can reflect on observations
            continue

        if message.content:
            print("=" * 70)
            print(f"[Final Response]:\n{message.content}")
            print("=" * 70)
            return message.content

    print("\n[!!!] Execution halted: Maximum turns exceeded without reaching a final answer.")
    return "Failed to resolve query within step limit."


if __name__ == "__main__":
    user_query = "What is the stock price of Apple, and how much would 15 shares cost including a flat 25 dollar transaction fee?"
    final_answer = run_react_agent(user_query)
    print(f"\n[Final Answer]: {final_answer}")


