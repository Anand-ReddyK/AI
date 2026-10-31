"""
Automated CI Regression Test Suite for Customer Support & Account Operations Agent.
Executes the agent across the Golden Benchmark Dataset, evaluates responses
using the calibrated LLM-as-a-Judge, and gates pull requests via Pytest assertions.
"""

import sys
import json
import time
import pytest
import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict
from langchain_core.messages import HumanMessage, AIMessage

# Ensure root directory is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.llm_provider import extract_text_from_response
from utils.config import PROVIDER_MODELS
from customer_support_account_operations_agent.database.seed_data import seed
from customer_support_account_operations_agent.agents.raw_support_agent import create_raw_support_agent
from customer_support_account_operations_agent.evaluation.judge import SupportAgentJudge, EvaluationVerdict
from customer_support_account_operations_agent.observability.native_tracer import (
    NativeTracer,
    NativeTracerCallback
)
from customer_support_account_operations_agent.observability.langfuse_tracer import (
    get_langfuse_callback
)

# ==========================================================================
# 1. Dataset Loader & Pytest Setup
# ==========================================================================

DATASET_PATH = Path(__file__).resolve().parent / "golden_support_dataset.json"

def load_golden_dataset() -> list[dict[str, Any]]:
    """Loads all test cases from the golden evaluation dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {DATASET_PATH}")
    
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


GOLDEN_TEST_CASES = load_golden_dataset()

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Initializes and seeds database before running CI evaluation."""
    print("\n[CI Fixture] Resetting SQLite test database with seed data...")
    seed()


@pytest.fixture(scope="session")
def agent_runner():
    """Compiles a single agent instance shared across the evaluation session."""
    return create_raw_support_agent()


@pytest.fixture(scope="session")
def judge():
    """Initializes the calibrated LLM Judge."""
    return SupportAgentJudge()


# In-memory scorecard accumulator for session summary
SESSION_SCORECARD = defaultdict(list)


# ===========================================================================
# 2. Helper: Extract Tool Execution Trajectory
# ===========================================================================

def extract_execution_trajectory(messages: list) -> tuple[str, list[dict[str, Any]]]:
    """
    Extracts the final response text and all intermediate tool calls from state messages.
    """
    executed_tools = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                executed_tools.append({
                    "tool_name": tc.get("name"),
                    "args": tc.get("args")
                })
    
    last_message = messages[-1]
    final_text = extract_text_from_response(last_message)
    return final_text, executed_tools

# =========================================================================
# 3. Pytest Test Cases (Parameterized)
# =========================================================================

@pytest.mark.parametrize(
    "test_case",
    GOLDEN_TEST_CASES,
    ids=[tc["test_id"] for tc in GOLDEN_TEST_CASES]
)
def test_agent_golden_benchmark(test_case: dict[str, Any], agent_runner, judge):
    """
    Executes the agent against a single golden test case and evaluates compliance.
    """
    test_id = test_case.get("test_id")
    category = test_case.get("category")
    user_input = test_case.get("user_input")

    # Local Native Tracer
    tracer = NativeTracer(trace_name=test_id + ":" + category + "_CI_Support_Agent")
    native_cb = NativeTracerCallback(tracer)
    callbacks = [native_cb]

    # Cloud Langfuse Tracer
    langfuse_cb = get_langfuse_callback()
    if langfuse_cb:
        callbacks.append(langfuse_cb)

    # 1. Execute Agent
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    final_state = agent_runner.invoke(
        initial_state,
        config={
            "callbacks": callbacks,
            "tags": ["ci-eval", category, test_id],
            "metadata": {"test_id": test_id, "suite": "golden_benchmark"}
        }
    )

    # Print local ASCII tree
    tracer.print_trace_tree()

    # Flush traces to Langfuse cloud
    if langfuse_cb:
        langfuse_cb._langfuse_client.flush()

    # 2. Extract Response & Tool Trajectoery
    actual_response, executed_tools = extract_execution_trajectory(final_state.get("messages", []))

    # 3. Evaluate via Calibrated LLM Judge
    verdict: EvaluationVerdict = judge.evaluate(
        test_case=test_case,
        actual_response=actual_response,
        executed_tools=executed_tools,
        experiment_tags=f"ci-{category}"
    )

    # 4. Record to Session Scorecard
    SESSION_SCORECARD[category].append({
        "test_id": test_id,
        "is_pass": verdict.is_pass,
        "factual_accuracy": verdict.factual_accuracy,
        "policy_adherence": verdict.policy_adherence,
        "forbidden_action_prevented": verdict.forbidden_action_prevented,
        "failure_reasons": verdict.failure_reasons,
        "rationale": verdict.thought_rationale
    })
    
    # 5. Assert Pass / Fail with Detailed Diagnostics
    error_msg = (
        f"\n[EVALUATION FAILED] {test_id} ({category})\n"
        f"Query: {user_input}\n"
        f"Agent Response: {actual_response}\n"
        f"Executed Tools: {executed_tools}\n"
        f"Judge Rationale: {verdict.thought_rationale}\n"
        f"Failure Reasons: {verdict.failure_reasons}"
    )

    assert verdict.is_pass, error_msg


# ==========================================================================
# 4. Standalone Runner & Markdown Scorecard Generator
# ==========================================================================

def run_evaluation_suite(categories: list[str] = None):
    """
    CLI runner to execute and print a structured scorecard without pytest.
    """
    print("=" * 80)
    print("STARTING CONTINUOUS EVALUATION HARNESS")
    print("=" * 80)

    seed()
    agent_provider = "groq"
    agent_model = None
    agent = create_raw_support_agent(provider=agent_provider, model_name=agent_model)

    judge_provider = "google"
    judge_model = None
    judge = SupportAgentJudge(provider=judge_provider, model_name=judge_model)

    cases = GOLDEN_TEST_CASES
    if categories:
        cases = [c for c in cases if c["category"] in categories]
    
    results = []
    print(f"Running evaluation over {len(cases)} test cases...\n")

    for idx, tc in enumerate(cases, 1):
        if idx % 5 == 0:
            print("[INTERUPT] 5 Test Cases Completed Pausing for 30 seconds")
            print("[RATE LIMIT PREVENTION]")
            time.sleep(30)
        test_id = tc["test_id"]
        print(f"[{idx}/{len(cases)}] Running {test_id} ({tc['category']})... ", end="", flush=True)

        # Local Native Tracer
        tracer = NativeTracer(trace_name=test_id + ":" + "_CI_Support_Agent")
        native_cb = NativeTracerCallback(tracer)
        callbacks = [native_cb]

        # Cloud Langfuse Tracer
        langfuse_cb = get_langfuse_callback()
        if langfuse_cb:
            callbacks.append(langfuse_cb)

        final_state = agent.invoke(
            {"messages": [HumanMessage(content=tc["user_input"])]},
            config={
                "callbacks": callbacks,
                "tags": ["ci-eval", tc['category'], test_id],
                "metadata": {"test_id": test_id, "suite": "golden_benchmark"}
            }
        )
        response, tools = extract_execution_trajectory(final_state["messages"])

        # Print local ASCII tree
        tracer.print_trace_tree()

        # Flush traces to Langfuse cloud
        if langfuse_cb:
            langfuse_cb._langfuse_client.flush()

        verdict = judge.evaluate(
            test_case=tc,
            actual_response=response,
            executed_tools=tools,
            experiment_tag="cli-eval"
        )

        status = "PASS" if verdict.is_pass else "FAIL"
        print(f"[{status}]")
        results.append({"test_case": tc, "verdict": verdict})
    
    # Print Summary Table
    total = len(results)
    passed = sum(1 for r in results if r["verdict"].is_pass)
    pass_rate = (passed / total) * 100 if total > 0 else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION SCORECARD SUMMARY")
    print("=" * 80)
    print(f"Total Cases: {total} | Passed: {passed} | Failed: {total - passed} | Pass Rate: {pass_rate:.1f}%\n")

    category_stats = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in results:
        cat = r["test_case"]["category"]
        category_stats[cat]["total"] += 1
        if r["verdict"].is_pass:
            category_stats[cat]["pass"] += 1
    
    header_fmt = "| {cat:<24} | {total:^7} | {passed:^7} | {rate:^10} |"
    divider    = "+--------------------------+---------+---------+------------+"

    print(divider)
    print(header_fmt.format(cat="Category", total="Total", passed="Passed", rate="Pass Rate"))
    print(divider)

    for cat, stats in category_stats.items():
        rate = (stats["pass"] / stats["total"]) * 100
        print(f"| {cat:<24} | {stats['total']:^7} | {stats['pass']:^7} | {rate:>8.1f}% |")
    
    print(divider)
    
    # List Failures
    failures = [r for r in results if not r["verdict"].is_pass]
    if failures:
        print("\n--- FAILURE DETAILS ---")
        for f in failures:
            tc = f["test_case"]
            v = f["verdict"]
            print(f"\n[FAILED]: {tc['test_id']} - {tc['category']}")
            print(f"Query: {tc['user_input']}")
            print(f"Reasons: {v.failure_reasons}")
            print(f"Judge Rationale: {v.thought_rationale[:200]}...")

    # Save test cases, verdicts, and metadata to JSON file
    agent_resolved_model = str(agent_model or PROVIDER_MODELS.get(agent_provider, {}).get("eval", "default"))
    judge_resolved_model = str(judge_model or PROVIDER_MODELS.get(judge_provider, {}).get("eval", "default"))

    # Sanitize slashes (e.g. 'openai/gpt-oss-20b' -> 'openai_gpt-oss-20b') so the OS doesn't treat them as directory paths
    safe_agent_model = agent_resolved_model.replace("/", "_")
    safe_judge_model = judge_resolved_model.replace("/", "_")

    output_filename = (
        f"baseline_evaluation_results_judge={safe_judge_model}_agent={safe_agent_model}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    output_path = Path(__file__).resolve().parent / output_filename


    export_payload = {
        "metadata": {
            "agent_version": "raw_support_agent_7.0",
            "agent_provider": agent_provider,
            "agent_model": agent_resolved_model,
            "evaluator_provider": judge.provider,
            "evaluator_model": judge_resolved_model,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_pct": round(pass_rate, 2),
            "category_summary": {
                cat: {
                    "total": stats["total"],
                    "passed": stats["pass"],
                    "pass_rate_pct": round((stats["pass"] / stats["total"]) * 100, 2)
                }
                for cat, stats in category_stats.items()
            }
        },
        "results": [
            {
                "test_case": r["test_case"],
                "verdict": r["verdict"].model_dump()
            }
            for r in results
        ]
    }

    with open(output_path, "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\n[Artifact Saved] Successfully saved evaluation metadata and {len(results)} test cases to:\n -> {output_path}")


if __name__ == "__main__":
    # By default, run a quick smoke test on 2 categories to save tokens during manual runs
    # Or pass None to run all 25 cases
    # run_evaluation_suite(categories=["policy_boundaries", "negative_constraints"])
    run_evaluation_suite()