"""
Demonstrates the native tracer wrapping simulated agent setups:
Parent Span (Agent Node) -> Child Span (Qdrant Retrieval) -> Child Span (Reranking).
"""

import sys
from pathlib import Path
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from customer_support_account_operations_agent.observability.native_tracer import NativeTracer

def demo():
    tracer = NativeTracer(trace_name="demo_agent_cycle")

    # 1. Top-Level Span: Agent Though & Routing
    with tracer.start_span("agent_reasoning", metadata={"model": "gemini-3.5-flash"}) as s1:
        time.sleep(0.12) # Simulate LLM deciding time
        s1.set_tokens(input_tokens=300, output_tokens=45)
    
    # 2. Nested Child Span: Qdrant Policy Retrieval
        with tracer.start_span("tool_search_billing_policy", metadata={"query": "refund terms"}) as s2:
            time.sleep(0.08)  # Simulating dense + BM25 vector search
            
            # 3. Nested Sub-Child Span: Cross-Encoder Reranking
            with tracer.start_span("cross_encoder_rerank", metadata={"candidates": 10, "top_k": 3}) as s3:
                time.sleep(0.05)  # Simulating cross-encoder scoring
    
    # 4. Sibling Top-Level Span: Final Answer Generation
    with tracer.start_span("llm_final_synthesis", metadata={"model": "gemini-3.5-flash"}) as s4:
        time.sleep(0.25)  # Simulating final response generation
        s4.set_tokens(input_tokens=650, output_tokens=180)
        
    # Print the resulting OpenTelemetry span tree
    tracer.print_trace_tree()


if __name__ == "__main__":
    demo()