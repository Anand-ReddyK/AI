"""
Native OpenTelemtry-style Distributed Tracer.
Implements the core primitives: Trace ID, Span ID, Parent-Child nesting.
latency calculations (ms), and token telemetry.
"""

import time
import uuid
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional, Any


@dataclass
class Span:
    name: str
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    
    def set_tokens(self, input_tokens: int = 0, output_tokens: int = 0):
        """Attaches token consumption telemetry to this span."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def set_metadata(self, key: str, value: Any):
        """Attaches arbitrary runtime metadata (e.g., tool arguments, model name)."""
        self.metadata[key] = value



class NativeTracer:
    def __init__(self, trace_name: str = "agent_run"):
        self.trace_id = uuid.uuid4().hex[:16]
        self.trace_name = trace_name
        self.spans: list[Span] = []
        self._active_span_stack: list[Span] = []
        self.start_time = time.perf_counter()
    
    @contextmanager
    def start_span(self, name: str, metadata: Optional[dict[str, Any]] = None):
        """
        Context manager to measure and nest execution spans.
        Automatically links chiuld spans to their parent span.
        """
        span_id = uuid.uuid4().hex[:16]
        # If there is already an active span on the stack, it is the parent!
        parent_span_id = self._active_span_stack[-1].span_id if self._active_span_stack else None

        span = Span(
            name=name,
            span_id=span_id,
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            metadata=metadata or {}
        )

        self._active_span_stack.append(span)
        span.start_time = time.perf_counter()

        try:
            yield span
        except Exception as e:
            span.error = str(e)
            raise
        finally:
            span.end_time = time.perf_counter()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            self._active_span_stack.pop()
            self.spans.append(span)
    

    def print_trace_tree(self):
        """Prints a clean ASCII hierarchy of the entire trace."""
        total_duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        total_in_tokens = sum(s.input_tokens for s in self.spans)
        total_out_tokens = sum(s.output_tokens for s in self.spans)

        print(f"\n{'='*60}")
        print(f"TRACE: {self.trace_name} | ID: {self.trace_id}")
        print(f"TOTAL DURATION: {total_duration_ms} ms | TOKENS: {total_in_tokens} in / {total_out_tokens} out")
        print(f"{'='*60}\n")

        # Build parent -> children map
        children_map: dict[Optional[str], list[Span]] = {}
        for s in self.spans:
            children_map.setdefault(s.parent_span_id, []).append(s)
        

        def _print_span(span: Span, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            token_info = f" | {span.input_tokens} in / {span.output_tokens} out" if (span.input_tokens or span.output_tokens) else ""
            error_info = f" [FAILED: {span.error}]" if span.error else ""

            print(f"{prefix}{connector}[Span: {span.name}] ({span.duration_ms}ms {token_info}){error_info}")

            # Print metadata if present (liek query or tool args)
            child_prefix = prefix + ("    " if is_last else "│   ")
            if span.metadata:
                for k, v in span.metadata.items():
                    print(f"{child_prefix}    │   {k}: {v}")
            

            children = children_map.get(span.span_id, [])
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                _print_span(child, child_prefix, is_last=(i == len(children) - 1))
        
        # Root spans have parent_span_id = None
        root_spans = children_map.get(None, [])
        for i, root in enumerate(root_spans):
            _print_span(root, prefix="", is_last=(i == len(root_spans) -1))

        print(f"{'='*60}\n")


from langchain_core.callbacks import BaseCallbackHandler


class NativeTracerCallback(BaseCallbackHandler):
    """
    Bridges LangGraph execution events directly to our NativeTracer.
    """
    def __init__(self, tracer: NativeTracer):
        super().__init__()
        self.tracer = tracer
        self._span_contexts: dict[str, Any] = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id, **kwargs):
        model_name = serialized.get("name", "llm_call")
        ctx = self.tracer.start_span(f"LLM: {model_name}", metadata={"prompts_count": len(prompts)})
        span = ctx.__enter__()
        self._span_contexts[str(run_id)] = (ctx, span)

    def on_llm_end(self, response, *, run_id, **kwargs):
        if str(run_id) in self._span_contexts:
            ctx, span = self._span_contexts.pop(str(run_id))
            
            in_tokens = 0
            out_tokens = 0
            
            # 1. Modern LangChain usage_metadata (Gemini, modern OpenAI)
            if hasattr(response, "generations") and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            in_tokens += msg.usage_metadata.get("input_tokens", 0)
                            out_tokens += msg.usage_metadata.get("output_tokens", 0)

            # 2. Legacy fallback (Groq / older OpenAI)
            if in_tokens == 0 and out_tokens == 0:
                llm_output = getattr(response, "llm_output", {}) or {}
                usage = llm_output.get("token_usage", {}) or {}
                in_tokens = usage.get("prompt_tokens", 0)
                out_tokens = usage.get("completion_tokens", 0)

            span.set_tokens(input_tokens=in_tokens, output_tokens=out_tokens)
            ctx.__exit__(None, None, None)


    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id, **kwargs):
        tool_name = serialized.get("name", "tool_call")
        ctx = self.tracer.start_span(f"Tool: {tool_name}", metadata={"input": input_str[:120]})
        span = ctx.__enter__()
        self._span_contexts[str(run_id)] = (ctx, span)

    def on_tool_end(self, output: str, *, run_id, **kwargs):
        if str(run_id) in self._span_contexts:
            ctx, span = self._span_contexts.pop(str(run_id))
            span.set_metadata("output_preview", str(output)[:100] + "...")
            ctx.__exit__(None, None, None)


        