import time
from dataclasses import dataclass, field
from langchain_core.messages import AIMessage

@dataclass
class ExecutionMetrics:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0

    @property
    def totak_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def latency_seconds(self) -> float:
        return round(self.end_time - self.start_time, 3)


    def record_call(self, response: AIMessage):
        """Extracts token usage directly from model response metadata."""
        self.llm_calls += 1
        
        # Check standard LangChain usage_metadata (Groq, OpenAI, Google)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
        else:
            # Fallback for provider-specific response metadata dicts
            resp_meta = getattr(response, "response_metadata", {})
            token_usage = resp_meta.get("token_usage", {})
            self.input_tokens += token_usage.get("prompt_tokens", 0)
            self.output_tokens += token_usage.get("completion_tokens", 0)    
    
    def finalize(self):
        self.end_time = time.perf_counter()
    
    def print_summary(self, architecture_name: str = "Supervisor Pattern"):
        duration = round(self.end_time - self.start_time, 3)
        print("\n" + "=" * 60)
        print(f"📊 PERFORMANCE METRICS: [{architecture_name.upper()}]")
        print("=" * 60)
        print(f"⏱️  Total Wall-Clock Latency : {duration}s")
        print(f"🔄  Total LLM Invocations   : {self.llm_calls} calls")
        print(f"📥  Input (Prompt) Tokens   : {self.input_tokens}")
        print(f"📤  Output (Completion)     : {self.output_tokens}")
        print(f"📦  Total Tokens Consumed   : {self.input_tokens + self.output_tokens}")
        print("=" * 60 + "\n")
    