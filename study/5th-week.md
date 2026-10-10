# Proposed Week 5 Execution Roadmap

## Milestone 5.1: The Mechanics of Native Tool Calling (No Frameworks Yet)

Master what happens at the HTTP payload level before hiding it behind abstractions.

### Concepts to master:

- OpenAI / Groq tool calling specification: tool_choice="auto", tools=[{"type": "function", ...}].

- Parallel tool calls: Modern models often return 2 or 3 tool calls in a single turn (e.g. querying revenue for 2023 AND 2024 simultaneously). If your loop only executes tool_calls[0], your agent silently drops information!

- Tool execution error handling: Feeding exceptions back to the model as observations ("Tool error: XYZ") so the agent can self-correct instead of crashing your Python script.

### Hands-on Task:

- Refactor `loop_chain_agent.py` to support parallel tool calls and robust exception recovery.

- Replace raw dictionary JSON schemas with Pydantic model reflection (`TypeAdapter` or `@tool` decorators) to auto-generate OpenAPI/JSON schemas.

## Milestone 5.2: The ReAct Pattern & Why Raw while-Loops Break

Analyze why your custom loop in `loop_chain.py` is great for learning, but fails in production.

### Concepts to master:

- **ReAct (Reasoning + Acting)**: Yao et al. (2022). Interleaving reasoning traces ("Thought:") with task-specific actions ("Action:" $\to$ "Observation:").
- **The 4 Classic Failure Modes of Agents**:
    - **Infinite Loops**: Repeating the same failed tool call with slightly different parameters.
    - **Context Window Exhaustion**: Churning through 10 tool calls until token limits are exceeded.
    - **Hallucinated Tool Arguments**: Calling a tool with keys that don't match the schema.
    - **State Loss**: If the server restarts midway through a multi-step task, the entire execution is lost.

- **Interview Takeaway:**
"Why not just use an infinite while-loop for an agent?" Answer: Raw while-loops lack deterministic state tracking, breakpoint resumption, human approval checkpoints, streaming hooks, and cyclic recursion limits.

## Milestone 5.3: Transition to LangGraph (State Machines as Code)

Why the industry migrated from LangChain chains to LangGraph state machines.

### Concepts to master:

- **Core LangGraph primitives:**
    - **State**: A shared typed schema (usually a `TypedDict` or `Pydantic` model) passed between nodes.
    - **Nodes**: Pure Python functions that receive the current state and return state updates.
    - **Edges**: Deterministic connections (`graph.add_edge("node_a", "node_b")`).
    - **Conditional Edges**: Routing logic (`graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})`).
    - **ToolNode**: Built-in execution of tool calls.

### Hands-on Task:

- Re-implement your shopping/HR agent from `loop_chain_agent.py`
 as a clean 3-node LangGraph:

```mermaid
[Start] --> [Agent Node] --(has tool calls?)--> [Tool Node] --> [Agent Node]
                                |
                         (no tool calls)
                                v
                              [End]
```

## Milestone 5.4: Bridge Week 4 to Week 5 — Building an "Agentic RAG" System

Implementing the vision from your `TODO.md`.

In Week 4, you built query rewriting, HyDE, hybrid retrieval, and reranking as isolated scripts in `pre_retrieval/` and `hybrid_rag/`. Now, wrap them into an intelligent system:

### Architecture (Adaptive / Corrective RAG):

- **Router Node**: The agent inspects the user query:
    - Simple greeting / chitchat $\to$ Answer immediately (no retrieval cost).
    - Domain query about COVID documents $\to$ Route to Local Qdrant Tool (your Week 4 `hybrid_retriever.py`).
    - General / real-time knowledge $\to$ Route to Web Search Tool (e.g. DuckDuckGo or Tavily).
- **Document Grader Node (Self-RAG)**:
    - After retrieval, an LLM evaluator grades: Are these retrieved chunks actually relevant to the question?
    - If Yes $\to$ Route to Generator with context.
    - If No $\to$ Trigger your `query_transformer.py` (HyDE / Rewrite) to reformulate the query and retry retrieval!

**Why this will stand out in interviews**: Freshers build basic RAG. Candidates who build Self-Correcting Agentic RAG with fallback to Web Search get fast-tracked into senior engineering interviews.

## Milestone 5.5: Short-Term Memory, Checkpointing & Human-in-the-Loop (HITL)

The enterprise pattern that separates toys from production agents.

### Concepts to master:

- **Short-term Memory (Checkpointers)**: Using `MemorySaver` or `SqliteSaver` in LangGraph with `thread_id` so conversations retain context across turns without manual message appending.
- **Human-in-the-Loop (Interrupts)**: `interrupt()` in LangGraph. Pausing the graph before executing high-impact tools (e.g., sending an email, deleting records, or placing an order) to wait for user confirmation.

### Hands-on Task:

Build a mini-agent where one tool requires explicit human approval (y/n) in the CLI before the agent resumes and updates its state.

## Milestone 5.6: The Week 5 Capstone Deliverable

An Autonomous Research & Synthesis Agent.

### The Objective:

An agent that takes an open-ended research question (e.g., "Compare the clinical efficacy and regulatory status of Drug X vs Drug Y").

### Graph Flow:

- **Plan**: Decomposes the research topic into sub-questions.
- **Retrieve**: Queries local Qdrant vectors for proprietary papers + searches the live web for recent updates.
- **Evaluate & Deduplicate**: Filters out low-signal search results.
- **Synthesize & Cite**: Produces a structured report complete with source citations and confidence notes.

### Deliverables:

- Clean repository with a modular architecture: agent/, tools/, graph.py.
- A test suite verifying that the agent terminates cleanly within bounded recursion limits.


# Week 5 Study Notes

## Milestone 5.1: The Mechanics of Native Tool Calling (No Frameworks Yet)

- **"How does Function / Tool Calling actually work under the hood without frameworks?"**
    - The LLM does not execute functions itself. The client passes JSON schemas describing available tools. When prompted, the model emits structured JSON arguments expressing its intent to call a tool. The client environment executes the function locally in Python, returns the observation in a message with `role: "tool"` matching the `tool_call_id`, and prompts the LLM again to synthesize the answer.

- **"Why must we support Parallel Tool Calling, and what happens if an agent ignores it?"**
    - Modern models often emit multiple tool calls in a single turn (e.g., fetching prices for two stocks simultaneously). The OpenAI/Groq wire protocol strictly requires that **every** `tool_call_id` produced by the assistant must have an associated `role: "tool"` response message in sequence. If your loop only executes `tool_calls[0]`, unhandled tool calls cause subsequent API calls to fail with HTTP 400 Bad Request.

- **"Why use Pydantic reflection over raw JSON schema dictionaries?"**
    - Handcrafted schema dictionaries are verbose, untyped, and break silently when function signatures change. Pydantic's `BaseModel.model_json_schema()` auto-generates OpenAPI-compliant parameter schemas from type annotations and `Field(description=...)`, while `model_validate()` validates incoming LLM arguments at runtime.

- **"How does Defensive Tool Execution enable agent self-correction?"**
    - Instead of letting Python raise unhandled exceptions (e.g., invalid mathematical syntax or missing database keys) that crash the script, the registry catches exceptions and returns the error string as the tool observation. The LLM reads the error in the subsequent turn and autonomously corrects its arguments, tries a fallback approach, or notifies the user.