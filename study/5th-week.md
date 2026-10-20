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


## Milestone 5.2: The ReAct Pattern & Why Raw while-Loops Break

- **"What is the core difference between Chain-of-Thought (CoT) and ReAct?"**
    - **Concept**: Chain-of-Thought is purely internal reasoning in a static context with no external feedback. ReAct alternates internal reasoning with external interactions (actions) and external updates (observations). The reasoning traces guide and interpret the actions, while the environment observations ground the reasoning, preventing hallucination.
- **"Why not just use an unconstrained while True loop for an agent?"**
    - **Concept**: Raw loops lack deterministic state preservation, recursion checkpoints, cycle detection, streaming hooks, and human approval gates. If any step fails or hangs, the entire state is unrecoverable, and without cycle detection, models easily get trapped in infinite action loops.
- **"How do you detect and mitigate action loops (thrashing) in an agent?"**
    - **Concept**: You maintain an action history cache and compare incoming tool calls against previous attempts. If the agent calls the same tool with identical arguments twice in a row without making progress, the loop controller intervenes by injecting a steering prompt (e.g., instructing the agent to choose an alternative approach or summarize its blockage) or terminates the execution.
- **"How does ReAct mitigate the 'Lost in the Middle' context problem?"**
    - **Concept**: Instead of appending raw, multi-megabyte tool payloads directly into the global conversation history, a ReAct architecture encourages the agent to synthesize observations into concise thoughts or summaries before moving to the next action, keeping the working memory dense with high-signal facts.

### Control Flow Guarantees & Termination Conditions

Understand the defensive mechanisms required to keep an autonomous agent bounded:

- **Cycle Detection**: Tracking whether identical tool arguments or identical observations have repeated across consecutive turns.
- **Budget Thresholds**: Enforcing both recursion limits (step counts) and cumulative token/cost ceilings.
- **Confidence & Stop Conditions**: Establishing unambiguous termination criteria so the agent knows exactly when its mission is completed versus when it must yield back to the user with a polite failure explanation.

## Milestone 5.3: Transition to LangGraph (State Machines as Code)

- **"What is the fundamental architectural difference between LangChain Chains (DAGs) and LangGraph?"**
    - Legacy LangChain Chains are Directed Acyclic Graphs (DAGs) designed for single-pass pipelines ( Input → Prompt → LLM → Output ). They cannot naturally handle cyclic recursion, backtracking, or dynamic branching.

    - LangGraph models agentic architectures as cyclic State Machines. It treats loops as first-class citizens (Node A → Node B → Node A), tracks state deterministically across transitions, and allows conditional branching, parallel fan-outs, and persistence.

- **"What is a State Reducer in LangGraph, and why is `Annotated[Sequence[BaseMessage], add_messages]` essential?"**
    - In LangGraph, when a node returns `{messages: [new_message]}`, default dictionary behavior would completely overwrite the previous list with only the new message, wiping out all history!
    - A Reducer defines the merge strategy. The `add_messages` reducer tells LangGraph: "Do not overwrite. Instead, append new messages to the existing list, and if a message has an existing ID, update that message in-place."
- **"How does LangGraph solve the infinite loop / thrashing problem that plagues raw `while`-loops?"**
    - LangGraph enforces a hard, configurable `recursion_limit` (default: 25 steps) at the graph executor level. If nodes cycle beyond this limit, LangGraph raises a `GraphRecursionError` rather than allowing unconstrained API spend.
    - Furthermore, conditional edges allow engineers to inspect state attributes (e.g., error counts or visit frequencies) and deterministically route to a fallback or human-review node if an agent repeats an action.
- **"What is the difference between a deterministic edge and a conditional edge?"**

    - Deterministic Edge (add_edge("tools", "agent")): A fixed transition. Once the tools node finishes executing, execution is guaranteed to move to agent.
    - Conditional Edge (add_conditional_edges("agent", router_func, path_map)): A dynamic transition. It executes a Python callable (router_func) that inspects the current state (e.g., checking if the model returned tool_calls vs text content) and selects the next destination node from a mapping dictionary.

- **"How does Human-in-the-Loop (HITL) work in LangGraph, and why can't we just use Python's `input()`?"**
    - In production, agents run on asynchronous web servers or background workers. Blocking the thread with `input()` is impossible when a user might take minutes or hours to respond.
    - LangGraph solves this using **Checkpointers (`MemorySaver` / `SqliteSaver`)** and **`interrupt()`**:
        - `interrupt(payload)` raises a control-flow signal that pauses execution, serializes the full state to the checkpointer under a `thread_id`, and yields control back to the caller.
        - The server can then shut down or serve other requests.
        - When the user responds via UI/API, the client resumes the workflow using `app.stream(Command(resume=user_response), config={"configurable": {"thread_id": ...}})`, picking up execution exactly where it paused.

- **"What does LangChain's `@tool` decorator actually pass to the LLM?"**
    - The LLM never sees the Python function code. `@tool` uses Pydantic reflection to inspect three elements:
        1. **Function name** $\to$ Tool identifier (`"name": "get_stock_price"`).
        2. **Docstring** $\to$ Tool description (`"description": "..."`), which the LLM reads to decide *when* and *why* to invoke the tool.
        3. **Type hints** $\to$ JSON Schema parameter definitions (`"parameters": {"type": "object", ...}`).
    - When passed to `llm.bind_tools()`, these are formatted into the provider's native wire schema (OpenAI/Groq tool specification).


## Milestone 5.4: Bridge Week 4 to Week 5 — Building an "Agentic RAG" System

- **"What is the difference between Self-RAG and Adaptive/Corrective RAG (CRAG), and why shouldn't an LLM simply decide everything inside a single prompt?"**

    - **Self-RAG**: introduces self-reflection tokens/nodes (retrieval necessity, document relevance grading, and hallucination verification).
    - **Corrective RAG (CRAG)**: specifically focuses on automated recovery: if retrieved documents fall below a confidence threshold, it intercepts the workflow and triggers query reformulation (e.g., HyDE or web search fallback) before calling the generator.
    - **Why separate into graph nodes?** Monolithic prompts try to retrieve, grade, reformulate, and synthesize in one context, leading to attention dilution ("lost in the middle"), inability to inspect intermediate failures, and zero deterministic routing boundaries.


- **"Why is static (naive) RAG considered an anti-pattern in production, and how does Adaptive RAG solve the 'Retrieval Tax'?"**
    - **The Problem (Retrieval Tax)**: In a naive RAG chain (`Query → Embed → Vector DB → LLM`), *every single query* triggers embedding generation and a vector database lookup. For greetings ("Hi"), chitchat, or out-of-domain questions, this incurs unnecessary latency (~200–500ms), API token spend, and injects irrelevant chunks into the prompt context, increasing hallucination rates.
    - **The Solution (Adaptive Routing)**: An upstream Router inspects the user's intent *before* any vector DB connection is established:
        1. *Chitchat / Small talk* $\to$ Direct generation node (0 vector DB calls, sub-second latency).
        2. *Domain Knowledge (e.g. COVID-19/Biomedical)* $\to$ Local Hybrid Retriever (Qdrant Dense + BM25 + Cross-Encoder).
        3. *Real-time / External facts* $\to$ Web Search tool (DuckDuckGo / Tavily).

- **"Why does Semantic Similarity not equal Retrieval Relevance? How does Self-RAG's Document Grader fix this?"**
    - **First-Principles**: Bi-encoders project sentences into high-dimensional geometric space. Cosine similarity measures *topical proximity*, not *logical entailment* or *factual sufficiency*. A chunk can have a 0.85 cosine similarity to "What is the clinical dosage of Drug X?" simply because it discusses Drug X's chemical formula, while having zero dosage information.
    - **The Grader Gate**: Self-RAG inserts a binary evaluator node between retrieval and synthesis. It grades whether the retrieved text actually contains factual grounding for the question. If irrelevant, it rejects the context *before* the generator sees it, eliminating context pollution and groundedness degradation.

- **"How does Corrective RAG (CRAG) autonomously recover when retrieval fails, and why is a Circuit Breaker mandatory?"**
    - **Automated Recovery Loop**: When the Document Grader rejects retrieved chunks as irrelevant, naive RAG simply outputs "I don't know" or hallucinates. CRAG intercepts the control flow and routes to a **Query Transformation Node**:
        - It reformulates the query using **HyDE (Hypothetical Document Embeddings)**: the LLM hallucinates a synthetic 2-3 sentence answer excerpt. Embedding this hypothetical answer matches true passage vectors far better than embedding raw, short user queries.
        - The graph cycles back to the retrieval node with the enhanced query.
    - **The Circuit Breaker (`rewrite_count`)**: Cyclic graphs without hard termination conditions risk infinite recursion loops (churning API spend if the knowledge base genuinely lacks the answer). Production CRAG enforces a strict counter (e.g., `max_retries = 1`). If the reformulated query fails grading a second time, the circuit breaker trips and falls back to live Web Search or gracefully reports missing data.

- **"Why must Agent tools implement Defensive Execution, and how does this impact LangGraph runtimes?"**
    - **The Failure Mode**: External tools (web search scrapers, live APIs, database connectors) routinely encounter transient DNS timeouts, rate limits, or network resets (e.g., `ddgs.exceptions.DDGSException`). If a tool raises an unhandled Python exception, the entire LangGraph execution thread crashes immediately.
    - **Defensive Pattern**: Tools must wrap external network boundaries in `try...except` blocks and return synthetic observation payloads:
      `{"payload": {"text": "Web search unavailable due to network timeout...", "source": "error"}}`.
      This preserves graph state continuity and allows the downstream Generator Node to explain the constraint to the user politely rather than terminating with an unhandled traceback.

- **"How do you resolve Pronoun Ambiguity in multi-turn Agentic RAG conversations?"**
    - **The Dilemma**: In Turn 1, the user asks: *"What is the ACE2 receptor binding affinity in SARS-CoV-2?"* In Turn 2, they ask: *"How does it enter human cells?"*
      If you pass Turn 2's raw text to the vector database, dense and BM25 retrievers fail because *"it"* carries no lexical or semantic signal for SARS-CoV-2.
    - **De-contextualization (Conversational Rewriting)**: Before routing or retrieval, the query transformer inspects the last $N$ turns from the LangGraph `messages` state history and rewrites the follow-up into a standalone, keyword-rich query:
      *"How does SARS-CoV-2 enter human cells using the ACE2 receptor?"*

- **"Why is using a 70B/120B Generative LLM for routing and grading computationally suboptimal, and how do System-1 decision engines (e.g., Jev / Small Classifiers) compare?"**
    - **System 1 (Discriminative) vs. System 2 (Generative)**:
        - *Generative LLMs (System 2)* decode tokens autoregressively one by one. Using a 120B model to emit `{"destination": "domain_rag"}` burns hundreds of milliseconds of compute, incurs Time-to-First-Token (TTFT) latency, and requires Pydantic schema validation that can fail if the model omits a field.
        - *Decision Engines (System 1, e.g., Jev, SetFit, or Cross-Encoder Classifiers)* evaluate candidate choices in a single forward pass over logit distributions without generating text tokens.
    - **Architectural Trade-offs**:
        - *Latency*: System 1 models execute in ~50ms vs. ~500–1,200ms for generative LLMs.
        - *Confidence Calibration*: Specialized decision primitives (like Jev's `Noul`) return mathematical probabilities ($p \in [0.0, 1.0]$), allowing precise thresholding (e.g., `if prob < 0.70: rewrite_query()`), whereas generative LLMs output uncalibrated binary text ("yes"/"no").
        - *Production Hybrid Standard*: Modern 2026 agent architectures use System 1 classifiers for edge routing and document grading, while reserving generative frontier LLMs solely for System 2 creative synthesis (HyDE generation and final answer drafting with citations).
