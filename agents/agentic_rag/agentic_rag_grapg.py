"""
Adaptive & Corrective RAG (CRAG) with LangGraph:
- Router Node: Inspects query -> 'direct_chat' vs 'domain_rag' vs 'web_search'
- Local Retrieval Node: Hybrid Dense + BM25 + Cross-Encoder Reranking
- Document Grader Node (Self-RAG): Assesses chunk relevence
- Query Transformer Node: Applies HyDE / Rewritting if retrieval was irrelevent
- Generation Node: Synthesizes grounded answer with citations
"""

from langsmith._openapi_client.types import run_query_params
import os 
import sys
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.agentic_rag.state import AgenticRAGState
from agents.agentic_rag.rag_tools import RAGToolKit

# Initialize shared toolkit & LLM
toolkit = RAGToolKit()
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.2
)


# =========================================================
# 1. Pydantic Schemas for Deterministic Routing & Grading
# =========================================================

class RouteDecision(BaseModel):
    """Routing classifier decision"""
    destination: Literal["direct_chat", "domain_rag", "web_search"] = Field(
        description="Choose 'domain_rag' for COVID-19/SARS-CoV-2 scientific questions, "
                    "'web_search' for general knowledge and real-time external topics, "
                    "'direct_chat' for conversational chitchat or greetings"
    )
    reasoning: str = Field(description="Explanation for routing choice.")


class GradeDecision(BaseModel):
    """Document relevence grade."""
    binary_score: Literal["yes", "no"] = Field(
        description="Does the document contain information relevent to the question?"
    )
    explanation: str = Field(description="Reason for grading yes or no")


# Structured LLM callers using json_mode
route_classification = llm.with_structured_output(RouteDecision, method="json_mode")
doc_grader = llm.with_structured_output(GradeDecision, method="json_mode")


# =========================================================
# 2. Node Implementations
# =========================================================

def router_node(state: AgenticRAGState):
    """
    Adaptive RAG Router: Analyzes incoming query to prevent unnecesary retrieval.
    """
    question = state["question"]
    messages = state.get("messages", [])
    
    # If there is prior chat history, convert messages to dict format for QueryTransformer
    history_dicts = []
    if len(messages) > 1:
        for msg in messages[:-1]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history_dicts.append({"role": role, "content": msg.content})
        
        # Rewrite conversational follow-ups into standalone queries
        standalone_query = toolkit.rewrite_search_query(question, history_dicts)
    else:
        standalone_query = question
    
    prompt = [
        SystemMessage(content=(
            "You are a query router. Classify the user query into one of three destinations:\n"
            "- 'direct_chat': for greetings, small talk, gratitude, or general assistant chitchat.\n"
            "- 'domain_rag': for questions about SARS-CoV-2, COVID-19, clinical symptoms, viral transmission, "
            "ACE2 receptor binding, antiviral drugs, or vaccine studies.\n"
            "- 'web_search': for current events, real-time news, sports, weather, or other topics unrelated to COVID-19.\n\n"
            "Return valid JSON with both keys:\n"
            "{\n"
            '  "destination": "direct_chat" | "domain_rag" | "web_search",\n'
            '  "reasoning": "brief explanation"\n'
            "}"
        )),
        HumanMessage(content=standalone_query)
    ]

    decision = route_classification.invoke(prompt)
    print(f"\n[Router Decision]: -> {decision.destination.upper()} (Reason: {decision.reasoning})")

    return {
        "route": decision.destination,
        "current_query": question,
        "rewrite_count": 0
    }


def local_retrieval_node(state: AgenticRAGState):
    """
    Executes Hybrid Search + Reranking using retriever
    """
    query = state["current_query"]
    print(f"\n[Local Retrieval]: Searching index for: '{query}'")

    docs = toolkit.retriever_and_rerank(query=query, retrieve_limit=5, rerank_top_k=3)
    return {"documents": docs}


def web_search_node(state: AgenticRAGState):
    """
    Web search fallback for real-time queries or failed local retrieval.
    """
    query = state["current_query"]
    print(f"\n[Web Search]: Searching live web for: '{query}'")

    docs = toolkit.search_web(query)
    return {"documents": docs, "grade": "relevant"}


def document_grader_node(state: AgenticRAGState):
    """
    Self-RAG Evaluator: Checks if retrieved chunks actually answer the question.
    """
    question = state["question"]
    docs = state.get("documents", [])

    if not docs:
        print("[Grader]: No documents retrieved. Marking NOT_RELEVENT")
        return {"grade": "not_relevent"}
    
    combined_text = "\n\n".join([d.get("payload", {}).get("text", "") for d in docs])

    prompt = [
        SystemMessage(content=(
            "You are a strict document evaluator.\n"
            "Assess whether the retrieved text contains relevent facts or answers to the user question.\n"
            "If it contains even partiual relevent context, grade 'yes'.\n"
            "If it is completely irrelevent or off-topic, grade 'no'.\n"
            "Return valid JSON with both keys:\n"
            "{\n"
            '  "binary_score": "yes" | "no",\n'
            '  "explanation": "brief explanation"\n'
            "}"
        )),
        HumanMessage(content=f"User Question: {question}\n\nRetrieved Text:\n{combined_text[:2000]}")
    ]

    grade_result = doc_grader.invoke(prompt)
    grade = "relevent" if grade_result.binary_score.lower().strip() == "yes" else "not_relevent"
    print(f"[Document Grader]: Grade = {grade.upper()} (Reason:{grade_result.explanation})")

    return {"grade": grade}


def transform_query_node(state: AgenticRAGState):
    """
    Corrective RAG (CRAG) loop: Rewrites the query using HyDE to improve retrieval.
    """
    question = state["question"]
    count = state.get("rewrite_count", 0) + 1

    print(f"\n[Query Transformer]: Attempt #{count} - Reformulating with HyDE...")
    # HyDE generates a hypothetical passage to improve dense vector match
    transformed = toolkit.transform_query_hyde(question)
    
    return {
        "current_query": transformed,
        "rewrite_count": count
    }


def direct_chat_node(state: AgenticRAGState):
    """
    Direct response without retrieval overhead for chitchat/greetings.
    """
    question = state["question"]
    # Send last few messages for conversational context
    recent_messages = list(state.get("messages", []))
    if len(recent_messages) > 1:
        recent_messages = recent_messages[-4:]


    prompt = [
        SystemMessage(content="You are a helpful, poliute AI research assistant. Respond warmly and consisely.")
    ] + recent_messages

    response = llm.invoke(prompt)
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content)]
        }


def generate_node(state: AgenticRAGState):
    """
    Synthesizes the grounded response with citations based on graded context.
    """
    question = state["question"]
    docs = state.get("documents", [])
    
    context_str = ""
    for idx, d in enumerate(docs, 1):
        text = d.get("payload", {}).get("text", "")
        source = d.get("payload", {}).get("source", f"Chunk {d.get('point_id', idx)}")
        context_str += f"\n--- Source [{idx}]: {source} ---\n{text}\n"
    
    prompt = [
        SystemMessage(content=(
            "You are a biomedical research synthesis assistant.\n"
            "1. Answer the question using ONLY the provided verified context.\n"
            "2. Cite your sources using bracket notations like [1], [2].\n"
            "3. If the context does not contain sufficent facts, clearly explain what is missing.\n"
            "4. If partial answer can be given then give it while stating what's missing"
        )),
        HumanMessage(content=f"Context:\n{context_str}\n\nQuestion:{question}\n\nAnswer:")
    ]

    response = llm.invoke(prompt)
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content)]
        }
    
        
# ===============================================================
# 3. Conditioal Edge Routing Functions
# ===============================================================

def route_destination(state: AgenticRAGState):
    """Routes based on the Router Node's classification."""
    route = state.get("route", "domain_rag")
    if route == "direct_chat":
        return "direct_chat"
    if route == "web_search":
        return "web_search"
    return "domain_rag"


def check_grading_result(state: AgenticRAGState) -> str:
    """
    Self-RAG Routing:
    - If documents are relevent -> proceed to generate
    - If not relevent & retry budget remains -> transform_query
    - If retry budget exhusted -> fallback to web_search
    """
    grade = state.get("grade", "relevent")
    rewrite_count = state.get("rewrite_count", 0)
    
    if grade == "relevent":
        return "generate"
    
    # Circuit breaker: max 1 HyDE rewrite attempt before falling back to Wen Search
    if rewrite_count < 1:
        return "transform_query"
    else:
        print('[Circuit Breaker] Max retries exceeded, fallback to web search...')
        return "web_search"


# ===============================================================
# 4. Graph Construction
# ===============================================================

workflow = StateGraph(AgenticRAGState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("local_retrieval", local_retrieval_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("document_grader", document_grader_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("direct_chat", direct_chat_node)
workflow.add_node("generate", generate_node)

# Add Edges
workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_destination,
    {
        "direct_chat": "direct_chat",
        "domain_rag": "local_retrieval",
        "web_search": "web_search"
    }
)


# After local retrival, always evaluate relevence with Grader
workflow.add_edge("local_retrieval", "document_grader")

# Conditional Edges from Grader
workflow.add_conditional_edges(
    "document_grader",
    check_grading_result,
    {
        "generate": "generate",
        "transform_query": "transform_query",
        "web_search": "web_search"
    }
)

# After query transformation, cycle back to local retrival
workflow.add_edge("transform_query", "local_retrieval")

# After web search, generate response
workflow.add_edge("web_search", "generate")

# Generation end the flow
workflow.add_edge("direct_chat", END)
workflow.add_edge("generate", END)

# Checkpointer preserves state across turns under a thread_id
checkpointer = MemorySaver()

# Compile
app = workflow.compile(checkpointer=checkpointer)


# ==============================================================
# 5. Interactive Chat Session Loop
# ==============================================================

def run_agentic_rag(thread_id: str = "session-101"):
    print("=" * 70)
    print("🤖 Agentic RAG Research Assistant")
    print("   - Type your question or chitchat.")
    print("   - Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 70)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        try:
            user_input = input("\nYou => ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nExiting session. Goodbye!")
            break
        
        # Pass the new user query into the state
        initial_state = {
            "question": user_input,
            "current_query": user_input,
            "documents": [],
            "route": "",
            "grade": "",
            "rewrite_count": 0,
            "generation": "",
            "messages": [HumanMessage(content=user_input)]
        }

        response = ""

        # Stream graph updates node-by-node
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                print(f"--- [Node Completed: {node_name}] ---")
                if "generation" in update:
                    response = update["generation"]

        print("\n" + "=" * 70)
        print("🤖 Assistant => " + (response or "No generation produced."))
        print("=" * 70)



if __name__ == "__main__":
    # # Test 1: Chitchat (Bypasses retrieval completely)
    # run_agentic_rag("Good morning! How are you doing today?")

    # # Test 2: In-Domain COVID-19 Query (Direct Hybrid + Rerank + Pass Grade)
    # run_agentic_rag("What ACE2 receptor binding affinity suggests high infectivity in SARS-CoV-2?")
    
    # # Test 3: Vague Query (Triggers Document Grader Failure -> HyDE Query Transformation -> Retrieval)
    # run_agentic_rag("Tell me about the enzyme binding")
    
    # # Test 4: External / Real-time Query (Routes directly to Web Search)
    # run_agentic_rag("What is the current stock price of Nvidia (NVDA)?")
    run_agentic_rag()