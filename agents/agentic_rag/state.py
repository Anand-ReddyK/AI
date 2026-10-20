"""
Defines the typed state passed across all nodes in the Agentic RAG workflow
"""

from typing import List, Dict, Any, Optional, Sequence
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgenticRAGState(TypedDict):
    """
    State tracking the lifecycle of a query through routing,
    retrieval, grading, query transformation, and answer synthesis.
    """

    question: str                       # Orignal user query
    current_query: str                  # Current query being processed (may be rewritten/HyDE)
    documents: List[Dict[str, Any]]     # Retrieved & reranked context chunks
    route: str                          # Decision: 'direct_chat' | 'domain_rag' | 'web_search'
    grade: str                          # Document quality: 'relevent' | 'not_relevent'
    rewrite_count: int                  # Circuit vbreak to prevent infinite rewiting loops
    generation: str                     # Final synthesized output
    messages: Annotated[Sequence[BaseMessage], add_messages] # Chat history across turns
    