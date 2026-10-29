"""
Policy RAG Tools:
Allows agent to search terms of service and billing policies.
"""

import sys
from pathlib import Path
from langchain_core.tools import tool

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from customer_support_account_operations_agent.retrieval.policy_retriever import PolicyRetriever

# Lazy singleton instance
_policy_retriever = None

def get_retriever() -> PolicyRetriever:
    """Lazy singleton for policy retriever."""
    global _policy_retriever
    if _policy_retriever is None:
        _policy_retriever = PolicyRetriever(use_reranker=True)
    return _policy_retriever


@tool
def search_billing_policy(query: str) -> str:
    """
    Searches the official SaaS Terms of Service and billing policies for rules regarding
    subscription renewals, refunds, cancellation windows, dispute fees, and service tiers.
    Always query this tool before making refund decisions or answering policdy questions.
    """
    retriever = get_retriever()
    return retriever.retrieve(query=query, top_k=3)