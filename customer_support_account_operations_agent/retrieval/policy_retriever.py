"""
Policy Retriever:
Wraps HybridRetriever (Dense + BM25 with RRF) and Cross-Encoder Reranker
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_rag.hybrid_retriever import HybridRetriever
from hybrid_rag.reranker import Reranker
from customer_support_account_operations_agent.retrieval.ingest_policies import DEFAULT_LOCAL_DB_PATH, COLLECTION_NAME

class PolicyRetriever:
    def __init__(self, use_reranker: bool = True):
        self.retriever = HybridRetriever(
            collection_name=COLLECTION_NAME,
            storage_path=DEFAULT_LOCAL_DB_PATH,
            rrf_k=60
        )
        self.reranker = Reranker() if use_reranker else None
    
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Executes hybrid search + reranking and formats chunks into a context string.
        """
        # 1. Hybrid Search (Dense + BM25)
        candidates = self.retriever.search_hybrid(query=query, limit=10)
        if not candidates:
            return "No matching policy terms found."
        

        # 2. Cross-Encoder Reranking
        if self.reranker:
            reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        else:
            reranked = candidates[:top_k]
        
        # 3. Format into grounded text block
        formatted_passages = []
        for i, item in enumerate(reranked, start=1):
            payload = item.get("payload", {})
            text = payload.get("text", "")
            meta = payload.get("metadata", {})
            page = meta.get("page", "?")
            header_path = meta.get("header_path", "Root")
            chunk_type = meta.get("chunk_type", "text")
            formatted_passages.append(f"[Policy Passage {i} (Page {page}) (Section: {header_path}) (Chunk Type: {chunk_type})]:\n{text}")
        

        return "\n\n".join(formatted_passages)
    