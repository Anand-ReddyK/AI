"""
A Wrapper Kit for the existing retrieval and transformation tools
- HybridRetriever (Dense Qdrant + BM25 + RRF)
- CrossEncoder Reranker
- QueryTransformer (HyDE + Conversational Rewriting + Query Decomposition)
- DuckDuckGo Web Search fallback
"""

import sys
import atexit
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add projects root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from hybrid_rag.hybrid_retriever import HybridRetriever
from hybrid_rag.reranker import Reranker
from pre_retrieval.query_transformer import QueryTransformer
from chunking_comparison_lab.config import COLL_COVID_6PAGE_STRUCTURE
from langchain_community.tools import DuckDuckGoSearchRun


DEFAULT_COLLECTION = COLL_COVID_6PAGE_STRUCTURE

class RAGToolKit:
    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        print("[RAGToolKit] Initializing Hybrid Retriever and Reranker...")
        self.retriever = HybridRetriever(collection_name=collection_name)
        self.reranker = Reranker()
        self.query_transformer = QueryTransformer()
        self.web_search = DuckDuckGoSearchRun()

        # Cleanly cloase local Qdrant before Python tears down sys.meta_path
        atexit.register(self.close)

    def close(self):
        try:
            self.retriever.client.close()
        except Exception:
            pass

    
    def retriever_and_rerank(
            self,
            query: str,
            retrieve_limit: int = 5,
            rerank_top_k: int = 3
        ) -> List[Dict[str, Any]]:
        """
        Executes hybrid dense+sparse retrieval with RRF, then scores with Cross-Encoder.
        """
        # step 1: Hybrid Search (RRF)
        raw_candidates = self.retriever.search_hybrid(
            query=query,
            limit=retrieve_limit,
            candidate_pool_size=20
        )

        if not raw_candidates:
            return []
        
        reranked = self.reranker.rerank(
            query=query,
            candidates=raw_candidates,
            top_k=rerank_top_k
        )

        return reranked
    

    def transform_query_hyde(self, query: str) -> Dict[str, Any]:
        """
        Uses HyDE to create a hypothetical passage for enhanced dense vector matching.
        """
        return self.query_transformer.generate_hyde_passage(query)
    

    def rewrite_search_query(self, query: str, chat_history: List[Dict[str, Any]] = None) -> str:
        """
        Rewrites conversational or vague questions into clear search queries.
        """
        return self.query_transformer.rewrite_query(query, chat_history)
    

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes DuckDuckGo search and formats results to match the chunk schema.
        Defensive handling catches DNS / rate-limit exceptions gracefully.
        """
        try:
            raw_text = self.web_search.run(query)
            if not raw_text or not raw_text.strip():
                raw_text = "No relevant search results found on the web."
        except Exception as e:
            print(f"[RAGToolKit Warning] Web search failed: {e}")
            raw_text = (
                f"Web search service is currently unavailable or encountered a network error: {e}. "
                "Unable to retrieve live external data at this moment."
            )

        return [{
            "point_id": "web_search",
            "payload": {"text": raw_text, "source": "DuckDuckGo Web Search"},
            "rerank_score": 1.0
        }]

        