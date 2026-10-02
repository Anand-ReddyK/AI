"""
Cross-Encoder Reranker for re-scoring retrieved chunks for better relevance.

- After a Hybrid search using Vector based (Dense) + Keyword (BM25) is conducted,
  we retrieve a set of top-k chunks (e.g. 25).
- This Cross-Encoder model re-scores these chunks based on the query + retrieved chunks.
- It outputs the top-k chunks with higher relevance.
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


# DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base" # Weight: ~2.2GB | High Quality Option
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Weight: ~0.3GB | Lightweight option


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        """
        Loads the cross-encoder model locally.
        Runs on CPU or GPU automatically.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        print(f"[Reranker] Loading Cross-Encoder model: {model_name} ...")
        self.model = CrossEncoder(model_name)
        print(f"[Reranker] Model loaded successfully.")
    

    def rerank(
            self,
            query: str,
            candidates: List[Dict[str, Any]],
            top_k: int = 4
        ) -> List[Dict[str, Any]]:
        """
        Takes a query and candidate chunks from Stage-1 retrieval (Hybrid/Dense/BM25),
        computes token-level cross-attention relevence scores, and returns top_k reordered.
        """
        if not candidates:
            return []
        
        # Prepare pairs: [ [query, chunk_text_1], [query, chunk_text_2], ...]
        pairs = []
        for c in candidates:
            payload = c.get("payload", ())
            text = payload.get("text", "")
            pairs.append([query, text])
        
        # Predict cross-encoder scores (logits/probabilities)
        scores = self.model.predict(pairs)

        # Attach rerank score to each candidate/chunk
        reranked_candidates = []
        for i, cand in enumerate(candidates):
            reranked_candidates.append({
                **cand,
                "rerank_score": float(scores[i]),
            })
        
        # Sort descending by cross-encoder score
        reranked_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_candidates[:top_k]
    

    
    
    