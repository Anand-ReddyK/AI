"""
Hybrid Retriever combining Qdrant Dense Vector Search with BM25 Keyword Search using
Reciprocal Rank Fusion (RRF).
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
# Add project root (/mnt/Storage/Projects/AI) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from chunking_comparison_lab.index_to_qdrant import (
    get_local_qdrant_client,
    get_embedding_model,
    DEFAULT_LOCAL_DB_PATH
)


def simple_tokenizer(text: str) -> List[str]:
    """Tokenize and lowercase text for BM25 matching."""
    return re.findall(r'\w+', text.lower())


class HybridRetriever:
    def __init__(
            self,
            collection_name: str,
            storage_path: str = DEFAULT_LOCAL_DB_PATH,
            rrf_k: int = 60
        ):
        """
        Args:
            collection_name: Qdrant collection name
            storage_path: Path to local Qdrant storage
            rrf_k: constant for RRF algorithm (default 60)
        """
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.rrf_k = rrf_k

        self.client: QdrantClient = get_local_qdrant_client(storage_path)
        self.embedding_model: SentenceTransformer = get_embedding_model()

        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[Dict[str, Any]] = []

        self._build_bm25_index_from_qdrant()
    

    def _build_bm25_index_from_qdrant(self):
        """Fetches stored text payloads from Qdrant to initialize the BM25 index."""
        if not self.client.collection_exists(self.collection_name):
            print(f"[WARNING] Collection '{self.collection_name}' not found.")
            return
        
        # Fetch points and their text payloads from Qdrant
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        self.corpus_chunks = []
        tokenized_corpus = []

        for p in points:
            payload = p.payload or {}
            text = payload.get("text", "")
            self.corpus_chunks.append({
                "point_id": p.id,
                "payload": payload,
                "text": text
            })
            tokenized_corpus.append(simple_tokenizer(text))
        
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"[HybridRetriever] Loaded {len(tokenized_corpus)} chunks into BM25 index.")
        
    
    def search_dense(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Dense semantic search using SentenceTransformer embeddings."""
        query_vector = self.embedding_model.encode(query).tolist()
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points

        results = []
        for rank, h in enumerate(hits, start=1):
            results.append({
                "point_id": h.id,
                "payload": h.payload,
                "dense_score": float(h.score),
                "dense_rank": rank
            })
        
        return results
    

    def search_bm25(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Sparse keyword search using BM25Okapi."""
        if not self.bm25 or not self.corpus_chunks:
            return []
        
        tokenized_query = simple_tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort chunks by score descending
        indexed_scores = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for rank, (idx, score) in enumerate(indexed_scores[:limit], start=1):
            if score <= 0.0:
                continue # Ignore chunks with 0 keyword overlap

            chunk = self.corpus_chunks[idx]
            results.append({
                "point_id": chunk["point_id"],
                "payload": chunk["payload"],
                "bm25_score": float(score),
                "bm25_rank": rank
            })
        
        return results
    

    def search_hybrid(
            self,
            query: str,
            limit: int = 5,
            candidate_pool_size: int = 25
        ) -> List[Dict[str, Any]]:
        """
        Runs both Dense and BM25 queries and combines result using RRF.
        Formula: RRF_SCORE = 1 / (rrf_k + rank)

        Args:
            query: Query to search for
            limit: Number of final results to return
            candidate_pool_size: Number of candidates to retrieve from each search
        
        Returns:
            List of retrieved chunks
        """
        if not self.bm25 or not self.corpus_chunks:
            raise ValueError("BM25 index not built. Run _build_bm25_index_from_qdrant first.")

        dense_results = self.search_dense(query, limit=candidate_pool_size)
        bm25_results = self.search_bm25(query, limit=candidate_pool_size)

        rrf_map: Dict[Any, Dict[str, Any]] = {}

        # 1. Add dense rank to RRF Score
        for item in dense_results:
            pid = item["point_id"]
            rank = item["dense_rank"]
            score = 1.0 / (self.rrf_k + rank)

            if pid not in rrf_map:
                rrf_map[pid] = {
                    "point_id" : pid,
                    "payload": item["payload"],
                    "rrf_score": 0.0,
                    "dense_rank": rank,
                    "bm25_rank": None
                }
            else:
                rrf_map[pid]["dense_rank"] = rank
            
            rrf_map[pid]["rrf_score"] += score
        
        # 2. Add BM25 rank to RRF Score
        for item in bm25_results:
            pid = item["point_id"]
            rank = item["bm25_rank"]
            score = 1.0 / (self.rrf_k + rank)

            if pid not in rrf_map:
                rrf_map[pid] = {
                    "point_id" : pid,
                    "payload": item["payload"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": rank
                }
            else:
                rrf_map[pid]["bm25_rank"] = rank
            
            rrf_map[pid]["rrf_score"] += score
        
        # 3. Sort by higher RRF score
        sorted_results = sorted(
            rrf_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return sorted_results[:limit]
        
        