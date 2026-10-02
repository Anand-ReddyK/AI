"""
Test Stage-1 (Hybrid RRF) vs Stage-2 (Cross-Encoder Reranked).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_rag.hybrid_retriever import HybridRetriever
from hybrid_rag.reranker import Reranker



COLLECTION_NAME = "doc_covid-6-page-rag_pdf" # A local collection

def main():
    retriver = HybridRetriever(collection_name=COLLECTION_NAME)
    reranker = Reranker()

    query = "Compare the growth advantage of XEC against KP.3.1.1: What was XEC's relative effective reproduction number in August 2024, and how did KP.3.1.1's global sequence proportion change from week 44 to week 47 in Table 1?"
    
    print("\n" + "=" * 50)
    print(f"QUERY: {query}")
    print("=" * 50)

    # 1. Stage 1: Retrieve 10 candidates via Hybrid Search
    print("\n[STAGE 1] Retrieving Top 10 candidates via Hybrid Search...")
    stage1_candidates = retriver.search_hybrid(query=query, limit=10, candidate_pool_size=25)

    print(f"\nRetrieved {len(stage1_candidates)} candidates.")
    for i, c in enumerate(stage1_candidates, start=1):
        print(f"\nRank {i}: {c['rrf_score']:.4f} | Dense: {c['dense_rank']} | BM25: {c['bm25_rank']} \n Text: {c['payload']['text'][:150]}...")
    
    # 2. Stage 2: Re-rank top candidates using Cross-Encoder
    print(f"[STAGE 2] Reranking candidates with Cross-Encoder...")
    reranked_top = reranker.rerank(query=query, candidates=stage1_candidates, top_k=4)

    print("\n STAGE 2 WINNERS (After Cross-Encoding Ranking) ---")
    for i, c in enumerate(reranked_top, start=1):
        print(f"\nRank: {i} | Rerank score: {c['rerank_score']:.4f} | RRF score: {c['rrf_score']:.4f} | Dense: {c['dense_rank']} | BM25: {c['bm25_rank']}")
        print(f"          Page: {c['payload'].get('page_number', 'N/A')} | Header Path: {c['payload'].get('header_path', 'N/A')} \n  Text: {c['payload']['text'][:120]}...")
    
    


if __name__ == "__main__":
    main()
    