"""
Test Script: Evaluates Query Rewriting
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pre_retrieval.query_transformer import QueryTransformer
from hybrid_rag.hybrid_retriever import HybridRetriever

COLLECTION_NAME = "doc_covid-6-page-rag_pdf"

def test_rewrite_query():
    transformer = QueryTransformer()
    print("=" * 50)
    print("DEMO : CONVERSATIONAL QUERY REWRITING")
    print("=" * 50)

    # Simulate a multi-turn chat where the user uses vague pronouns
    mock_history = [
        {"role": "user", "content": "Tell me about Tocilizumb in severe COVID-19."},
        {"role": "assistant", "content": "Tocilizumab is an IL-6 receptor antagonist used for severe patients."}
    ]

    vague_query = "What is its recommended single dose and criteria for a second dose?"

    print(f"Original Query: {vague_query}")
    rewritten_query = transformer.rewrite_query(query=vague_query, chat_history=mock_history)
    print(f"Rewritten Query: {rewritten_query}")
    print('-' * 50)

def test_hyde_query():
    print("=" * 50)
    print("DEMO: HYPOTHETICAL DOCUMENT EMBEDDINGS")
    print("=" * 50)

    transformer = QueryTransformer()
    retriever = HybridRetriever(collection_name=COLLECTION_NAME)

    conceptual_query = "What specific laboratory marker values define cytokine release syndrome?"

    print(f"Query: {conceptual_query}\n")
    hypothetical_passage = transformer.generate_hyde_passage(query=conceptual_query)
    print(f"Generated HyDE passage:\n{hypothetical_passage}\n")


    # ? Maybe this should be tested for the comparision in confidence score and results between raw query and HyDE passage for dense search.
    #* HyDE search: We use the hypothetical passage for dense search, but the original query for BM25
    print("HyDE-Assisted Search (Dense on hypothetical passage + BM25 on original query)...")
    dense_hyde_hits = retriever.search_dense(query=hypothetical_passage, limit=10)
    bm25_raw_hits = retriever.search_bm25(query=conceptual_query, limit=10)

    hybrid_hits = retriever.rrf_combine(
        dense_results=dense_hyde_hits,
        bm25_results=bm25_raw_hits,
        limit=10
    )
    print("\n --- Top Winner with (HyDE + Dense) + BM25 ---")
    print(f"\n{hybrid_hits[0]['payload']['text']}")
    print(f"Score: {hybrid_hits[0]['rrf_score']:.4f}")
    print(f"Dense Rank: {hybrid_hits[0]['dense_rank']}")
    print(f"BM25 Rank: {hybrid_hits[0]['bm25_rank']}")

    print('-' * 50)


def test_decompose_query():
    print("=" * 60)
    print("DEMO: QUERY DECOMPOSITION BENCHMARK")
    print("=" * 60)

    transformer = QueryTransformer()

    queries = {
        # Level 1: Simple Atomic Queries (Should return 1 query)
        "Simple Tech": "What is the maximum payload size supported by AWS Lambda?",
        "Simple Finance": "What was Apple's gross profit margin in Q3 2024?",

        # Level 2: Dual Comparison (Should return 2 sub-queries)
        "Cloud Comparison": "Compare the pricing models and replication latency of AWS Aurora vs Google Cloud Spanner.",

        # Level 3: Multi-Aspect Regulatory (Should return 3 sub-queries)
        "Compliance & Privacy": "What are the breach notification deadlines, financial penalties, and DPO requirements under GDPR?",

        # Level 4: Complex Multi-Variable Architecture (Should return 4-5 sub-queries)
        "System Architecture": (
            "Evaluate whether an enterprise fintech app should migrate from monolithic PostgreSQL to distributed CockroachDB, "
            "considering multi-region write latency, ACID compliance guarantees, schema migration complexity, and licensing costs."
        )
    }

    for label, query in queries.items():
        print(f"\n[Category: {label}]")
        print(f"Original : {query}")
        decomposed = transformer.decompose_query(query=query)
        print(f"Count    : {len(decomposed)} sub-queries")
        for i, sq in enumerate(decomposed, 1):
            print(f"   {i}. {sq}")
        print("-" * 60)

    

def main():
    # test_rewrite_query()
    # test_hyde_query()
    test_decompose_query()

if __name__ == "__main__":
    main()