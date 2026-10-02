from hybrid_retriever import HybridRetriever

# Use any collection in the local Qdrant
COLLECTION_NAME = "doc_covid-6-page-rag_pdf"

def main():
    retriever = HybridRetriever(collection_name=COLLECTION_NAME)

    queries = [
        "what are the diagnostic features of COVID-19",
        "Can SARS-CoV-2 infect the central nervous system?",
        "What ACE2 receptor binding affinity suggests high infectivity?",
        "list the antiviral drugs that inhibit SARS-CoV-2 replication",
    ]


    for q in queries:
        print("\n" + "=" * 70)
        print(f"Query: {q}")
        print("=" * 70)

        dense = retriever.search_dense(q, limit=2)
        bm25 = retriever.search_bm25(q, limit=2)
        hybrid = retriever.search_hybrid(q, limit=2)

        print("\n[TOP DENSE MATCH]")
        if dense:
            print(f"Score: {dense[0]['dense_score']:.4f}")
            print(f"Text: {dense[0]['payload'].get('text', '')[:150]}...")
        else:
            print("No dense match.")
        
        print("\n[TOP BM25 MATCH]")
        if bm25:
            print(f"Score: {bm25[0]['bm25_score']:.4f}")
            print(f"Text: {bm25[0]['payload'].get('text', '')[:150]}...")
        else:
            print("No BM25 match.")
        
        print("\n[TOP HYBRID RRF MATCH]")
        if hybrid:
            print(f"Score: {hybrid[0]['rrf_score']:.4f}")
            print(f"Dense Rank: {hybrid[0]['dense_rank']}")
            print(f"BM25 Rank: {hybrid[0]['bm25_rank']}")
            print(f"Text: {hybrid[0]['payload'].get('text', '')[:150]}...")
        else:
            print("No hybrid match.")
    
if __name__ == "__main__":
    main()