"""
RAGAS Evaluation Pipeline for RAG v2
Computes:
1. Context Recall (Retriever)
2. Context Precision (Retriever)
3. Faithfulness (Generator)
4. Answer Relevance (Generator)
"""

import time
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

from hybrid_rag.hybrid_retriever import HybridRetriever
from hybrid_rag.reranker import Reranker
from chunking_comparison_lab.config import (
    GOLDEN_DATASET_COVID_6PAGE,
    LOCAL_COLL_COVID_6PAGE,
    LOCAL_VECTOR_STORE_PATH
)
from utils.llm_provider import get_eval_llm

GOLDEN_DATASET_PATH = GOLDEN_DATASET_COVID_6PAGE
COLLECTION_NAME = LOCAL_COLL_COVID_6PAGE
LOCAL_DB_PATH = LOCAL_VECTOR_STORE_PATH

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


def generate_answer_with_groq(groq_client: Groq, query: str, contexts: list) -> str:
    """Standard generator prompt: answer strictly based on retrieved chunks."""
    joined_context = "\n\n ----- \n\n".join(contexts)
    prompt = f"""You are a strict technical document assistant.
Answer the user's question using ONLY the provided context. If the context does not contain
sufficient information, state clearly that it cannot be answered from the document.

[Context]
{joined_context}

[Question]
{query}

Provide a concise, factual answer."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def run_ragas_benchmark(max_test_samples: int = 4):
    print("=" * 75)
    print("🚀 RUNNING RAGAS BENCHMARK ON RAG v2 PIPELINE")
    print("=" * 75)

    # 1. Initialize Components
    print("\n[1/4] Initializing Retrievers, Rerankers, and Groq Clients....")
    retriever = HybridRetriever(collection_name=COLLECTION_NAME, storage_path=LOCAL_DB_PATH)
    reranker = Reranker()
    groq_client = Groq(api_key=GROQ_API_KEY)

    # LLM Judge (Groq) & Embedding Model for RAGAS
    eval_llm = get_eval_llm()
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load Golden Dataset
    print(f"[2/4] Loading Golden Dataset from {GOLDEN_DATASET_PATH}...")
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    test_samples = golden_data[:max_test_samples]

    questions = []
    ground_truths = []
    contexts_list = []
    answers = []

    # 3. Execute Pipeline Across Golden Dataset
    print(f"[3/4] Running RAG v2 pipeline (Hybrid -> Rerank -> Generate) for {len(test_samples)} samples...")
    for idx, sample in enumerate(test_samples, start=1):
        q = sample["question"]
        gt = sample["ground_truth_answer"]
        print(f"  Processing Question {idx}/{len(test_samples)}: {q[:60]}...")

        # Stage 1: Hybrid retrieval (10 candidates)
        candidates = retriever.search_hybrid(query=q, limit=10, candidate_pool_size=20)

        # Stage 2: Cross-Encoder Reranking (top 3 winners)
        reranked = reranker.rerank(query=q, candidates=candidates, top_k=3)

        # Extract string text from payloads (Fix: not dicts)
        retrieved_text = [r["payload"].get("text", "") for r in reranked]

        # Stage 3: LLM Generation
        gen_answer = generate_answer_with_groq(groq_client=groq_client, query=q, contexts=retrieved_text)

        questions.append(q)
        ground_truths.append(gt)
        contexts_list.append(retrieved_text)
        answers.append(gen_answer)

        # Rate limit safety pause between samples
        print("-------------PAUSE 2 SECONDS-------------")
        time.sleep(2)

    # 4. Construct RAGAS Dataset
    ragas_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(ragas_dict)

    # 5. Compute Metrics with LLM judge
    print("\n[4/4] Computing RAGAS metrics...")
    print("Running LLM-as-judge scoring across all 4 dimensions....\n")

    # FIX: Groq only supports n=1. Ragas answer_relevancy defaults to strictness=3 (which sends n=3)
    answer_relevancy.strictness = 1


    # Configure strictly sequential execution with backoff retries
    rate_limit_config = RunConfig(
        max_workers=1,   # Prevents parallel request bursts
        max_wait=60,     # Maximum wait backoff in seconds
        max_retries=10,  # Retry automatically if temporary 429 occurs
        timeout=180
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=rate_limit_config
    )

    # 6. Display Results
    print("=" * 75)
    print("📊 RAGAS EVALUATION SCORECARD")
    print("=" * 75)
    df_results = results.to_pandas()

    summary_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    for col in summary_cols:
        if col in df_results.columns:
            avg_score = df_results[col].mean()
            print(f"  • {col.replace('_', ' ').title():<22}: {avg_score:.4f} / 1.0000")


    print("\nDetailed Question-by-Question Breakdown:")
    for i, row in df_results.iterrows():
        print(f"\n--- Question #{i + 1} ---")
        # Supports both modern 'user_input' and legacy 'question'
        q_text = row.get("user_input", row.get("question", "N/A"))
        print(f"Q: {q_text[:80]}...")
        print(
            f"Precision: {row.get('context_precision', 0):.2f} | "
            f"Recall: {row.get('context_recall', 0):.2f} | "
            f"Faithfulness: {row.get('faithfulness', 0):.2f} | "
            f"Relevancy: {row.get('answer_relevancy', 0):.2f}"
        )

    # Optional: Save results to CSV for your portfolio/github!
    output_csv = PROJECT_ROOT / "evaluation/ragas_scorecard.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\n📁 Saved full evaluation scorecard to: {output_csv}")

    return df_results



if __name__ == "__main__":
    run_ragas_benchmark(max_test_samples=15)
