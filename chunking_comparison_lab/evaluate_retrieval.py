import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from config import STRATEGIES_TO_COLLECTIONS_COVID_6PAGE


# ============================================================================
# 1. Configuration
# ============================================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")

# Initialize Clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 3

GOLDEN_DATASET_PATH = "golden_dataset/covid_6page.json"

STRATEGIES = STRATEGIES_TO_COLLECTIONS_COVID_6PAGE


# ============================================================================
# 2. Retrieval Helper
# ============================================================================

def retrieve_context(collection_name: str, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Retrives top-k chunks from Qdrant and gathers metadata."""
    query_vector = embedding_model.encode(query).tolist()

    hits = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    chunks = []
    retrieved_pages = set()
    combined_texts = []

    for h in hits:
        payload = h.payload or {}
        text = payload.get("text", "")

        page = payload.get("page_number")
        if page is not None:
            retrieved_pages.add(page)
        
        chunks.append({
            "score": round(float(h.score), 4),
            "page_number": page,
            "header_path": payload.get("header_path", "N/A"),
            "chunk_type": payload.get("chunk_type", "text"),
            "text": text
        })
        combined_texts.append(text)
    
    
    return {
        "chunks": chunks,
        "combined_text": "\n\n--- CHUNK SEPARATOR ---\n\n".join(combined_texts),
        "retrieved_pages": sorted(list(retrieved_pages)),
        "total_chars": sum([len(c["text"]) for c in chunks]),
        "avg_score": round(sum([c["score"] for c in chunks]) / len(chunks), 4) if chunks else 0.0
    }


# ============================================================================
# 3. Deterministic Fact Coverage Check
# ============================================================================

def compute_fact_coverage(context: str, mandatory_facts: List[str]) -> Dict[str, Any]:
    """Checks whether the mandatory ground-truth facts appear in retrieved context."""
    context_lower = context.lower().replace("-", " ")
    found = []
    missing = []
    
    for fact in mandatory_facts:
        fact_lower = fact.lower().replace("-", " ")
        # Check direct substring
        if fact_lower in context_lower:
            found.append(fact)
        else:
            # Check composite facts separated by comma or colon
            sub_parts = [p.strip() for p in fact_lower.replace(":", ",").split(",") if p.strip()]
            if len(sub_parts) > 1 and all(p in context_lower for p in sub_parts):
                found.append(fact)
            else:
                missing.append(fact)
                
    coverage_rate = len(found) / len(mandatory_facts) if mandatory_facts else 1.0
    return {
        "found_facts": found,
        "missing_facts": missing,
        "coverage_rate": round(coverage_rate, 2)
    }

# ====================================================================
# 4. LLM Generation (RAG Answer)
# ====================================================================

def generate_rag_answer(question: str, context:str) -> str:
    """Prompts LLM to answer using ONLY the retrieved context."""

    prompt = f"""
    You are an strict assistant. Answer the user's question using ONLY the provided context.
    If the context does not contain sufficent facts to answer completely, state exactly what information is missing.

    [Context]
    {context}

    [Question]
    {question}

    Provide a concise, factual answer.
    """

    message = {
        "role": "user",
        "content": prompt
    }

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[message],
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(str(e))
        return f"Error in LLM call: {e}"


# =================================================================
# 5. LLM-as-a-Judge Evaluation
# =================================================================

def evaluation_with_llm_judge(
    question: str,
    ground_truth: str,
    mandatory_facts: List[str],
    retrieved_context: str,
    generated_answer: str
    ) -> Dict[str, Any]:
    """Evaluates the generated answer and context against the ground truth using LLM judge."""

    judge_prompt = f"""
    You are an AI evaluator assessing RAG retrieval accuracy and answer quality.

    [Question]
    {question}

    [Ground Truth Answer]
    {ground_truth}

    [Mandatory Facts Required]
    {json.dumps(mandatory_facts, indent=2)}

    [Retrieved Context]
    {retrieved_context}

    [Generated Answer]
    {generated_answer}

    Evaluate the generated answer on two criterias:
    1. "context_accuracy_score" (0 to 5):
        - 5: The Retrieved Context contains all mandatory facts, the Generated Answer is factually correct and complete.
        - 4: The Retrieved Context contains most facts, answer is good but minor details missing.
        - 3: The Retrieved Context is partially relevant, answer has mixed accuracy.
        - 1-2: The Retrieved Context misses key facts, answer is mostly incomplete.
        - 0: The Retrieved Context is irrelevant or wrong, answer is incorrect.

    2. "generated_answer_faithfulness" (0 to 5):
        - 5: The Generated Answer is factually correct with respect to the question and consistent with the retrieved context.
             or If the Generated Answer states that the context is not sufficient to answer the question when the retrieved context doesn't have enough facts to answer the question.
        - 4: The Generated Answer is good but minor details and mostly consistent with the retrieved context.
        - 3: The Generated Answer is partially relevant and has mixed accuracy with respect to the retrieved context.
        - 1-2: The Generated Answer misses key facts and is mostly inconsistent with the retrieved context.
        - 0: The Generated Answer is irrelevant or wrong and contradicts the retrieved context.

    
    Provide your evaluation in a JSON format:
    {{
        "context_accuracy_score": <int>,
        "generated_answer_faithfulness": <int>,
        "reasoning": "<A short explanation in 3-4 sentences>"
    }}
    
    Return ONLY the JSON object, without markdown formatting.
    """

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.2,  # Low temp for consistent judging
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean any markdown code blocks
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        return json.loads(raw)
    except Exception as e:
        print(str(e))
        return {"error": str(e), "context_accuracy_score": 0, "generated_answer_faithfulness": 0, "reasoning": "Error in LLM call"}

# =================================================================
# 6. Orchestration
# =================================================================


def load_golden_dataset() -> Dict[str, Any]:
    try:
        with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def evaluate_strategy(strategy: str, collection: str, golden_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates a single strategy across all golden dataset questions."""

    print("="*60)
    print(f"Strategy: {strategy}")
    print("="*60)

    total_context_score = 0
    total_faithfulness_score = 0
    total_fact_coverage = 0
    
    for entry in golden_data:
        id = entry["id"]
        difficulty = entry["difficulty"]
        question_type = entry["question_type"]
        question = entry["question"]
        target_pages = set(entry["target_pages"]) if entry["target_pages"] else set()
        ground_truth = entry["ground_truth_answer"]
        mandatory_facts = entry["mandatory_facts"]
        retrieval_challenge = entry["retrieval_challenge"]

        print(f"\n {'='*30} Evaluating {id} : {difficulty} | {question_type}" + "="*30)
        print(f"QUESTION: {question}\n")

        retrieved_context = retrieve_context(collection, question)

        retrieved_context_pages = set(retrieved_context["retrieved_pages"]) if retrieved_context["retrieved_pages"] else set()

        fact_coverage = compute_fact_coverage(retrieved_context["combined_text"], mandatory_facts)
        total_fact_coverage += fact_coverage["coverage_rate"]

        generated_answer = generate_rag_answer(question, retrieved_context["combined_text"])

        evaluation = evaluation_with_llm_judge(question, ground_truth, mandatory_facts, retrieved_context["combined_text"], generated_answer)

        total_context_score += evaluation["context_accuracy_score"]
        total_faithfulness_score += evaluation["generated_answer_faithfulness"]

        print("="*60)
        print(f"Context Accuracy Score: {evaluation['context_accuracy_score']}")
        print(f"Generated Answer Faithfulness: {evaluation['generated_answer_faithfulness']}")
        print(f"Reasoning: {evaluation['reasoning']}")
        print(f"Target Pages: {target_pages}")
        print(f"Retrieved Pages: {retrieved_context_pages}")
        print(f"Missing Target Pages: {target_pages.difference(retrieved_context_pages)}")        
        print("="*60)
    
    avg_context_score = total_context_score / len(golden_data)
    avg_faithfulness_score = total_faithfulness_score / len(golden_data)
    avg_fact_coverage = total_fact_coverage / len(golden_data)

    print("="*60)
    print(f"Strategy: {strategy}")
    print("="*60)
    print(f"Average Context Accuracy Score: {avg_context_score}")
    print(f"Average Generated Answer Faithfulness: {avg_faithfulness_score}")
    print(f"Average Fact Coverage: {avg_fact_coverage}")
    print("="*60)
    print("+"*60)
    print("*"*60)
    return {
        "strategy": strategy,
        "avg_context_score": avg_context_score,
        "avg_faithfulness_score": avg_faithfulness_score,
        "avg_fact_coverage": avg_fact_coverage
    }
    
        

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 COMPLETE RAG EVALUATION SUITE")
    print("=" * 60)

    # Load Golden Dataset
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    results = []

for strat, collection in STRATEGIES.items():

    result = evaluate_strategy(strat, collection, golden_data)
    results.append(result)

print("="*60)
print("Final Results:")
print("="*60)
for result in results:
    print(f"Strategy: {result['strategy']}")
    print(f"Average Context Accuracy Score: {result['avg_context_score']}")
    print(f"Average Generated Answer Faithfulness: {result['avg_faithfulness_score']}")
    print(f"Average Fact Coverage: {result['avg_fact_coverage']}")
    print("="*60)