from pandas._libs.tslibs.offsets import YearBegin
from huggingface_hub.hf_api import api
import os
from typing import List, Dict, Generator
from dotenv import load_dotenv
from groq import Groq
from index_to_qdrant import search_chunks, DEFAULT_LOCAL_DB_PATH

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

def get_groq_model() -> Groq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found")
    return Groq(api_key=GROQ_API_KEY)


def build_rag_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Constructs a structured context prompt for the LLM with citations and metadata.
    """

    context_blocks = []
    for idx, item in enumerate(retrieved_chunks):
        payload = item.get("payload", {})
        
        page = payload.get("page_number", "N/A")
        header = payload.get("header_path", "Root Document")
        chunk_type = payload.get("chunk_type", "text")
        text = payload.get("text", "").strip()

        block = f"--- [SOURCE {idx + 1}: Page {page} | Section {header} | Type: {chunk_type.upper()}] ---\n{text}"
        context_blocks.append(block)

    joined_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a Q&A document assistant. Your answers must be faithful, factual, and strictly"
        " based on the provided context.\n"
        "Guidelines:\n"
        "1. Always cite your sources in the text using bracketed citations, e.g. [Page X, Section: ...].\n"
        "2. If the context contains a table relevent to the query, format your explanation clearly or summarize the numbers accurately.\n"
        "3. If the context does not contain enough information to answer the question, state honestly that the document does not contain that information.\n"
        "4. If you the context contains information that can only partially answer the given question, then state that while providing the answer."
        "5. DO NOT make up facts outside the provided context, and DO NOT hallucinate."
    )

    prompt = f"""
    {system_prompt}

    CONTEXT EVIDENCE:
    {joined_context}

    USER QUESTION:
    {query}
    """

    return prompt


def query_rag_stream(
        collection_name: str,
        query:str,
        top_k: int = 4,
        storage_path: str = DEFAULT_LOCAL_DB_PATH,
        model: str = DEFAULT_GROQ_MODEL
    ) -> tuple[Generator[str, None, None], List[Dict]]:
    """
    Retrieves top_k chunks from local Qdrant and streams response from Groq LLM.

    Args:
        collection_name: Name of the Qdrant collection.
        query: User's query.
        top_k: Number of chunks to retrieve.
        storage_path: Path to the local Qdrant instance.
        model: Name of the Groq LLM to use.

    Returns:
        {stream_generator, retrieved_chunks}
    """
    # 1. Retrieve relevant chunks from local Qdrant
    retrieved_chunks = search_chunks(
        collection_name=collection_name,
        query=query,
        limit=top_k,
        storage_path=storage_path
    )

    if not retrieved_chunks:
        def empty_generator():
            yield "No relevant information found in the document for your query."
        return empty_generator(), []

    full_prompt = build_rag_prompt(query, retrieved_chunks)
    client = get_groq_model()

    response_stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
        stream=True
    )

    def text_generator():
        for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    
    return text_generator(), retrieved_chunks

        
        