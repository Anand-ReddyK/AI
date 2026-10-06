from collections.abc import Callable
from typing import List, Dict, Optional
import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from pathlib import Path

# Anchor DB path to project root regardless of where python is executed from
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

DEFAULT_LOCAL_DB_PATH = str(PROJECT_ROOT / "local_qdrant_db")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384
_embedding_model: Optional[SentenceTransformer] = None

# Define the 3 JSON files for the chosen document (e.g., covid-6-page)


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def get_local_qdrant_client(storage_path: str = DEFAULT_LOCAL_DB_PATH) -> QdrantClient:
    """Returns a local embedded Qdrant client presisting to disk."""
    os.makedirs(storage_path, exist_ok=True)
    return QdrantClient(path=storage_path)


def index_chunks_to_local_qdrant(
        collection_name: str,
        chunks: List[Dict],
        storage_path: str = DEFAULT_LOCAL_DB_PATH,
        batch_size: int = 32,
        recreate_collection: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
    """
    Embeds and uploads chunks directly to a local Qdrant instance
    
    
    Args:
        collection_name: Name of the collection to create
        chunks: List of chunk dictionaries
        storage_path: Path to the local Qdrant instance
        batch_size: Batch size for embedding generation
        recreate_collection: Whether to recreate the collection
        progress_callback: Optional callback function to report progress func(current, total, message)
    """

    if not chunks:
        print("[WARNING] No chunks provided for indexing.")
        return 0
    
    client = get_local_qdrant_client(storage_path)
    model = get_embedding_model()

    # Recreate collection if required
    if recreate_collection and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
        )
    
    texts = [c["text"] for c in chunks]
    total_texts = len(texts)

    if progress_callback:
        progress_callback(0, total_texts, f"Starting embedding for {total_texts} chunks...")
    
    # Generate embeddings in batches so progress can be tracked. (Just for fun 😋)
    embeddings = []
    for i in range(0, total_texts, batch_size):
        batch = texts[i: i + batch_size]
        batch_vecs = model.encode(batch, show_progress_bar=True)
        embeddings.extend(batch_vecs)

        current_count = min(i + batch_size, total_texts)
        if progress_callback:
            progress_callback(current_count, total_texts, f"Embeded {current_count} of {total_texts} chunks...")
    
    points = [
        PointStruct(
            id = idx + 1,
            vector=embeddings[idx].tolist(),
            payload=chunk
        )
        for idx, chunk in enumerate(chunks)
    ]

    if progress_callback:
        progress_callback(total_texts, total_texts, "Upserting points into local Qdrant....")
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"[DONE] Indexed {len(points)} chunks into local collection '{collection_name}' at {storage_path}")

    return len(points)




def upload_json_to_qdrant(
        collection_name: str,
        json_path: str,
        storage_path: str = DEFAULT_LOCAL_DB_PATH
    ):
    """Helper to upload from an existing JSON file."""
    if not os.path.exists(json_path):
        print(f"[SKIPPING] JSON file not found at {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    return index_chunks_to_local_qdrant(collection_name, chunks, storage_path)
    

def search_chunks(
        collection_name: str,
        query: str,
        limit: int = 4,
        storage_path: str = DEFAULT_LOCAL_DB_PATH,
        ) -> List[Dict]:
        """
        Encodes query and retrieves the most relevent chunks from local Qdrant.
        """
        client = get_local_qdrant_client(storage_path)
        if not client.collection_exists(collection_name):
            print(f"[WARNING] Collection '{collection_name}' does not exist.")
            return []
        
        model = get_embedding_model()
        query_vector = model.encode(query).tolist()

        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        )

        results = []
        for hit in search_result.points:
            results.append({
                "score": hit.score,
                "payload": hit.payload
            })
        
        return results
    
    

def upload_strategy_to_qdrant(collection_name: str, json_path: str):
    if not os.path.exists(json_path):
        print(f"[SKIPPING] JSON file not found at {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"\n[Processing] Collection {collection_name} with {len(chunks)} chunks")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedding_model = get_embedding_model()

    # Recreate collection cleanly
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
    )

    # Generate embeddings
    texts = [c["text"] for c in chunks]
    print(f"[Generating] {len(texts)} embeddings using all-MiniLM-L6-v2...")
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)

    # Prepare Qdrant Points
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=i+1,
                vector=embeddings[i].tolist(),
                payload=chunk
            )
        )
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"[DONE] uploaded {len(points)} chunks to {collection_name}")



if __name__ == "__main__":
    # Main Execution
    print("=" * 60)
    print("🚀 MULTI-STRATEGY UPLOADER")
    print("=" * 60)

    from config import STRATEGY_COLLECTION_TO_CHUNKED_FILE_COVID_6PAGE

    for strategy, json_file in STRATEGY_COLLECTION_TO_CHUNKED_FILE_COVID_6PAGE.items():
        upload_strategy_to_qdrant(strategy, json_file)
    
