# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
print("Connected to Qdrant Cloud!")

# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

model = SentenceTransformer("all-MiniLM-L6-v2")

if not client.collection_exists(COLLECTION_NAME):
    # print(f"Deleting existing collection: {COLLECTION_NAME}")
    # client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

    print(f"Created collection: {COLLECTION_NAME}")
    print(f"Vector size: {EMBEDDING_SIZE}")
    print("Distance: COSINE")

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD
    )

    # ============================================================
    # PART 4 — LOAD OUR KNOWLEDGE
    # ============================================================

    with open("knowledge.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents")

    # ============================================================
    # PART 5 — CREATE EMBEDDINGS
    # ============================================================

    print("Loading embeddings...")
    
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embeddings size: {len(embeddings[0])}")

    # ============================================================
    # PART 6 - CREATE QDRANT POINTS
    # ============================================================

    points = []

    for i in range(len(documents)):
        points.append(
            PointStruct(
                id=i+1,
                vector=embeddings[i].tolist(),
                payload=documents[i]
            )
        )

    # ============================================================
    # PART 7 - UPLOAD TO QDRANT
    # ============================================================

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Uploaded {len(points)} documents to Qdrant!")
else:
    print(f"Collection {COLLECTION_NAME} already exists.")

# ============================================================
# PART 8 - SEARCH QDRANT
# ============================================================

def search(query, top_k=3):
    # Convert the question into embedding
    query_vector = model.encode(query).tolist()

    # Search Qdrant for similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results

# ============================================================
# PART 9 - TEST SEARCH
# ============================================================

# query = "I don't want to go to office today, tommorow is weekend?"
# query = "I have a fever, can i get a day off?"
# results = search(query)

# print("\nSearch results:")
# for result in results:
#     print(f"Score: {result.score:.3f}")
#     print(result.payload["text"])
#     print()

# ============================================================
# PART 10 - CONNECT TO GROQ
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# PART 11 - ASK THE LLM
# ============================================================

def ask_llm(question, context):
    prompt = f"""
    Ask the question using only the provided context.

    Question: {question}
    Context: {context}

    Rules:
        - Only use the information provided in the context.
        - If the context doesn't contain the answer, respond with "I don't know."
    """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"{'*'*30} Reasoning {'*'*30}\nreasoning: {response.choices[0].message.content} {'*'*70}")
    return response.choices[0].message.content

# ============================================================
# PART 12 - COMPLETE RAG PIPELINE
# ============================================================

# question = "How many vacation days do I get?"
# result = search(question)
# # print(result)
# context = "\n".join([r.payload["text"] for r in result])

# print("\nSearch context:")
# for r in result:
#     print(f"Score: {r.score:.3f}")
#     print(r.payload["text"])
#     print(r.payload["category"])
#     print()

# answer = ask_llm(question, context)

# print(f"\nAnswer: {answer}")

# ============================================================
# PART 13 - Search with filter
# ============================================================

def search_with_filter(query, query_filter=None, top_k=3):
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter
    )

    return results.points

reimbursement_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="leave")
        )
    ]
)

print(f"{'*'*30} Now with search filter {'*'*30}")

question = "How many vacation days do I get?"
result = search_with_filter(question, query_filter=reimbursement_filter)
context = "\n".join([r.payload["text"] for r in result])

print("\nSearch context:")
for r in result:
    print(f"Score: {r.score:.3f}")
    print(r.payload["text"])
    print(r.payload["category"])
    print()

answer = ask_llm(question, context)

print(f"\nAnswer: {answer}")