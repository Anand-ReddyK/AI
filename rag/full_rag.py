import os
import sys
import time
import numpy as np
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is required")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=api_key)
groqmodel = "openai/gpt-oss-20b"

documents = [
    "Employees recieve 24 days of paid leave per year.",
    
    "Employees work from office on Tuesday, Wednesday and Thursday.",
    "Monday and Friday are optional work-from-home days.",
       
    "Employees receive Rs 3000 per month for gym reimbursement.",
    
    "Employees can claim Rs 2000 per month for home internet.",
    
    "Employees have a 90 day notice period."
]

s = time.time()
documents_embeddings = model.encode(documents)
e = time.time()
print(f"Embedding Time: {e - s:.4f} seconds")

print(sys.getsizeof(documents_embeddings))


def consine_similarity(a, b):
     return np.dot(a, b) / (
         np.linalg.norm(a) * np.linalg.norm(b)
     )

def retrieve(qembedding: str):
    qembedding = model.encode(qembedding)
    scores = []
    for i, document in enumerate(documents_embeddings):
        score = consine_similarity(qembedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    context = "\n".join([scores[i][1] for i in range(3)])
    return context

def ask_llm(question):
    s = time.time()
    context = retrieve(question)
    e = time.time()
    print(f"Processing Time: {e - s:.4f} seconds")
    print("The context sent to LLM:", context)
    print("*"*30)
    system_prompt = f"""
    - Answer the question in a single sentence.
    - Answer the question only based on the given context.
    - If the context is not available only answer if the question is a general one and can be answer without specific context or information.
    - If the question cannot be answered with confidence or if the required context is not given, or if the given context is not sufficient or relevent then answer something in the lines of "I don't have enough information about xyz to answer this question.

    Context:
        {context}
    """
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    user_message = {
        "role": "user",
        "content": question
    }
    messages = [system_message, user_message]

    response = client.chat.completions.create(model=groqmodel, messages=messages)
    answer = response.choices[0].message
    
    print("*"*30, "Reasoning", "*"*30)
    print(answer.reasoning)
    
    print("*"*30, "Answer", "*"*30)
    print(answer.content)

    return answer.content

query = "what does my wife like?"
ask_llm(query)