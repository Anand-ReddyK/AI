import os
import numpy as np
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

def consine_similarity(a, b):
     return np.dot(a, b) / (
         np.linalg.norm(a) * np.linalg.norm(b)
     )

model = SentenceTransformer("all-MiniLM-L6-v2")
text = "Machine learning is fun."

# embedding = model.encode(text)
# print(embedding.shape)
# print(embedding[:10])

a1 = "vacation policy"
t1 = "gym"
# t1 = "There are 24 paid leaves"
# t1 = "I have fever"
# t2 = "I want to take a leave tomorrow"
t2 = "exercise"

va1 = model.encode(a1)
v1 = model.encode(t1)
v2 = model.encode(t2)
# print(v1, v2)
print(consine_similarity(v1, v2))
# print(consine_similarity(va1, v1))