import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is required")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-20b"

prompt = "Explain how the future of tech might look like mainly how the AI and LLM models, give me a browd view in 2 parahs"
message = {
    "role": 'user',
    "content": prompt
}

messages = [message]
stream = client.chat.completions.create(model=model, messages=messages, stream=True)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)