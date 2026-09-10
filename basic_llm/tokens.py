import requests
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("No API key")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-20b"

role = "user"
prompts = [
    "Hello",
    "Give me 5 Si-Fi books recommondations for beginners",
    "When will the next Haikyu anime movie will be coming out?"
]

for prompt in prompts:
    message = {
        "role": role,
        "content": prompt
    }
    messages = [message]

    response = client.chat.completions.create(model=model, messages=messages, max_tokens=500)
    print(response, "\n\n")
    usage = response.usage

    print("-"*10, "Resoning", "-"*10)
    print(response.choices[0].message.reasoning, "\n Final Reason:", response.choices[0].finish_reason)

    print("-"*10, "Answer", "-"*10)
    print(response.choices[0].message.content)

    print("-"*10, "Tokens", "-"*10)
    print(f"Prompt Tokens: {usage.prompt_tokens} \nCompletion Tokens: {usage.completion_tokens} \nTotal Tokens: {usage.total_tokens}")

    print("-"*200)
    