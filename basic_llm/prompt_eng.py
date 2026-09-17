import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is required")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-20b"

def llm_ans(prompt):
    message = {
        "role": "user",
        "content": prompt
    }

    messages = [message]
    response = client.chat.completions.create(model=model, messages=messages)
    message = response.choices[0].message
    print("**"*100, "Reasoning", "**"*100)
    print(message.reasoning)
    
    print("**"*100, "Answer", "**"*100)
    print(message.content)
    
    return message.content


prompt = """
#ROLE:
You are a support assistant at a mobile/laptop company

#TASK:
You have to classify the issue in a category

#CONSTRAINT:
You have to classify the issue in one of three categories namely billing, technical, return.

#OUTPUT FORMAT:
Your answershould be in one word only. The word should be one of the categories given in constraints

#EXAMPLE:
For instance if a user complain says he wants a refund then the category is 'Return'

#FALLBACK
If the issue is unrelated to any of the categories mentioned in constraints, then the naswer should be 'Other'

This is a user complaint:
My marriage is broken
"""

llm_ans(prompt)