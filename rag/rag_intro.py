import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is required")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-20b"

knowledge_base = {
    "age": "The age of anand is 25 years",
    "net worth": "The net worth of anand is 2000"
}

def retrive_info(question: str):
    question = question.lower()
    for key, value in knowledge_base.items():
        if key in question:
            return value

    return "Context not available for this question!"

def ask_llm(question):
    context = retrive_info(question)

    system_prompt = f"""
    - Answer the question in a single sentence.
    - Answer the question only based on the given context.
    - If the context is not available only answer if the question is a general one and can be answer without specific context or information.
    - If the question cannot be answered with confidence or id the required cantext is not given, or if the given context is not sufficient then answer something in the lines of "I don't have enough information about xyz to answer this question.

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

    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message
    
    print("*"*30, "Reasoning", "*"*30)
    print(answer.reasoning)
    
    print("*"*30, "Answer", "*"*30)
    print(answer.content)

    return answer.content

ask_llm("What is anand's worth?")
    