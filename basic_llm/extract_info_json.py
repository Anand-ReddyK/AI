import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("No API key")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-20b"

class Ticket(BaseModel):
    name: str
    email: str
    issue: str

schema = Ticket.model_json_schema()

# print(schema)

response_format = {
    "type": "json_object"
}

system_prompt = f"""
Your job is to extract personal information from the ticket strictly based on the schema and respond in json output.
{schema}
"""

system_message = {
    "role": "system",
    "content": system_prompt
}

text="Hello My name is Pratyush. Yesterday I broke up with my girlfriend sheetal I have an iphone which is not working at all. My address is delhi. My email is abc@gmail.com. My contact number is 82134"

prompt = f"""
This is a customer ticket. Please extract the personal information from this.
{text}
"""

message = {
    "role": "user",
    "content": prompt
}

messages = [system_message, message]

response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)

answer = response.choices[0].message.content
reason = response.choices[0].message.reasoning

print(answer)
print(reason)

import json
raw_json = answer
data = json.loads(raw_json)
ticket = Ticket(**data)

print(ticket, "\n\n", ticket.name, ticket.email, ticket.issue)
