import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print(GROQ_API_KEY)

if not GROQ_API_KEY:
  print("No API KEY Found")

client = Groq(api_key=GROQ_API_KEY)

chat_completion = client.chat.completions.create(
  messages=[
    {
      "role": "user",
      "content": "what is your name?"
    }
  ],
  model="openai/gpt-oss-20b",
)

print("Answer: ", chat_completion.choices[0].message.content)
print("Reasoning: ", chat_completion.choices[0].message.reasoning)


