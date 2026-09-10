import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print(GROQ_API_KEY)

if not GROQ_API_KEY:
  print("No API KEY Found")

client = Groq(api_key=GROQ_API_KEY)

model = "openai/gpt-oss-20b"

role = "user"
prompt = "Can you suggest me some names for my clothing company that attracts Gen-Z customers?"
# SYSTEN PROMPT
message_system = {
  "role": "system",
  "content": """
  You are a brand manager, who suggests names. Only respond names.

  The Respons should have only the names seperated by commas.

  For example:
  name1, name2, name3
  """
}

# Message role and content
message = {
  "role": role,
  "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model=model, messages=messages, temperature=0)
print("#######################################")
names = response.choices[0].message.content
print("Answer: ", names)
print("Reasoning: ", response.choices[0].message.reasoning)

# names = "VibeThreads, FluxWear, NeonNest, PulsePop, EchoEdge, TrendTide, HypeHue, UrbanPulse, NovaNook, FluxFash, ZestZone, PrismPop, RebelRidge, SyncStyle, KineticKouture"

# li_name = list(names)
names = names.split(',')
for i in range(len(names)):
  names[i] = names[i].strip()

print(names)
