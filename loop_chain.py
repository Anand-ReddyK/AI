import re
import os
import math
import json
from sqlite3.dbapi2 import connect
from groq import Groq
from time import sleep
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is required")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-20b"

def get_product_price(product):
    if product == "iPhone 17":
        return 1000

    if product == "iPhone 15":
        return 500
    
    return 200

def calculator(expression):
    try:
        return eval(expression)
    except Exception as e:
        return "calculator error!" + str(e)

tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}

system_prompt = """
You are a shopping assistant.

You have these tools:
get_product_price(product)
calculator(expression)

Format:
1. The tool call should be passed through the 'tool_calls' in the format mentioned above.
2. The response should be in JSON Format.
Ex:
    {
        "message": "what you need to do",
        "isfinal": false
    }
    OR
    {
        "message": "This is the final answer",
        "isfinal": true
    }
3. In the Response the 'message' should be the reasoning of 'what to do now' or 'the final answer' 
    and 'isfinal' should be a boolean string value indicating if the given response is the final answer or not


Follow the rules:
1. Decide what you need to do next.
2. Call ONLY ONE tool at a time.
3. Never guess or invent a tool result, if a tool result fails please respond with the tool name and that if failed.
4. Wait until you receive an observation.
5. Then decide your next action.
6. When the task is complete, give the Final Answer.

The Response should be in JSON format.
And the response should be provided exactly as given above.

IMPORTANT:
In the response 'isfinal' needs to be true:
    - When you can answer the question that was given from the user, even if you get to use the tools or not.
    - If you are unable to get the final answer, then explain that in the message and keep the 'isfinal' as true.
    - If you need to get addintional info from the user, then the 'isfinal' needs to be true
    - 'isfinal' only needs to be false when a tool so that you can get the response to get to the answer, if no tool is called that would mean the 'isfinal' needs to be true.
    - 'isfinal' does not only represent if you response did or did not have the 'final answer', but have you gone to a point where you used the tool and reasoning to get enough information to get to the answer as much you could
       and if there is no more things you could do with your tools and constricted capabilities, just provide what you did and what you need more to get to the answer, so here too you need to give 'isfinal' as true.
    - If a tool's result is failed then respond with the explaining that and keep 'isfinal' as true.
      
"""

groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product" : {"type": "string"}
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

response_type = {
    "type": "json_object"
}

def run_agent(question):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(5):

        print("\n", "-"*20)
        print("STEP:", step + 1)
        print("-"*20)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
            temperature=0
        )
        
        answer = response.choices[0].message or ""
        print("**"*30, "Reasoning", "**"*30)
        print(answer.reasoning)

        content = answer.content
        if content:
            content = json.loads(answer.content or "")
        
        print("**"*30, "Tools", "**"*30)
        print(answer.tool_calls)
        
        print("**"*30, "Message", "**"*30)
        print(content)
        
        if content and content["isfinal"] == True:
            print("**"*30, "Final Response", "**"*30)
            print("Response >>>", content["message"])
            break

        messages.append(answer)
    
        print("**"*30, "Observation sent back", "**"*30)
        if answer.tool_calls:
            id = answer.tool_calls[0].id
            function_name = answer.tool_calls[0].function.name
            
            if function_name not in tools:
                observation = "The called tool in not available, these are the available tools: "
                for key in tools.keys():
                    observation += key + "|"
            else:
                function_name = tools[function_name]
                kwargs = json.loads(answer.tool_calls[0].function.arguments)
                observation = function_name(**kwargs)
            
            print(observation)
            messages.append({
                "role": "tool",
                "tool_call_id": id,
                "content": str(observation)
            })
        else:
            observation = "Nither a tool was called nor isfinal is true. Either should be provided."
            print(observation)
            messages.append({
                "role": "system",
                "content": observation
            })
            
            
            
            

# prompt = """
# I have 5000 rupees.
# and how mouch money will i have left?
# """
prompt = """
I have 5000 rupees. What is the price of an iphone 17?
and how mouch money will i have left if i bought it?
"""
run_agent(prompt)
