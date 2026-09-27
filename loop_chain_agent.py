from operator import mod
import os
import json
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

JOB_DESCRIPTION = """
We are hiring a Backend Python Developer.

Requirements:
- Strong Python
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""

RESUME = """
Name: Rahul Sharma

Experience:
3 years as a Software Developer.

Skills:
Python, FastAPI, MySQL, Docker,
REST APIs, Git

Projects:
Built a food delivery backend using
FastAPI and MySQL.

Deployed applications using Docker.
"""

def ask_llm(system_prompt, user_prompt, response_format=None):
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    user_message = {
        "role": "user",
        "content": user_prompt
    }

    messages = [system_message, user_message]
    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    answer = response.choices[0].message
    print("**"*100, "Reasoning", "**"*100)
    print(answer.reasoning)
    
    print("**"*100, "Answer", "**"*100)
    print(answer.content)

    content = answer.content

    if response_format and response_format["type"] == "json_object":
        content = json.loads(content or "{}")

    return content

def resume_skills_extract_agent(RESUME):
    print("------------- RESUME SKILLS EXTRACT ----------------")
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the candidates resume provided.
    Only return the skills no other information. Do not invent any skills by yourself.
    Output Fomrat:
        Skills should be seperated by comas. Just return comma seperated skills do not return any other filler information.
    EX:
        Git, Bash, Docker, Linux, Css, HTML, C++, Cursor
    """

    user_prompt = f"""
    Extract the skills from the resume
    {RESUME}
    """

    return ask_llm(system_prompt, user_prompt)

def job_description_skills_extract_agent(JOB_DESCRIPTION):
    print("------------- JOB DESCRIPTION SKILLS EXTRACT ----------------")
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the Job description provided.
    Only return the skills no other information. Do not invent any skills by yourself.
    Output Fomrat:
        Skills should be seperated by comas. Just return comma seperated skills do not return any other filler information.
    EX:
        Git, Bash, Docker, Linux, Css, HTML, C++, Cursor
    """

    user_prompt = f"""
    Extract the skills from the job description
    {JOB_DESCRIPTION}
    """

    return ask_llm(system_prompt, user_prompt)


def job_description_and_resume_skills_match_agent(JOB_DESCRIPTION_SKILLS, RESUME_SKILLS):
    print("------------- JOB DESCRIPTION SKILLS AND RESUME SKILLS MATCHING ----------------")
    system_prompt = """
    You are a professional HR assistant. Compare the skills of the candidate against the Job description skills and provide a final score
    between 1 and 100. Also produce a short verdict whther the candidate is a good fit for the role.
    
    Output Fomrat:
        - The response should be in JSON fromat
    Ex:
        {
            "score": 89,
            "vardict": "The verdict for if the candidate is a goog fot or not"
        }
    """

    user_prompt = f"""
    Compare and match the skills
    
    RESUME SKILLS:
    {RESUME_SKILLS}

    JOB DESCRIPTION SKILLS:
    {JOB_DESCRIPTION_SKILLS}
    """
    response_fomat = {
        "type": "json_object"
    }
    
    return ask_llm(system_prompt, user_prompt, response_format=response_fomat)


groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "resume_skills_extract_agent",
            "descriptioin": "This is an agentic tool that extracts skills from a resume text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "RESUME" : {"type": "string"}
                },
                "required": ["RESUME"]
            },
            "returns": "String of skills seperated by commas"
        }
    },
    {
        "type": "function",
        "function": {
            "name": "job_description_skills_extract_agent",
            "descriptioin": "This is an agentic tool that extracts skills from a job description text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "JOB_DESCRIPTION" : {"type": "string"}
                },
                "required": ["JOB_DESCRIPTION"]
            },
            "returns": "String of skills seperated by commas"
        }
    },
    {
        "type": "function",
        "function": {
            "name": "job_description_and_resume_skills_match_agent",
            "descriptioin": "This is an agentic tool that matches resume skills against job description skills and provides a verdic and a score from 1 to 100.",
            "parameters": {
                "type": "object",
                "properties": {
                    "JOB_DESCRIPTION_SKILLS" : {"type": "string"},
                    "RESUME_SKILLS" : {"type": "string"}
                },
                "required": ["JOB_DESCRIPTION_SKILLS", "RESUME_SKILLS"]
            },
            "returns": "A Json object with two keys vardict and score"
        }
    }
]

tools = {
    "resume_skills_extract_agent": resume_skills_extract_agent,
    "job_description_skills_extract_agent": job_description_skills_extract_agent,
    "job_description_and_resume_skills_match_agent": job_description_and_resume_skills_match_agent
}

system_prompt = """
You are a HR manager. You have a few tools at your disposal.
Since you are a manager your job is to use these tools to get a match score and verdict by matching a resume against a job description.

These are the tools you have:
1. resume_skills_extract_agent
2. job_description_skills_extract_agent
3. job_description_and_resume_skills_match_agent


IMPORTANT:
- Call only 1 tool at a time.
- You do not do any grunt work, your job is to delegate the work to the tools.
- In case you do not have a tool that you need in the process of completing the task, then will ask for this tool that you need with a name and a short description.
- This does not mean you do not do 'any' work. You get the results from the tools and decide what you need to do next, and if the result you got is enough so that you can return the result to the user.

"""

user_prompt = f"""
Match this Job description agains the candidate resume and provide a verdict and a score.
Job description:
{JOB_DESCRIPTION}
Resume:
{RESUME}
"""

system_message = {
    "role": "system",
    "content": system_prompt
}
user_message = {
    "role": "user",
    "content": user_prompt
}

messages = [system_message, user_message]

for step in range(1, 6):
    print("--------------------------------\n", f"STEP: {step}\n", "---------------------------\n")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=groq_tools,
        tool_choice="auto",
        temperature=0
    )
    
    answer = response.choices[0].message
    content = answer.content
    tool_calls = answer.tool_calls
    
    print("**"*30, "Reasoning", "**"*30)
    print(answer.reasoning)
    
    print("**"*30, "Tools", "**"*30)
    print(tool_calls)
    
    print("**"*30, "Answer", "**"*30)
    print(content)
    
    messages.append(answer)
    if content:
        print("+"*50)
        print("Answer: ", content)
        print("+"*50)
        break
    elif tool_calls:
        id = tool_calls[0].id
        function = tool_calls[0].function.name

        if function not in tools:
            observation = "The called tool in not available, these are the available tools: "
            for key in tools:
                observation += key + "|"
        else:
            function_name = tools[function]
            kwargs = json.loads(tool_calls[0].function.arguments)
            observation = function_name(**kwargs)

        print("observation:", observation)
        messages.append({
            "role": "tool",
            "tool_call_id": str(id),
            "content": str(observation)
        })
    else:
        observation = "Nither a tool was called nor a response was given. Either should be provided."
        print("observation:", observation)
        messages.append({
            "role": "system",
            "content": observation
        })
        
print("******************CAME OUT OF THE LOOP******************")