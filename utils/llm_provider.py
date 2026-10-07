"""
LLM Provider Abstraction Layer:
Support for OpenAI, Groq and Gemini via a global configuration
"""
import warnings
import logging
# 1. Suppress the LangChain 'temperature will be ignored' UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
# 2. Suppress Google SDK's internal AFC (Automatic Function Calling) logger notice
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)


import os
from typing import Optional, List, Dict, Literal, Union
from dotenv import load_dotenv

from utils.config import PROVIDER_MODELS

load_dotenv()


# 1. Global Provider Configuration

DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", "google").lower()

# 2. Factory for LangChain Chat Models (Used by RAGAS & LangGraph)

def get_chat_model(
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        provider: Literal["google", "groq"] = DEFAULT_PROVIDER,
        **kwargs
    ):
    """
    Returns a unified LangChain ChatModel instance for Ragas evaluation or LangGraph

    Args:
        model_name: Optional model override. If None, uses default for that provider
        temperature: sampling temperature
        provider: "google" or "groq". Defaults to global AI_PROVIDER    
    """
    selected_provider = provider.lower()

    if selected_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in enironment variable")
        
        chosen_model = model_name if model_name else PROVIDER_MODELS["google"]["eval"]
        print(f"[LLM Factory] Active Evaluator: Google Gemini {chosen_model}")
        return ChatGoogleGenerativeAI(
            model=chosen_model,
            google_api_key=api_key,
            temperature=temperature,
            max_retries=kwargs.pop("max_retries", 5),
            **kwargs
        )
    
    elif selected_provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in enironment variable")
        
        chosen_model = model_name if model_name else PROVIDER_MODELS["groq"]["eval"]
        print(f"[LLM Factory] Active Evaluator: Groq {chosen_model}")
        return ChatGroq(
            model=chosen_model,
            api_key=api_key,
            temperature=temperature,
            max_retries=kwargs.pop("max_retries", 5),
            **kwargs
        )
    else:
        raise ValueError(f"Invalid provider: {selected_provider}. Must be 'google' or 'groq'")


get_eval_llm = get_chat_model


def extract_text_from_response(response) -> str:
    """Universally extracts clean string content across Groq, Gemini, and OpenAI."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Extract text from block lists: [{'type': 'text', 'text': '...'}]
        text_parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return "".join(text_parts).strip()
    return str(content).strip()

    

# 3. Flexible Chat Completion with Multi-Turn & Auto-Fallback
def chat_completion(
    messages: Union[str, List[Dict[str, str]]],
    system_prompt: Optional[str] = None,
    provider: Literal["google", "groq"] = DEFAULT_PROVIDER,
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    fallback_on_error: bool = True,
    **kwargs
) -> str:
    """
    Flexible completion function:
    - If 'messages' is a string: automatically wraps it into a user message.
    - If 'messages' is a list of dicts: preserves full multi-turn conversation history.
    - Forwards arbitrary parameters (**kwargs) like max_tokens, stop, etc.
    - Automatically falls back to the alternative provider on 429 rate limit errors.
    """
    active_provider = (provider or DEFAULT_PROVIDER).lower()

    # Normalize input: allow passing either a single string or a full messages list
    formatted_messages: List[Dict[str, str]] = []

    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    

    if isinstance(messages, str):
        formatted_messages.append({"role": "user", "content": messages})
    
    elif isinstance(messages, list):
        formatted_messages.extend(messages)
    
    else:
        raise TypeError("messages must be a string or a list of message dictionaries.")
    
    try:
        model = get_chat_model(
            provider=active_provider,
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )

        response = model.invoke(formatted_messages)

        return extract_text_from_response(response)
    
    except Exception as e:
        print(f"[LLM Factory] Warning: Provider '{active_provider}' encountered an error: {e}")
        
        # Automatic failover: If Groq hits rate limits, failover to Gemini
        if fallback_on_error and active_provider == "groq":
            print("[LLM Factory] ⚡ Automatically falling back to Google Gemini...")
            return chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                provider="google",
                temperature=temperature,
                fallback_on_error=False,
                **kwargs
            )
        
        raise e


if __name__ == "__main__":
    # Testing the functions
    answer = chat_completion("Hello how are you?")
    print("Answer: ", answer)

    answer = chat_completion(
    messages="Give me a 1-sentence summary.",
    max_tokens=200
    )
    print("Answer: ", answer)
