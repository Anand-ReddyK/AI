"""
Production Observability with LangFuse.
Wraps the official Langfuse CallbackHandler for LangGraph.
Automatically logs traces,  spans, token counts, and costs to the Langfuse dashboard.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Check if langfuse is steup
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    print("WARNING: Langfuse is not configured. Skipping Langfuse tracer initialization.")
    LANGFUSE_ENABLED = False
else:
    LANGFUSE_ENABLED = True

try:
    from langfuse.langchain import CallbackHandler
except ImportError:
    CallbackHandler = None
    LANGFUSE_ENABLED = False

def get_langfuse_callback() -> Optional[CallbackHandler]:
    """
    Returns a configured Langfuse CallackHandler for LangChain / LagGraph.
    Automatically picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from .env.
    """

    if not LANGFUSE_ENABLED or CallbackHandler is None:
        print("[OBSERVABILITY] Langfuse is not enabled or credentials are missing from .env.")
        return None

    # In Langfuse v5, CallbackHandler automatically connects using your .env credentials!
    return CallbackHandler()