"""
Typed State Definition for Customer Support & Operations Agent.
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Currently there are no security flags or trace tokens stored here yet
class SupportAgantState(TypedDict):
    """
    MessagesState holding conversational turns and tool calls
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]