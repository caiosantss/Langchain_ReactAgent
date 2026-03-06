from typing import Annotated, Sequence, TypedDict
from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]