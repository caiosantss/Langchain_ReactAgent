from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

def load_llm() -> BaseChatModel:
    return ChatGroq(
        model="openai/gpt-oss-20b",
        )