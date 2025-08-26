from utils.logger import logger
from langchain_groq import ChatGroq


def build_model(
    model: str = "llama3-70b-8192",
) -> ChatGroq:
    """
    Build and return ChatGroq model instance.
    """
    try:
        llm = ChatGroq(
            model=model,
        )
        logger.info("Model was built.")
        return llm
    except Exception as e:
        raise ValueError(f"Error building model '{model}': {e}.")