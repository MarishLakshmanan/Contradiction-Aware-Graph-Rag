from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


class EmbeddingClient:
    """
    A class for connecting with any embedding model that supports the OpenAI API
    """

    def __init__(self, model_url: str, model_id: str):
        self.model = OpenAIEmbeddings(
            model=model_id,
            base_url=model_url,
            api_key=SecretStr("local"),
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )
