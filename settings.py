from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # populates process env for libraries (e.g. langchain) that read os.environ directly


class Settings(BaseSettings):
    """
    Central .env-backed config. Instantiating raises pydantic.ValidationError
    listing every missing required variable, so a bad .env fails at import
    time instead of mid-pipeline.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langgraph_client_url: str = Field(
        description="The URL to connect with langgraph client"
    )
    langgraph_client_api: str = Field(
        default="", description="The API key for accessing the langgraph client"
    )
    # required -- pipeline cannot run without these
    embedding_model_id: str = Field(
        description="HF embedding model id, e.g. BAAI/bge-m3"
    )
    chat_model_id: str = Field(
        description="Claude model id used for query generation, the RAG agent, and answer generation"
    )
    anthropic_api_key: str = Field(
        description="Required by every langchain init_chat_model call (read implicitly from process env)"
    )
    chroma_api_key: str = Field(description="Chroma Cloud API key")
    chroma_tenant_id: str = Field(description="Chroma Cloud tenant id")
    chroma_db_name: str = Field(description="Chroma Cloud database name")

    # optional
    semantic_scholar_api_key: str | None = Field(
        default=None,
        description="Optional but recommended; avoids 429s on Semantic Scholar's public endpoint",
    )
    log_level: str = Field(
        default="INFO", description="Python logging level for make.py's basicConfig"
    )

    eval_chat_model_id: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Chat model used only by tests/eval_vectordb.py to synthesize eval queries",
    )


settings = Settings()  # type: ignore[call-arg]
