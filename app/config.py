from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    MISTRAL_API_KEY: str
    MISTRAL_CHAT_MODEL: str = "mistral-large-latest"
    MISTRAL_EMBED_MODEL: str = "mistral-embed"

    # Qdrant Cloud
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str = "SmartAiChatbot"

    # Embedding
    EMBEDDING_DIM: int = 1024 

    # Retrieval
    RETRIEVAL_SCORE_THRESHOLD: float = 0.65
    RETRIEVAL_TOP_K: int = 3

    # Memory
    MEMORY_MAX_TURNS: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Singleton — import this everywhere
settings = Settings()
