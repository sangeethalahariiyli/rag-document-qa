from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "models/embedding-001"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retriever_k: int = 4

    # Storage
    vectorstore_path: str = "./vectorstore"
    upload_dir: str = "./data/uploads"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Single shared instance — import this everywhere
settings = Settings()