"""Application settings."""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Environment-backed settings."""

    vision_agent_api_key: str = Field(default="not-configured", alias="VISION_AGENT_API_KEY")
    anthropic_api_key: str = Field(default="not-configured", alias="ANTHROPIC_API_KEY")

    elasticsearch_url: str = Field(default="http://localhost:39200", alias="ELASTICSEARCH_URL")
    elasticsearch_index: str = Field(default="documents", alias="ELASTICSEARCH_INDEX")

    embedding_provider: str = Field(default="openai", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")
    search_mode: str = Field(default="semantic", alias="SEARCH_MODE")
    semantic_weight: float = Field(default=0.7, alias="SEMANTIC_WEIGHT")
    semantic_index_name: str = Field(default="documents_semantic", alias="SEMANTIC_INDEX_NAME")

    database_url: str = Field(default="sqlite:///database.db", alias="DATABASE_URL")

    api_key: str = Field(default="not-configured", alias="API_KEY")
    pdf_storage_path: str = Field(default="./data/pdfs", alias="PDF_STORAGE_PATH")
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    max_file_size_mb: int = Field(default=5120, alias="MAX_FILE_SIZE_MB")  # 5 GB default
    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    final_context_chunks: int = Field(default=5, alias="FINAL_CONTEXT_CHUNKS")
    max_context_chars: int = Field(default=7500, alias="MAX_CONTEXT_CHARS")
    max_chars_per_chunk: int = Field(default=1500, alias="MAX_CHARS_PER_CHUNK")
    enable_latency_logs: bool = Field(default=False, alias="ENABLE_LATENCY_LOGS")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    ollama_num_predict: int = Field(default=700, alias="OLLAMA_NUM_PREDICT")
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")
    demo_ollama_model: str = Field(default="phi3", alias="DEMO_OLLAMA_MODEL")
    pdf_highlight_lazy: bool = Field(default=True, alias="PDF_HIGHLIGHT_LAZY")

    environment: str = Field(default="development", alias="ENVIRONMENT")
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v_upper

    @validator("max_file_size_mb")
    def validate_max_file_size(cls, v: int) -> int:
        """Validate max file size."""
        if v < 1 or v > 5120:
            raise ValueError("MAX_FILE_SIZE_MB must be between 1 and 5120 (5GB)")
        return v

    @validator("retrieval_top_k", "final_context_chunks")
    def validate_positive_counts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Context limits must be >= 1")
        return v

    @validator("max_context_chars")
    def validate_context_chars(cls, v: int) -> int:
        if v < 1000:
            raise ValueError("MAX_CONTEXT_CHARS must be >= 1000")
        return v

    @validator("max_chars_per_chunk")
    def validate_chunk_chars(cls, v: int) -> int:
        if v < 400:
            raise ValueError("MAX_CHARS_PER_CHUNK must be >= 400")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


settings = get_settings()
