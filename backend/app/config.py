from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    huggingfacehub_api_token: str = Field(
        ...,
        min_length=1,
    )

    hf_embedding_model: str = Field(
        ...,
        min_length=1,
    )

    hf_chat_model: str = Field(
        ...,
        min_length=1,
    )

    api_key: str = Field(
        ...,
        min_length=32,
    )

    repo_path: str = "./repo"

    chroma_path: str = "./db"

    chunk_size: int = Field(
        default=2000,
        gt=0,           # Must be greater than 0
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,           # Must be greater than or equal to 0
    )

    retrieval_k: int = Field(
        default=8,
        gt=0,           # Must be greater than 0
    )

    # cors_origins: str = "http://localhost:3000"
    cors_origins: str = Field(
        ...,
        min_length=1,
    )

    max_repo_size_mb: int = Field(
        default=500,
        gt=0,           # Must be greater than 0
    )

    max_python_files: int = Field(
        default=2000,
        gt=0,           # Must be greater than 0
    )

    max_chunks: int = Field(
        default=20000,
        gt=0,           # Must be greater than 0
    )
    
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_settings(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be less than chunk_size."
            )

        return self
    
    @property
    def repo_dir(self) -> Path:
        return BASE_DIR / self.repo_path

    @property
    def chroma_dir(self) -> Path:
        return BASE_DIR / self.chroma_path

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


settings = Settings()