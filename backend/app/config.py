from pathlib import Path

from pydantic import Field
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
        gt=0,
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
    )

    retrieval_k: int = Field(
        default=8,
        gt=0,
    )

    # cors_origins: str = "http://localhost:3000"
    cors_origins: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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