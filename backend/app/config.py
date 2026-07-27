from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    huggingfacehub_api_token: str
    hf_embedding_model: str
    hf_chat_model: str
    repo_path: str
    chroma_path: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    cors_origins: str

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def repo_dir(self) -> Path:
        return BASE_DIR / self.repo_path

    @property
    def chroma_dir(self) -> Path:
        return BASE_DIR / self.chroma_path

    @property
    def allowed_origins(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

settings = Settings()
