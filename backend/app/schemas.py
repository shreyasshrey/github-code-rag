from pydantic import BaseModel, Field

class RepositoryIngestRequest(BaseModel):
    repo_url: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)

class ApiResponse(BaseModel):
    message: str
