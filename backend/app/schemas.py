from pydantic import BaseModel, Field


class RepositoryIngestRequest(BaseModel):
    repo_url: str = Field(
        ...,
        min_length=1,
        description="Public GitHub repository URL",
        examples=[
            "https://github.com/psf/requests"
        ],
    )


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about the indexed repository",
        examples=[
            "Where is the main entry point?"
        ],
    )


class ApiResponse(BaseModel):
    message: str