from fastapi import APIRouter, HTTPException
from app.config import settings
from app.schemas import ApiResponse, ChatRequest, RepositoryIngestRequest
from app.services.repository_service import RepositoryService
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api")
repository_service = RepositoryService(settings.repo_dir, settings.chroma_dir)
ingestion_service = IngestionService(settings)
rag_service = RAGService(settings)

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/repository/ingest")
def ingest_repository(request: RepositoryIngestRequest):
    try:
        repo_dir = repository_service.clone(request.repo_url)
        result = ingestion_service.build_index(repo_dir)
        rag_service.vectorstore = result["vectorstore"]
        rag_service.clear_memory()
        return {"message": "Repository indexed successfully.", "files": result["files"], "chunks": result["chunks"]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/chat")
def chat(request: ChatRequest):
    try:
        if request.question.strip().lower() == "clear":
            repository_service.clear()
            rag_service.vectorstore = None
            rag_service.clear_memory()
            return {"answer": "Repository and vector index cleared.", "sources": []}
        return rag_service.answer(request.question)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.delete("/repository", response_model=ApiResponse)
def clear_repository():
    repository_service.clear()
    rag_service.vectorstore = None
    rag_service.clear_memory()
    return {"message": "Repository and vector index cleared."}
