import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import verify_api_key

from app.schemas import ApiResponse, ChatRequest, RepositoryIngestRequest
from app.config import settings
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.services.repository_service import RepositoryService


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api",
)


repository_service = RepositoryService(
    settings.repo_dir,
    settings.chroma_dir,
    settings.max_repo_size_mb,
)


ingestion_service = IngestionService(
    settings,
)


rag_service = RAGService(
    settings,
)


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
def health():
    return {
        "status": "ok",
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
def readiness():

    if not settings.repo_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository directory not found.",
        )

    if not settings.chroma_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChromaDB directory not found.",
        )

    if not settings.huggingfacehub_api_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hugging Face API key is missing.",
        )

    return {
        "status": "ready",
    }
    

@router.post(
    "/repository/ingest",
    # response_model=RepositoryIngestResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
def ingest_repository(
    request: RepositoryIngestRequest,
):
    try:
        logger.info(
            "Starting repository ingestion: %s",
            request.repo_url,
        )

        repo_dir = repository_service.clone(
            request.repo_url,
        )

        result = ingestion_service.build_index(
            repo_dir,
        )

        rag_service.vectorstore = result["vectorstore"]

        rag_service.clear_memory()

        logger.info(
            "Repository indexed successfully. Files: %s, Chunks: %s",
            result["files"],
            result["chunks"],
        )

        return {
            "message": "Repository indexed successfully.",
            "files": result["files"],
            "chunks": result["chunks"],
        }

    except ValueError as exc:
        logger.warning(
            "Invalid repository ingestion request: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "Repository ingestion failed",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Repository ingestion failed.",
        )


@router.post(
    "/chat",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
def chat(
    request: ChatRequest,
):
    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty.",
            )

        if question.lower() == "clear":
            repository_service.clear()

            rag_service.vectorstore = None

            rag_service.clear_memory()

            return {
                "answer": "Repository and vector index cleared.",
                "sources": [],
            }

        return rag_service.answer(
            question,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid chat request: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception(
            "Chat request failed",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat request failed.",
        )


@router.delete(
    "/repository",
    response_model=ApiResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
def clear_repository():
    try:
        repository_service.clear()

        rag_service.vectorstore = None

        rag_service.clear_memory()

        logger.info(
            "Repository and vector index cleared successfully",
        )

        return {
            "message": "Repository and vector index cleared.",
        }

    except Exception:
        logger.exception(
            "Failed to clear repository",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear repository.",
        )