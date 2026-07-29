import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.logging_config import configure_logging

# Must run before any other module in the app creates a logger and starts
# calling .info()/.warning()/etc. on it — otherwise those calls hit the
# default unconfigured root logger (WARNING level, no handlers) and are
# silently dropped. This was previously never imported anywhere, so none
# of the app's logging was actually being written to file or console.
configure_logging()

from app.api.routes import router
from app.config import settings


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GitHub Code RAG Assistant API")

    yield

    logger.info("Shutting down GitHub Code RAG Assistant API")


app = FastAPI(
    title="GitHub Code RAG Assistant",
    description=(
        "A Retrieval-Augmented Generation API for asking questions "
        "about indexed GitHub repositories."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)


app.include_router(router)


@app.get(
    "/",
    tags=["Root"],
)
def root():
    return {
        "message": "GitHub Code RAG API is running.",
        "docs": "/docs",
    }