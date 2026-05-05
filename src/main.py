"""FastAPI app entry point."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.utils.logging import setup_logging, get_logger, set_request_id


# Set up logging
setup_logging(log_level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("Starting Financial Document Assistant")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"PDF Storage Path: {settings.pdf_storage_path}")
    yield
    logger.info("Shutting down Financial Document Assistant")


app = FastAPI(
    title="Financial Document Assistant",
    description="Local assistant for tax, accounting, and financial documents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Attach a request ID for logging and tracing."""
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Financial Document Assistant"
    }


@app.get("/", tags=["Root"])
async def root():
    """Serve the main UI."""
    static_dir = Path(__file__).parent.parent / "static"
    index_path = static_dir / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "service": "Financial Document Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_version": "/api/v1"
    }


@app.get("/pitch.html", tags=["Root"])
async def pitch():
    """Serve optional pitch page if present."""
    static_dir = Path(__file__).parent.parent / "static"
    pitch_path = static_dir / "pitch.html"

    if pitch_path.exists():
        return FileResponse(pitch_path)

    return {"error": "Pitch page not found"}


@app.get("/semantic_search.html", tags=["Root"])
async def semantic_search_page():
    """Serve optional semantic search page if present."""
    static_dir = Path(__file__).parent.parent / "static"
    search_path = static_dir / "semantic_search.html"

    if search_path.exists():
        return FileResponse(search_path)

    return {"error": "Semantic search page not found"}


@app.get("/index.html", tags=["Root"])
async def index_page():
    """Serve the main chat page explicitly."""
    static_dir = Path(__file__).parent.parent / "static"
    index_path = static_dir / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {"error": "Index page not found"}


static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

source_files_dir = Path(__file__).parent.parent / "Source_files"
if source_files_dir.exists():
    app.mount("/source_files", StaticFiles(directory=str(source_files_dir)), name="source_files")

existing_files_dir = Path(__file__).parent.parent / "Existing_files"
if existing_files_dir.exists():
    app.mount("/existing_files", StaticFiles(directory=str(existing_files_dir)), name="existing_files")

from src.api.search import router as search_router
from src.api.documents import router as documents_router
from src.api.feedback import router as feedback_router
from src.api.ai import router as ai_router
from src.api.semantic_search import router as semantic_search_router
from src.api.llama_api import router as llama_router
from src.api.semantic_search_api import router as semantic_search_api_router

app.include_router(search_router)
app.include_router(documents_router)
app.include_router(feedback_router)
app.include_router(ai_router)
app.include_router(semantic_search_router)
app.include_router(llama_router)
app.include_router(semantic_search_api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
