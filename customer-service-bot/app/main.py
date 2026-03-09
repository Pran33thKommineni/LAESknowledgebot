"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app import __version__
from app.api.routes import admin_router, router
from app.config import get_config
from app.core.knowledge import get_knowledge_hub
from app.utils.logging import setup_logging, get_logger

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Runs startup and shutdown logic.
    """
    # Startup
    logger = get_logger("main")
    logger.info("Starting Customer Service Bot...")
    
    # Initialize knowledge hub and index documents
    try:
        knowledge_hub = get_knowledge_hub()
        doc_count = knowledge_hub.index_documents()
        logger.info(f"Indexed {doc_count} document chunks")
    except Exception as e:
        logger.warning(f"Could not index documents: {e}")
    
    config = get_config()
    logger.info(f"Configured for: {config.company.name}")
    logger.info(f"Default LLM provider: {config.providers.default_provider}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Service Bot...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    # Setup logging
    config = get_config()
    setup_logging(level=config.settings.log_level)
    
    # Create FastAPI app
    app = FastAPI(
        title="Customer Service Bot API",
        description="""
        Enterprise-ready customer service bot with multi-LLM provider support.
        
        ## Features
        
        - **Multi-provider LLM support**: Switch between OpenAI and Anthropic
        - **Hybrid knowledge base**: FAQ matching + RAG document retrieval
        - **Conversation memory**: Maintains context across messages
        - **Customizable**: Configure via YAML files
        
        ## Quick Start
        
        1. Send a message to `/chat` with your question
        2. Use the returned `session_id` for follow-up messages
        3. View conversation history at `/chat/{session_id}/history`
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(router)
    app.include_router(admin_router)
    
    # Serve static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        
        @app.get("/", include_in_schema=False)
        async def serve_ui():
            """Serve the chat UI."""
            return FileResponse(STATIC_DIR / "index.html")
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.settings.host,
        port=config.settings.port,
        reload=config.settings.debug,
    )
