"""API routes for the customer service bot."""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app import __version__
from app.api.models import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    ErrorResponse,
    HealthResponse,
    MessageItem,
    NewConversationRequest,
    NewConversationResponse,
    ProviderInfo,
    ProvidersResponse,
    ReloadKnowledgeResponse,
)
from app.config import get_config, reload_config
from app.core.conversation import get_conversation_manager
from app.core.knowledge import get_knowledge_hub
from app.core.llm_providers import get_llm_factory
from app.utils.logging import get_logger

logger = get_logger("api")

# Create routers
router = APIRouter()
admin_router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# Health Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the service is running",
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
    )


# ============================================================================
# Chat Endpoints
# ============================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Send a message",
    description="Send a message and receive a response from the customer service bot",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message and return a response.
    
    - **message**: The user's message (required)
    - **session_id**: Optional session ID for conversation continuity
    - **provider**: Optional LLM provider override (openai, anthropic)
    """
    try:
        conversation_manager = get_conversation_manager()
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process the message
        response = await conversation_manager.process_message(
            session_id=session_id,
            message=request.message,
            provider=request.provider,
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id,
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/chat/new",
    response_model=NewConversationResponse,
    tags=["chat"],
    summary="Start new conversation",
    description="Start a new conversation and receive a greeting",
)
async def new_conversation(
    request: Optional[NewConversationRequest] = None,
) -> NewConversationResponse:
    """Start a new conversation with a greeting message."""
    try:
        conversation_manager = get_conversation_manager()
        
        session_id = request.session_id if request else None
        session_id, greeting = conversation_manager.start_new_conversation(session_id)
        
        return NewConversationResponse(
            session_id=session_id,
            greeting=greeting,
        )
    
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/chat/{session_id}/history",
    response_model=ConversationHistoryResponse,
    tags=["chat"],
    summary="Get conversation history",
    description="Retrieve the message history for a conversation",
    responses={
        404: {"model": ErrorResponse, "description": "Conversation not found"},
    },
)
async def get_history(session_id: str) -> ConversationHistoryResponse:
    """Get the conversation history for a session."""
    conversation_manager = get_conversation_manager()
    
    history = conversation_manager.get_conversation_history(session_id)
    
    if not history:
        # Return empty history (conversation may be new or not exist)
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=[],
            message_count=0,
        )
    
    messages = [MessageItem(role=m["role"], content=m["content"]) for m in history]
    
    return ConversationHistoryResponse(
        session_id=session_id,
        messages=messages,
        message_count=len(messages),
    )


@router.delete(
    "/chat/{session_id}",
    tags=["chat"],
    summary="Clear conversation",
    description="Delete a conversation and its history",
)
async def clear_conversation(session_id: str) -> dict:
    """Clear/delete a conversation."""
    conversation_manager = get_conversation_manager()
    
    deleted = conversation_manager.clear_conversation(session_id)
    
    return {
        "success": deleted,
        "session_id": session_id,
        "message": "Conversation cleared" if deleted else "Conversation not found",
    }


# ============================================================================
# Admin Endpoints
# ============================================================================

@admin_router.post(
    "/reload-knowledge",
    response_model=ReloadKnowledgeResponse,
    summary="Reload knowledge base",
    description="Reload configuration files and re-index documents",
)
async def reload_knowledge() -> ReloadKnowledgeResponse:
    """Reload configuration and re-index documents."""
    try:
        # Reload configuration
        reload_config()
        
        # Re-index documents
        knowledge_hub = get_knowledge_hub()
        count = knowledge_hub.reindex()
        
        return ReloadKnowledgeResponse(
            success=True,
            documents_indexed=count,
            message="Knowledge base reloaded successfully",
        )
    
    except Exception as e:
        logger.error(f"Error reloading knowledge: {e}")
        return ReloadKnowledgeResponse(
            success=False,
            documents_indexed=0,
            message=f"Error reloading knowledge: {str(e)}",
        )


@admin_router.get(
    "/providers",
    response_model=ProvidersResponse,
    summary="List LLM providers",
    description="Get information about available LLM providers",
)
async def list_providers() -> ProvidersResponse:
    """List available LLM providers."""
    config = get_config()
    factory = get_llm_factory()
    
    providers = []
    for name, provider_config in config.providers.providers.items():
        providers.append(ProviderInfo(
            name=name,
            enabled=provider_config.enabled,
            model=provider_config.model,
        ))
    
    return ProvidersResponse(
        default_provider=config.providers.default_provider,
        available_providers=providers,
    )


@admin_router.get(
    "/config",
    summary="Get current configuration",
    description="Get the current company and prompt configuration",
)
async def get_current_config() -> dict:
    """Get current configuration (excluding sensitive data)."""
    config = get_config()
    
    return {
        "company": {
            "name": config.company.name,
            "industry": config.company.industry,
            "tone": config.company.tone,
            "business_hours": config.company.business_hours,
        },
        "faq_count": len(config.faqs.faqs),
        "default_provider": config.providers.default_provider,
        "embeddings_provider": config.providers.embeddings.provider,
    }
