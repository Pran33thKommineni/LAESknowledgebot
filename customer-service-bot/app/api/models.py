"""Request and response models for the API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# Chat Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    provider: Optional[str] = Field(None, description="LLM provider to use (openai, anthropic)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is your return policy?",
                    "session_id": "user-123-session-456",
                    "provider": None,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID for this conversation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "We offer a 30-day return policy on all items...",
                    "session_id": "user-123-session-456",
                    "timestamp": "2024-01-15T10:30:00Z",
                }
            ]
        }
    }


class NewConversationRequest(BaseModel):
    """Request model for starting a new conversation."""
    
    session_id: Optional[str] = Field(None, description="Optional custom session ID")


class NewConversationResponse(BaseModel):
    """Response model for new conversation."""
    
    session_id: str = Field(..., description="Session ID for the new conversation")
    greeting: str = Field(..., description="Initial greeting message")


# ============================================================================
# History Models
# ============================================================================

class MessageItem(BaseModel):
    """A single message in conversation history."""
    
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    
    session_id: str
    messages: list[MessageItem]
    message_count: int


# ============================================================================
# Admin Models
# ============================================================================

class ReloadKnowledgeResponse(BaseModel):
    """Response model for knowledge reload."""
    
    success: bool
    documents_indexed: int
    message: str


class ProviderInfo(BaseModel):
    """Information about an LLM provider."""
    
    name: str
    enabled: bool
    model: str


class ProvidersResponse(BaseModel):
    """Response model for listing providers."""
    
    default_provider: str
    available_providers: list[ProviderInfo]


# ============================================================================
# Health Models
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
