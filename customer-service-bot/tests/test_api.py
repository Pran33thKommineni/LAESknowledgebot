"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health endpoint."""
    
    def test_health_check(self, client):
        """Test that health check returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestChatEndpoints:
    """Tests for chat endpoints."""
    
    def test_new_conversation(self, client):
        """Test starting a new conversation."""
        response = client.post("/chat/new")
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "greeting" in data
        assert len(data["session_id"]) > 0
        assert len(data["greeting"]) > 0
    
    def test_new_conversation_with_custom_id(self, client):
        """Test starting a new conversation with custom session ID."""
        response = client.post(
            "/chat/new",
            json={"session_id": "custom-session-123"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "custom-session-123"
    
    def test_get_empty_history(self, client):
        """Test getting history for non-existent session."""
        response = client.get("/chat/nonexistent-session/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []
        assert data["message_count"] == 0
    
    def test_clear_conversation(self, client):
        """Test clearing a conversation."""
        # First create a conversation
        new_response = client.post("/chat/new")
        session_id = new_response.json()["session_id"]
        
        # Then clear it
        response = client.delete(f"/chat/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_clear_nonexistent_conversation(self, client):
        """Test clearing a non-existent conversation."""
        response = client.delete("/chat/nonexistent-session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestAdminEndpoints:
    """Tests for admin endpoints."""
    
    def test_list_providers(self, client):
        """Test listing LLM providers."""
        response = client.get("/admin/providers")
        
        assert response.status_code == 200
        data = response.json()
        assert "default_provider" in data
        assert "available_providers" in data
        assert isinstance(data["available_providers"], list)
    
    def test_get_config(self, client):
        """Test getting current configuration."""
        response = client.get("/admin/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "company" in data
        assert "faq_count" in data
        assert "default_provider" in data


class TestChatValidation:
    """Tests for chat request validation."""
    
    def test_empty_message_rejected(self, client):
        """Test that empty messages are rejected."""
        response = client.post(
            "/chat",
            json={"message": ""},
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_message_required(self, client):
        """Test that message field is required."""
        response = client.post(
            "/chat",
            json={},
        )
        
        assert response.status_code == 422  # Validation error
