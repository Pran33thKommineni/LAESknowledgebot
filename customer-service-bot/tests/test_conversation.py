"""Tests for conversation management."""

import pytest

from app.core.conversation import (
    Conversation,
    ConversationStore,
    Message,
)


class TestMessage:
    """Tests for Message dataclass."""
    
    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
        assert msg.metadata == {}
    
    def test_message_with_metadata(self):
        """Test creating a message with metadata."""
        msg = Message(
            role="assistant",
            content="Hi there!",
            metadata={"source": "faq"},
        )
        
        assert msg.metadata == {"source": "faq"}


class TestConversation:
    """Tests for Conversation class."""
    
    def test_create_conversation(self):
        """Test creating a conversation."""
        conv = Conversation(session_id="test-123")
        
        assert conv.session_id == "test-123"
        assert conv.messages == []
        assert conv.created_at is not None
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation(session_id="test-123")
        
        msg = conv.add_message("user", "Hello")
        
        assert len(conv.messages) == 1
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_get_history(self):
        """Test getting conversation history."""
        conv = Conversation(session_id="test-123")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")
        
        history = conv.get_history()
        
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}
    
    def test_get_history_with_limit(self):
        """Test getting limited conversation history."""
        conv = Conversation(session_id="test-123")
        for i in range(10):
            conv.add_message("user", f"Message {i}")
        
        history = conv.get_history(max_messages=3)
        
        assert len(history) == 3
        assert history[0]["content"] == "Message 7"
        assert history[2]["content"] == "Message 9"


class TestConversationStore:
    """Tests for ConversationStore class."""
    
    def test_create_conversation(self):
        """Test creating a conversation in the store."""
        store = ConversationStore()
        
        conv = store.create("test-123")
        
        assert conv.session_id == "test-123"
        assert store.get("test-123") is conv
    
    def test_create_with_auto_id(self):
        """Test creating a conversation with auto-generated ID."""
        store = ConversationStore()
        
        conv = store.create()
        
        assert conv.session_id is not None
        assert len(conv.session_id) > 0
    
    def test_get_nonexistent(self):
        """Test getting a non-existent conversation."""
        store = ConversationStore()
        
        result = store.get("nonexistent")
        
        assert result is None
    
    def test_get_or_create_existing(self):
        """Test get_or_create with existing conversation."""
        store = ConversationStore()
        original = store.create("test-123")
        
        result = store.get_or_create("test-123")
        
        assert result is original
    
    def test_get_or_create_new(self):
        """Test get_or_create with new conversation."""
        store = ConversationStore()
        
        result = store.get_or_create("new-session")
        
        assert result.session_id == "new-session"
    
    def test_delete_conversation(self):
        """Test deleting a conversation."""
        store = ConversationStore()
        store.create("test-123")
        
        result = store.delete("test-123")
        
        assert result is True
        assert store.get("test-123") is None
    
    def test_delete_nonexistent(self):
        """Test deleting a non-existent conversation."""
        store = ConversationStore()
        
        result = store.delete("nonexistent")
        
        assert result is False
    
    def test_list_sessions(self):
        """Test listing all sessions."""
        store = ConversationStore()
        store.create("session-1")
        store.create("session-2")
        store.create("session-3")
        
        sessions = store.list_sessions()
        
        assert len(sessions) == 3
        assert "session-1" in sessions
        assert "session-2" in sessions
        assert "session-3" in sessions
