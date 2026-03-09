"""Conversation manager with session handling and context building."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.config import get_config
from app.core.knowledge import get_knowledge_hub
from app.core.llm_providers import get_llm_factory
from app.utils.logging import get_logger

logger = get_logger("conversation")


@dataclass
class Message:
    """A single message in a conversation."""
    
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """A conversation session with message history."""
    
    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        return message
    
    def get_history(self, max_messages: Optional[int] = None) -> list[dict]:
        """
        Get conversation history as list of dicts.
        
        Args:
            max_messages: Maximum number of messages to return
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = self.messages
        if max_messages:
            messages = messages[-max_messages:]
        
        return [{"role": m.role, "content": m.content} for m in messages]


class ConversationStore:
    """
    In-memory conversation storage.
    
    For production, replace with Redis or database storage.
    """
    
    def __init__(self):
        """Initialize the conversation store."""
        self._conversations: dict[str, Conversation] = {}
    
    def create(self, session_id: Optional[str] = None) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            New Conversation instance
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        conversation = Conversation(session_id=session_id)
        self._conversations[session_id] = conversation
        logger.debug(f"Created conversation: {session_id}")
        
        return conversation
    
    def get(self, session_id: str) -> Optional[Conversation]:
        """
        Get a conversation by session ID.
        
        Args:
            session_id: Session ID to look up
            
        Returns:
            Conversation or None if not found
        """
        return self._conversations.get(session_id)
    
    def get_or_create(self, session_id: str) -> Conversation:
        """
        Get existing conversation or create new one.
        
        Args:
            session_id: Session ID
            
        Returns:
            Conversation instance
        """
        conversation = self.get(session_id)
        if conversation is None:
            conversation = self.create(session_id)
        return conversation
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.debug(f"Deleted conversation: {session_id}")
            return True
        return False
    
    def list_sessions(self) -> list[str]:
        """Get list of all session IDs."""
        return list(self._conversations.keys())


class ConversationManager:
    """
    Manages conversations, context building, and LLM interactions.
    """
    
    def __init__(
        self,
        max_history_messages: int = 10,
        provider: Optional[str] = None,
    ):
        """
        Initialize the conversation manager.
        
        Args:
            max_history_messages: Maximum messages to include in context
            provider: LLM provider to use (uses default if not specified)
        """
        self.max_history_messages = max_history_messages
        self.provider = provider
        
        self.store = ConversationStore()
        self.knowledge_hub = get_knowledge_hub()
        self.llm_factory = get_llm_factory()
        self._config = get_config()
    
    def _build_system_prompt(self, context: str = "") -> str:
        """
        Build the full system prompt with context.
        
        Args:
            context: Retrieved context to include
            
        Returns:
            Complete system prompt
        """
        base_prompt = self._config.get_formatted_system_prompt()
        
        if context:
            context_prompt = self._config.prompts.context_prompt.format(
                context=context
            )
            return f"{base_prompt}\n\n{context_prompt}"
        
        return base_prompt
    
    def _build_escalation_response(self, topic: str) -> str:
        """
        Build an escalation response for sensitive topics.
        
        Args:
            topic: The escalation topic matched
            
        Returns:
            Escalation response text
        """
        return self._config.prompts.escalation_prompt.format(
            topic=topic,
            escalation_email=self._config.company.escalation_email,
            business_hours=self._config.company.business_hours,
        )
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        provider: Optional[str] = None,
    ) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            session_id: Conversation session ID
            message: User message
            provider: Optional LLM provider override
            
        Returns:
            Assistant response
        """
        # Get or create conversation
        conversation = self.store.get_or_create(session_id)
        
        # Check for escalation topics
        escalation_topic = self.knowledge_hub.check_escalation_topic(message)
        if escalation_topic:
            response = self._build_escalation_response(escalation_topic)
            conversation.add_message("user", message)
            conversation.add_message("assistant", response, escalated=True, topic=escalation_topic)
            logger.info(f"Escalated topic detected: {escalation_topic}")
            return response
        
        # Retrieve relevant context
        context = self.knowledge_hub.get_formatted_context(message)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(context)
        
        # Add user message to history
        conversation.add_message("user", message)
        
        # Get conversation history
        history = conversation.get_history(self.max_history_messages)
        
        # Generate response
        try:
            llm_provider = self.llm_factory.get_provider(provider or self.provider)
            response = await llm_provider.generate(history, system_prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = (
                "I apologize, but I'm having trouble processing your request right now. "
                f"Please try again or contact us at {self._config.company.escalation_email}."
            )
        
        # Add assistant response to history
        conversation.add_message("assistant", response)
        
        return response
    
    def get_greeting(self) -> str:
        """
        Get the greeting message for new conversations.
        
        Returns:
            Greeting message
        """
        return self._config.get_formatted_greeting()
    
    def get_conversation_history(self, session_id: str) -> list[dict]:
        """
        Get the full conversation history.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of message dicts
        """
        conversation = self.store.get(session_id)
        if conversation is None:
            return []
        return conversation.get_history()
    
    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear/delete a conversation.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            True if cleared, False if not found
        """
        return self.store.delete(session_id)
    
    def start_new_conversation(self, session_id: Optional[str] = None) -> tuple[str, str]:
        """
        Start a new conversation with greeting.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Tuple of (session_id, greeting_message)
        """
        conversation = self.store.create(session_id)
        greeting = self.get_greeting()
        conversation.add_message("assistant", greeting)
        
        return conversation.session_id, greeting


# Global conversation manager instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """
    Get the global conversation manager instance.
    
    Returns:
        ConversationManager instance
    """
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
