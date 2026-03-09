"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")


@pytest.fixture(scope="session")
def project_root_path():
    """Get the project root path."""
    return project_root


@pytest.fixture
def sample_faq():
    """Sample FAQ item for testing."""
    return {
        "question": "What is your return policy?",
        "answer": "We offer 30-day returns on all items.",
        "keywords": ["return", "refund", "exchange"],
    }


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "What is your return policy?"},
    ]
