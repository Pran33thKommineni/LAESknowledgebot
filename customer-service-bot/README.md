# Customer Service Bot

An enterprise-ready AI customer service bot with multi-LLM provider support, hybrid knowledge management, and a scalable REST API.

## Features

- **Multi-Provider LLM Support**: Switch between OpenAI (GPT-4) and Anthropic (Claude) at runtime
- **Hybrid Knowledge Base**: Combines static FAQs with RAG document retrieval
- **Conversation Memory**: Maintains context across messages within a session
- **Fully Customizable**: All company-specific settings via YAML configuration files
- **Production Ready**: Async FastAPI, structured logging, health checks
- **Extensible**: Easy to add new LLM providers or knowledge sources

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd customer-service-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# At minimum, set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 3. Run the Server

```bash
# Start the server
uvicorn app.main:app --reload

# Or run directly
python -m app.main
```

The API will be available at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send a message and get a response |
| `/chat/new` | POST | Start a new conversation with greeting |
| `/chat/{session_id}/history` | GET | Get conversation history |
| `/chat/{session_id}` | DELETE | Clear a conversation |

### Admin

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/reload-knowledge` | POST | Reload config and re-index documents |
| `/admin/providers` | GET | List available LLM providers |
| `/admin/config` | GET | Get current configuration |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |

## Usage Examples

### Send a Message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?"}'
```

Response:
```json
{
  "response": "We offer a 30-day return policy on all items...",
  "session_id": "abc123-def456",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Continue a Conversation

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I initiate a return?",
    "session_id": "abc123-def456"
  }'
```

### Use a Specific Provider

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "provider": "anthropic"
  }'
```

## Configuration

### Company Settings (`config/company.yaml`)

Customize your company's identity and behavior:

```yaml
company:
  name: "Your Company"
  industry: "Your Industry"
  tone: "friendly and professional"
  escalation_email: "support@yourcompany.com"
  business_hours: "9am-5pm EST"
  
  # Topics that should be escalated to humans
  escalation_topics:
    - billing disputes
    - legal matters
```

### FAQ Knowledge Base (`config/faqs.yaml`)

Add frequently asked questions:

```yaml
faqs:
  - question: "What is your return policy?"
    answer: "We offer 30-day returns..."
    keywords:
      - return
      - refund
      - exchange
```

### LLM Providers (`config/providers.yaml`)

Configure AI model settings:

```yaml
default_provider: "openai"

providers:
  openai:
    enabled: true
    model: "gpt-4o"
    temperature: 0.7
    
  anthropic:
    enabled: true
    model: "claude-3-5-sonnet-20241022"
```

### System Prompts (`config/prompts.yaml`)

Customize the AI's personality and behavior with template variables.

## Document Ingestion (RAG)

Place documents in the `documents/` folder to automatically index them:

- Supported formats: `.txt`, `.md`, `.pdf`, `.docx`
- Documents are chunked and embedded for semantic search
- Reload with: `POST /admin/reload-knowledge`

## Project Structure

```
customer-service-bot/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Request/Response schemas
│   ├── core/
│   │   ├── conversation.py  # Conversation manager
│   │   ├── llm_providers.py # LLM abstraction
│   │   └── knowledge.py     # Knowledge hub
│   ├── rag/
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   └── utils/
│       └── logging.py
├── config/                  # YAML configuration files
├── documents/               # Documents for RAG
├── tests/                   # Test suite
├── requirements.txt
└── .env.example
```

## Extending the Bot

### Adding a New LLM Provider

1. Create a new provider class in `app/core/llm_providers.py`:

```python
class MyProvider(LLMProvider):
    def _create_model(self) -> BaseChatModel:
        # Return your LangChain model
        pass
```

2. Register it:

```python
LLMProviderFactory.register_provider("myprovider", MyProvider)
```

3. Add configuration in `config/providers.yaml`

### Adding Custom Knowledge Sources

Extend the `KnowledgeHub` class in `app/core/knowledge.py` to add new retrieval methods.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_api.py -v
```

## Production Deployment

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes* |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes* |
| `HOST` | Server host | No (default: 0.0.0.0) |
| `PORT` | Server port | No (default: 8000) |
| `LOG_LEVEL` | Logging level | No (default: INFO) |

*At least one LLM provider API key is required.

### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Scaling Considerations

- **Session Storage**: Replace in-memory `ConversationStore` with Redis for horizontal scaling
- **Vector Store**: ChromaDB supports persistent storage; consider Pinecone or Weaviate for larger deployments
- **Rate Limiting**: Add rate limiting middleware for production
- **Authentication**: Implement API key or OAuth authentication

## License

MIT License - feel free to use and modify for your needs.
