# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Setup and Installation:**
```bash
# Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Development dependencies
pip install -e ".[dev]"
```

**Testing:**
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_knowledge_graph.py

# Run with asyncio support
pytest -v tests/
```

**Code Quality:**
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

**Configuration Testing:**
```bash
# Test configuration loading
python -c "from src.config import initialize_config; config = initialize_config(); print('Configuration loaded successfully!')"

# Test Neo4j connection
python -c "
import asyncio
from src.knowledge_graph import KnowledgeGraph

async def test():
    async with KnowledgeGraph() as kg:
        stats = await kg.get_knowledge_statistics()
        print(f'Connected to Neo4j! Stats: {stats}')

asyncio.run(test())
"
```

**Usage Monitoring:**
```bash
# Check API usage and costs
python -c "
import asyncio
from src.cloud_nlp import OpenRouterProcessor

async def check_usage():
    async with OpenRouterProcessor() as processor:
        summary = processor.get_usage_summary()
        print(f'Daily: \${summary[\"costs\"][\"daily_cost\"]:.2f}')
        print(f'Monthly: \${summary[\"costs\"][\"monthly_cost\"]:.2f}')

asyncio.run(check_usage())
"
```

## Architecture Overview

This is a Personal Knowledge Graph Server that transforms files and web content into an intelligent knowledge network using AI. The system follows a layered architecture:

**Core Components:**
- `src/config.py`: Central configuration management with environment variable substitution and validation
- `src/cloud_nlp.py`: OpenRouter API integration for AI-powered entity extraction and relationship mapping
- `src/knowledge_graph.py`: Neo4j graph database operations for storing and querying knowledge

**Data Flow:**
1. **Input Processing**: Files from `E:\GraphKnowledge` or web URLs are processed
2. **AI Analysis**: OpenRouter API (Claude/GPT-4/Mixtral) extracts entities and relationships
3. **Graph Storage**: Neo4j stores the interconnected knowledge with full-text search
4. **Query Interface**: MCP tools provide search and discovery for AI assistants

**Configuration System:**
- YAML-based configuration with environment variable substitution (`${VAR_NAME}`)
- Multiple model selection based on task complexity (simple/complex/batch)
- Budget tracking and cost management with configurable limits
- Privacy filtering with sensitive data detection and redaction

**Key Abstractions:**
- `Entity`: Represents extracted entities (people, concepts, organizations) with confidence scores
- `Relationship`: Represents connections between entities with types and context
- `KnowledgeGraph`: Async context manager for Neo4j operations
- `OpenRouterProcessor`: Handles AI API requests with cost tracking and privacy filtering

**Development Status:**
Currently in Phase 1 (Foundation). Core components implemented: configuration, OpenRouter integration, Neo4j operations. In progress: file monitoring, MCP server, web scraping.

## Environment Setup

**Required Environment Variables:**
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=personal-knowledge
WATCH_DIRECTORY=E:\GraphKnowledge
PROCESSED_DIRECTORY=E:\GraphKnowledge\processed
INBOX_DIRECTORY=E:\GraphKnowledge\inbox
```

**Directory Structure:**
```
E:\GraphKnowledge\
├── inbox/          # Files to be processed
├── processed/      # Completed files
├── test/          # Test files
├── sensitive/     # Local processing only
└── private/       # Local processing only
```

**Neo4j Schema:**
- `Entity` nodes with properties: name, type, description, confidence, mentions
- `RELATES_TO` relationships with type, confidence, context
- `SourceFile` nodes for provenance tracking
- Full-text indexes for search functionality

## Testing Patterns

**File Processing Test:**
```python
# Create test file and process
echo "# AI Research Notes
Key person: @AndrewNg
Organization: OpenAI
Concept: transformer architecture" > E:\GraphKnowledge\test\sample.md

# Process with cloud_nlp
python -c "
import asyncio
from src.cloud_nlp import process_file_content

async def test():
    with open('E:/GraphKnowledge/test/sample.md', 'r') as f:
        content = f.read()
    entities, relationships = await process_file_content(
        'E:/GraphKnowledge/test/sample.md', content
    )
    print(f'Entities: {len(entities)}, Relationships: {len(relationships)}')

asyncio.run(test())
"
```

**Knowledge Graph Queries:**
```python
# Search and analyze
python -c "
import asyncio
from src.knowledge_graph import search_knowledge_graph, get_entity_info

async def test():
    results = await search_knowledge_graph('AI', limit=5)
    print('Search results:', [r['name'] for r in results])
    
    if results:
        info = await get_entity_info(results[0]['name'])
        print(f'Connections: {info[\"total_connections\"]}')

asyncio.run(test())
"
```

## Important Implementation Notes

- All async operations use context managers (`async with`) for proper resource management
- Budget limits are enforced before API calls to prevent cost overruns  
- Sensitive data is automatically filtered using regex patterns before cloud processing
- Neo4j operations include proper error handling and schema initialization
- Configuration validation ensures required environment variables are set
- Model selection is automatic based on task complexity to optimize cost/quality