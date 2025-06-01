# Advanced Knowledge Graph Features 🧠✨

This document describes the enhanced capabilities that transform your Personal Knowledge Graph Server into an intelligent knowledge discovery system.

## 🌟 Overview

The advanced features add sophisticated AI-powered analysis on top of the basic entity extraction and relationship mapping:

- **Advanced Relationship Detection**: Temporal, causal, and hierarchical relationships
- **Semantic Search Engine**: Vector embeddings for intelligent content discovery  
- **Entity Disambiguation**: Automatic duplicate detection and merging
- **Concept Clustering**: Thematic grouping of related entities
- **Knowledge Gap Detection**: Identify missing connections and concepts
- **Enhanced MCP Tools**: Powerful AI assistant integration

## 🧠 Advanced Relationship Detection

### Temporal Relationships
Detects time-based relationships between entities:

- **Before/After**: "X happened before Y", "Y followed X"
- **During**: "X occurred during Y", "while X was happening"
- **Concurrent**: "X and Y happened simultaneously"

```python
async with AdvancedRelationshipDetector() as detector:
    temporal_rels = await detector.detect_temporal_relationships(entities, text)
    
    for rel in temporal_rels:
        print(f"{rel.source_entity} {rel.temporal_indicator} {rel.target_entity}")
        # "GPT-3 before GPT-4" (confidence: 0.85)
```

### Causal Relationships  
Identifies cause-and-effect connections:

- **Causes**: "X causes Y", "X leads to Y"
- **Enables**: "X enables Y", "X makes Y possible"
- **Prevents**: "X prevents Y", "X blocks Y"

```python
causal_rels = await detector.detect_causal_relationships(entities, text)

for rel in causal_rels:
    print(f"{rel.source_entity} {rel.relationship_subtype} {rel.target_entity}")
    # "Transformer architecture enables GPT models" (confidence: 0.92)
```

### Hierarchical Relationships
Discovers structural relationships:

- **Contains**: "X contains Y", "X includes Y"
- **Part Of**: "X is part of Y", "X belongs to Y"  
- **Subtype Of**: "X is a type of Y", "X is an instance of Y"

```python
hierarchical_rels = await detector.detect_hierarchical_relationships(entities, text)

for rel in hierarchical_rels:
    print(f"{rel.source_entity} {rel.relationship_subtype} {rel.target_entity}")
    # "Deep Learning subtype_of Machine Learning" (confidence: 0.88)
```

## 🔍 Semantic Search Engine

### Vector Embeddings
Uses sentence transformers for semantic understanding:

```python
async with SemanticSearchEngine() as search_engine:
    # Generate embeddings with caching
    embeddings = await search_engine.generate_embeddings(texts)
    
    # Semantic search
    results = await search_engine.semantic_search(
        "machine learning algorithms", 
        limit=10,
        similarity_threshold=0.3
    )
```

### Related Entity Discovery
Find conceptually similar entities:

```python
related_entities = await search_engine.find_conceptually_related_entities(
    "OpenAI", 
    limit=10,
    similarity_threshold=0.4
)

for entity in related_entities:
    print(f"{entity['name']} - similarity: {entity['similarity_score']:.3f}")
    # "Anthropic - similarity: 0.754"
    # "Google DeepMind - similarity: 0.691"
```

### Concept Clustering
Automatically group related concepts:

```python
clusters = await search_engine.cluster_concepts(
    min_cluster_size=3,
    eps=0.3
)

for cluster in clusters:
    print(f"Theme: {cluster.dominant_theme}")
    print(f"Entities: {', '.join(cluster.entities)}")
    print(f"Coherence: {cluster.coherence_score:.3f}")
    # Theme: "AI Research Companies"
    # Entities: OpenAI, Anthropic, Google DeepMind, Meta AI
    # Coherence: 0.782
```

## 🎯 Entity Disambiguation

### Conflict Resolution
Automatically resolve entity naming conflicts:

```python
async with EntityDisambiguator() as disambiguator:
    # Input: ["OpenAI", "Open AI", "OpenAI Inc."]
    resolved = await disambiguator.resolve_entity_conflicts(entities)
    # Output: Single consolidated "OpenAI" entity
```

### Alias Detection
Discover alternative names and abbreviations:

```python
aliases = await disambiguator.detect_entity_aliases("OpenAI")
# Returns: ["Open AI", "OpenAI Inc.", "Open Artificial Intelligence"]
```

### Similarity-Based Merging
Merge entities based on semantic similarity:

```python
merged_count = await disambiguator.merge_similar_entities(
    confidence_threshold=0.85
)
print(f"Merged {merged_count} duplicate entities")
```

## 🔬 Knowledge Gap Detection

### Domain Analysis
Identify missing concepts in specific domains:

```python
gaps = await search_engine.detect_knowledge_gaps("machine learning")

for gap in gaps:
    print(f"Gap: {gap['gap_type']}")
    print(f"Description: {gap['description']}")
    print(f"Suggestion: {gap['suggestion']}")
    # Gap: missing_concept
    # Description: No entities found for reinforcement learning
    # Suggestion: Add content about Q-learning, policy gradients, and RL applications
```

### Isolated Entity Detection
Find entities with few connections:

```python
# Automatically detects entities that might benefit from more context
gaps = await search_engine.detect_knowledge_gaps("AI research")
# Returns suggestions for connecting isolated entities
```

## 🛠️ Enhanced MCP Tools

The advanced features are exposed through powerful MCP tools for Claude integration:

### find_related_concepts
```python
@app.call_tool()
async def find_related_concepts(
    entity_name: str, 
    relationship_types: Optional[List[str]] = None,
    limit: int = 10
) -> str:
```

**Usage with Claude:**
```
You: "Find concepts related to machine learning that I might have missed"
Claude: *uses find_related_concepts*
"Based on your knowledge graph, you have strong coverage of neural networks and deep learning, but I notice gaps in reinforcement learning and computer vision..."
```

### detect_knowledge_gaps
```python
@app.call_tool()
async def detect_knowledge_gaps(domain: str, limit: int = 10) -> str:
```

**Usage with Claude:**
```
You: "What knowledge gaps exist in my AI research notes?"
Claude: *uses detect_knowledge_gaps*
"I found several areas for improvement: 1) Missing connections between transformer architecture and attention mechanisms, 2) Limited coverage of AI safety concepts..."
```

### suggest_connections
```python
@app.call_tool()
async def suggest_connections(entity1: str, entity2: str) -> str:
```

**Usage with Claude:**
```
You: "Are there any hidden connections between my AI research and business strategy notes?"
Claude: *uses suggest_connections*
"I found interesting semantic connections! Your notes on 'attention mechanisms' relate to your business strategy around 'focused execution'..."
```

### semantic_search_entities
```python
@app.call_tool()
async def semantic_search_entities(
    query: str, 
    limit: int = 10,
    similarity_threshold: float = 0.3
) -> str:
```

**Usage with Claude:**
```
You: "Find entities related to 'neural network optimization'"
Claude: *uses semantic_search_entities*
"Found 8 relevant entities: 1. Gradient Descent (89% relevant), 2. Backpropagation (84% relevant)..."
```

### merge_similar_entities
```python
@app.call_tool()
async def merge_similar_entities(confidence_threshold: float = 0.85) -> str:
```

**Usage with Claude:**
```
You: "Clean up duplicate entities in my knowledge graph"
Claude: *uses merge_similar_entities*
"Successfully merged 12 duplicate entities. Consolidated 'AI' and 'Artificial Intelligence', merged 'OpenAI' and 'Open AI'..."
```

### detect_advanced_relationships
```python
@app.call_tool()
async def detect_advanced_relationships(
    filepath: str, 
    text_content: str = ""
) -> str:
```

**Usage with Claude:**
```
You: "Analyze this research paper for advanced relationships"
Claude: *uses detect_advanced_relationships*
"Found 15 advanced relationships: 5 temporal (transformers→BERT→GPT), 8 causal (attention mechanism→improved performance), 2 hierarchical..."
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

New dependencies include:
- `sentence-transformers>=2.2.2` - For embeddings
- `scikit-learn>=1.3.0` - For clustering and similarity
- `numpy>=1.24.0` - For vector operations
- `torch>=2.0.0` - Required by sentence-transformers

### 2. Run the Demo
```bash
python demo_advanced_features.py
```

This comprehensive demo showcases all advanced features with sample data.

### 3. Start Enhanced MCP Server
```bash
python -m src.mcp_server
```

The MCP server now includes all advanced tools for Claude integration.

### 4. Configure OpenRouter (Optional)
Set your OpenRouter API key for AI-enhanced analysis:
```bash
export OPENROUTER_API_KEY="your-key-here"
```

## 📊 Performance Considerations

### Embedding Cache
- Embeddings are automatically cached to disk
- Reduces computation for repeated texts
- Cache file: `embeddings_cache.pkl`

### Model Selection
- Default: `all-MiniLM-L6-v2` (fast, lightweight)
- For better accuracy: Can be configured to use larger models
- Embeddings run locally (no API calls)

### Memory Usage
- Embedding model: ~50MB RAM
- Cache size grows with unique texts processed
- Clustering scales to thousands of entities

### Cost Management
- Basic embeddings: Free (local processing)
- AI-enhanced analysis: Uses OpenRouter API
- Configurable budget limits and model selection

## 🎯 Use Cases

### Research & Learning
- **Literature Review**: Find related papers and concepts
- **Knowledge Gaps**: Identify areas needing more research
- **Concept Mapping**: Understand relationships between ideas

### Content Creation
- **Topic Discovery**: Find related concepts to explore
- **Content Connections**: Link ideas across different domains
- **Outline Generation**: Use hierarchical relationships

### Business Intelligence
- **Competitor Analysis**: Find related companies and technologies
- **Market Research**: Identify gaps and opportunities
- **Strategic Planning**: Connect business concepts and trends

### Personal Knowledge Management
- **Note Organization**: Automatically cluster related notes
- **Knowledge Review**: Find forgotten connections
- **Learning Optimization**: Focus on knowledge gaps

## 🔧 Configuration

### Semantic Search Settings
```yaml
semantic_search:
  embedding_model: "all-MiniLM-L6-v2"
  similarity_threshold: 0.3
  cache_embeddings: true
  max_results: 50
```

### Clustering Parameters
```yaml
clustering:
  min_cluster_size: 3
  eps: 0.3  # Distance threshold
  metric: "cosine"
```

### Entity Disambiguation
```yaml
disambiguation:
  merge_threshold: 0.85
  alias_detection: true
  confidence_boost: 0.1  # Per additional mention
```

## 🐛 Troubleshooting

### Common Issues

**1. Embedding Model Download**
```
# If sentence-transformers fails to download
pip install --upgrade sentence-transformers
# Or set HF_HOME environment variable
```

**2. Memory Issues**
```python
# For large knowledge graphs, use batch processing
async with SemanticSearchEngine() as engine:
    # Process in smaller chunks
    results = await engine.semantic_search(query, limit=10)
```

**3. No Clusters Found**
```python
# Try adjusting clustering parameters
clusters = await search_engine.cluster_concepts(
    min_cluster_size=2,  # Reduce minimum size
    eps=0.4  # Increase distance threshold
)
```

**4. Poor Semantic Search Results**
```python
# Lower similarity threshold
results = await search_engine.semantic_search(
    query, 
    similarity_threshold=0.2  # More permissive
)
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-modal Embeddings**: Support for images and audio
- **Dynamic Clustering**: Real-time cluster updates
- **Graph Neural Networks**: Advanced relationship prediction
- **Federated Learning**: Privacy-preserving knowledge sharing

### Integration Roadmap
- **Web Interface**: Visual knowledge graph explorer
- **API Endpoints**: RESTful access to advanced features
- **Jupyter Integration**: Interactive analysis notebooks
- **Enterprise Features**: Team collaboration and sharing

## 📚 Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/)
- [Vector Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [DBSCAN Clustering Algorithm](https://scikit-learn.org/stable/modules/clustering.html#dbscan)

---

*The advanced features represent a significant leap in knowledge graph intelligence, transforming static entity storage into dynamic knowledge discovery. With these tools, your personal knowledge graph becomes a true AI-powered research assistant.* 