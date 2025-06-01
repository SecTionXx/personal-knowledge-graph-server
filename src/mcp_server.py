"""MCP Server for Personal Knowledge Graph - Expose knowledge graph capabilities to AI assistants."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from urllib.parse import urlparse
import re

# HTTP and web scraping
import httpx
from bs4 import BeautifulSoup

# MCP imports
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent

# Local imports
from .config import get_config, initialize_config
from .file_processor import process_single_file, FileMonitor
from .knowledge_graph import (
    search_knowledge_graph, 
    get_entity_info, 
    store_extracted_data,
    KnowledgeGraph
)
from .cloud_nlp import process_file_content
from .advanced_nlp import AdvancedRelationshipDetector, EntityDisambiguator
from .semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("personal-knowledge-graph")

class WebScraper:
    """Web content scraper for processing URLs."""
    
    def __init__(self):
        self.config = get_config()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "User-Agent": "Personal-Knowledge-Graph-Server/1.0 (+https://github.com/user/repo)"
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    async def scrape_url(self, url: str) -> Tuple[str, str, str]:
        """Scrape content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Tuple of (title, content, metadata)
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            return title, content, metadata
            
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {e}")
            raise
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try different title sources
        title_sources = [
            soup.find('title'),
            soup.find('h1'),
            soup.find('meta', property='og:title'),
            soup.find('meta', name='twitter:title')
        ]
        
        for source in title_sources:
            if source:
                title = source.get('content') if source.get('content') else source.get_text()
                if title and title.strip():
                    return title.strip()
        
        return "Unknown Title"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try different content selectors
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '#content',
            '.article-body'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(strip=True, separator='\n')
                if len(content) > 200:  # Ensure substantial content
                    return self._clean_content(content)
        
        # Fallback: extract all text from body
        body = soup.find('body')
        if body:
            content = body.get_text(strip=True, separator='\n')
            return self._clean_content(content)
        
        return soup.get_text(strip=True, separator='\n')
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common noise patterns
        noise_patterns = [
            r'Subscribe to our newsletter.*',
            r'Sign up for.*',
            r'Follow us on.*',
            r'Share this article.*',
            r'Related articles.*',
            r'Advertisement.*',
            r'Cookie policy.*'
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> str:
        """Extract metadata from the page."""
        metadata = []
        
        # Author
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '.author',
            '.byline'
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                author = author_elem.get('content') or author_elem.get_text()
                if author:
                    metadata.append(f"Author: {author.strip()}")
                    break
        
        # Publication date
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish_date"]',
            'time[datetime]',
            '.publication-date',
            '.date'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date = date_elem.get('datetime') or date_elem.get('content') or date_elem.get_text()
                if date:
                    metadata.append(f"Published: {date.strip()}")
                    break
        
        # Source domain
        domain = urlparse(url).netloc
        metadata.append(f"Source: {domain}")
        
        return " | ".join(metadata)

# Response formatting functions
def format_search_results(entities: List[Dict]) -> str:
    """Format search results for AI consumption."""
    if not entities:
        return "No entities found matching your search criteria."
    
    result = f"Found {len(entities)} entities:\n\n"
    
    for i, entity in enumerate(entities, 1):
        result += f"{i}. **{entity['name']}** ({entity['type']})\n"
        result += f"   Description: {entity['description']}\n"
        result += f"   Confidence: {entity['confidence']:.2f}\n"
        
        if 'relevance_score' in entity:
            result += f"   Relevance: {entity['relevance_score']:.2f}\n"
        
        if entity.get('mentions'):
            mentions = entity['mentions'][:3]  # Show first 3 mentions
            result += f"   Mentions: {', '.join(mentions)}\n"
        
        result += "\n"
    
    return result

def format_entity_connections(connections: Dict) -> str:
    """Format entity relationship data for AI."""
    if 'error' in connections:
        return f"Error: {connections['error']}"
    
    center = connections['center_entity']
    connected = connections['connected_entities']
    relationships = connections['relationships']
    
    result = f"**{center['name']}** ({center['type']})\n"
    result += f"Description: {center['description']}\n"
    result += f"Confidence: {center['confidence']:.2f}\n\n"
    
    if not connected:
        result += "No connections found for this entity.\n"
        return result
    
    result += f"**Connections ({len(connected)} entities):**\n\n"
    
    # Group relationships by type
    rel_by_type = {}
    for rel in relationships:
        rel_type = rel['type']
        if rel_type not in rel_by_type:
            rel_by_type[rel_type] = []
        rel_by_type[rel_type].append(rel)
    
    for rel_type, rels in rel_by_type.items():
        result += f"**{rel_type}:**\n"
        for rel in rels[:5]:  # Limit to 5 per type
            if rel['source'] == center['name']:
                result += f"  → {rel['target']}\n"
            else:
                result += f"  ← {rel['source']}\n"
        
        if len(rels) > 5:
            result += f"  ... and {len(rels) - 5} more\n"
        result += "\n"
    
    return result

def format_processing_result(result, url: str = "", title: str = "") -> str:
    """Format file/URL processing results for AI."""
    if not result.success:
        return f"❌ Processing failed: {result.error_message}"
    
    source = title if title else Path(result.file_path).name if result.file_path else url
    
    response = f"✅ Successfully processed: **{source}**\n\n"
    response += f"**Results:**\n"
    response += f"• Entities extracted: {result.entities_count}\n"
    response += f"• Relationships found: {result.relationships_count}\n"
    response += f"• Processing time: {result.processing_time:.2f} seconds\n\n"
    
    if result.entities_count > 0:
        response += "The content has been analyzed and integrated into your knowledge graph. "
        response += "You can now search for the extracted entities and explore their connections.\n\n"
        
        if result.entities_count > 10:
            response += f"This was a substantial document with {result.entities_count} entities. "
            response += "Consider exploring the main concepts to discover new insights.\n"
    
    return response

def format_knowledge_stats(stats: Dict) -> str:
    """Format knowledge graph statistics for AI."""
    response = "📊 **Knowledge Graph Statistics**\n\n"
    response += f"**Content:**\n"
    response += f"• Total entities: {stats['total_entities']:,}\n"
    response += f"• Total relationships: {stats['total_relationships']:,}\n"
    response += f"• Source files: {stats['total_files']:,}\n\n"
    
    if stats['entity_types']:
        response += f"**Entity Types:** {', '.join(stats['entity_types'])}\n\n"
    
    response += f"**Quality Metrics:**\n"
    response += f"• Average entity confidence: {stats['avg_entity_confidence']:.3f}\n"
    response += f"• Max connections per entity: {stats['max_entity_connections']}\n"
    response += f"• Min connections per entity: {stats['min_entity_connections']}\n\n"
    
    # Add insights
    if stats['total_entities'] > 1000:
        response += "🎉 You have a substantial knowledge base! "
    elif stats['total_entities'] > 100:
        response += "📈 Your knowledge graph is growing well! "
    else:
        response += "🌱 Your knowledge graph is just getting started. "
    
    response += "The more content you add, the more valuable connections and insights you'll discover."
    
    return response

# MCP Tool Implementations

@app.call_tool()
async def process_file(filepath: str) -> str:
    """Process a file and add to knowledge graph.
    
    Args:
        filepath: Path to the file to process
        
    Returns:
        Processing result summary
    """
    try:
        # Validate file path
        file_path = Path(filepath)
        if not file_path.exists():
            return f"❌ Error: File not found: {filepath}"
        
        if not file_path.is_file():
            return f"❌ Error: Path is not a file: {filepath}"
        
        # Check file extension
        config = get_config()
        if file_path.suffix.lower() not in config.file_monitoring.supported_extensions:
            return f"❌ Error: Unsupported file type: {file_path.suffix}. Supported: {', '.join(config.file_monitoring.supported_extensions)}"
        
        # Process the file
        result = await process_single_file(filepath)
        
        return format_processing_result(result)
        
    except Exception as e:
        logger.error(f"Error in process_file tool: {e}")
        return f"❌ Error processing file: {str(e)}"

@app.call_tool()
async def search_knowledge(query: str, limit: int = 10) -> str:
    """Search entities and relationships in knowledge graph.
    
    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)
        
    Returns:
        Formatted search results
    """
    try:
        if not query.strip():
            return "❌ Error: Search query cannot be empty."
        
        if limit < 1 or limit > 50:
            limit = 10
        
        # Search the knowledge graph
        entities = await search_knowledge_graph(query, limit)
        
        return format_search_results(entities)
        
    except Exception as e:
        logger.error(f"Error in search_knowledge tool: {e}")
        return f"❌ Error searching knowledge graph: {str(e)}"

@app.call_tool()
async def get_entity_connections(entity_name: str, depth: int = 2) -> str:
    """Get connections for a specific entity.
    
    Args:
        entity_name: Name of the entity to explore
        depth: Relationship depth to explore (default: 2)
        
    Returns:
        Formatted entity connections
    """
    try:
        if not entity_name.strip():
            return "❌ Error: Entity name cannot be empty."
        
        if depth < 1 or depth > 5:
            depth = 2
        
        # Get entity connections
        connections = await get_entity_info(entity_name)
        
        return format_entity_connections(connections)
        
    except Exception as e:
        logger.error(f"Error in get_entity_connections tool: {e}")
        return f"❌ Error getting entity connections: {str(e)}"

@app.call_tool()
async def scrape_and_process_url(url: str, context: str = "") -> str:
    """Scrape web content and process into knowledge graph.
    
    Args:
        url: URL to scrape and process
        context: Additional context about the content
        
    Returns:
        Processing result summary
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return f"❌ Error: Invalid URL format: {url}"
        
        if parsed_url.scheme not in ['http', 'https']:
            return f"❌ Error: Only HTTP/HTTPS URLs are supported: {url}"
        
        # Scrape the content
        async with WebScraper() as scraper:
            title, content, metadata = await scraper.scrape_url(url)
        
        if not content or len(content) < 100:
            return f"❌ Error: Could not extract sufficient content from URL: {url}"
        
        # Process the scraped content
        full_context = f"Web article: {title}"
        if context:
            full_context += f" | {context}"
        if metadata:
            full_context += f" | {metadata}"
        
        entities, relationships = await process_file_content(
            file_path=url,
            content=content,
            context=full_context
        )
        
        # Store in knowledge graph
        entity_ids, relationship_ids = await store_extracted_data(
            entities=entities,
            relationships=relationships,
            source_file=url
        )
        
        # Create result object
        from .file_processor import ProcessingResult
        result = ProcessingResult(
            file_path=url,
            success=True,
            entities_count=len(entities),
            relationships_count=len(relationships),
            processing_time=0.0  # Not tracking for web scraping
        )
        
        return format_processing_result(result, url=url, title=title)
        
    except Exception as e:
        logger.error(f"Error in scrape_and_process_url tool: {e}")
        return f"❌ Error processing URL: {str(e)}"

@app.call_tool()
async def get_knowledge_stats() -> str:
    """Get knowledge graph statistics and health.
    
    Returns:
        Formatted knowledge graph statistics
    """
    try:
        # Get knowledge graph statistics
        async with KnowledgeGraph() as kg:
            stats = await kg.get_knowledge_statistics()
        
        return format_knowledge_stats(stats)
        
    except Exception as e:
        logger.error(f"Error in get_knowledge_stats tool: {e}")
        return f"❌ Error getting knowledge graph statistics: {str(e)}"

@app.call_tool()
async def start_file_monitoring() -> str:
    """Start continuous file monitoring for automatic processing.
    
    Returns:
        Status message about file monitoring
    """
    try:
        config = get_config()
        
        # Create file monitor
        monitor = FileMonitor()
        
        # Start monitoring in background
        asyncio.create_task(monitor.start_monitoring())
        
        return f"""File monitoring started successfully!

Monitoring directory: {config.directories.watch_directory}
Processed files will be moved to: {config.directories.processed_directory}

Supported file types: .md, .txt, .pdf, .docx, .json
File changes will be processed automatically with a 2-second debounce delay.

The monitor will run in the background and process new/modified files automatically."""
        
    except Exception as e:
        error_msg = f"Failed to start file monitoring: {e}"
        logger.error(error_msg)
        return error_msg

# Enhanced Advanced Tools

@app.call_tool()
async def find_related_concepts(
    entity_name: str, 
    relationship_types: Optional[List[str]] = None,
    limit: int = 10
) -> str:
    """Find semantically related concepts using advanced NLP and embeddings.
    
    Args:
        entity_name: Name of the entity to find related concepts for
        relationship_types: Optional filter for specific relationship types (temporal, causal, hierarchical, semantic)
        limit: Maximum number of related concepts to return
        
    Returns:
        Formatted list of related concepts with similarity scores and relationship types
    """
    try:
        async with SemanticSearchEngine() as search_engine:
            # Find semantically related entities
            related_entities = await search_engine.find_conceptually_related_entities(
                entity_name, limit=limit, similarity_threshold=0.3
            )
            
            if not related_entities:
                return f"No related concepts found for '{entity_name}'. Try adding more content about this entity or check the spelling."
            
            # Format results
            result_lines = [f"🔗 Related Concepts for '{entity_name}':\n"]
            
            for i, entity in enumerate(related_entities, 1):
                similarity_percent = int(entity['similarity_score'] * 100)
                result_lines.append(
                    f"{i}. **{entity['name']}** ({entity['type']}) - {similarity_percent}% similar"
                )
                
                if entity.get('description'):
                    result_lines.append(f"   📝 {entity['description'][:150]}{'...' if len(entity['description']) > 150 else ''}")
                
                result_lines.append("")  # Empty line for separation
            
            # Add summary
            result_lines.append(f"Found {len(related_entities)} semantically related concepts.")
            
            return "\n".join(result_lines)
            
    except Exception as e:
        error_msg = f"Failed to find related concepts for '{entity_name}': {e}"
        logger.error(error_msg)
        return error_msg

@app.call_tool()
async def detect_knowledge_gaps(domain: str, limit: int = 10) -> str:
    """Identify areas where more knowledge could be valuable.
    
    Args:
        domain: Domain or topic to analyze for knowledge gaps
        limit: Maximum number of gaps to identify
        
    Returns:
        List of identified knowledge gaps with suggestions
    """
    try:
        async with SemanticSearchEngine() as search_engine:
            gaps = await search_engine.detect_knowledge_gaps(domain)
            
            if not gaps:
                return f"✅ No significant knowledge gaps detected for '{domain}'. Your knowledge graph appears to have good coverage in this area."
            
            result_lines = [f"🔍 Knowledge Gaps Analysis for '{domain}':\n"]
            
            for i, gap in enumerate(gaps[:limit], 1):
                confidence_percent = int(gap.get('confidence', 0.5) * 100)
                
                result_lines.append(f"{i}. **{gap['gap_type'].replace('_', ' ').title()}** ({confidence_percent}% confidence)")
                result_lines.append(f"   📋 {gap['description']}")
                result_lines.append(f"   💡 **Suggestion**: {gap['suggestion']}")
                
                if gap.get('entity'):
                    result_lines.append(f"   🎯 **Entity**: {gap['entity']}")
                
                result_lines.append("")  # Empty line for separation
            
            result_lines.append(f"💡 **Recommendation**: Focus on the highest confidence gaps first to maximize knowledge graph value.")
            
            return "\n".join(result_lines)
            
    except Exception as e:
        error_msg = f"Failed to detect knowledge gaps for domain '{domain}': {e}"
        logger.error(error_msg)
        return error_msg

@app.call_tool()
async def suggest_connections(entity1: str, entity2: str) -> str:
    """Suggest potential relationships between two entities.
    
    Args:
        entity1: First entity name
        entity2: Second entity name
        
    Returns:
        Analysis of potential relationships between the entities
    """
    try:
        # Get information about both entities
        async with KnowledgeGraph() as kg:
            entity1_info = await kg.get_entity_connections(entity1)
            entity2_info = await kg.get_entity_connections(entity2)
            
            if not entity1_info:
                return f"Entity '{entity1}' not found in knowledge graph."
            
            if not entity2_info:
                return f"Entity '{entity2}' not found in knowledge graph."
        
        # Analyze with semantic search and relationship detection
        async with SemanticSearchEngine() as search_engine:
            # Calculate semantic similarity
            related_entities = await search_engine.find_conceptually_related_entities(entity1, limit=50)
            
            # Check if entity2 is in the related entities
            entity2_relation = None
            for related in related_entities:
                if related['name'].lower() == entity2.lower():
                    entity2_relation = related
                    break
            
            result_lines = [f"🔗 Connection Analysis: '{entity1}' ↔ '{entity2}'\n"]
            
            if entity2_relation:
                similarity_percent = int(entity2_relation['similarity_score'] * 100)
                result_lines.append(f"**Semantic Similarity**: {similarity_percent}%")
                
                if similarity_percent >= 70:
                    result_lines.append("🟢 **Strong conceptual relationship detected**")
                elif similarity_percent >= 40:
                    result_lines.append("🟡 **Moderate conceptual relationship detected**")
                else:
                    result_lines.append("🔴 **Weak conceptual relationship**")
            else:
                result_lines.append("**Semantic Similarity**: Not in top related entities (<30%)")
                result_lines.append("🔴 **No strong semantic relationship detected**")
            
            result_lines.append("")
            
            # Check existing relationships
            existing_relationships = []
            for rel in entity1_info.get('relationships', []):
                if (rel.get('target_entity', '').lower() == entity2.lower() or 
                    rel.get('source_entity', '').lower() == entity2.lower()):
                    existing_relationships.append(rel)
            
            if existing_relationships:
                result_lines.append("**Existing Relationships**:")
                for rel in existing_relationships:
                    result_lines.append(f"• {rel['type']}: {rel.get('description', 'No description')}")
                result_lines.append("")
            else:
                result_lines.append("**No existing direct relationships found**\n")
            
            # Suggest potential relationship types
            result_lines.append("**Potential Connection Types**:")
            
            # Entity type-based suggestions
            type1 = entity1_info.get('type', 'unknown')
            type2 = entity2_info.get('type', 'unknown')
            
            suggestions = []
            
            if type1 == type2:
                suggestions.append(f"• **Peer Relationship**: Both are {type1}s - they might be competitors, alternatives, or part of the same category")
            
            if type1 in ['person', 'organization'] and type2 in ['person', 'organization']:
                suggestions.append("• **Collaboration**: They might work together, have partnerships, or professional relationships")
            
            if type1 == 'technology' and type2 == 'technology':
                suggestions.append("• **Technical Relationship**: One might be built on the other, or they might integrate together")
            
            if type1 == 'concept' and type2 == 'concept':
                suggestions.append("• **Conceptual Relationship**: They might be related theories, methodologies, or ideas")
            
            if not suggestions:
                suggestions.append("• **Contextual Relationship**: They might appear in similar contexts or domains")
                suggestions.append("• **Temporal Relationship**: One might have influenced or preceded the other")
                suggestions.append("• **Hierarchical Relationship**: One might be a part of or contained within the other")
            
            result_lines.extend(suggestions)
            
            result_lines.append("")
            result_lines.append("💡 **Recommendation**: Look for content that mentions both entities together to establish concrete relationships.")
            
            return "\n".join(result_lines)
            
    except Exception as e:
        error_msg = f"Failed to analyze connection between '{entity1}' and '{entity2}': {e}"
        logger.error(error_msg)
        return error_msg

@app.call_tool()
async def discover_concept_clusters(min_cluster_size: int = 3, max_clusters: int = 10) -> str:
    """Discover clusters of conceptually related entities.
    
    Args:
        min_cluster_size: Minimum number of entities in a cluster
        max_clusters: Maximum number of clusters to return
        
    Returns:
        Overview of discovered concept clusters
    """
    try:
        async with SemanticSearchEngine() as search_engine:
            clusters = await search_engine.cluster_concepts(
                min_cluster_size=min_cluster_size, eps=0.35
            )
            
            if not clusters:
                return f"No significant concept clusters found with minimum size {min_cluster_size}. Try reducing the minimum cluster size or adding more diverse content."
            
            result_lines = [f"🎯 Discovered Concept Clusters (minimum size: {min_cluster_size}):\n"]
            
            for i, cluster in enumerate(clusters[:max_clusters], 1):
                coherence_percent = int(cluster.coherence_score * 100)
                
                result_lines.append(f"**Cluster {i}: {cluster.dominant_theme}** ({coherence_percent}% coherence)")
                result_lines.append(f"📊 **Entities ({len(cluster.entities)})**: {', '.join(cluster.entities[:8])}")
                
                if len(cluster.entities) > 8:
                    result_lines.append(f"   ... and {len(cluster.entities) - 8} more")
                
                result_lines.append("")
            
            if len(clusters) > max_clusters:
                result_lines.append(f"... and {len(clusters) - max_clusters} more clusters")
                result_lines.append("")
            
            # Summary insights
            total_entities_clustered = sum(len(c.entities) for c in clusters)
            avg_coherence = sum(c.coherence_score for c in clusters) / len(clusters)
            
            result_lines.append("📈 **Cluster Analysis Summary**:")
            result_lines.append(f"• Total clusters: {len(clusters)}")
            result_lines.append(f"• Entities clustered: {total_entities_clustered}")
            result_lines.append(f"• Average coherence: {int(avg_coherence * 100)}%")
            result_lines.append("")
            result_lines.append("💡 These clusters represent conceptually related groups in your knowledge graph. High coherence indicates strong thematic consistency.")
            
            return "\n".join(result_lines)
            
    except Exception as e:
        error_msg = f"Failed to discover concept clusters: {e}"
        logger.error(error_msg)
        return error_msg

@app.call_tool()
async def semantic_search_entities(
    query: str, 
    limit: int = 10,
    similarity_threshold: float = 0.3
) -> str:
    """Perform semantic search to find entities related to a query concept.
    
    Args:
        query: Search query describing the concept you're looking for
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0.0-1.0) for results
        
    Returns:
        Semantically relevant entities with similarity scores
    """
    try:
        async with SemanticSearchEngine() as search_engine:
            results = await search_engine.semantic_search(
                query, limit=limit, similarity_threshold=similarity_threshold
            )
            
            if not results:
                return f"No entities found matching '{query}' with similarity ≥ {int(similarity_threshold * 100)}%. Try broadening your search terms or lowering the similarity threshold."
            
            result_lines = [f"🔍 Semantic Search Results for: '{query}'\n"]
            
            for i, result in enumerate(results, 1):
                relevance_percent = int(result.relevance_score * 100)
                
                result_lines.append(f"{i}. **{result.entity_name}** ({result.entity_type}) - {relevance_percent}% relevant")
                
                if result.description:
                    result_lines.append(f"   📝 {result.description[:200]}{'...' if len(result.description) > 200 else ''}")
                
                # Show relationship count
                rel_count = len(result.relationships)
                if rel_count > 0:
                    result_lines.append(f"   🔗 {rel_count} relationship{'s' if rel_count != 1 else ''}")
                
                result_lines.append("")  # Empty line for separation
            
            # Search insights
            avg_relevance = sum(r.relevance_score for r in results) / len(results)
            
            result_lines.append(f"📊 **Search Insights**:")
            result_lines.append(f"• Results found: {len(results)}")
            result_lines.append(f"• Average relevance: {int(avg_relevance * 100)}%")
            result_lines.append(f"• Similarity threshold: {int(similarity_threshold * 100)}%")
            
            return "\n".join(result_lines)
            
    except Exception as e:
        error_msg = f"Semantic search failed for query '{query}': {e}"
        logger.error(error_msg)
        return error_msg

@app.call_tool()
async def merge_similar_entities(confidence_threshold: float = 0.85) -> str:
    """Automatically merge similar entities to reduce duplicates.
    
    Args:
        confidence_threshold: Minimum similarity score (0.0-1.0) required to merge entities
        
    Returns:
        Summary of merge operations performed
    """
    try:
        async with EntityDisambiguator() as disambiguator:
            merged_count = await disambiguator.merge_similar_entities(confidence_threshold)
            
            if merged_count == 0:
                return f"✅ No similar entities found with similarity ≥ {int(confidence_threshold * 100)}%. Your knowledge graph appears to have minimal duplicates at this threshold."
            
            result_lines = [
                f"🔄 Entity Merge Operation Completed\n",
                f"**Confidence Threshold**: {int(confidence_threshold * 100)}%",
                f"**Entities Merged**: {merged_count}",
                "",
                "✅ **Benefits**:",
                "• Reduced duplicate entities",
                "• Consolidated entity information",
                "• Improved knowledge graph consistency",
                "• Enhanced search and relationship accuracy",
                "",
                "💡 **Note**: Merged entities retain all relationships and mentions from both original entities. Check the logs for specific merge details."
            ]
            
            return "\n".join(result_lines)
            
    except Exception as e:
        error_msg = f"Failed to merge similar entities: {e}"
        logger.error(error_msg)
        return error_msg

@app.call_tool()
async def detect_advanced_relationships(filepath: str, text_content: str = "") -> str:
    """Detect advanced relationships (temporal, causal, hierarchical) in text content.
    
    Args:
        filepath: Path to the file to analyze (optional if text_content provided)
        text_content: Direct text content to analyze (optional if filepath provided)
        
    Returns:
        Analysis of advanced relationships found in the content
    """
    try:
        # Get content from file or use provided text
        if text_content:
            content = text_content
            context = "Direct text input"
        elif filepath:
            file_path = Path(filepath)
            if not file_path.exists():
                return f"File not found: {filepath}"
            
            content = file_path.read_text(encoding='utf-8')
            context = f"File: {filepath}"
        else:
            return "Either filepath or text_content must be provided"
        
        # Extract entities first
        from .cloud_nlp import process_file_content
        entities, basic_relationships = await process_file_content(filepath or "direct_input", content, context)
        
        if len(entities) < 2:
            return f"Insufficient entities found in content ({len(entities)} entities). Need at least 2 entities to detect relationships."
        
        # Detect advanced relationships
        async with AdvancedRelationshipDetector() as detector:
            temporal_rels = await detector.detect_temporal_relationships(entities, content)
            causal_rels = await detector.detect_causal_relationships(entities, content)
            hierarchical_rels = await detector.detect_hierarchical_relationships(entities, content)
        
        # Format results
        result_lines = [f"🧠 Advanced Relationship Analysis\n**Source**: {context}\n"]
        
        # Basic stats
        total_advanced = len(temporal_rels) + len(causal_rels) + len(hierarchical_rels)
        result_lines.append(f"**Entities Analyzed**: {len(entities)}")
        result_lines.append(f"**Basic Relationships**: {len(basic_relationships)}")
        result_lines.append(f"**Advanced Relationships**: {total_advanced}")
        result_lines.append("")
        
        # Temporal relationships
        if temporal_rels:
            result_lines.append(f"⏰ **Temporal Relationships ({len(temporal_rels)})**:")
            for rel in temporal_rels[:5]:  # Limit to top 5
                confidence_percent = int(rel.confidence * 100)
                result_lines.append(f"• {rel.source_entity} → {rel.target_entity} ({rel.relationship_subtype}) - {confidence_percent}%")
            if len(temporal_rels) > 5:
                result_lines.append(f"  ... and {len(temporal_rels) - 5} more")
            result_lines.append("")
        
        # Causal relationships
        if causal_rels:
            result_lines.append(f"🎯 **Causal Relationships ({len(causal_rels)})**:")
            for rel in causal_rels[:5]:  # Limit to top 5
                confidence_percent = int(rel.confidence * 100)
                result_lines.append(f"• {rel.source_entity} → {rel.target_entity} ({rel.relationship_subtype}) - {confidence_percent}%")
            if len(causal_rels) > 5:
                result_lines.append(f"  ... and {len(causal_rels) - 5} more")
            result_lines.append("")
        
        # Hierarchical relationships
        if hierarchical_rels:
            result_lines.append(f"🏗️ **Hierarchical Relationships ({len(hierarchical_rels)})**:")
            for rel in hierarchical_rels[:5]:  # Limit to top 5
                confidence_percent = int(rel.confidence * 100)
                result_lines.append(f"• {rel.source_entity} → {rel.target_entity} ({rel.relationship_subtype}) - {confidence_percent}%")
            if len(hierarchical_rels) > 5:
                result_lines.append(f"  ... and {len(hierarchical_rels) - 5} more")
            result_lines.append("")
        
        if total_advanced == 0:
            result_lines.append("ℹ️ No advanced relationships detected. Try content with more explicit temporal, causal, or hierarchical language.")
        else:
            result_lines.append("💡 **Note**: Advanced relationships enhance understanding of entity interactions beyond basic connections.")
        
        return "\n".join(result_lines)
            
    except Exception as e:
        error_msg = f"Failed to detect advanced relationships: {e}"
        logger.error(error_msg)
        return error_msg

# Server lifecycle functions
async def init_server():
    """Initialize the MCP server."""
    try:
        # Initialize configuration
        config = initialize_config()
        logger.info("MCP Server initialized successfully")
        
        # Ensure directories exist
        Path(config.file_monitoring.watch_directory).mkdir(parents=True, exist_ok=True)
        Path(config.file_monitoring.inbox_directory).mkdir(parents=True, exist_ok=True)
        Path(config.file_monitoring.processed_directory).mkdir(parents=True, exist_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {e}")
        return False

async def run_server():
    """Run the MCP server."""
    if not await init_server():
        logger.error("Server initialization failed")
        return
    
    try:
        logger.info("Starting Personal Knowledge Graph MCP Server...")
        logger.info("Available tools: process_file, search_knowledge, get_entity_connections, scrape_and_process_url, get_knowledge_stats, start_file_monitoring")
        
        # Run the server
        await app.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("MCP Server shutdown complete")

def main():
    """Main entry point for the MCP server."""
    asyncio.run(run_server())

if __name__ == "__main__":
    main() 