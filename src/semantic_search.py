"""Semantic search engine with embeddings for intelligent knowledge discovery."""

import asyncio
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from .config import get_config
from .cloud_nlp import Entity, Relationship, OpenRouterProcessor
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with relevance score."""
    entity_name: str
    entity_type: str
    description: str
    relevance_score: float
    context: str
    relationships: List[Dict[str, Any]]

@dataclass
class ConceptCluster:
    """Cluster of related concepts."""
    cluster_id: int
    entities: List[str]
    centroid_embedding: np.ndarray
    dominant_theme: str
    coherence_score: float

class SemanticSearchEngine:
    """Semantic search engine with embeddings for intelligent knowledge discovery."""
    
    def __init__(self):
        self.config = get_config()
        self.knowledge_graph = None
        self.openrouter = None
        
        # Embedding model
        self.embedding_model = None
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight but effective
        
        # Cache for embeddings
        self.embeddings_cache = {}
        self.cache_file = Path("embeddings_cache.pkl")
        
        # Load cached embeddings if available
        self._load_embedding_cache()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.knowledge_graph = KnowledgeGraph()
        self.openrouter = OpenRouterProcessor()
        await self.knowledge_graph.__aenter__()
        await self.openrouter.__aenter__()
        
        # Initialize embedding model in a separate thread to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None, self._initialize_embedding_model
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.knowledge_graph:
            await self.knowledge_graph.__aexit__(exc_type, exc_val, exc_tb)
        if self.openrouter:
            await self.openrouter.__aexit__(exc_type, exc_val, exc_tb)
        
        # Save embedding cache
        self._save_embedding_cache()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self.embedding_model:
            logger.warning("Embedding model not available, falling back to AI-based similarity")
            return []
        
        try:
            # Check cache first
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.embeddings_cache:
                    cached_embeddings.append((i, self.embeddings_cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            new_embeddings = []
            if uncached_texts:
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.embedding_model.encode, uncached_texts
                )
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, embeddings):
                    self.embeddings_cache[text] = embedding.tolist()
                
                new_embeddings = embeddings.tolist()
            
            # Combine cached and new embeddings in correct order
            all_embeddings = [None] * len(texts)
            
            # Place cached embeddings
            for i, embedding in cached_embeddings:
                all_embeddings[i] = embedding
            
            # Place new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[idx] = embedding
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    async def semantic_search(
        self, 
        query: str, 
        limit: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[SearchResult]:
        """Perform semantic search on the knowledge graph."""
        if not self.knowledge_graph:
            raise RuntimeError("Knowledge graph not initialized")
        
        try:
            # Get all entities from knowledge graph
            entities_data = await self.knowledge_graph.search_entities("", limit=1000)
            
            if not entities_data:
                return []
            
            # Generate query embedding
            query_embeddings = await self.generate_embeddings([query])
            if not query_embeddings:
                # Fallback to keyword search
                return await self._fallback_keyword_search(query, entities_data, limit)
            
            query_embedding = np.array(query_embeddings[0])
            
            # Prepare entity texts for embedding
            entity_texts = []
            entities_info = []
            
            for entity_data in entities_data:
                # Combine entity information for better semantic matching
                entity_text = f"{entity_data['name']} {entity_data.get('description', '')} {entity_data.get('context', '')}"
                entity_texts.append(entity_text)
                entities_info.append(entity_data)
            
            # Generate embeddings for entities
            entity_embeddings = await self.generate_embeddings(entity_texts)
            
            if not entity_embeddings:
                return await self._fallback_keyword_search(query, entities_data, limit)
            
            # Calculate similarities
            similarities = []
            for i, entity_embedding in enumerate(entity_embeddings):
                if entity_embedding:
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [np.array(entity_embedding)]
                    )[0][0]
                    
                    if similarity >= similarity_threshold:
                        similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create search results
            search_results = []
            for i, similarity in similarities[:limit]:
                entity_data = entities_info[i]
                
                # Get entity relationships
                relationships = await self._get_entity_relationships(entity_data['name'])
                
                result = SearchResult(
                    entity_name=entity_data['name'],
                    entity_type=entity_data['type'],
                    description=entity_data.get('description', ''),
                    relevance_score=float(similarity),
                    context=entity_data.get('context', ''),
                    relationships=relationships
                )
                
                search_results.append(result)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def find_conceptually_related_entities(
        self, 
        entity_name: str,
        limit: int = 10,
        similarity_threshold: float = 0.4
    ) -> List[Dict]:
        """Find entities conceptually related to a given entity."""
        if not self.knowledge_graph:
            raise RuntimeError("Knowledge graph not initialized")
        
        try:
            # Get the target entity
            entity_info = await self.knowledge_graph.get_entity_connections(entity_name)
            if not entity_info:
                return []
            
            # Create entity description for embedding
            entity_text = f"{entity_name} {entity_info.get('description', '')} {entity_info.get('context', '')}"
            
            # Get all other entities
            all_entities = await self.knowledge_graph.search_entities("", limit=1000)
            
            # Filter out the target entity
            other_entities = [e for e in all_entities if e['name'] != entity_name]
            
            if not other_entities:
                return []
            
            # Generate embeddings
            target_embedding = await self.generate_embeddings([entity_text])
            if not target_embedding:
                return []
            
            entity_texts = [
                f"{e['name']} {e.get('description', '')} {e.get('context', '')}"
                for e in other_entities
            ]
            
            other_embeddings = await self.generate_embeddings(entity_texts)
            if not other_embeddings:
                return []
            
            # Calculate similarities
            target_emb = np.array(target_embedding[0])
            similarities = []
            
            for i, other_embedding in enumerate(other_embeddings):
                if other_embedding:
                    similarity = cosine_similarity(
                        [target_emb], 
                        [np.array(other_embedding)]
                    )[0][0]
                    
                    if similarity >= similarity_threshold:
                        similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create results
            related_entities = []
            for i, similarity in similarities[:limit]:
                entity_data = other_entities[i]
                related_entities.append({
                    "name": entity_data['name'],
                    "type": entity_data['type'],
                    "description": entity_data.get('description', ''),
                    "similarity_score": float(similarity),
                    "context": entity_data.get('context', '')
                })
            
            return related_entities
        
        except Exception as e:
            logger.error(f"Failed to find related entities for {entity_name}: {e}")
            return []
    
    async def cluster_concepts(
        self, 
        min_cluster_size: int = 3,
        eps: float = 0.3
    ) -> List[ConceptCluster]:
        """Cluster entities into conceptually similar groups."""
        if not self.knowledge_graph:
            raise RuntimeError("Knowledge graph not initialized")
        
        try:
            # Get all entities
            entities_data = await self.knowledge_graph.search_entities("", limit=1000)
            
            if len(entities_data) < min_cluster_size:
                return []
            
            # Generate embeddings for all entities
            entity_texts = [
                f"{e['name']} {e.get('description', '')} {e.get('context', '')}"
                for e in entities_data
            ]
            
            embeddings = await self.generate_embeddings(entity_texts)
            if not embeddings:
                return []
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings)
            
            # Perform clustering
            clustering = DBSCAN(
                eps=eps, 
                min_samples=min_cluster_size, 
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(embedding_matrix)
            
            # Group entities by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append((entities_data[i], embeddings[i]))
            
            # Create ConceptCluster objects
            concept_clusters = []
            for cluster_id, cluster_entities in clusters.items():
                entity_names = [e[0]['name'] for e in cluster_entities]
                cluster_embeddings = [e[1] for e in cluster_entities]
                
                # Calculate centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Determine dominant theme using AI
                dominant_theme = await self._determine_cluster_theme(entity_names)
                
                # Calculate coherence score
                coherence_score = self._calculate_cluster_coherence(cluster_embeddings)
                
                cluster = ConceptCluster(
                    cluster_id=cluster_id,
                    entities=entity_names,
                    centroid_embedding=centroid,
                    dominant_theme=dominant_theme,
                    coherence_score=coherence_score
                )
                
                concept_clusters.append(cluster)
            
            # Sort by coherence score
            concept_clusters.sort(key=lambda c: c.coherence_score, reverse=True)
            
            return concept_clusters
        
        except Exception as e:
            logger.error(f"Concept clustering failed: {e}")
            return []
    
    async def detect_knowledge_gaps(
        self, 
        domain: str,
        gap_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect areas where knowledge might be incomplete."""
        try:
            # Get entities related to the domain
            domain_entities = await self.semantic_search(domain, limit=50)
            
            if len(domain_entities) < 5:
                return [{
                    "gap_type": "insufficient_coverage",
                    "description": f"Very few entities found for domain '{domain}'",
                    "suggestion": f"Consider adding more content related to {domain}",
                    "confidence": 0.8
                }]
            
            gaps = []
            
            # Check for isolated entities (few connections)
            for entity in domain_entities[:20]:
                connections_count = len(entity.relationships)
                if connections_count < 2:
                    gaps.append({
                        "gap_type": "isolated_entity",
                        "entity": entity.entity_name,
                        "description": f"Entity '{entity.entity_name}' has few connections",
                        "suggestion": f"Add more content explaining how {entity.entity_name} relates to other concepts",
                        "confidence": 0.7
                    })
            
            # Check for missing common relationships using AI
            if self.openrouter:
                ai_gaps = await self._ai_gap_detection(domain, [e.entity_name for e in domain_entities[:10]])
                gaps.extend(ai_gaps)
            
            return gaps[:10]  # Limit to top 10 gaps
        
        except Exception as e:
            logger.error(f"Knowledge gap detection failed for domain '{domain}': {e}")
            return []
    
    async def _fallback_keyword_search(
        self, 
        query: str, 
        entities_data: List[Dict], 
        limit: int
    ) -> List[SearchResult]:
        """Fallback keyword-based search when embeddings are not available."""
        query_words = set(query.lower().split())
        
        scored_entities = []
        for entity_data in entities_data:
            # Calculate keyword overlap score
            entity_text = f"{entity_data['name']} {entity_data.get('description', '')} {entity_data.get('context', '')}"
            entity_words = set(entity_text.lower().split())
            
            overlap = len(query_words.intersection(entity_words))
            if overlap > 0:
                score = overlap / len(query_words.union(entity_words))
                scored_entities.append((entity_data, score))
        
        # Sort by score
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # Create search results
        results = []
        for entity_data, score in scored_entities[:limit]:
            relationships = await self._get_entity_relationships(entity_data['name'])
            
            result = SearchResult(
                entity_name=entity_data['name'],
                entity_type=entity_data['type'],
                description=entity_data.get('description', ''),
                relevance_score=score,
                context=entity_data.get('context', ''),
                relationships=relationships
            )
            results.append(result)
        
        return results
    
    async def _get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get relationships for an entity."""
        try:
            entity_info = await self.knowledge_graph.get_entity_connections(entity_name)
            return entity_info.get('relationships', [])
        except Exception as e:
            logger.error(f"Failed to get relationships for {entity_name}: {e}")
            return []
    
    async def _determine_cluster_theme(self, entity_names: List[str]) -> str:
        """Use AI to determine the dominant theme of a cluster."""
        if not self.openrouter:
            return "Unknown Theme"
        
        try:
            prompt = f"""Analyze the following entities and determine their dominant theme or category:

Entities: {', '.join(entity_names[:10])}

Return a single, concise theme/category name (2-4 words max) that best describes what these entities have in common.
Examples: "Machine Learning", "Programming Languages", "Business Strategy", "Scientific Research"
"""
            
            response = await self.openrouter._make_api_request(
                model=self.config.get_model_for_task("simple"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            theme = content.strip().strip('"\'')
            
            return theme if theme else "Unknown Theme"
        
        except Exception as e:
            logger.error(f"Failed to determine cluster theme: {e}")
            return "Unknown Theme"
    
    def _calculate_cluster_coherence(self, embeddings: List[List[float]]) -> float:
        """Calculate how coherent a cluster is."""
        if len(embeddings) < 2:
            return 0.0
        
        embeddings_array = np.array(embeddings)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings_array)):
            for j in range(i + 1, len(embeddings_array)):
                sim = cosine_similarity([embeddings_array[i]], [embeddings_array[j]])[0][0]
                similarities.append(sim)
        
        # Return average similarity as coherence score
        return float(np.mean(similarities))
    
    async def _ai_gap_detection(self, domain: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Use AI to detect knowledge gaps."""
        if not self.openrouter:
            return []
        
        try:
            prompt = f"""Given the domain "{domain}" and the following entities: {', '.join(entities[:8])}

Identify potential knowledge gaps - important concepts, relationships, or entities that might be missing.

Return JSON format with gaps:
[{{"gap_type": "missing_concept", "description": "...", "suggestion": "...", "confidence": 0.7}}]

Focus on:
- Missing fundamental concepts
- Absent relationships between entities
- Important subcategories not covered
- Key methodologies or techniques missing

Maximum 5 gaps.
"""
            
            response = await self.openrouter._make_api_request(
                model=self.config.get_model_for_task("simple"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse JSON response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                gaps = json.loads(json_content)
                
                # Validate and filter gaps
                valid_gaps = []
                for gap in gaps:
                    if all(key in gap for key in ["gap_type", "description", "suggestion"]):
                        gap["confidence"] = gap.get("confidence", 0.6)
                        valid_gaps.append(gap)
                
                return valid_gaps[:5]
        
        except Exception as e:
            logger.error(f"AI gap detection failed: {e}")
        
        return []
    
    def _load_embedding_cache(self):
        """Load cached embeddings from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            self.embeddings_cache = {}
    
    def _save_embedding_cache(self):
        """Save embeddings cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}") 