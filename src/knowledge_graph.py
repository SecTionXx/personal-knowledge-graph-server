"""Neo4j knowledge graph implementation for storing and querying entities and relationships."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from .config import get_config
from .cloud_nlp import Entity, Relationship

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Neo4j knowledge graph implementation."""
    
    def __init__(self):
        """Initialize knowledge graph connection."""
        self.config = get_config()
        self.driver = None
        self._is_connected = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.username, self.config.neo4j.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=self.config.neo4j.max_connections,
                connection_timeout=self.config.neo4j.timeout
            )
            
            # Test connection
            async with self.driver.session(database=self.config.neo4j.database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            self._is_connected = True
            logger.info("Connected to Neo4j database")
            
            # Initialize database schema
            await self._initialize_schema()
            
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close database connection."""
        if self.driver:
            await self.driver.close()
            self._is_connected = False
            logger.info("Closed Neo4j database connection")
    
    async def _initialize_schema(self):
        """Initialize database schema with constraints and indexes."""
        schema_queries = [
            # Entity constraints
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)",
            "CREATE INDEX source_file_index IF NOT EXISTS FOR (s:SourceFile) ON (s.path)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
            "CREATE FULLTEXT INDEX file_fulltext IF NOT EXISTS FOR (f:SourceFile) ON EACH [f.path, f.content_preview]"
        ]
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                    logger.debug(f"Executed schema query: {query}")
                except Exception as e:
                    # Constraint/index might already exist
                    logger.debug(f"Schema query warning: {e}")
    
    async def store_entities(self, entities: List[Entity], source_file: str = "") -> List[str]:
        """Store entities in the knowledge graph.
        
        Args:
            entities: List of entities to store
            source_file: Source file path for provenance
            
        Returns:
            List of entity IDs that were created or updated
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        stored_entities = []
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            for entity in entities:
                try:
                    # Create or update entity
                    query = """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.description = $description,
                        e.confidence = $confidence,
                        e.mentions = $mentions,
                        e.context = $context,
                        e.updated_at = datetime(),
                        e.last_seen = datetime()
                    ON CREATE SET e.created_at = datetime()
                    RETURN e.name as entity_name
                    """
                    
                    result = await session.run(query, {
                        "name": entity.name,
                        "type": entity.type,
                        "description": entity.description,
                        "confidence": entity.confidence,
                        "mentions": entity.mentions,
                        "context": entity.context
                    })
                    
                    record = await result.single()
                    if record:
                        stored_entities.append(record["entity_name"])
                    
                    # Link to source file if provided
                    if source_file:
                        await self._link_entity_to_file(session, entity.name, source_file)
                    
                except Exception as e:
                    logger.error(f"Failed to store entity {entity.name}: {e}")
        
        logger.info(f"Stored {len(stored_entities)} entities")
        return stored_entities
    
    async def store_relationships(self, relationships: List[Relationship], source_file: str = "") -> List[str]:
        """Store relationships in the knowledge graph.
        
        Args:
            relationships: List of relationships to store
            source_file: Source file path for provenance
            
        Returns:
            List of relationship IDs that were created
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        stored_relationships = []
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            for relationship in relationships:
                try:
                    # Create relationship between entities
                    query = """
                    MATCH (source:Entity {name: $source_entity})
                    MATCH (target:Entity {name: $target_entity})
                    MERGE (source)-[r:RELATES_TO {
                        type: $relationship_type,
                        source_entity: $source_entity,
                        target_entity: $target_entity
                    }]->(target)
                    SET r.description = $description,
                        r.confidence = $confidence,
                        r.context = $context,
                        r.updated_at = datetime()
                    ON CREATE SET r.created_at = datetime()
                    RETURN id(r) as relationship_id
                    """
                    
                    result = await session.run(query, {
                        "source_entity": relationship.source_entity,
                        "target_entity": relationship.target_entity,
                        "relationship_type": relationship.relationship_type,
                        "description": relationship.description,
                        "confidence": relationship.confidence,
                        "context": relationship.context
                    })
                    
                    record = await result.single()
                    if record:
                        stored_relationships.append(str(record["relationship_id"]))
                    
                    # Link relationship to source file
                    if source_file:
                        await self._link_relationship_to_file(
                            session,
                            relationship.source_entity,
                            relationship.target_entity,
                            relationship.relationship_type,
                            source_file
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to store relationship {relationship.source_entity} -> {relationship.target_entity}: {e}")
        
        logger.info(f"Stored {len(stored_relationships)} relationships")
        return stored_relationships
    
    async def _link_entity_to_file(self, session, entity_name: str, file_path: str):
        """Link an entity to its source file."""
        query = """
        MERGE (f:SourceFile {path: $file_path})
        ON CREATE SET f.created_at = datetime()
        SET f.updated_at = datetime()
        WITH f
        MATCH (e:Entity {name: $entity_name})
        MERGE (e)-[:EXTRACTED_FROM]->(f)
        """
        
        await session.run(query, {
            "entity_name": entity_name,
            "file_path": file_path
        })
    
    async def _link_relationship_to_file(self, session, source_entity: str, target_entity: str, relationship_type: str, file_path: str):
        """Link a relationship to its source file."""
        query = """
        MERGE (f:SourceFile {path: $file_path})
        ON CREATE SET f.created_at = datetime()
        SET f.updated_at = datetime()
        WITH f
        MATCH (source:Entity {name: $source_entity})-[r:RELATES_TO {type: $relationship_type}]->(target:Entity {name: $target_entity})
        MERGE (r)-[:FOUND_IN]->(f)
        """
        
        await session.run(query, {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relationship_type": relationship_type,
            "file_path": file_path
        })
    
    async def search_entities(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for entities using full-text search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching entities with their properties
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        search_query = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
        YIELD node, score
        RETURN node.name as name,
               node.type as type,
               node.description as description,
               node.confidence as confidence,
               node.mentions as mentions,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            result = await session.run(search_query, {"query": query, "limit": limit})
            entities = []
            
            async for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "confidence": record["confidence"],
                    "mentions": record["mentions"],
                    "relevance_score": record["score"]
                })
            
            return entities
    
    async def get_entity_connections(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get all connections for an entity up to a specified depth.
        
        Args:
            entity_name: Name of the entity
            max_depth: Maximum relationship depth to explore
            
        Returns:
            Dictionary containing the entity and its connections
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        query = """
        MATCH (center:Entity {name: $entity_name})
        OPTIONAL MATCH path = (center)-[r:RELATES_TO*1..$max_depth]-(connected:Entity)
        RETURN center,
               COLLECT(DISTINCT connected) as connected_entities,
               COLLECT(DISTINCT r) as relationships,
               COLLECT(DISTINCT path) as paths
        """
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            result = await session.run(query, {
                "entity_name": entity_name,
                "max_depth": max_depth
            })
            
            record = await result.single()
            if not record:
                return {"error": f"Entity '{entity_name}' not found"}
            
            center_entity = record["center"]
            connected_entities = record["connected_entities"] or []
            relationships = []
            
            # Extract relationships from paths
            for path in record["paths"] or []:
                if path:
                    for rel in path.relationships:
                        relationships.append({
                            "source": rel.start_node["name"],
                            "target": rel.end_node["name"],
                            "type": rel["type"],
                            "description": rel.get("description", ""),
                            "confidence": rel.get("confidence", 0.0)
                        })
            
            return {
                "center_entity": {
                    "name": center_entity["name"],
                    "type": center_entity["type"],
                    "description": center_entity["description"],
                    "confidence": center_entity["confidence"]
                },
                "connected_entities": [
                    {
                        "name": entity["name"],
                        "type": entity["type"],
                        "description": entity["description"],
                        "confidence": entity["confidence"]
                    }
                    for entity in connected_entities
                ],
                "relationships": relationships,
                "total_connections": len(connected_entities)
            }
    
    async def get_similar_entities(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find entities similar to the given entity based on relationships and type.
        
        Args:
            entity_name: Name of the entity to find similar entities for
            limit: Maximum number of similar entities to return
            
        Returns:
            List of similar entities with similarity scores
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        query = """
        MATCH (target:Entity {name: $entity_name})
        MATCH (target)-[:RELATES_TO]-(intermediate:Entity)-[:RELATES_TO]-(similar:Entity)
        WHERE similar <> target
        WITH similar, target, COUNT(intermediate) as common_connections
        MATCH (similar)-[:RELATES_TO]-(all_similar_connections:Entity)
        WITH similar, target, common_connections, COUNT(all_similar_connections) as similar_total_connections
        MATCH (target)-[:RELATES_TO]-(all_target_connections:Entity)
        WITH similar, target, common_connections, similar_total_connections, COUNT(all_target_connections) as target_total_connections
        WITH similar, 
             toFloat(common_connections) / (similar_total_connections + target_total_connections - common_connections) as jaccard_similarity,
             common_connections,
             CASE WHEN similar.type = target.type THEN 0.2 ELSE 0.0 END as type_bonus
        RETURN similar.name as name,
               similar.type as type,
               similar.description as description,
               similar.confidence as confidence,
               jaccard_similarity + type_bonus as similarity_score,
               common_connections
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            result = await session.run(query, {"entity_name": entity_name, "limit": limit})
            similar_entities = []
            
            async for record in result:
                similar_entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "confidence": record["confidence"],
                    "similarity_score": record["similarity_score"],
                    "common_connections": record["common_connections"]
                })
            
            return similar_entities
    
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        stats_query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATES_TO]-()
        WITH e, COUNT(r) as relationship_count
        RETURN COUNT(e) as total_entities,
               AVG(e.confidence) as avg_entity_confidence,
               COLLECT(DISTINCT e.type) as entity_types,
               SUM(relationship_count) / 2 as total_relationships,
               MAX(relationship_count) as max_entity_connections,
               MIN(relationship_count) as min_entity_connections
        """
        
        files_query = """
        MATCH (f:SourceFile)
        RETURN COUNT(f) as total_files
        """
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            # Get main statistics
            result = await session.run(stats_query)
            stats_record = await result.single()
            
            # Get file count
            result = await session.run(files_query)
            files_record = await result.single()
            
            return {
                "total_entities": stats_record["total_entities"] or 0,
                "total_relationships": stats_record["total_relationships"] or 0,
                "total_files": files_record["total_files"] or 0,
                "entity_types": stats_record["entity_types"] or [],
                "avg_entity_confidence": round(stats_record["avg_entity_confidence"] or 0, 3),
                "max_entity_connections": stats_record["max_entity_connections"] or 0,
                "min_entity_connections": stats_record["min_entity_connections"] or 0
            }
    
    async def delete_entity(self, entity_name: str) -> bool:
        """Delete an entity and all its relationships.
        
        Args:
            entity_name: Name of the entity to delete
            
        Returns:
            True if the entity was deleted, False if not found
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        query = """
        MATCH (e:Entity {name: $entity_name})
        DETACH DELETE e
        RETURN COUNT(e) as deleted_count
        """
        
        async with self.driver.session(database=self.config.neo4j.database) as session:
            result = await session.run(query, {"entity_name": entity_name})
            record = await result.single()
            
            deleted = record["deleted_count"] > 0
            if deleted:
                logger.info(f"Deleted entity: {entity_name}")
            else:
                logger.warning(f"Entity not found for deletion: {entity_name}")
            
            return deleted

# Convenience functions
async def store_extracted_data(
    entities: List[Entity],
    relationships: List[Relationship],
    source_file: str = ""
) -> Tuple[List[str], List[str]]:
    """Store extracted entities and relationships in the knowledge graph.
    
    Args:
        entities: List of entities to store
        relationships: List of relationships to store
        source_file: Source file path for provenance
        
    Returns:
        Tuple of (stored_entity_ids, stored_relationship_ids)
    """
    async with KnowledgeGraph() as kg:
        entity_ids = await kg.store_entities(entities, source_file)
        relationship_ids = await kg.store_relationships(relationships, source_file)
        return entity_ids, relationship_ids

async def search_knowledge_graph(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search the knowledge graph for entities matching the query.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching entities
    """
    async with KnowledgeGraph() as kg:
        return await kg.search_entities(query, limit)

async def get_entity_info(entity_name: str) -> Dict[str, Any]:
    """Get detailed information about an entity and its connections.
    
    Args:
        entity_name: Name of the entity
        
    Returns:
        Dictionary containing entity information and connections
    """
    async with KnowledgeGraph() as kg:
        return await kg.get_entity_connections(entity_name) 