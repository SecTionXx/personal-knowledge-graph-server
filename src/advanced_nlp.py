"""Advanced NLP capabilities for enhanced relationship detection and semantic analysis."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import httpx

from .config import get_config
from .cloud_nlp import Entity, Relationship, OpenRouterProcessor
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

@dataclass
class AdvancedRelationship(Relationship):
    """Extended relationship with additional semantic properties."""
    temporal_indicator: Optional[str] = None
    causal_strength: float = 0.0
    semantic_similarity: float = 0.0
    relationship_subtype: Optional[str] = None

class AdvancedRelationshipDetector:
    """Advanced relationship detection with temporal, causal, and hierarchical analysis."""
    
    def __init__(self):
        self.config = get_config()
        self.openrouter = None
        
        # Temporal relationship patterns
        self.temporal_patterns = {
            "before": [
                r"\b(?:before|prior to|earlier than|preceding)\b",
                r"\b(?:then|afterwards|subsequently|later)\b",
                r"\b(?:first.*then|initially.*followed by)\b"
            ],
            "after": [
                r"\b(?:after|following|subsequent to|later than)\b",
                r"\b(?:caused by|resulted from|stemmed from)\b"
            ],
            "during": [
                r"\b(?:during|while|throughout|amid)\b",
                r"\b(?:at the same time|simultaneously|concurrently)\b"
            ],
            "concurrent": [
                r"\b(?:simultaneously|at the same time|together)\b",
                r"\b(?:in parallel|alongside|concurrently)\b"
            ]
        }
        
        # Causal relationship patterns
        self.causal_patterns = {
            "causes": [
                r"\b(?:causes?|leads? to|results? in|brings? about)\b",
                r"\b(?:triggers?|initiates?|generates?|produces?)\b",
                r"\b(?:because of|due to|as a result of)\b"
            ],
            "enables": [
                r"\b(?:enables?|allows?|facilitates?|makes possible)\b",
                r"\b(?:supports?|helps|assists)\b"
            ],
            "prevents": [
                r"\b(?:prevents?|stops?|blocks?|inhibits?)\b",
                r"\b(?:avoids?|eliminates?|reduces?)\b"
            ]
        }
        
        # Hierarchical relationship patterns
        self.hierarchical_patterns = {
            "contains": [
                r"\b(?:contains?|includes?|comprises?|encompasses?)\b",
                r"\b(?:consists? of|made up of|composed of)\b"
            ],
            "part_of": [
                r"\b(?:part of|component of|element of|member of)\b",
                r"\b(?:belongs to|within|inside)\b"
            ],
            "subtype_of": [
                r"\b(?:type of|kind of|form of|variety of)\b",
                r"\b(?:instance of|example of|case of)\b"
            ]
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.openrouter = OpenRouterProcessor()
        await self.openrouter.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.openrouter:
            await self.openrouter.__aexit__(exc_type, exc_val, exc_tb)
    
    async def detect_temporal_relationships(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[AdvancedRelationship]:
        """Detect temporal relationships between entities."""
        temporal_relationships = []
        
        # Pattern-based detection
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                relationship = await self._analyze_temporal_pattern(
                    entity1, entity2, text
                )
                if relationship:
                    temporal_relationships.append(relationship)
        
        # AI-enhanced temporal analysis
        if len(entities) >= 2:
            ai_temporal = await self._ai_temporal_analysis(entities, text)
            temporal_relationships.extend(ai_temporal)
        
        return temporal_relationships
    
    async def detect_causal_relationships(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[AdvancedRelationship]:
        """Detect causal relationships between entities."""
        causal_relationships = []
        
        # Pattern-based detection
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                relationship = await self._analyze_causal_pattern(
                    entity1, entity2, text
                )
                if relationship:
                    causal_relationships.append(relationship)
        
        # AI-enhanced causal analysis
        if len(entities) >= 2:
            ai_causal = await self._ai_causal_analysis(entities, text)
            causal_relationships.extend(ai_causal)
        
        return causal_relationships
    
    async def detect_hierarchical_relationships(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[AdvancedRelationship]:
        """Detect hierarchical relationships between entities."""
        hierarchical_relationships = []
        
        # Pattern-based detection
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                relationship = await self._analyze_hierarchical_pattern(
                    entity1, entity2, text
                )
                if relationship:
                    hierarchical_relationships.append(relationship)
        
        # AI-enhanced hierarchical analysis
        if len(entities) >= 2:
            ai_hierarchical = await self._ai_hierarchical_analysis(entities, text)
            hierarchical_relationships.extend(ai_hierarchical)
        
        return hierarchical_relationships
    
    async def detect_semantic_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate semantic similarity between two entities."""
        try:
            # Combine entity information for comparison
            text1 = f"{entity1.name} {entity1.description} {entity1.context}"
            text2 = f"{entity2.name} {entity2.description} {entity2.context}"
            
            # Use AI for semantic similarity if available
            if self.openrouter:
                similarity = await self._ai_semantic_similarity(text1, text2)
                if similarity is not None:
                    return similarity
            
            # Fallback to basic text similarity
            return self._basic_text_similarity(text1, text2)
        
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def _analyze_temporal_pattern(
        self, 
        entity1: Entity, 
        entity2: Entity, 
        text: str
    ) -> Optional[AdvancedRelationship]:
        """Analyze text for temporal patterns between entities."""
        entity1_indices = [m.start() for m in re.finditer(
            re.escape(entity1.name), text, re.IGNORECASE
        )]
        entity2_indices = [m.start() for m in re.finditer(
            re.escape(entity2.name), text, re.IGNORECASE
        )]
        
        if not entity1_indices or not entity2_indices:
            return None
        
        # Find closest entity mentions
        min_distance = float('inf')
        best_pair = None
        
        for i1 in entity1_indices:
            for i2 in entity2_indices:
                distance = abs(i1 - i2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (i1, i2)
        
        if not best_pair or min_distance > 200:  # Too far apart
            return None
        
        # Extract context around entities
        start_idx = max(0, min(best_pair) - 100)
        end_idx = min(len(text), max(best_pair) + 100)
        context = text[start_idx:end_idx]
        
        # Check for temporal patterns
        for temporal_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    confidence = 0.7 + (0.3 * (1 - min_distance / 200))
                    
                    # Determine relationship direction
                    if best_pair[0] < best_pair[1]:  # entity1 comes first
                        source, target = entity1.name, entity2.name
                    else:
                        source, target = entity2.name, entity1.name
                    
                    return AdvancedRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type="temporal",
                        description=f"Temporal relationship: {temporal_type}",
                        confidence=confidence,
                        context=context,
                        temporal_indicator=temporal_type,
                        relationship_subtype=temporal_type
                    )
        
        return None
    
    async def _analyze_causal_pattern(
        self, 
        entity1: Entity, 
        entity2: Entity, 
        text: str
    ) -> Optional[AdvancedRelationship]:
        """Analyze text for causal patterns between entities."""
        # Similar to temporal analysis but for causal patterns
        entity1_indices = [m.start() for m in re.finditer(
            re.escape(entity1.name), text, re.IGNORECASE
        )]
        entity2_indices = [m.start() for m in re.finditer(
            re.escape(entity2.name), text, re.IGNORECASE
        )]
        
        if not entity1_indices or not entity2_indices:
            return None
        
        min_distance = float('inf')
        best_pair = None
        
        for i1 in entity1_indices:
            for i2 in entity2_indices:
                distance = abs(i1 - i2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (i1, i2)
        
        if not best_pair or min_distance > 150:  # Causal relationships typically closer
            return None
        
        start_idx = max(0, min(best_pair) - 80)
        end_idx = min(len(text), max(best_pair) + 80)
        context = text[start_idx:end_idx]
        
        for causal_type, patterns in self.causal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    confidence = 0.8 + (0.2 * (1 - min_distance / 150))
                    
                    # Determine causal direction
                    pattern_match = re.search(pattern, context, re.IGNORECASE)
                    pattern_pos = pattern_match.start() + start_idx
                    
                    if best_pair[0] < pattern_pos < best_pair[1]:
                        source, target = entity1.name, entity2.name
                    elif best_pair[1] < pattern_pos < best_pair[0]:
                        source, target = entity2.name, entity1.name
                    else:
                        # Default to text order
                        if best_pair[0] < best_pair[1]:
                            source, target = entity1.name, entity2.name
                        else:
                            source, target = entity2.name, entity1.name
                    
                    return AdvancedRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type="causal",
                        description=f"Causal relationship: {causal_type}",
                        confidence=confidence,
                        context=context,
                        causal_strength=confidence,
                        relationship_subtype=causal_type
                    )
        
        return None
    
    async def _analyze_hierarchical_pattern(
        self, 
        entity1: Entity, 
        entity2: Entity, 
        text: str
    ) -> Optional[AdvancedRelationship]:
        """Analyze text for hierarchical patterns between entities."""
        # Similar approach for hierarchical relationships
        entity1_indices = [m.start() for m in re.finditer(
            re.escape(entity1.name), text, re.IGNORECASE
        )]
        entity2_indices = [m.start() for m in re.finditer(
            re.escape(entity2.name), text, re.IGNORECASE
        )]
        
        if not entity1_indices or not entity2_indices:
            return None
        
        min_distance = float('inf')
        best_pair = None
        
        for i1 in entity1_indices:
            for i2 in entity2_indices:
                distance = abs(i1 - i2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (i1, i2)
        
        if not best_pair or min_distance > 100:  # Hierarchical relationships very close
            return None
        
        start_idx = max(0, min(best_pair) - 60)
        end_idx = min(len(text), max(best_pair) + 60)
        context = text[start_idx:end_idx]
        
        for hierarchical_type, patterns in self.hierarchical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    confidence = 0.85 + (0.15 * (1 - min_distance / 100))
                    
                    # Determine hierarchical direction
                    pattern_match = re.search(pattern, context, re.IGNORECASE)
                    pattern_pos = pattern_match.start() + start_idx
                    
                    if hierarchical_type in ["contains", "comprises"]:
                        # Container comes first
                        if best_pair[0] < best_pair[1]:
                            source, target = entity1.name, entity2.name
                        else:
                            source, target = entity2.name, entity1.name
                    else:  # part_of, subtype_of
                        # Part/subtype comes first
                        if best_pair[0] < best_pair[1]:
                            source, target = entity1.name, entity2.name
                        else:
                            source, target = entity2.name, entity1.name
                    
                    return AdvancedRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type="hierarchical",
                        description=f"Hierarchical relationship: {hierarchical_type}",
                        confidence=confidence,
                        context=context,
                        relationship_subtype=hierarchical_type
                    )
        
        return None
    
    async def _ai_temporal_analysis(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[AdvancedRelationship]:
        """Use AI to detect complex temporal relationships."""
        if not self.openrouter:
            return []
        
        try:
            entity_names = [e.name for e in entities]
            
            prompt = f"""Analyze the following text for temporal relationships between entities.
            
Entities: {', '.join(entity_names)}

Text: {text[:2000]}  # Limit text for cost control

Identify temporal relationships with:
1. Which entity/event happened first
2. The temporal relationship type (before, after, during, concurrent)
3. Confidence (0.0-1.0)

Return JSON format:
[{{"source": "entity1", "target": "entity2", "type": "before", "confidence": 0.8, "context": "relevant sentence"}}]
"""
            
            response = await self.openrouter._make_api_request(
                model=self.config.get_model_for_task("simple"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            # Parse AI response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            relationships = self._parse_ai_relationships(content, "temporal")
            
            return relationships
        
        except Exception as e:
            logger.error(f"AI temporal analysis failed: {e}")
            return []
    
    async def _ai_causal_analysis(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[AdvancedRelationship]:
        """Use AI to detect complex causal relationships."""
        if not self.openrouter:
            return []
        
        try:
            entity_names = [e.name for e in entities]
            
            prompt = f"""Analyze the following text for causal relationships between entities.
            
Entities: {', '.join(entity_names)}

Text: {text[:2000]}

Identify causal relationships with:
1. Which entity causes or influences another
2. The causal type (causes, enables, prevents)
3. Confidence (0.0-1.0)

Return JSON format:
[{{"source": "entity1", "target": "entity2", "type": "causes", "confidence": 0.9, "context": "relevant sentence"}}]
"""
            
            response = await self.openrouter._make_api_request(
                model=self.config.get_model_for_task("simple"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            relationships = self._parse_ai_relationships(content, "causal")
            
            return relationships
        
        except Exception as e:
            logger.error(f"AI causal analysis failed: {e}")
            return []
    
    async def _ai_hierarchical_analysis(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[AdvancedRelationship]:
        """Use AI to detect complex hierarchical relationships."""
        if not self.openrouter:
            return []
        
        try:
            entity_names = [e.name for e in entities]
            
            prompt = f"""Analyze the following text for hierarchical relationships between entities.
            
Entities: {', '.join(entity_names)}

Text: {text[:2000]}

Identify hierarchical relationships with:
1. Parent-child, container-contained, or category-member relationships
2. The hierarchical type (contains, part_of, subtype_of)
3. Confidence (0.0-1.0)

Return JSON format:
[{{"source": "entity1", "target": "entity2", "type": "contains", "confidence": 0.8, "context": "relevant sentence"}}]
"""
            
            response = await self.openrouter._make_api_request(
                model=self.config.get_model_for_task("simple"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            relationships = self._parse_ai_relationships(content, "hierarchical")
            
            return relationships
        
        except Exception as e:
            logger.error(f"AI hierarchical analysis failed: {e}")
            return []
    
    async def _ai_semantic_similarity(self, text1: str, text2: str) -> Optional[float]:
        """Calculate semantic similarity using AI."""
        if not self.openrouter:
            return None
        
        try:
            prompt = f"""Rate the semantic similarity between these two descriptions on a scale of 0.0 to 1.0:

Description 1: {text1[:500]}
Description 2: {text2[:500]}

Consider:
- Conceptual similarity
- Domain relevance
- Contextual relationship

Return only a decimal number between 0.0 and 1.0.
"""
            
            response = await self.openrouter._make_api_request(
                model=self.config.get_model_for_task("simple"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract numeric value
            import re
            match = re.search(r'(\d+\.?\d*)', content)
            if match:
                similarity = float(match.group(1))
                return min(1.0, max(0.0, similarity))
            
            return None
        
        except Exception as e:
            logger.error(f"AI semantic similarity failed: {e}")
            return None
    
    def _parse_ai_relationships(
        self, 
        content: str, 
        relationship_category: str
    ) -> List[AdvancedRelationship]:
        """Parse AI response into AdvancedRelationship objects."""
        relationships = []
        
        try:
            # Try to extract JSON from response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                parsed_relationships = json.loads(json_content)
                
                for rel in parsed_relationships:
                    if all(key in rel for key in ["source", "target", "type", "confidence"]):
                        advanced_rel = AdvancedRelationship(
                            source_entity=rel["source"],
                            target_entity=rel["target"],
                            relationship_type=relationship_category,
                            description=f"{relationship_category.title()} relationship: {rel['type']}",
                            confidence=float(rel["confidence"]),
                            context=rel.get("context", ""),
                            relationship_subtype=rel["type"]
                        )
                        
                        # Set category-specific attributes
                        if relationship_category == "temporal":
                            advanced_rel.temporal_indicator = rel["type"]
                        elif relationship_category == "causal":
                            advanced_rel.causal_strength = float(rel["confidence"])
                        
                        relationships.append(advanced_rel)
        
        except Exception as e:
            logger.error(f"Failed to parse AI relationships: {e}")
        
        return relationships
    
    def _basic_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class EntityDisambiguator:
    """Entity disambiguation system for resolving conflicts and merging similar entities."""
    
    def __init__(self):
        self.config = get_config()
        self.openrouter = None
        self.knowledge_graph = None
        
        # Common entity aliases and variations
        self.known_aliases = {
            # Technology companies
            "openai": ["open ai", "openai inc", "openai inc.", "open artificial intelligence"],
            "google": ["alphabet", "alphabet inc", "google inc", "google llc"],
            "microsoft": ["microsoft corp", "microsoft corporation", "msft"],
            "meta": ["facebook", "meta platforms", "facebook inc"],
            
            # Programming languages
            "javascript": ["js", "ecmascript", "node.js", "nodejs"],
            "python": ["py", "python3", "cpython"],
            "typescript": ["ts", "typescript language"],
            
            # AI/ML terms
            "artificial intelligence": ["ai", "machine intelligence", "artificial intelligence"],
            "machine learning": ["ml", "statistical learning", "automated learning"],
            "neural network": ["neural net", "nn", "artificial neural network"],
            "deep learning": ["dl", "deep neural networks"],
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.openrouter = OpenRouterProcessor()
        self.knowledge_graph = KnowledgeGraph()
        await self.openrouter.__aenter__()
        await self.knowledge_graph.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.openrouter:
            await self.openrouter.__aexit__(exc_type, exc_val, exc_tb)
        if self.knowledge_graph:
            await self.knowledge_graph.__aexit__(exc_type, exc_val, exc_tb)
    
    async def resolve_entity_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """Resolve conflicts between entities with similar names."""
        if not entities:
            return []
        
        # Group entities by normalized names
        entity_groups = {}
        
        for entity in entities:
            normalized_name = self._normalize_entity_name(entity.name)
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        resolved_entities = []
        
        for normalized_name, group in entity_groups.items():
            if len(group) == 1:
                # No conflicts
                resolved_entities.append(group[0])
            else:
                # Resolve conflicts within the group
                resolved_entity = await self._resolve_entity_group(group)
                resolved_entities.append(resolved_entity)
        
        return resolved_entities
    
    async def merge_similar_entities(self, confidence_threshold: float = 0.85) -> int:
        """Merge similar entities in the knowledge graph."""
        if not self.knowledge_graph:
            raise RuntimeError("Knowledge graph not initialized")
        
        # Get all entities from the knowledge graph
        entities_data = await self.knowledge_graph.search_entities("", limit=1000)
        
        if len(entities_data) < 2:
            return 0
        
        # Convert to Entity objects
        entities = []
        for entity_data in entities_data:
            entity = Entity(
                name=entity_data['name'],
                type=entity_data['type'],
                description=entity_data.get('description', ''),
                confidence=entity_data.get('confidence', 0.0),
                mentions=entity_data.get('mentions', []),
                context=entity_data.get('context', '')
            )
            entities.append(entity)
        
        # Find similar entities
        merge_pairs = []
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                similarity = await self._calculate_entity_similarity(entity1, entity2)
                
                if similarity >= confidence_threshold:
                    merge_pairs.append((entity1, entity2, similarity))
        
        # Sort by similarity (highest first)
        merge_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Perform merges
        merged_count = 0
        already_merged = set()
        
        for entity1, entity2, similarity in merge_pairs:
            if entity1.name in already_merged or entity2.name in already_merged:
                continue
            
            try:
                await self._merge_entities(entity1, entity2)
                already_merged.add(entity2.name)  # Mark the merged entity
                merged_count += 1
                logger.info(f"Merged entities: {entity1.name} <- {entity2.name} (similarity: {similarity:.3f})")
            
            except Exception as e:
                logger.error(f"Failed to merge entities {entity1.name} and {entity2.name}: {e}")
        
        return merged_count
    
    async def detect_entity_aliases(self, entity_name: str) -> List[str]:
        """Detect potential aliases for an entity."""
        aliases = []
        
        # Check known aliases
        normalized_name = entity_name.lower().strip()
        for canonical, alias_list in self.known_aliases.items():
            if normalized_name == canonical or normalized_name in alias_list:
                aliases.extend([canonical] + alias_list)
                break
        
        # Remove the original name and duplicates
        aliases = list(set(aliases))
        if entity_name.lower() in aliases:
            aliases.remove(entity_name.lower())
        
        # AI-based alias detection
        if self.openrouter:
            ai_aliases = await self._ai_alias_detection(entity_name)
            aliases.extend(ai_aliases)
        
        # Remove duplicates and normalize
        final_aliases = []
        seen = set()
        for alias in aliases:
            normalized = alias.lower().strip()
            if normalized not in seen and normalized != entity_name.lower():
                final_aliases.append(alias)
                seen.add(normalized)
        
        return final_aliases
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'\b(inc|inc\.|corp|corp\.|ltd|ltd\.|llc|company|co\.)$', '', normalized)
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    async def _resolve_entity_group(self, entities: List[Entity]) -> Entity:
        """Resolve conflicts within a group of similar entities."""
        if len(entities) == 1:
            return entities[0]
        
        # Sort by confidence and mentions count
        entities.sort(key=lambda e: (e.confidence, len(e.mentions)), reverse=True)
        
        # Use the highest confidence entity as the base
        resolved_entity = entities[0]
        
        # Merge information from other entities
        for other_entity in entities[1:]:
            # Combine mentions
            resolved_entity.mentions.extend(other_entity.mentions)
            
            # Combine context if significantly different
            if (other_entity.context and 
                len(other_entity.context) > len(resolved_entity.context) * 0.5):
                resolved_entity.context += f" {other_entity.context}"
            
            # Use better description if available
            if (len(other_entity.description) > len(resolved_entity.description) and
                other_entity.confidence >= resolved_entity.confidence * 0.8):
                resolved_entity.description = other_entity.description
        
        # Remove duplicate mentions
        resolved_entity.mentions = list(set(resolved_entity.mentions))
        
        # Update confidence based on multiple mentions
        resolved_entity.confidence = min(1.0, resolved_entity.confidence + 
                                       (len(entities) - 1) * 0.1)
        
        return resolved_entity
    
    async def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between two entities."""
        # Name similarity
        name_sim = self._calculate_name_similarity(entity1.name, entity2.name)
        
        # Type similarity
        type_sim = 1.0 if entity1.type == entity2.type else 0.5
        
        # Description similarity
        desc_sim = self._calculate_text_similarity(entity1.description, entity2.description)
        
        # Context similarity
        context_sim = self._calculate_text_similarity(entity1.context, entity2.context)
        
        # Weighted average
        similarity = (name_sim * 0.4 + type_sim * 0.2 + desc_sim * 0.3 + context_sim * 0.1)
        
        return similarity
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between entity names."""
        norm1 = self._normalize_entity_name(name1)
        norm2 = self._normalize_entity_name(name2)
        
        if norm1 == norm2:
            return 1.0
        
        # Check for abbreviations
        if self._is_abbreviation(norm1, norm2) or self._is_abbreviation(norm2, norm1):
            return 0.9
        
        # Calculate edit distance similarity
        return self._edit_distance_similarity(norm1, norm2)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between text descriptions."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _is_abbreviation(self, short: str, long: str) -> bool:
        """Check if short string is an abbreviation of long string."""
        if len(short) >= len(long):
            return False
        
        # Simple check: first letters of words
        long_words = long.split()
        short_clean = short.replace('.', '')
        
        if len(short_clean) == len(long_words):
            return all(short_clean[i].lower() == word[0].lower() 
                      for i, word in enumerate(long_words))
        
        return False
    
    def _edit_distance_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on edit distance."""
        if not s1 or not s2:
            return 0.0
        
        # Simple Levenshtein distance implementation
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        edit_distance = dp[m][n]
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0
    
    async def _merge_entities(self, primary_entity: Entity, secondary_entity: Entity):
        """Merge two entities in the knowledge graph."""
        # This would involve:
        # 1. Combining entity attributes
        # 2. Updating all relationships pointing to secondary entity
        # 3. Deleting the secondary entity
        
        # For now, log the action (actual implementation would require 
        # more complex Neo4j operations)
        logger.info(f"Would merge: {secondary_entity.name} -> {primary_entity.name}")
        
        # TODO: Implement actual merge logic in knowledge graph
        # This requires updating relationships and removing duplicates
    
    async def _ai_alias_detection(self, entity_name: str) -> List[str]:
        """Use AI to detect potential aliases."""
        if not self.openrouter:
            return []
        
        try:
            prompt = f"""List potential aliases, abbreviations, and alternative names for the entity "{entity_name}".

Consider:
- Common abbreviations
- Alternative spellings
- Former names
- Industry-specific terms
- Regional variations

Return only a JSON array of strings: ["alias1", "alias2", ...]
Maximum 10 aliases.
"""
            
            response = await self.openrouter._make_api_request(
                model=self.config.get_model_for_task("simple"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract JSON array
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                aliases = json.loads(json_content)
                
                # Filter and validate aliases
                valid_aliases = []
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip() and len(alias) < 100:
                        valid_aliases.append(alias.strip())
                
                return valid_aliases[:10]  # Limit to 10
        
        except Exception as e:
            logger.error(f"AI alias detection failed for {entity_name}: {e}")
        
        return [] 