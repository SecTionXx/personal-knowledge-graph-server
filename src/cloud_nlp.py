"""OpenRouter API integration for AI-powered entity extraction and relationship mapping."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import httpx
import re

from .config import get_config

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    description: str
    confidence: float
    mentions: List[str]
    context: str

@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    confidence: float
    context: str

@dataclass
class UsageStats:
    """Tracks API usage and costs."""
    date: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float

class CostTracker:
    """Track API usage and costs to stay within budget."""
    
    def __init__(self):
        self.config = get_config()
        self.usage_log: List[UsageStats] = []
        
        # Model pricing (per 1M tokens)
        self.pricing = {
            "anthropic/claude-3-haiku:beta": {"input": 0.25, "output": 1.25},
            "anthropic/claude-3-sonnet:beta": {"input": 3.0, "output": 15.0},
            "mistralai/mixtral-8x7b-instruct": {"input": 0.6, "output": 0.6},
        }
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API usage."""
        if model not in self.pricing:
            logger.warning(f"Unknown model pricing: {model}")
            return 0.0
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def log_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Log API usage."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        usage = UsageStats(
            date=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )
        
        self.usage_log.append(usage)
        logger.info(f"API usage: {model} - ${cost:.4f} ({input_tokens} in, {output_tokens} out)")
    
    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get total cost for a specific day."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        return sum(
            usage.cost for usage in self.usage_log
            if usage.date.startswith(date)
        )
    
    def get_weekly_cost(self) -> float:
        """Get total cost for the current week."""
        week_ago = datetime.now() - timedelta(days=7)
        week_ago_str = week_ago.isoformat()
        
        return sum(
            usage.cost for usage in self.usage_log
            if usage.date >= week_ago_str
        )
    
    def get_monthly_cost(self) -> float:
        """Get total cost for the current month."""
        month_ago = datetime.now() - timedelta(days=30)
        month_ago_str = month_ago.isoformat()
        
        return sum(
            usage.cost for usage in self.usage_log
            if usage.date >= month_ago_str
        )
    
    def check_budget_limits(self) -> Dict[str, bool]:
        """Check if usage is within budget limits."""
        daily_cost = self.get_daily_cost()
        weekly_cost = self.get_weekly_cost()
        monthly_cost = self.get_monthly_cost()
        
        return {
            "daily_ok": daily_cost <= self.config.budget.daily_limit,
            "weekly_ok": weekly_cost <= self.config.budget.weekly_limit,
            "monthly_ok": monthly_cost <= self.config.budget.monthly_limit,
            "daily_cost": daily_cost,
            "weekly_cost": weekly_cost,
            "monthly_cost": monthly_cost
        }

class PrivacyFilter:
    """Filter sensitive information from text before sending to API."""
    
    def __init__(self):
        self.config = get_config()
        self.compiled_patterns = [
            re.compile(pattern) for pattern in self.config.privacy.sensitive_patterns
        ]
    
    def filter_sensitive_data(self, text: str) -> str:
        """Remove sensitive information from text."""
        filtered_text = text
        
        for pattern in self.compiled_patterns:
            filtered_text = pattern.sub(self.config.privacy.redaction_placeholder, filtered_text)
        
        return filtered_text
    
    def has_sensitive_data(self, text: str) -> bool:
        """Check if text contains sensitive information."""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False

class OpenRouterProcessor:
    """Main class for OpenRouter API integration."""
    
    def __init__(self):
        self.config = get_config()
        self.cost_tracker = CostTracker()
        self.privacy_filter = PrivacyFilter()
        
        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {self.config.openrouter.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Personal-Knowledge-Graph-Server/1.0"
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    async def _make_api_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make a request to the OpenRouter API."""
        # Check budget limits
        budget_status = self.cost_tracker.check_budget_limits()
        if not budget_status["monthly_ok"]:
            raise ValueError(f"Monthly budget exceeded: ${budget_status['monthly_cost']:.2f}")
        
        if not budget_status["daily_ok"]:
            logger.warning(f"Daily budget exceeded: ${budget_status['daily_cost']:.2f}")
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.openrouter.max_tokens,
            "temperature": self.config.openrouter.temperature
        }
        
        try:
            response = await self.client.post(
                f"{self.config.openrouter.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Log usage
            if "usage" in result:
                usage = result["usage"]
                self.cost_tracker.log_usage(
                    model=model,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0)
                )
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            raise
    
    async def extract_entities(
        self,
        text: str,
        context: str = "",
        complexity: str = "simple"
    ) -> List[Entity]:
        """Extract entities from text using AI."""
        # Filter sensitive information
        filtered_text = self.privacy_filter.filter_sensitive_data(text)
        
        # Choose model based on complexity
        model = self.config.get_model_for_task(complexity)
        
        # Prepare prompt
        system_prompt = """You are an expert at extracting entities from text. Extract all important entities including:
- People (names, roles, titles)
- Organizations (companies, institutions, groups)
- Concepts (ideas, theories, methodologies)
- Technologies (software, tools, platforms)
- Projects (initiatives, research, products)
- Locations (places, addresses, regions)
- Events (meetings, conferences, incidents)

For each entity, provide:
1. Name (canonical form)
2. Type (person/organization/concept/technology/project/location/event)
3. Description (brief explanation)
4. Confidence (0.0-1.0)
5. Mentions (how it appears in text)
6. Context (surrounding information)

Return JSON format only."""
        
        user_prompt = f"""Context: {context}

Text to analyze:
{filtered_text}

Extract entities in this JSON format:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type",
      "description": "brief description",
      "confidence": 0.9,
      "mentions": ["mention1", "mention2"],
      "context": "surrounding context"
    }}
  ]
}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self._make_api_request(model, messages)
            
            content = response["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                data = json.loads(content)
                entities = []
                
                for entity_data in data.get("entities", []):
                    entity = Entity(
                        name=entity_data.get("name", ""),
                        type=entity_data.get("type", ""),
                        description=entity_data.get("description", ""),
                        confidence=float(entity_data.get("confidence", 0.0)),
                        mentions=entity_data.get("mentions", []),
                        context=entity_data.get("context", "")
                    )
                    
                    # Filter by confidence threshold
                    if entity.confidence >= self.config.entity_extraction.confidence_threshold:
                        entities.append(entity)
                
                logger.info(f"Extracted {len(entities)} entities from text")
                return entities
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse entity extraction response: {e}")
                logger.debug(f"Raw response: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def extract_relationships(
        self,
        entities: List[Entity],
        text: str,
        context: str = ""
    ) -> List[Relationship]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []
        
        # Filter sensitive information
        filtered_text = self.privacy_filter.filter_sensitive_data(text)
        
        # Use complex model for relationship extraction
        model = self.config.get_model_for_task("complex")
        
        entity_list = "\n".join([f"- {entity.name} ({entity.type}): {entity.description}" for entity in entities])
        
        system_prompt = """You are an expert at identifying relationships between entities. Analyze the provided text and entities to find meaningful connections.

Relationship types include:
- works_for, collaborates_with, reports_to
- influences, depends_on, contradicts
- part_of, contains, related_to
- created_by, used_by, implements
- mentioned_in, discussed_in, references

For each relationship, provide:
1. Source entity name (exactly as provided)
2. Target entity name (exactly as provided)
3. Relationship type
4. Description
5. Confidence (0.0-1.0)
6. Context (evidence from text)

Return JSON format only."""
        
        user_prompt = f"""Context: {context}

Entities found:
{entity_list}

Text to analyze:
{filtered_text}

Find relationships in this JSON format:
{{
  "relationships": [
    {{
      "source_entity": "entity name",
      "target_entity": "entity name",
      "relationship_type": "relationship type",
      "description": "relationship description",
      "confidence": 0.9,
      "context": "evidence from text"
    }}
  ]
}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self._make_api_request(model, messages)
            content = response["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                data = json.loads(content)
                relationships = []
                
                entity_names = {entity.name for entity in entities}
                
                for rel_data in data.get("relationships", []):
                    # Validate that entities exist
                    source = rel_data.get("source_entity", "")
                    target = rel_data.get("target_entity", "")
                    
                    if source in entity_names and target in entity_names and source != target:
                        relationship = Relationship(
                            source_entity=source,
                            target_entity=target,
                            relationship_type=rel_data.get("relationship_type", ""),
                            description=rel_data.get("description", ""),
                            confidence=float(rel_data.get("confidence", 0.0)),
                            context=rel_data.get("context", "")
                        )
                        
                        # Filter by confidence threshold
                        if relationship.confidence >= self.config.entity_extraction.confidence_threshold:
                            relationships.append(relationship)
                
                logger.info(f"Extracted {len(relationships)} relationships")
                return relationships
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse relationship extraction response: {e}")
                logger.debug(f"Raw response: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    async def process_text(
        self,
        text: str,
        context: str = "",
        file_path: str = ""
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Complete processing of text to extract entities and relationships."""
        logger.info(f"Processing text: {len(text)} characters")
        
        # Check if we should process locally for sensitive content
        if file_path and self.config.is_sensitive_directory(file_path):
            logger.info("Sensitive directory detected - skipping cloud processing")
            return [], []
        
        if self.privacy_filter.has_sensitive_data(text):
            logger.warning("Sensitive data detected in text")
        
        # Extract entities first
        entities = await self.extract_entities(text, context, "simple")
        
        if not entities:
            logger.warning("No entities extracted")
            return [], []
        
        # Then extract relationships
        relationships = await self.extract_relationships(entities, text, context)
        
        return entities, relationships
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of API usage and costs."""
        budget_status = self.cost_tracker.check_budget_limits()
        
        return {
            "costs": budget_status,
            "total_requests": len(self.cost_tracker.usage_log),
            "models_used": list(set(usage.model for usage in self.cost_tracker.usage_log)),
            "budget_limits": {
                "daily": self.config.budget.daily_limit,
                "weekly": self.config.budget.weekly_limit,
                "monthly": self.config.budget.monthly_limit
            }
        }

# Convenience functions for easy use
async def process_file_content(file_path: str, content: str, context: str = "") -> Tuple[List[Entity], List[Relationship]]:
    """Process file content and extract entities and relationships."""
    async with OpenRouterProcessor() as processor:
        return await processor.process_text(content, context, file_path)

async def extract_entities_from_text(text: str, context: str = "") -> List[Entity]:
    """Extract entities from text."""
    async with OpenRouterProcessor() as processor:
        entities, _ = await processor.process_text(text, context)
        return entities 