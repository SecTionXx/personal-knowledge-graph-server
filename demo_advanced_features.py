#!/usr/bin/env python3
"""
Demo script for Advanced Knowledge Graph Features

This script demonstrates the enhanced capabilities of the Personal Knowledge Graph Server:
- Advanced relationship detection (temporal, causal, hierarchical)
- Semantic search with embeddings
- Entity disambiguation and merging
- Concept clustering
- Knowledge gap detection

Run this script to see all the advanced features in action!
"""

import asyncio
import logging
from pathlib import Path
from typing import List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our advanced modules
from src.config import initialize_config
from src.cloud_nlp import Entity, process_file_content
from src.advanced_nlp import AdvancedRelationshipDetector, EntityDisambiguator
from src.semantic_search import SemanticSearchEngine
from src.knowledge_graph import store_extracted_data

class AdvancedFeaturesDemo:
    """Comprehensive demo of advanced knowledge graph features."""
    
    def __init__(self):
        self.sample_content = """
        # AI and Machine Learning Research

        Artificial Intelligence has revolutionized technology in recent years. OpenAI developed GPT models that use transformer architecture, which was first introduced by Google's research team in 2017. 

        Machine Learning enables these systems to learn from data. Deep Learning, a subset of machine learning, uses neural networks with multiple layers. Before deep learning became popular, traditional machine learning algorithms dominated the field.

        The transformer architecture caused a major breakthrough in natural language processing. This innovation led to the development of BERT by Google and GPT by OpenAI. Subsequently, these models influenced the creation of Claude by Anthropic.

        Neural networks are the foundation of deep learning systems. They consist of interconnected nodes that process information. Convolutional Neural Networks (CNNs) are particularly effective for image processing, while Recurrent Neural Networks (RNNs) excel at sequential data processing.

        Companies like Google, OpenAI, and Anthropic are leading AI research. Google created TensorFlow, an open-source machine learning framework. OpenAI focuses on developing safe artificial general intelligence. Anthropic specializes in AI safety research and developing helpful AI assistants.

        The field of artificial intelligence encompasses machine learning, which includes deep learning as a specialized area. Natural language processing is an application area within AI that benefits from transformer models.
        """
    
    async def run_full_demo(self):
        """Run comprehensive demo of all advanced features."""
        print("🚀 Advanced Knowledge Graph Features Demo")
        print("=" * 50)
        
        # Initialize configuration
        try:
            initialize_config()
            print("✅ Configuration initialized")
        except Exception as e:
            print(f"⚠️  Using default configuration: {e}")
        
        await self.demo_basic_processing()
        await self.demo_advanced_relationships()
        await self.demo_entity_disambiguation()
        await self.demo_semantic_search()
        await self.demo_concept_clustering()
        await self.demo_knowledge_gaps()
        await self.demo_real_file_processing()
        
        print("\n🎉 Demo completed! Check the results above to see the advanced features in action.")
    
    async def demo_basic_processing(self):
        """Demonstrate basic entity and relationship extraction."""
        print("\n1️⃣ Basic Entity & Relationship Extraction")
        print("-" * 40)
        
        try:
            entities, relationships = await process_file_content(
                "demo_content.md", 
                self.sample_content, 
                "AI Research Demo"
            )
            
            print(f"✅ Extracted {len(entities)} entities and {len(relationships)} relationships")
            
            # Show sample entities
            print("\n📋 Sample Entities:")
            for entity in entities[:5]:
                print(f"  • {entity.name} ({entity.type}) - {entity.confidence:.2f}")
            
            # Show sample relationships
            print("\n🔗 Sample Relationships:")
            for rel in relationships[:3]:
                print(f"  • {rel.source_entity} → {rel.target_entity} ({rel.relationship_type})")
            
            # Store for later demos
            await store_extracted_data(entities, relationships, "demo_content.md")
            print("💾 Data stored in knowledge graph")
            
        except Exception as e:
            print(f"❌ Basic processing failed: {e}")
    
    async def demo_advanced_relationships(self):
        """Demonstrate advanced relationship detection."""
        print("\n2️⃣ Advanced Relationship Detection")
        print("-" * 40)
        
        try:
            # First extract basic entities
            entities, _ = await process_file_content(
                "demo_content.md", 
                self.sample_content, 
                "AI Research Demo"
            )
            
            async with AdvancedRelationshipDetector() as detector:
                # Detect temporal relationships
                temporal_rels = await detector.detect_temporal_relationships(
                    entities, self.sample_content
                )
                
                # Detect causal relationships
                causal_rels = await detector.detect_causal_relationships(
                    entities, self.sample_content
                )
                
                # Detect hierarchical relationships
                hierarchical_rels = await detector.detect_hierarchical_relationships(
                    entities, self.sample_content
                )
                
                print(f"⏰ Temporal relationships: {len(temporal_rels)}")
                for rel in temporal_rels[:3]:
                    print(f"  • {rel.source_entity} {rel.temporal_indicator} {rel.target_entity} ({rel.confidence:.2f})")
                
                print(f"\n🎯 Causal relationships: {len(causal_rels)}")
                for rel in causal_rels[:3]:
                    print(f"  • {rel.source_entity} {rel.relationship_subtype} {rel.target_entity} ({rel.confidence:.2f})")
                
                print(f"\n🏗️  Hierarchical relationships: {len(hierarchical_rels)}")
                for rel in hierarchical_rels[:3]:
                    print(f"  • {rel.source_entity} {rel.relationship_subtype} {rel.target_entity} ({rel.confidence:.2f})")
                
                # Demonstrate semantic similarity
                if len(entities) >= 2:
                    similarity = await detector.detect_semantic_similarity(entities[0], entities[1])
                    print(f"\n🔍 Semantic similarity between '{entities[0].name}' and '{entities[1].name}': {similarity:.3f}")
        
        except Exception as e:
            print(f"❌ Advanced relationship detection failed: {e}")
    
    async def demo_entity_disambiguation(self):
        """Demonstrate entity disambiguation and merging."""
        print("\n3️⃣ Entity Disambiguation & Merging")
        print("-" * 40)
        
        try:
            # Create some entities with potential duplicates
            duplicate_entities = [
                Entity("OpenAI", "organization", "AI research company", 0.9, ["OpenAI"], ""),
                Entity("Open AI", "organization", "Artificial intelligence research lab", 0.85, ["Open AI"], ""),
                Entity("Google", "organization", "Technology company", 0.95, ["Google"], ""),
                Entity("Alphabet", "organization", "Parent company of Google", 0.9, ["Alphabet"], ""),
                Entity("AI", "concept", "Artificial Intelligence", 0.8, ["AI"], ""),
                Entity("Artificial Intelligence", "concept", "Machine intelligence", 0.9, ["Artificial Intelligence"], ""),
            ]
            
            async with EntityDisambiguator() as disambiguator:
                # Resolve conflicts
                resolved_entities = await disambiguator.resolve_entity_conflicts(duplicate_entities)
                
                print(f"📊 Original entities: {len(duplicate_entities)}")
                print(f"📊 After conflict resolution: {len(resolved_entities)}")
                
                print("\n🔍 Resolved entities:")
                for entity in resolved_entities:
                    print(f"  • {entity.name} ({entity.type}) - confidence: {entity.confidence:.2f}")
                
                # Detect aliases
                print("\n🏷️  Alias detection:")
                for entity_name in ["OpenAI", "Google", "AI"]:
                    aliases = await disambiguator.detect_entity_aliases(entity_name)
                    if aliases:
                        print(f"  • {entity_name}: {', '.join(aliases[:3])}")
        
        except Exception as e:
            print(f"❌ Entity disambiguation failed: {e}")
    
    async def demo_semantic_search(self):
        """Demonstrate semantic search capabilities."""
        print("\n4️⃣ Semantic Search with Embeddings")
        print("-" * 40)
        
        try:
            async with SemanticSearchEngine() as search_engine:
                # Semantic search
                search_queries = [
                    "machine learning algorithms",
                    "neural network architecture",
                    "AI research companies"
                ]
                
                for query in search_queries:
                    print(f"\n🔍 Search: '{query}'")
                    results = await search_engine.semantic_search(query, limit=3)
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            print(f"  {i}. {result.entity_name} ({result.entity_type}) - {result.relevance_score:.3f}")
                    else:
                        print("  No results found")
                
                # Find related entities
                print(f"\n🔗 Related to 'OpenAI':")
                related = await search_engine.find_conceptually_related_entities("OpenAI", limit=3)
                for entity in related:
                    print(f"  • {entity['name']} - similarity: {entity['similarity_score']:.3f}")
        
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
    
    async def demo_concept_clustering(self):
        """Demonstrate concept clustering."""
        print("\n5️⃣ Concept Clustering")
        print("-" * 40)
        
        try:
            async with SemanticSearchEngine() as search_engine:
                clusters = await search_engine.cluster_concepts(min_cluster_size=2, eps=0.4)
                
                if clusters:
                    print(f"🎯 Found {len(clusters)} concept clusters:")
                    
                    for i, cluster in enumerate(clusters[:3], 1):
                        print(f"\n  Cluster {i}: {cluster.dominant_theme}")
                        print(f"    Entities: {', '.join(cluster.entities[:5])}")
                        print(f"    Coherence: {cluster.coherence_score:.3f}")
                else:
                    print("🔍 No clusters found (this is normal with limited demo data)")
        
        except Exception as e:
            print(f"❌ Concept clustering failed: {e}")
    
    async def demo_knowledge_gaps(self):
        """Demonstrate knowledge gap detection."""
        print("\n6️⃣ Knowledge Gap Detection")
        print("-" * 40)
        
        try:
            async with SemanticSearchEngine() as search_engine:
                domains = ["machine learning", "neural networks", "AI safety"]
                
                for domain in domains:
                    print(f"\n🔍 Analyzing domain: '{domain}'")
                    gaps = await search_engine.detect_knowledge_gaps(domain)
                    
                    if gaps:
                        for gap in gaps[:2]:
                            print(f"  • {gap['gap_type']}: {gap['description'][:100]}...")
                            print(f"    Suggestion: {gap['suggestion'][:100]}...")
                    else:
                        print("  ✅ No significant gaps detected")
        
        except Exception as e:
            print(f"❌ Knowledge gap detection failed: {e}")
    
    async def demo_real_file_processing(self):
        """Demonstrate processing a real file if available."""
        print("\n7️⃣ Real File Processing")
        print("-" * 40)
        
        # Create a sample file for demonstration
        sample_file = Path("demo_ai_research.md")
        
        try:
            # Write sample content
            sample_file.write_text(self.sample_content, encoding='utf-8')
            print(f"📄 Created sample file: {sample_file}")
            
            # Process the file
            entities, relationships = await process_file_content(
                str(sample_file), 
                self.sample_content, 
                "Real file processing demo"
            )
            
            print(f"✅ Processed file: {len(entities)} entities, {len(relationships)} relationships")
            
            # Advanced analysis on the file
            async with AdvancedRelationshipDetector() as detector:
                temporal_rels = await detector.detect_temporal_relationships(entities, self.sample_content)
                print(f"⏰ Detected {len(temporal_rels)} temporal relationships from file")
            
            # Clean up
            sample_file.unlink()
            print(f"🗑️  Cleaned up sample file")
        
        except Exception as e:
            print(f"❌ Real file processing failed: {e}")
            # Clean up on error
            if sample_file.exists():
                sample_file.unlink()

def print_feature_summary():
    """Print a summary of available advanced features."""
    print("\n📚 Advanced Features Summary")
    print("=" * 50)
    
    features = [
        ("🧠 Advanced Relationship Detection", [
            "Temporal relationships (before, after, during, concurrent)",
            "Causal relationships (causes, enables, prevents)",
            "Hierarchical relationships (contains, part_of, subtype_of)",
            "AI-enhanced pattern recognition"
        ]),
        ("🔍 Semantic Search Engine", [
            "Vector embeddings for semantic similarity",
            "Contextual entity search",
            "Related concept discovery",
            "Intelligent query understanding"
        ]),
        ("🎯 Entity Disambiguation", [
            "Automatic duplicate detection",
            "Entity merging and consolidation",
            "Alias detection and resolution",
            "Confidence-based conflict resolution"
        ]),
        ("🎪 Concept Clustering", [
            "Automatic thematic grouping",
            "Coherence scoring",
            "Dominant theme identification",
            "Scalable clustering algorithms"
        ]),
        ("🔬 Knowledge Gap Detection", [
            "Missing concept identification",
            "Relationship gap analysis",
            "Domain coverage assessment",
            "Improvement suggestions"
        ]),
        ("🛠️ Enhanced MCP Tools", [
            "find_related_concepts",
            "detect_knowledge_gaps", 
            "suggest_connections",
            "semantic_search_entities",
            "merge_similar_entities",
            "detect_advanced_relationships"
        ])
    ]
    
    for category, items in features:
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\n💡 Integration Benefits:")
    print(f"  • More accurate relationship mapping")
    print(f"  • Reduced entity duplication")
    print(f"  • Intelligent content discovery")
    print(f"  • Enhanced AI assistant interactions")
    print(f"  • Automated knowledge quality improvement")

async def main():
    """Main demo function."""
    print_feature_summary()
    
    print(f"\n🤖 Starting Advanced Features Demo...")
    print(f"This demo will showcase the enhanced knowledge graph capabilities.")
    print(f"Note: Some features require OpenRouter API access and may have reduced functionality in demo mode.")
    
    demo = AdvancedFeaturesDemo()
    await demo.run_full_demo()
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Set up your OpenRouter API key for full functionality")
    print(f"2. Install sentence-transformers for local embeddings")
    print(f"3. Start your MCP server to use these tools with Claude")
    print(f"4. Try processing your own documents for real insights!")

if __name__ == "__main__":
    asyncio.run(main()) 