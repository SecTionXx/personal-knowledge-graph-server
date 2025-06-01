#!/usr/bin/env python3
"""
Quick Start Script for Personal Knowledge Graph Server

This script helps you test the basic functionality of your knowledge graph system.
Run this after setting up your environment variables and Neo4j database.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    print("🚀 Personal Knowledge Graph Server - Quick Start")
    print("=" * 50)
    
    # Check environment variables
    print("\n📋 Checking environment...")
    
    required_vars = [
        "OPENROUTER_API_KEY",
        "NEO4J_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Please create a .env file with these variables.")
        print("   See README.md for setup instructions.")
        return False
    
    print("✅ Environment variables found")
    
    # Test configuration loading
    print("\n📋 Testing configuration...")
    try:
        from config import initialize_config
        config = initialize_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test Neo4j connection
    print("\n📋 Testing Neo4j connection...")
    try:
        from knowledge_graph import KnowledgeGraph
        async with KnowledgeGraph() as kg:
            stats = await kg.get_knowledge_statistics()
            print(f"✅ Connected to Neo4j!")
            print(f"   Current entities: {stats['total_entities']}")
            print(f"   Current relationships: {stats['total_relationships']}")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("   Make sure Neo4j is running and credentials are correct")
        return False
    
    # Test entity extraction
    print("\n📋 Testing AI entity extraction...")
    try:
        from cloud_nlp import OpenRouterProcessor
        
        sample_text = """
        # AI Research Meeting Notes
        
        Today we discussed transformer architecture with Dr. Sarah Chen from MIT.
        OpenAI's GPT models have revolutionized natural language processing.
        We're planning to collaborate with Google Brain on attention mechanisms.
        The project timeline is 6 months, starting in January 2024.
        """
        
        async with OpenRouterProcessor() as processor:
            entities, relationships = await processor.process_text(
                sample_text,
                context="Quick start test document",
                file_path="quick_start_test.md"
            )
            
            print(f"✅ Entity extraction working!")
            print(f"   Extracted {len(entities)} entities:")
            for entity in entities[:5]:  # Show first 5
                print(f"     • {entity.name} ({entity.type})")
            
            if len(entities) > 5:
                print(f"     ... and {len(entities) - 5} more")
            
            print(f"   Found {len(relationships)} relationships")
            for rel in relationships[:3]:  # Show first 3
                print(f"     • {rel.source_entity} -> {rel.target_entity} ({rel.relationship_type})")
            
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
        return False
    
    # Test storing in knowledge graph
    print("\n📋 Testing knowledge graph storage...")
    try:
        from knowledge_graph import store_extracted_data
        
        entity_ids, rel_ids = await store_extracted_data(
            entities, relationships, "quick_start_test.md"
        )
        
        print(f"✅ Successfully stored data!")
        print(f"   Stored {len(entity_ids)} entities")
        print(f"   Stored {len(rel_ids)} relationships")
        
    except Exception as e:
        print(f"❌ Knowledge graph storage failed: {e}")
        return False
    
    # Test search functionality
    print("\n📋 Testing search functionality...")
    try:
        from knowledge_graph import search_knowledge_graph, get_entity_info
        
        # Search for AI-related entities
        search_results = await search_knowledge_graph("AI", limit=3)
        
        print(f"✅ Search functionality working!")
        print(f"   Found {len(search_results)} entities matching 'AI':")
        
        for result in search_results:
            print(f"     • {result['name']} ({result['type']}) - Score: {result['relevance_score']:.2f}")
        
        # Get detailed info about first entity
        if search_results:
            entity_name = search_results[0]['name']
            entity_info = await get_entity_info(entity_name)
            
            if 'center_entity' in entity_info:
                connections = entity_info['total_connections']
                print(f"   Entity '{entity_name}' has {connections} connections")
            
    except Exception as e:
        print(f"❌ Search functionality failed: {e}")
        return False
    
    # Test cost tracking
    print("\n📋 Testing cost tracking...")
    try:
        async with OpenRouterProcessor() as processor:
            usage = processor.get_usage_summary()
            costs = usage['costs']
            
            print(f"✅ Cost tracking working!")
            print(f"   Daily cost: ${costs['daily_cost']:.4f}")
            print(f"   Weekly cost: ${costs['weekly_cost']:.4f}")
            print(f"   Total requests: {usage['total_requests']}")
            
    except Exception as e:
        print(f"❌ Cost tracking failed: {e}")
        return False
    
    # Success summary
    print("\n🎉 Quick Start Complete!")
    print("=" * 50)
    print("Your Personal Knowledge Graph Server is ready!")
    print("\n📋 Next Steps:")
    print("1. Add files to E:/GraphKnowledge/inbox for automatic processing")
    print("2. Run the full test suite: python tests/test_basic_functionality.py")
    print("3. Start building your knowledge graph with real documents")
    print("4. Explore the README.md for advanced usage examples")
    
    return True

def setup_environment_check():
    """Check if basic setup requirements are met."""
    print("🔧 Environment Setup Check")
    print("-" * 30)
    
    # Check Python version
    import sys
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required. You have {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("   Create .env file with your configuration")
        print("   See README.md for required variables")
        return False
    print("✅ .env file found")
    
    # Check if requirements are installed
    try:
        import httpx, neo4j, yaml, watchdog
        print("✅ Main dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    print("Personal Knowledge Graph Server - Quick Start")
    print("=" * 50)
    
    # Environment check first
    if not setup_environment_check():
        print("\n❌ Environment setup incomplete. Please check the issues above.")
        sys.exit(1)
    
    # Run main test
    try:
        success = asyncio.run(main())
        if success:
            print("\n✅ All tests passed! Your system is ready.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 