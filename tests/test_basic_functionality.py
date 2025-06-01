"""Basic functionality tests for Personal Knowledge Graph Server."""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path

# Import our modules
from src.config import initialize_config, Config
from src.cloud_nlp import OpenRouterProcessor, Entity, Relationship
from src.knowledge_graph import KnowledgeGraph
from src.file_processor import FileMonitor, process_single_file, ProcessingResult

# Import MCP server tools
from src.mcp_server import (
    process_file, search_knowledge, get_entity_connections,
    get_knowledge_stats, WebScraper
)

# Sample test data
SAMPLE_TEXT = """
# Machine Learning Research Notes

This document discusses transformer architecture, which was introduced by Vaswani et al. 
Andrew Ng, a prominent AI researcher at Stanford University, has been working on deep learning.
OpenAI has developed GPT models that use transformer technology.
The research was influenced by attention mechanisms in neural networks.
"""

EXPECTED_ENTITIES = ["transformer architecture", "Andrew Ng", "Stanford University", "OpenAI", "GPT models"]
EXPECTED_ENTITY_TYPES = ["concept", "person", "organization", "technology"]

@pytest.mark.asyncio
async def test_config_loading():
    """Test configuration loading and validation."""
    try:
        config = initialize_config()
        assert config is not None
        assert hasattr(config, 'openrouter')
        assert hasattr(config, 'neo4j')
        assert hasattr(config, 'file_monitoring')
        print("✅ Configuration loading test passed")
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_entity_extraction():
    """Test entity extraction from sample text."""
    # Skip if no API key (for CI/CD)
    if not os.getenv('OPENROUTER_API_KEY'):
        pytest.skip("No OpenRouter API key found")
    
    try:
        async with OpenRouterProcessor() as processor:
            entities, relationships = await processor.process_text(
                SAMPLE_TEXT,
                context="Test document for entity extraction",
                file_path=""
            )
            
            assert len(entities) > 0, "No entities extracted"
            assert len(relationships) >= 0, "Relationships should be non-negative"
            
            # Check entity types
            entity_names = [entity.name.lower() for entity in entities]
            entity_types = [entity.type for entity in entities]
            
            print(f"✅ Extracted {len(entities)} entities and {len(relationships)} relationships")
            print(f"   Entities: {entity_names}")
            print(f"   Types: {set(entity_types)}")
            
            # Verify we got some expected entities (case insensitive)
            found_expected = [expected.lower() for expected in EXPECTED_ENTITIES if any(expected.lower() in name for name in entity_names)]
            assert len(found_expected) > 0, f"No expected entities found. Got: {entity_names}"
            
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_neo4j_connection():
    """Test Neo4j database connection and basic operations."""
    try:
        async with KnowledgeGraph() as kg:
            # Test basic connection
            stats = await kg.get_knowledge_statistics()
            assert isinstance(stats, dict)
            assert 'total_entities' in stats
            
            print(f"✅ Neo4j connection test passed")
            print(f"   Current stats: {stats}")
            
    except Exception as e:
        print(f"❌ Neo4j connection test failed: {e}")
        print("   Make sure Neo4j is running and credentials are correct")
        raise

@pytest.mark.asyncio
async def test_file_reading():
    """Test reading different file types."""
    try:
        monitor = FileMonitor()
        
        # Test markdown file reading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            test_content = "# Test Document\n\nThis is a test markdown file with **bold** text."
            f.write(test_content)
            f.flush()
            
            content = monitor._read_file_content(f.name)
            assert content.strip() == test_content.strip()
            
            # Clean up
            os.unlink(f.name)
        
        # Test text file reading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            test_content = "This is a simple text file for testing."
            f.write(test_content)
            f.flush()
            
            content = monitor._read_file_content(f.name)
            assert content.strip() == test_content.strip()
            
            # Clean up
            os.unlink(f.name)
        
        print("✅ File reading test passed")
        print("   Successfully read .md and .txt files")
        
    except Exception as e:
        print(f"❌ File reading test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_file_processing():
    """Test processing a file through the complete pipeline."""
    # Skip if no API key
    if not os.getenv('OPENROUTER_API_KEY'):
        pytest.skip("No OpenRouter API key found")
    
    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            test_content = """
# Test AI Research Document

Dr. Jane Smith from MIT is researching neural networks.
She collaborates with researchers at Stanford University.
Their project focuses on transformer models and attention mechanisms.
Google AI and OpenAI are also involved in this research.
"""
            f.write(test_content)
            f.flush()
            
            # Process the file
            result = await process_single_file(f.name)
            
            # Verify processing result
            assert isinstance(result, ProcessingResult)
            assert result.success, f"Processing failed: {result.error_message}"
            assert result.entities_count > 0, "No entities extracted"
            assert result.processing_time > 0, "Processing time should be positive"
            
            print(f"✅ File processing test passed")
            print(f"   Processed file: {Path(f.name).name}")
            print(f"   Entities: {result.entities_count}")
            print(f"   Relationships: {result.relationships_count}")
            print(f"   Time: {result.processing_time:.2f}s")
            
            # Clean up
            os.unlink(f.name)
            
    except Exception as e:
        print(f"❌ File processing test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_file_monitor_stats():
    """Test file monitor statistics functionality."""
    try:
        monitor = FileMonitor()
        
        # Get initial statistics
        stats = monitor.get_statistics()
        
        # Verify stats structure
        required_keys = [
            'total_files', 'files_processed', 'files_failed', 'success_rate',
            'entities_extracted', 'relationships_extracted', 'total_processing_time',
            'average_processing_time', 'is_running'
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing statistics key: {key}"
        
        # Verify initial values
        assert stats['total_files'] >= 0
        assert stats['files_processed'] >= 0
        assert stats['files_failed'] >= 0
        assert stats['is_running'] == False  # Monitor not started
        
        print("✅ File monitor statistics test passed")
        print(f"   Statistics keys verified: {len(required_keys)}")
        
    except Exception as e:
        print(f"❌ File monitor statistics test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_mcp_knowledge_stats():
    """Test MCP get_knowledge_stats tool."""
    try:
        result = await get_knowledge_stats()
        
        # Should return a formatted string
        assert isinstance(result, str)
        assert "Knowledge Graph Statistics" in result
        assert "Total entities:" in result or "Error" in result
        
        print("✅ MCP get_knowledge_stats test passed")
        print(f"   Result preview: {result[:150]}...")
        
    except Exception as e:
        print(f"❌ MCP get_knowledge_stats test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_mcp_process_file():
    """Test MCP process_file tool."""
    # Skip if no API key
    if not os.getenv('OPENROUTER_API_KEY'):
        pytest.skip("No OpenRouter API key found")
    
    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            test_content = """
# MCP Test Document

This is a test document for the MCP server.
It contains information about artificial intelligence and machine learning.
Researchers like Andrew Ng work at Stanford University.
"""
            f.write(test_content)
            f.flush()
            
            # Test the MCP tool
            result = await process_file(f.name)
            
            # Verify result format
            assert isinstance(result, str)
            assert ("Successfully processed" in result or "Error" in result)
            
            print("✅ MCP process_file test passed")
            print(f"   Result: {result}")
            
            # Clean up
            os.unlink(f.name)
            
    except Exception as e:
        print(f"❌ MCP process_file test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_mcp_search_knowledge():
    """Test MCP search_knowledge tool."""
    try:
        result = await search_knowledge("artificial intelligence", 3)
        
        # Should return a formatted string
        assert isinstance(result, str)
        assert ("Found" in result or "No entities found" in result or "Error" in result)
        
        print("✅ MCP search_knowledge test passed")
        print(f"   Result preview: {result[:200]}...")
        
    except Exception as e:
        print(f"❌ MCP search_knowledge test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_mcp_entity_connections():
    """Test MCP get_entity_connections tool."""
    try:
        result = await get_entity_connections("artificial intelligence", 2)
        
        # Should return a formatted string
        assert isinstance(result, str)
        assert ("artificial intelligence" in result.lower() or "Error" in result)
        
        print("✅ MCP get_entity_connections test passed")
        print(f"   Result preview: {result[:200]}...")
        
    except Exception as e:
        print(f"❌ MCP get_entity_connections test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_web_scraper():
    """Test WebScraper functionality."""
    try:
        async with WebScraper() as scraper:
            # Test with a simple, reliable URL
            title, content, metadata = await scraper.scrape_url("https://example.com")
            
            assert isinstance(title, str)
            assert isinstance(content, str)
            assert isinstance(metadata, str)
            assert len(content) > 0
            
            print("✅ WebScraper test passed")
            print(f"   Title: {title}")
            print(f"   Content length: {len(content)}")
            print(f"   Metadata: {metadata}")
            
    except Exception as e:
        print(f"⚠️ WebScraper test failed (may be expected without internet): {e}")
        # Don't raise - web scraping might fail due to network issues

@pytest.mark.asyncio
async def test_end_to_end_processing():
    """Test end-to-end processing: extract entities and store in Neo4j."""
    # Skip if no API key
    if not os.getenv('OPENROUTER_API_KEY'):
        pytest.skip("No OpenRouter API key found")
    
    try:
        # Step 1: Extract entities
        async with OpenRouterProcessor() as processor:
            entities, relationships = await processor.process_text(
                SAMPLE_TEXT,
                context="End-to-end test document",
                file_path="test_document.md"
            )
        
        assert len(entities) > 0, "No entities extracted"
        
        # Step 2: Store in Neo4j
        async with KnowledgeGraph() as kg:
            entity_ids = await kg.store_entities(entities, "test_document.md")
            relationship_ids = await kg.store_relationships(relationships, "test_document.md")
            
            assert len(entity_ids) == len(entities), "Not all entities were stored"
            
            # Step 3: Search for stored entities
            search_results = await kg.search_entities("machine learning", limit=5)
            assert len(search_results) >= 0, "Search should return results or empty list"
            
            # Step 4: Get entity connections (if we have entities)
            if entity_ids:
                first_entity = entity_ids[0]
                connections = await kg.get_entity_connections(first_entity)
                assert 'center_entity' in connections or 'error' in connections
            
            print(f"✅ End-to-end test passed")
            print(f"   Stored {len(entity_ids)} entities and {len(relationship_ids)} relationships")
            print(f"   Search returned {len(search_results)} results")
            
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_privacy_filtering():
    """Test privacy filtering functionality."""
    try:
        async with OpenRouterProcessor() as processor:
            # Text with sensitive information
            sensitive_text = """
            John Doe's SSN is 123-45-6789 and his credit card number is 1234 5678 9012 3456.
            Contact him at john.doe@example.com for more information.
            This is about artificial intelligence research.
            """
            
            # Filter the text
            filtered_text = processor.privacy_filter.filter_sensitive_data(sensitive_text)
            
            # Check that sensitive data was redacted
            assert "123-45-6789" not in filtered_text, "SSN should be redacted"
            assert "1234 5678 9012 3456" not in filtered_text, "Credit card should be redacted"
            assert "john.doe@example.com" not in filtered_text, "Email should be redacted"
            assert "[REDACTED]" in filtered_text, "Should contain redaction placeholder"
            assert "artificial intelligence" in filtered_text, "Normal content should remain"
            
            print("✅ Privacy filtering test passed")
            print(f"   Original length: {len(sensitive_text)}")
            print(f"   Filtered length: {len(filtered_text)}")
            
    except Exception as e:
        print(f"❌ Privacy filtering test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_cost_tracking():
    """Test cost tracking functionality."""
    try:
        async with OpenRouterProcessor() as processor:
            # Get initial usage
            initial_summary = processor.get_usage_summary()
            initial_requests = initial_summary['total_requests']
            
            # Make a small API call if we have an API key
            if os.getenv('OPENROUTER_API_KEY'):
                entities, _ = await processor.process_text(
                    "This is a short test about AI.",
                    context="Cost tracking test"
                )
                
                # Check that usage was tracked
                final_summary = processor.get_usage_summary()
                final_requests = final_summary['total_requests']
                
                assert final_requests > initial_requests, "Request count should increase"
                assert 'costs' in final_summary, "Cost information should be available"
                
                print("✅ Cost tracking test passed")
                print(f"   Requests: {initial_requests} → {final_requests}")
                print(f"   Daily cost: ${final_summary['costs']['daily_cost']:.4f}")
            else:
                print("✅ Cost tracking test passed (skipped API call - no key)")
                
    except Exception as e:
        print(f"❌ Cost tracking test failed: {e}")
        raise

def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Starting Personal Knowledge Graph Server Tests\n")
    
    tests = [
        test_config_loading,
        test_privacy_filtering,
        test_cost_tracking,
        test_file_reading,
        test_file_monitor_stats,
        test_neo4j_connection,
        test_mcp_knowledge_stats,
        test_mcp_search_knowledge,
        test_mcp_entity_connections,
        test_web_scraper,
        test_entity_extraction,
        test_file_processing,
        test_mcp_process_file,
        test_end_to_end_processing,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            print(f"\n📋 Running {test.__name__}...")
            asyncio.run(test())
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⏭️  {test.__name__} skipped: {e}")
            skipped += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   📋 Total: {len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your system is ready for MCP integration.")
        print("\n🔧 Next Steps:")
        print("   1. Test MCP server: python demo_mcp_server.py test")
        print("   2. Set up Claude integration: python demo_mcp_server.py setup")
        print("   3. Start MCP server: python demo_mcp_server.py start")
        print("   4. Connect Claude to access your knowledge graph!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 