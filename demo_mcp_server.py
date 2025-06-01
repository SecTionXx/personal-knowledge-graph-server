#!/usr/bin/env python3
"""
Demonstration script for the MCP Server

This script shows how to:
1. Start the MCP server
2. Test individual tools manually
3. Set up integration with Claude

Usage:
    python demo_mcp_server.py start       # Start the MCP server
    python demo_mcp_server.py test        # Test MCP tools manually
    python demo_mcp_server.py setup       # Show setup instructions for Claude
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mcp_server import (
    process_file, search_knowledge, get_entity_connections,
    scrape_and_process_url, get_knowledge_stats, start_file_monitoring,
    run_server
)
from src.config import initialize_config

async def test_tools():
    """Test MCP tools manually to verify functionality."""
    print("🧪 Testing MCP Server Tools")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = initialize_config()
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return
    
    # Test 1: Get knowledge statistics
    print("\n📊 Testing get_knowledge_stats...")
    try:
        stats_result = await get_knowledge_stats()
        print("✅ get_knowledge_stats works")
        print(f"Preview: {stats_result[:200]}...")
    except Exception as e:
        print(f"❌ get_knowledge_stats failed: {e}")
    
    # Test 2: Create a test file and process it
    print("\n📝 Testing process_file...")
    test_content = """
# AI Research Notes

Dr. Sarah Chen from MIT is working on transformer models.
She collaborates with researchers at Stanford University and OpenAI.
Their latest project focuses on attention mechanisms in deep learning.
Google AI and Meta are also involved in this research area.
"""
    
    # Create test file
    inbox_dir = Path(config.file_monitoring.inbox_directory)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = inbox_dir / "mcp_test_file.md"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    try:
        process_result = await process_file(str(test_file))
        print("✅ process_file works")
        print(f"Result: {process_result}")
    except Exception as e:
        print(f"❌ process_file failed: {e}")
    
    # Test 3: Search knowledge
    print("\n🔍 Testing search_knowledge...")
    try:
        search_result = await search_knowledge("artificial intelligence", 5)
        print("✅ search_knowledge works")
        print(f"Preview: {search_result[:300]}...")
    except Exception as e:
        print(f"❌ search_knowledge failed: {e}")
    
    # Test 4: Web scraping (optional, might fail without internet)
    print("\n🌐 Testing scrape_and_process_url...")
    try:
        # Use a simple, reliable URL for testing
        url_result = await scrape_and_process_url(
            "https://example.com", 
            "Test web scraping functionality"
        )
        print("✅ scrape_and_process_url works")
        print(f"Preview: {url_result[:300]}...")
    except Exception as e:
        print(f"⚠️ scrape_and_process_url failed (expected if no internet): {e}")
    
    # Test 5: Get entity connections (if we have entities)
    print("\n🔗 Testing get_entity_connections...")
    try:
        connections_result = await get_entity_connections("artificial intelligence")
        print("✅ get_entity_connections works")
        print(f"Preview: {connections_result[:300]}...")
    except Exception as e:
        print(f"❌ get_entity_connections failed: {e}")
    
    print("\n🎉 MCP Tools Testing Complete!")
    print("\nIf most tests passed, your MCP server is ready for Claude integration.")

async def start_mcp_server():
    """Start the MCP server."""
    print("🚀 Starting Personal Knowledge Graph MCP Server...")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        await run_server()
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")

def show_setup_instructions():
    """Show setup instructions for Claude integration."""
    print("⚙️ MCP Server Setup Instructions for Claude")
    print("=" * 50)
    
    print("""
🔧 **Prerequisites:**
1. Make sure you have the 'mcp' package installed:
   pip install mcp

2. Your Personal Knowledge Graph Server is set up and working:
   python tests/test_basic_functionality.py

📋 **MCP Server Configuration:**

Create or update your Claude Desktop config file:

**Windows:** %APPDATA%\\Claude\\claude_desktop_config.json
**macOS:** ~/Library/Application Support/Claude/claude_desktop_config.json

Add this configuration:

```json
{
  "mcpServers": {
    "personal-knowledge-graph": {
      "command": "python",
      "args": ["{path_to_your_project}/src/mcp_server.py"],
      "env": {
        "OPENROUTER_API_KEY": "your_openrouter_key_here",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your_neo4j_password",
        "NEO4J_DATABASE": "personal-knowledge"
      }
    }
  }
}
```

🔄 **Alternative: Start Server Manually**

1. Start the MCP server in one terminal:
   python demo_mcp_server.py start

2. The server will be available for Claude to connect to

🛠️ **Available Tools for Claude:**

Once connected, Claude will have access to these tools:

• **process_file(filepath)** - Process files and add to knowledge graph
• **search_knowledge(query, limit)** - Search your knowledge base
• **get_entity_connections(entity_name, depth)** - Explore entity relationships
• **scrape_and_process_url(url, context)** - Process web content
• **get_knowledge_stats()** - View knowledge graph statistics
• **start_file_monitoring()** - Get file monitoring setup instructions

💬 **Example Claude Interactions:**

You: "Process my research notes from yesterday"
Claude: *uses process_file tool* "I've processed your notes and found 15 entities..."

You: "What do I know about machine learning?"
Claude: *uses search_knowledge tool* "Based on your knowledge graph, you have..."

You: "Show me connections for transformer architecture"
Claude: *uses get_entity_connections tool* "Transformer architecture is connected to..."

You: "Process this research paper URL for me"
Claude: *uses scrape_and_process_url tool* "I've scraped and processed the paper..."

🔍 **Testing the Integration:**

1. Restart Claude Desktop after updating the config
2. Start a new conversation
3. Ask Claude: "Can you show me my knowledge graph statistics?"
4. Claude should use the get_knowledge_stats tool automatically

✅ **Success Indicators:**

• Claude mentions using "personal-knowledge-graph" tools
• You can ask Claude to process files and search your knowledge
• Claude can answer questions about your stored knowledge
• Web URLs can be processed through Claude conversations

🆘 **Troubleshooting:**

• Check that Neo4j is running
• Verify your OpenRouter API key is valid
• Ensure all environment variables are set correctly
• Check Claude Desktop logs for connection errors
• Test individual tools first: python demo_mcp_server.py test

🎯 **Ready to Use:**

Once connected, you'll have an AI assistant that can:
• Process any file you give it
• Search through all your knowledge
• Discover connections between concepts
• Learn from web content in real-time
• Help you explore and understand your knowledge graph

Your personal knowledge graph is now AI-powered! 🚀
""")

def print_usage():
    """Print usage information."""
    print("MCP Server Demo and Setup")
    print("=" * 30)
    print()
    print("Usage:")
    print("  python demo_mcp_server.py start       # Start the MCP server")
    print("  python demo_mcp_server.py test        # Test MCP tools manually")
    print("  python demo_mcp_server.py setup       # Show Claude integration setup")
    print()
    print("Examples:")
    print("  python demo_mcp_server.py start")
    print("  python demo_mcp_server.py test")
    print("  python demo_mcp_server.py setup")

async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        await start_mcp_server()
        
    elif command == "test":
        await test_tools()
        
    elif command == "setup":
        show_setup_instructions()
        
    else:
        print(f"❌ Error: Unknown command '{command}'")
        print_usage()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 