#!/usr/bin/env python3
"""
Demonstration script for the File Monitoring System

This script shows how to use the file processor to:
1. Process individual files
2. Start continuous monitoring
3. Check processing statistics

Usage:
    python demo_file_monitor.py single <file_path>    # Process one file
    python demo_file_monitor.py monitor               # Start continuous monitoring
    python demo_file_monitor.py test                  # Run a quick test
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.file_processor import FileMonitor, process_single_file
from src.config import initialize_config

async def demo_single_file(file_path: str):
    """Demonstrate processing a single file."""
    print(f"🔄 Processing single file: {file_path}")
    print("-" * 50)
    
    try:
        result = await process_single_file(file_path)
        
        if result.success:
            print(f"✅ Successfully processed: {Path(file_path).name}")
            print(f"   Entities extracted: {result.entities_count}")
            print(f"   Relationships found: {result.relationships_count}")
            print(f"   Processing time: {result.processing_time:.2f} seconds")
        else:
            print(f"❌ Failed to process: {Path(file_path).name}")
            print(f"   Error: {result.error_message}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

async def demo_continuous_monitoring():
    """Demonstrate continuous file monitoring."""
    print("🔄 Starting continuous file monitoring...")
    print("   Drop files into the inbox directory to see them processed automatically")
    print("   Press Ctrl+C to stop")
    print("-" * 50)
    
    monitor = FileMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
        await monitor.stop_monitoring()
        
        # Show final statistics
        stats = monitor.get_statistics()
        print("\n📊 Final Statistics:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Files failed: {stats['files_failed']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Entities extracted: {stats['entities_extracted']}")
        print(f"   Relationships found: {stats['relationships_extracted']}")
        print(f"   Total processing time: {stats['total_processing_time']:.2f}s")

async def demo_test():
    """Run a quick test with a sample file."""
    print("🧪 Running file processor test...")
    print("-" * 50)
    
    # Initialize configuration
    try:
        config = initialize_config()
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return
    
    # Create a test file
    test_content = """
# AI Research Meeting Notes

## Attendees
- Dr. Sarah Chen (MIT)
- Prof. Andrew Ng (Stanford)
- John Smith (OpenAI)

## Discussion Topics

### Transformer Architecture
We discussed the transformer architecture introduced by Vaswani et al. in 2017.
The attention mechanism is revolutionizing natural language processing.

### Current Projects
- GPT model improvements at OpenAI
- BERT research at Google
- Multi-modal AI at Meta

### Collaborations
Stanford and MIT are collaborating on attention mechanisms.
OpenAI is working with academic institutions on safety research.

## Action Items
1. Schedule follow-up meeting
2. Share research papers
3. Begin joint project proposal
"""
    
    # Write test file to inbox
    inbox_dir = Path(config.file_monitoring.inbox_directory)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = inbox_dir / "test_ai_meeting_notes.md"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"📝 Created test file: {test_file}")
    
    # Process the test file
    result = await process_single_file(str(test_file))
    
    if result.success:
        print(f"✅ Test completed successfully!")
        print(f"   Entities extracted: {result.entities_count}")
        print(f"   Relationships found: {result.relationships_count}")
        print(f"   Processing time: {result.processing_time:.2f} seconds")
        
        # Show what was extracted (if we can access the data)
        if result.entities_count > 0:
            print("\n🔍 Processing completed - check your Neo4j database for:")
            print("   • People: Dr. Sarah Chen, Prof. Andrew Ng, John Smith")
            print("   • Organizations: MIT, Stanford, OpenAI, Google, Meta")
            print("   • Concepts: transformer architecture, attention mechanism")
            print("   • Technologies: GPT, BERT")
            
    else:
        print(f"❌ Test failed: {result.error_message}")

def print_usage():
    """Print usage information."""
    print("File Monitoring System Demo")
    print("=" * 30)
    print()
    print("Usage:")
    print("  python demo_file_monitor.py single <file_path>    # Process one file")
    print("  python demo_file_monitor.py monitor               # Start continuous monitoring")
    print("  python demo_file_monitor.py test                  # Run a quick test")
    print()
    print("Examples:")
    print('  python demo_file_monitor.py single "E:/GraphKnowledge/inbox/notes.md"')
    print("  python demo_file_monitor.py monitor")
    print("  python demo_file_monitor.py test")

async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "single":
        if len(sys.argv) < 3:
            print("❌ Error: Please specify a file path")
            print('Usage: python demo_file_monitor.py single "path/to/file.md"')
            return
        
        file_path = sys.argv[2]
        if not Path(file_path).exists():
            print(f"❌ Error: File not found: {file_path}")
            return
        
        await demo_single_file(file_path)
        
    elif command == "monitor":
        await demo_continuous_monitoring()
        
    elif command == "test":
        await demo_test()
        
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