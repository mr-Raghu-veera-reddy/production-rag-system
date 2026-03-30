"""
Test complete document processing pipeline
"""

from document_loader import DocumentLoader
from text_chunker import TextChunker

print("\n" + "=" * 70)
print("🧪 TESTING COMPLETE PIPELINE")
print("=" * 70)

# Step 1: Load documents
print("\n📂 Step 1: Loading documents...")
loader = DocumentLoader()
documents = loader.load_directory("./data")

if not documents:
    print("❌ No documents found!")
    exit(1)

print(f"✅ Loaded {len(documents)} documents")

# Step 2: Chunk documents
print("\n✂️  Step 2: Chunking documents...")
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_documents(documents)

print(f"✅ Created {len(chunks)} chunks")

# Step 3: Verify chunks
print("\n🔍 Step 3: Verifying chunks...")

# Check all chunks have required fields
required_fields = ['text', 'source', 'chunk_id', 'word_count']
all_valid = True

for chunk in chunks:
    for field in required_fields:
        if field not in chunk:
            print(f"❌ Chunk missing field: {field}")
            all_valid = False
            break

if all_valid:
    print("✅ All chunks valid")
else:
    print("❌ Some chunks invalid")
    exit(1)

# Print success
print("\n" + "=" * 70)
print("✅ PIPELINE TEST PASSED!")
print("=" * 70)
print("\nSummary:")
print(f"  • Documents loaded: {len(documents)}")
print(f"  • Chunks created: {len(chunks)}")
print(f"  • Average chunk size: {sum(c['word_count'] for c in chunks) / len(chunks):.0f} words")
print("\n🎉 Day 1 objectives complete!")