# Production RAG System

A production-grade Retrieval Augmented Generation (RAG) system.

## Progress

### ✅ Day 1 - Document Processing

- [x] Environment setup
- [x] OpenAI API integration
- [x] PDF document loader
- [x] Text chunking with overlap
- [x] Basic testing

### 🔄 Day 2 - Embeddings & Vector Store

- [ ] Embedding generation
- [ ] ChromaDB setup
- [ ] Document ingestion pipeline

## Current Status

Can load PDFs and split them into chunks ready for embedding.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "OPENAI_API_KEY=your-key" > .env

# Test
python src/test_pipeline.py
```

## Structure

```
src/
├── document_loader.py  # Load PDFs
├── text_chunker.py     # Split into chunks
└── test_*.py           # Test files
```
