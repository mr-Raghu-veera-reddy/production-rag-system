# 🤖 Production RAG System

A complete production-grade Retrieval Augmented Generation (RAG) system with semantic search, LLM-powered answers, and a chat interface.

[![Live Demo](https://img.shields.io/badge/demo-coming_soon-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## ✨ Features

- **📄 Document Processing**: Load and parse PDF documents
- **✂️ Smart Chunking**: Split documents with overlap for context
- **🧮 Embeddings**: Generate semantic embeddings using OpenAI
- **💾 Vector Storage**: ChromaDB for persistent vector search
- **🔍 Semantic Retrieval**: Find relevant chunks by meaning, not keywords
- **🤖 LLM Answers**: GPT-powered answers with source citations
- **💬 Chat Interface**: Clean Streamlit UI for conversations
- **📊 Metrics**: Track costs, tokens, and performance

## 🎯 Progress

### ✅ Day 1 - Document Processing

- [x] Environment setup
- [x] PDF document loader
- [x] Text chunking with overlap

### ✅ Day 2 - Embeddings & Vector Store

- [x] OpenAI embedding generation
- [x] ChromaDB vector database
- [x] Complete ingestion pipeline

### ✅ Day 3 - Retrieval & QA

- [x] Semantic retriever
- [x] QA generator with citations
- [x] Complete RAG system
- [x] Streamlit chat UI

## 📊 Performance

| Metric            | Value          |
| ----------------- | -------------- |
| Answer Quality    | High (GPT-3.5) |
| Retrieval Speed   | ~1-2s          |
| Answer Generation | ~2-3s          |
| Total Latency     | ~3-5s          |
| Cost per Query    | $0.0005-0.001  |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
OpenAI API key ($5 credit recommended)
```

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/production-rag-system
cd production-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "OPENAI_API_KEY=your-key-here" > .env

# Add PDFs to data folder
cp your-documents.pdf ./data/

# Ingest documents
python src/ingestion_pipeline.py

# Run UI
streamlit run app.py
```

## 💻 Usage

### Chat Interface

```bash
streamlit run app.py
# Opens browser automatically
# Ask questions about your documents
```

### Python API

```python
from src.rag_system import RAGSystem

# Initialize
rag = RAGSystem()

# Ask question
result = rag.query("What is machine learning?")

# Print answer
print(result['answer'])
print(result['sources'])
```

### Command Line

```python
python src/rag_system.py
# Follow prompts for interactive mode
```

## 🏗️ Architecture

```
┌─────────────┐
│   PDF Docs  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Loader    │ (PyPDF2)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunker   │ (500 words, 50 overlap)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embeddings │ (OpenAI text-embedding-3-small)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ChromaDB   │ (Vector storage)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retriever  │ (Semantic search)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ QA Generator│ (GPT-3.5-turbo)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Answer    │ (With citations)
└─────────────┘
```

## 📁 Project Structure

```
production-rag-system/
├── src/
│   ├── document_loader.py      # PDF loading
│   ├── text_chunker.py         # Text chunking
│   ├── embeddings.py           # Embedding generation
│   ├── vector_store.py         # ChromaDB interface
│   ├── retriever.py            # Semantic retrieval
│   ├── qa_generator.py         # Answer generation
│   ├── rag_system.py           # Complete system
│   └── ingestion_pipeline.py   # Document ingestion
├── tests/
│   ├── test_day2.py
│   └── test_complete_system.py
├── data/                        # Your PDF documents
├── chroma_db/                   # Vector database
├── app.py                       # Streamlit UI
├── requirements.txt
└── README.md
```

## 🧪 Testing

```bash
# Test individual components
python src/embeddings.py
python src/vector_store.py
python src/retriever.py
python src/qa_generator.py

# Test complete system
python tests/test_complete_system.py
```

## 💰 Cost Estimate

**For 3-5 PDFs (~50 chunks):**

- Initial ingestion: ~$0.08
- Per query: ~$0.0005-0.001
- 100 queries: ~$0.05-0.10

**Total for development: ~$0.20-0.50**

## 🔧 Configuration

Edit in code or UI sidebar:

- **Model**: `gpt-3.5-turbo` (fast) or `gpt-4` (better)
- **Top K**: Number of chunks to retrieve (1-10)
- **Temperature**: 0 (focused) to 1 (creative)
- **Chunk Size**: Default 500 words
- **Chunk Overlap**: Default 50 words

## 🚀 Next Steps

- [ ] Deploy to cloud (Streamlit Cloud)
- [ ] Add user authentication
- [ ] Support more file types (DOCX, TXT)
- [ ] Add conversation memory
- [ ] Implement query rewriting
- [ ] Add hybrid search (keyword + semantic)
- [ ] Build evaluation framework

## 🙏 Acknowledgments

- OpenAI for embeddings and GPT models
- ChromaDB for vector storage
- Streamlit for UI framework

---

**Built with ❤️ in 3 days**
