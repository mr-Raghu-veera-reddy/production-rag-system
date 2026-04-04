"""
Streamlit UI for RAG System
Simple chat interface for asking questions
"""

import streamlit as st
from src.rag_system import RAGSystem
from src.ingestion_pipeline import IngestionPipeline
import os


@st.cache_resource
def load_rag_system(model, top_k, temperature, use_advanced):
    """Cache RAG system initialization"""
    return RAGSystem(
        model=model,
        top_k=top_k,
        temperature=temperature,
        use_advanced_retrieval=use_advanced
    )



# Get API key from secrets or environment
try:
    # Try Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    # Fall back to .env file (for local development)
    elif os.getenv('OPENAI_API_KEY'):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    else:
        st.error("⚠️ OpenAI API key not found! Please configure it in Streamlit secrets.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading API key: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Production RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        st.session_state.rag_system = RAGSystem(
            model="gpt-3.5-turbo",
            top_k=5,
            temperature=0.3
        )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# Sidebar
with st.sidebar:
    st.title("🤖 RAG System")
    st.markdown("---")
    
    # Document upload section
    st.header("📁 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDFs to add to the knowledge base"
    )
    
    if uploaded_files:
        if st.button("📤 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                import tempfile
                import shutil
                
                # 1. Create a secure temporary directory
                temp_dir = tempfile.mkdtemp()
                
                # 2. Save uploaded files to the temp directory
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # 3. Run ingestion from the temp directory
                pipeline = IngestionPipeline()
                pipeline.ingest_directory(temp_dir, clear_existing=False)
                
                # 4. Clean up the temp folder so your cloud storage doesn't fill up!
                shutil.rmtree(temp_dir)
                
                # 5. Reinitialize RAG system
                st.session_state.rag_system = RAGSystem()
                
                st.success(f"✅ Processed {len(uploaded_files)} documents!")
                st.rerun()
    
    
    st.markdown("---")
    
    # Settings
    st.header("⚙️ Settings")
    
        # Advanced mode toggle
    use_advanced = st.checkbox(
        "🚀 Advanced Mode",
        value=False,
        help="Enable query rewriting and re-ranking (slower but better results)"
    )
    
    model_choice = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="gpt-4 is more accurate but 20x more expensive"
    )
    
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to use for context"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="0 = focused, 1 = creative"
    )
    
    if st.button("Apply Settings"):
        st.session_state.rag_system = RAGSystem(
            model=model_choice,
            top_k=top_k,
            temperature=temperature,
            use_advanced_retrieval=use_advanced
            
        )
        st.success("Settings applied!")
    
    st.markdown("---")
    
     # Show current mode
    if 'rag_system' in st.session_state:
        mode = "Advanced" if use_advanced else "Basic"
        st.info(f"**Current Mode:** {mode}")
    
    # System Metrics
    st.header("📊 System Metrics")
    if 'rag_system' in st.session_state:
        # Get stats from monitoring
        try:
            stats = st.session_state.rag_system.monitor.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", stats.get('total_queries', 0))
                st.metric("Success Rate", f"{stats.get('success_rate', 0):.1%}")
            with col2:
                st.metric("Avg Latency", f"{stats.get('avg_latency', 0):.2f}s")
                st.metric("Avg Cost", f"${stats.get('avg_cost_per_query', 0):.4f}")
        except:
            # Fallback just in case the monitor isn't tracking yet
            st.metric("Total Queries", len(st.session_state.chat_history))
            st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")

    # Keep the clear chat button, it's super useful!
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.total_cost = 0.0
        st.rerun()
    
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.info(f"""
    **Mode:** {'Cloud' if 'STREAMLIT_SHARING' in os.environ else 'Local'}  
    **Version:** 1.0  
    **Status:** 🟢 Online
    """)

# Main chat interface / Landing Page
st.title("🤖 Production RAG System")
st.markdown("""
**An intelligent document Q&A system powered by:**
- 📄 PDF document processing
- 🧠 OpenAI GPT-4 & embeddings
- 🔍 Semantic search with ChromaDB
- 🎯 Query rewriting & re-ranking
- ⚡ Real-time answer generation with citations

**Try it:** Ask a question about your documents!
""")
st.markdown("---")

# Example Questions (Only show if chat history is empty!)
if len(st.session_state.chat_history) == 0:
    st.subheader("💡 Example Questions")
    example_questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key findings?",
        "What are the conclusions?",
        "What is machine learning?"
    ]
    
    cols = st.columns(2)
    for i, ex_q in enumerate(example_questions):
        col_idx = i % 2
        if cols[col_idx].button(ex_q, key=f"example_{i}"):
            st.session_state.example_clicked = ex_q
            st.rerun()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")
            
            # Show metadata
            if "metadata" in message:
                with st.expander("📊 Metadata"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks", message["metadata"]["chunks"])
                    with col2:
                        st.metric("Tokens", message["metadata"]["tokens"])
                    with col3:
                        st.metric("Cost", f"${message['metadata']['cost']:.4f}")

# Chat input
question = st.chat_input("Ask a question about your documents...")

# Auto-fill and trigger if an example question was clicked
if 'example_clicked' in st.session_state:
    question = st.session_state.example_clicked
    del st.session_state.example_clicked

if question:
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get answer
    with st.chat_message("assistant"):
        try:
            # Added the upgraded spinner here
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.rag_system.query(question)
                
            # Display answer
            st.markdown(result['answer'])
            
            # Show sources
            with st.expander("📚 Sources"):
                for source in result['sources']:
                    st.write(f"• {source}")
            
            # Show metadata
            with st.expander("📊 Metadata"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chunks Used", result['chunks_used'])
                with col2:
                    st.metric("Tokens Used", result['tokens_used'])
                with col3:
                    st.metric("Cost", f"${result['cost']:.4f}")
                    
            # We moved the history and cost updates INSIDE the try block.
            # This way, if an error happens, it won't save a broken message to your chat history!
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources'],
                "metadata": {
                    "chunks": result['chunks_used'],
                    "tokens": result['tokens_used'],
                    "cost": result['cost']
                }
            })
            st.session_state.total_cost += result['cost']
            
        # Added the error handling here
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Try rephrasing your question or check your API key.")
            
    

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding-bottom: 20px;'>
    Built with ❤️ using Streamlit, OpenAI, and ChromaDB<br>
    <a href='https://github.com/mr-Raghu-veera-reddy/production-rag-system' target='_blank'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)