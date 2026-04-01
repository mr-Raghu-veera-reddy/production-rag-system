"""
Streamlit UI for RAG System
Simple chat interface for asking questions
"""

import streamlit as st
from src.rag_system import RAGSystem
from src.ingestion_pipeline import IngestionPipeline
import os

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
                # Save uploaded files
                for file in uploaded_files:
                    save_path = os.path.join("./data", file.name)
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # Run ingestion
                pipeline = IngestionPipeline()
                pipeline.ingest_directory("./data", clear_existing=False)
                
                # Reinitialize RAG system
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
    
    # Statistics
    st.header("📊 Statistics")
    
    if st.button("📈 Show Stats"):
     st.session_state.rag_system.show_stats()
    
    st.metric("Total Queries", len(st.session_state.chat_history))
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.total_cost = 0.0
        st.rerun()

# Main chat interface
st.title("💬 Ask Questions About Your Documents")
st.markdown("*Powered by GPT-3.5 and semantic search*")

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
if question := st.chat_input("Ask a question about your documents..."):
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
        with st.spinner("Thinking..."):
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
    
    # Add assistant message to history
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
    
    # Update total cost
    st.session_state.total_cost += result['cost']

# Footer
st.markdown("---")
st.markdown(
    "*Built with ❤️ using Streamlit, OpenAI, and ChromaDB*",
    unsafe_allow_html=True
)