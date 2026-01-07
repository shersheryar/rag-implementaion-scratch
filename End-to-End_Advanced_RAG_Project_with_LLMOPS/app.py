import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to python path to ensure src modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data_ingestion import DataIngestion
from src.model_loader import ModelLoader
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")


def load_css():
    with open("src/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    st.markdown("---")
    
    # Status Indicator
    status_container = st.empty()
    status_container.status("System starting...", state="running")
    
    # Model Information
    st.markdown("### Model Info")
    st.info("Using **SmolLM2-360M** + **MPNet** embeddings.")
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("Built with **LlamaIndex** & **Streamlit**")

@st.cache_resource
def load_components():
    """
    Load data, models, and create index. Cached to prevent reloading on every run.
    """
    # 1. Load Data
    ingestion = DataIngestion()
    documents = ingestion.load_data()
    
    # 2. Load Models
    model_loader = ModelLoader()
    llm = model_loader.load_llm()
    embed_model = model_loader.load_embedding_model()
    
    # 3. Create RAG Pipeline
    pipeline = RAGPipeline(llm, embed_model)
    index = pipeline.create_index(documents)
    query_engine = pipeline.create_query_engine(index)
    
    return query_engine

# Main App Layout
st.title("Advanced RAG Assistant")
st.markdown("""
<div style="text-align: center; color: #aaa; margin-bottom: 10px; max-width: 700px; margin-left: auto; margin-right: auto;">
    <p style="font-size: 1.1em; margin-bottom: 15px;">
        <b>Explore the "Machine Translation with Attention" Research Paper</b>
    </p>
    <p style="font-size: 0.95em; line-height: 1.6;">
        This RAG-powered assistant is trained on the seminal paper introducing attention mechanisms 
        in neural machine translation. Ask questions about encoder-decoder architectures, 
        attention weights, BLEU scores, or any concept from the paper.
    </p>
    <p style="font-size: 0.85em; color: #666; margin-top: 15px;">
        <b>Try asking:</b> "How does attention improve translation?" or "What is the BLEU score?"
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the query engine (only once)
try:
    with st.spinner("Initializing system components..."):
        query_engine = load_components()
    status_container.success("System Ready!")
except Exception as e:
    status_container.error(f"Error: {e}")
    st.error(f"Error loading system: {e}")
    st.stop()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            response = query_engine.query(prompt)
            response_text = str(response)
        except Exception as e:
            response_text = f"An error occurred: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
