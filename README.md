# 🤖 Generative AI & RAG Projects

A comprehensive collection of **Retrieval-Augmented Generation (RAG)** implementations and **LangChain** projects demonstrating modern AI application development patterns.

---

## 📁 Projects Overview

### 1. 🚀 End-to-End Advanced RAG Project with LLMOps
**Path:** `End-to-End_Advanced_RAG_Project_with_LLMOPS/`

A production-ready RAG application built with **Streamlit**, **LlamaIndex**, and **HuggingFace** models.

**Features:**
- Interactive chat interface with conversation history
- Uses SmolLM2-360M LLM with MPNet embeddings
- Document ingestion and vector indexing pipeline
- Modular architecture with separate components for data ingestion, model loading, and RAG pipeline

**Tech Stack:** `LlamaIndex` • `Streamlit` • `HuggingFace Transformers` • `Sentence Transformers`

---

### 2. 📚 RAG Application using LangChain, Mistral AI & Weaviate
**Path:** `RAG_Application_using_Langchain_Mistral_AI_and_Weviate_db/`

A RAG implementation leveraging **Mistral AI** for generation and **Weaviate** as a vector database.

**Features:**
- PDF document processing and chunking
- Vector storage with Weaviate DB
- Mistral AI integration for high-quality responses

**Tech Stack:** `LangChain` • `Mistral AI` • `Weaviate`

---

### 3. 🔍 RAG Application using LangChain, OpenAI & FAISS
**Path:** `RAG_Application_using_‪LangChain‬_OpenAI‬_and_FAISS/`

A classic RAG setup using **OpenAI** embeddings and **FAISS** for efficient similarity search.

**Features:**
- OpenAI embeddings for semantic understanding
- FAISS vector store for fast retrieval
- Document QA capabilities

**Tech Stack:** `LangChain` • `OpenAI` • `FAISS`

---

### 4. 🛠️ RAG from Scratch
**Path:** `Rag_from_scratch/`

Learn the fundamentals of RAG by building it from scratch without high-level abstractions.

**Features:**
- Understanding RAG internals
- Custom implementation of retrieval mechanisms
- Educational deep-dive into RAG architecture

**Resources:** Includes `RAG_From_Scratch.pdf` guide

---

### 5. 💬 Chatbot using LangChain with Memory
**Path:** `chatbot_using_langchain_with_memory/`

A conversational AI chatbot with **persistent memory** for context-aware responses.

**Features:**
- Conversation memory management
- Context-aware dialogue
- LangChain memory modules

**Tech Stack:** `LangChain` • `Memory Modules`

---

### 6. 📖 LangChain Essentials
**Path:** `langchain/`

A comprehensive guide to **LangChain** fundamentals including chains, prompts, and LangServe.

**Features:**
- LangChain core concepts
- Chain composition patterns
- LangServe deployment example

**Tech Stack:** `LangChain` • `LangServe`

---

### 7. 🦙 RAG Application with LlamaIndex & Mistral
**Path:** `rag_application_with_llamaindex_mistral/`

A RAG application built with **LlamaIndex** framework and **Mistral AI** models.

**Features:**
- LlamaIndex data connectors
- Mistral AI integration
- Advanced indexing strategies

**Tech Stack:** `LlamaIndex` • `Mistral AI`

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip or uv package manager

### General Setup

```bash
# Clone the repository
git clone https://github.com/shersheryar/rag-implementaion-scratch.git
cd rag-implementaion-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for a specific project
cd <project-directory>
pip install -r requirements.txt
```

### Environment Variables

Most projects require API keys. Create a `.env` file in the project directory:

```bash
# Example .env file
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key
HUGGINGFACE_TOKEN=your_hf_token
```

---

## 🚀 Quick Start

### Running the Advanced RAG Project

```bash
cd End-to-End_Advanced_RAG_Project_with_LLMOPS
pip install -r requirements.txt
streamlit run app.py
```

### Running Jupyter Notebooks

```bash
jupyter notebook
# Navigate to the desired .ipynb file
```

---

## 📚 Learning Path

| Order | Project | Focus Area |
|-------|---------|------------|
| 1 | LangChain Essentials | Fundamentals |
| 2 | Chatbot with Memory | Conversation Memory |
| 3 | RAG from Scratch | Core Concepts |
| 4 | RAG with FAISS | Vector Search |
| 5 | RAG with Weaviate | Vector Databases |
| 6 | RAG with LlamaIndex | Advanced Indexing |
| 7 | End-to-End RAG | Production Deployment |

---

## 🔧 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Frameworks** | LangChain, LlamaIndex |
| **LLMs** | OpenAI, Mistral AI, HuggingFace |
| **Vector Stores** | FAISS, Weaviate |
| **Embeddings** | OpenAI, Sentence Transformers, MPNet |
| **UI** | Streamlit |
| **Deployment** | LangServe |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Add new RAG implementations

---

## 👨‍💻 Author

**Shersheryar**

[![GitHub](https://img.shields.io/badge/GitHub-shersheryar-black?style=flat&logo=github)](https://github.com/shersheryar)

---

<p align="center">
  <b>⭐ Star this repository if you find it helpful!</b>
</p>
