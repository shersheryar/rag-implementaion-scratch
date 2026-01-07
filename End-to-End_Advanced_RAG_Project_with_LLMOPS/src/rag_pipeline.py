from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from src.config import CHUNK_SIZE

# Custom prompt template for cleaner responses
QA_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant answering questions about a research paper.\n"
    "Use ONLY the relevant English content from the context below to answer the question.\n"
    "Do NOT include file paths, page numbers, or metadata in your response.\n"
    "Do NOT include examples in Spanish or other languages.\n"
    "Provide a clear, concise answer in English only.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer: "
)

class RAGPipeline:
    def __init__(self, llm, embed_model):
        self.llm = llm
        self.embed_model = embed_model
    
    def create_index(self, documents):
        """Creates a VectorStoreIndex from documents."""
        print("Creating Vector Store Index...")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = CHUNK_SIZE

        index = VectorStoreIndex.from_documents(
            documents,
        )
        print("Index created successfully.")
        return index

    def create_query_engine(self, index):
        """Creates a query engine from the index with custom prompt."""
        qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)
        return index.as_query_engine(text_qa_template=qa_prompt)
