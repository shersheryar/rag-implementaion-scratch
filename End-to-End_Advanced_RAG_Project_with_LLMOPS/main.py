import sys
from pathlib import Path

# Add project root to python path to ensure src modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data_ingestion import DataIngestion
from src.model_loader import ModelLoader
from src.rag_pipeline import RAGPipeline

def main():
    print("Starting End-to-End RAG Pipeline...")

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

    # 4. Execute Query
    query = "what is attention"
    print(f"\nQuerying: '{query}'")
    response = query_engine.query(query)
    
    print("\nResponse:")
    print("=" * 50)
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    main()
