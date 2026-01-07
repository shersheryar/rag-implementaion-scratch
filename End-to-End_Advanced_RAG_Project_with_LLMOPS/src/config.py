import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Model Configuration
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# RAG Parameters
CHUNK_SIZE = 1024
CONTEXT_WINDOW = 4096
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
