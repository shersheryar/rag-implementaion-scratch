from llama_index.core import SimpleDirectoryReader
from src.config import DATA_DIR

class DataIngestion:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def load_data(self):
        """Loads data from the configured data directory."""
        print(f"Loading data from {self.data_dir}...")
        reader = SimpleDirectoryReader(input_dir=str(self.data_dir))
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents.")
        return documents
