import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from src.config import (
    LLM_MODEL_NAME, 
    EMBEDDING_MODEL_NAME, 
    CONTEXT_WINDOW, 
    MAX_NEW_TOKENS, 
    TEMPERATURE
)

class ModelLoader:
    def __init__(self):
        self.device_map = "auto"
        self.system_prompt = """You are a helpful AI assistant for RAG."""
        self.query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    def load_llm(self):
        """Initializes and returns the HuggingFaceLLM."""
        print(f"Loading LLM: {LLM_MODEL_NAME}...")
        llm = HuggingFaceLLM(
            context_window=CONTEXT_WINDOW,
            max_new_tokens=MAX_NEW_TOKENS,
            system_prompt=self.system_prompt,
            generate_kwargs={"temperature": TEMPERATURE, "do_sample": False},
            query_wrapper_prompt=self.query_wrapper_prompt,
            tokenizer_name=LLM_MODEL_NAME,
            model_name=LLM_MODEL_NAME,
            device_map=self.device_map,
            tokenizer_kwargs={"max_length": CONTEXT_WINDOW},
        )
        return llm

    def load_embedding_model(self):
        """Initializes and returns the Embedding Model."""
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        return embed_model
