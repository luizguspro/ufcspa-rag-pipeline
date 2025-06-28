import os
from dotenv import load_dotenv

load_dotenv()

# Chaves de API e Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ufcspa-index")

# Modelos de IA
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATIVE_MODEL_FOR_METADATA = "gpt-4o-mini" # Rápido e econômico

# Configurações do Pipeline
DATA_DIRECTORY = "data/processed/"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
UPLOAD_BATCH_SIZE = 100