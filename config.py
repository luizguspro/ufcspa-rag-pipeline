import os
from dotenv import load_dotenv

load_dotenv()

# Chaves de API e Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ufcspa-index") # Nome do índice que vamos criar

# --- MUDANÇA CRÍTICA: DEFINIÇÃO DO MODELO E DIMENSÃO ---
# Agora usamos o modelo da OpenAI para gerar os vetores
EMBEDDING_MODEL = "text-embedding-3-small"
# A dimensão correspondente a este modelo é 1536
EMBEDDING_DIMENSION = 1536

# Modelo da OpenAI para enriquecimento de metadados
GENERATIVE_MODEL_FOR_METADATA = "gpt-4o-mini"

# Configurações do Pipeline
DATA_DIRECTORY = "data/processed/"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
UPLOAD_BATCH_SIZE = 100