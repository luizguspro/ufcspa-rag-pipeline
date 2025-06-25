"""
Exemplo de arquivo de configuração.
Renomeie para config.py e preencha com suas chaves.
"""

import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente (opcional)
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-key-here")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-key")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ufcspa-docs")
PINECONE_HOST = os.getenv("PINECONE_HOST", "https://your-index.pinecone.io")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))