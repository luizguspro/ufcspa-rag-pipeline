import os
import pinecone
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (se você estiver usando um arquivo .env)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "ufcspa-index" # << CONFIRME SE ESTE É O NOME DO SEU ÍNDICE

# CUIDADO: O SCRIPT A SEGUIR DELETA DADOS PERMANENTEMENTE

print("Iniciando conexão com o Pinecone...")
try:
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )

    if INDEX_NAME in pinecone.list_indexes():
        print(f"Encontrado o índice '{INDEX_NAME}'. Preparando para deletar...")
        
        # Ação de deletar o índice
        pinecone.delete_index(INDEX_NAME)
        
        print(f"✅ O índice '{INDEX_NAME}' foi deletado com sucesso.")
    else:
        print(f"⚠️ O índice '{INDEX_NAME}' não foi encontrado. Nenhuma ação foi tomada.")

except Exception as e:
    print(f"❌ Ocorreu um erro: {e}")