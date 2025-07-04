# delete_index.py
import pinecone
import config  # Importa suas configurações
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def delete_pinecone_index():
    """Conecta ao Pinecone e deleta o índice especificado no config."""
    try:
        logger.info("Iniciando conexão com o Pinecone para deletar índice...")
        
        pc = pinecone.Pinecone(
            api_key=config.PINECONE_API_KEY
        )
        
        index_name = config.PINECONE_INDEX_NAME

        if index_name in pc.list_indexes().names():
            logger.warning(f"Índice '{index_name}' encontrado. Deletando agora...")
            
            pc.delete_index(index_name)
            
            logger.info(f"✅ O índice '{index_name}' foi deletado com sucesso.")
        else:
            logger.info(f"✅ O índice '{index_name}' não foi encontrado. Nenhuma ação foi necessária.")

    except Exception as e:
        logger.error(f"❌ Ocorreu um erro ao tentar deletar o índice: {e}")

if __name__ == "__main__":
    confirm = input(f"Você tem CERTEZA que deseja deletar o índice '{config.PINECONE_INDEX_NAME}'? Esta ação é irreversível. (s/n): ")
    if confirm.lower() == 's':
        delete_pinecone_index()
    else:
        print("Operação cancelada.")