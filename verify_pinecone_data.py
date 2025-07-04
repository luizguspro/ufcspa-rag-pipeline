import logging
import pprint
import pinecone
import config

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter(indent=4)

def verify_pinecone_data():
    """
    Conecta ao Pinecone, exibe as estatísticas do índice e realiza uma
    busca genérica para verificar se o índice contém dados.
    """
    logger.info("--- INICIANDO FERRAMENTA DE VERIFICAÇÃO ROBUSTA ---")
    
    if not all([config.PINECONE_API_KEY, config.PINECONE_ENVIRONMENT, config.PINECONE_INDEX_NAME]):
        logger.critical("ERRO: Verifique as chaves de API do Pinecone no seu arquivo .env!")
        return

    try:
        # 1. Conecta ao Pinecone
        pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Verifica se o índice existe antes de tentar conectar
        if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"O índice '{config.PINECONE_INDEX_NAME}' não foi encontrado na sua conta Pinecone.")
            logger.error("Verifique se o nome no seu arquivo .env está correto e se o script de ingestão já foi executado.")
            return
            
        index = pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(f"Conectado com sucesso ao índice '{config.PINECONE_INDEX_NAME}'.")

        # 2. Mostra as Estatísticas do Índice (A Pista Mais Importante)
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        print("\n" + "="*50)
        print("Estatísticas do Índice no Pinecone:")
        print(f"  - Total de Vetores no Índice: {vector_count}")
        print("="*50 + "\n")

        # 3. Realiza uma busca genérica para ver se retorna QUALQUER COISA
        if vector_count > 0:
            logger.info("Realizando uma busca genérica para confirmar que o índice está populado...")
            
            # Cria um vetor de busca "aleatório" do tamanho correto
            # para ver quais são os vizinhos mais próximos.
            dimension = stats.get('dimension', config.EMBEDDING_DIMENSION)
            dummy_vector = [0.0] * dimension
            
            query_response = index.query(
                vector=dummy_vector,
                top_k=3,  # Pede para trazer os 3 vetores mais próximos
                include_metadata=True
            )
            
            if query_response['matches']:
                logger.info("✅ SUCESSO! A busca genérica encontrou dados. O índice está funcionando.")
                print("\n--- DETALHES DOS VETORES ENCONTRADOS ---\n")
                for match in query_response['matches']:
                    print(f"ID do Vetor: {match['id']}")
                    print(f"Score de Similaridade: {match['score']:.4f}")
                    print("Metadados:")
                    pp.pprint(match['metadata'])
                    print("-" * 50 + "\n")
            else:
                logger.error("❌ FALHA! O índice reporta ter vetores, mas a busca não retornou nada.")
                logger.error("Isso pode indicar um problema de indexação no Pinecone. Tente novamente em alguns minutos.")

        else:
            logger.warning("O índice está vazio. Nenhum dado para visualizar.")
            logger.info("Execute o script 'ingest_enriched_to_pinecone.py' para popular o índice.")

    except Exception as e:
        logger.critical(f"Ocorreu um erro ao se comunicar com o Pinecone: {e}", exc_info=True)


if __name__ == "__main__":
    verify_pinecone_data()