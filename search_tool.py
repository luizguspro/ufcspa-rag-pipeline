import requests
from pinecone import Pinecone
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorSearchTool:
    def __init__(self):
        try:
            # Configuração para usar requests com OpenAI
            self.openai_api_key = config.OPENAI_API_KEY
            self.headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Nova sintaxe do Pinecone
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.pinecone_index = self.pc.Index(config.PINECONE_INDEX_NAME)
            logging.info("VectorSearchTool inicializada com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao inicializar a VectorSearchTool: {e}")
            raise

    def _get_embedding(self, text: str) -> list[float]:
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=self.headers,
                json={
                    "input": text,
                    "model": config.EMBEDDING_MODEL
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            else:
                logging.error(f"Erro na API OpenAI: {response.status_code} - {response.text}")
                raise Exception("Erro ao gerar embedding")
                
        except Exception as e:
            logging.error(f"Erro ao gerar embedding: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> list[str]:
        logging.info(f"Recebida nova busca: '{query}'")
        try:
            query_vector = self._get_embedding(query)
            results = self.pinecone_index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            context_list = [match['metadata']['text'] for match in results['matches'] if 'metadata' in match and 'text' in match['metadata']]
            logging.info(f"Encontrados {len(context_list)} resultados relevantes.")
            return context_list
        except Exception as e:
            logging.error(f"Erro durante a busca: {e}")
            return []

search_tool_instance = VectorSearchTool()

def search_vectorstore(query: str) -> list[str]:
    return search_tool_instance.search(query)

if __name__ == '__main__':
    print("Executando teste da ferramenta de busca...")
    pergunta_exemplo = "Quais são as normas para atividades de extensão universitária?"
    
    try:
        contextos = search_vectorstore(pergunta_exemplo)
        
        print(f"\n--- Contextos Encontrados para a pergunta: '{pergunta_exemplo}' ---\n")
        if not contextos:
            print("Nenhum contexto foi encontrado. Verifique se o índice Pinecone está populado e se as chaves de API estão corretas.")
        else:
            for i, contexto in enumerate(contextos, 1):
                print(f"Resultado {i}:\n{contexto}\n---")
    except Exception as e:
        print(f"\n❌ Erro ao executar busca: {e}")
        print("\n💡 Verifique:")
        print("1. Se o arquivo .env está configurado corretamente")
        print("2. Se o índice Pinecone existe e tem o nome correto")
        print("3. Se as chaves de API estão válidas")