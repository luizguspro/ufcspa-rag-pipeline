import requests
from pinecone import Pinecone
import config

# --- Inicialização dos Clientes ---
# Headers para OpenAI
headers = {
    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Inicializa Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)
pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)

# --- Função Ferramenta para o CrewAI ---
def search_vectorstore(query: str) -> list[str]:
    """
    Recebe uma pergunta, gera seu embedding com a OpenAI, busca os k textos
    mais relevantes no Pinecone e retorna uma lista com esses textos.
    """
    print(f"Recebida nova busca: '{query}'")

    # 1. Gerar o embedding da pergunta usando a API da OpenAI
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json={
                "input": query,
                "model": "text-embedding-3-small"
            }
        )
        
        if response.status_code != 200:
            print(f"Erro na API OpenAI: {response.status_code}")
            print(f"Resposta: {response.text}")
            return []
        
        result = response.json()
        query_vector = result['data'][0]['embedding']
        
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return []

    # 2. Fazer a busca de similaridade no Pinecone
    try:
        results = pinecone_index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        
        # 3. Extrair e retornar o conteúdo de texto dos metadados
        context_list = []
        for match in results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                context_list.append(match['metadata']['text'])
        
        print(f"Encontrados {len(context_list)} resultados relevantes.")
        return context_list
        
    except Exception as e:
        print(f"Erro na busca Pinecone: {e}")
        return []

# --- Bloco de teste para validar a função ---
if __name__ == '__main__':
    pergunta_exemplo = "Quais são as normas para atividades de extensão universitária?"
    contextos = search_vectorstore(pergunta_exemplo)
    
    print("\n--- Contextos Encontrados ---\n")
    if not contextos:
        print("Nenhum contexto foi encontrado.")
    else:
        for i, contexto in enumerate(contextos, 1):
            print(f"Resultado {i}:\n{contexto}\n")
            print("-" * 80)