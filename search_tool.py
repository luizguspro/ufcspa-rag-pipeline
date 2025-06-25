"""
Ferramenta de busca vetorial para documentos da UFCSPA.

Esta ferramenta é projetada para ser usada por agentes CrewAI,
permitindo busca semântica em documentos normativos da universidade.
"""

import os
import logging
from typing import List, Dict, Optional, Union
from functools import lru_cache
import time

import requests
from pinecone import Pinecone
import config

# Configuração de logging mais profissional
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Evita múltiplas inicializações em logs
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class VectorSearchTool:
    """
    Ferramenta de busca vetorial otimizada para uso com CrewAI.
    
    Esta classe encapsula toda a lógica de busca, permitindo
    configuração flexível e reuso eficiente de conexões.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
        min_score: float = 0.0
    ):
        """
        Inicializa a ferramenta de busca.
        
        Args:
            openai_api_key: Chave API OpenAI (usa config.py se não fornecida)
            pinecone_api_key: Chave API Pinecone (usa config.py se não fornecida)
            pinecone_index_name: Nome do índice Pinecone (usa config.py se não fornecida)
            embedding_model: Modelo OpenAI para embeddings
            top_k: Número de resultados a retornar
            min_score: Score mínimo para considerar um resultado relevante
        """
        # Configurações com fallback para config.py ou variáveis de ambiente
        self.openai_api_key = openai_api_key or getattr(config, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = pinecone_api_key or getattr(config, 'PINECONE_API_KEY', None) or os.getenv('PINECONE_API_KEY')
        self.pinecone_index_name = pinecone_index_name or getattr(config, 'PINECONE_INDEX_NAME', None) or os.getenv('PINECONE_INDEX_NAME')
        
        if not all([self.openai_api_key, self.pinecone_api_key, self.pinecone_index_name]):
            raise ValueError("API keys e index name são obrigatórios. Configure via parâmetros, config.py ou variáveis de ambiente.")
        
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.min_score = min_score
        
        # Headers para OpenAI
        self.headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Inicializa Pinecone
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.pinecone_index_name)
            logger.info(f"Conectado ao índice Pinecone: {self.pinecone_index_name}")
        except Exception as e:
            logger.error(f"Erro ao conectar ao Pinecone: {e}")
            raise
    
    @lru_cache(maxsize=128)
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Gera embedding para um texto usando cache LRU.
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Lista de floats representando o embedding ou None se erro
        """
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=self.headers,
                json={
                    "input": text,
                    "model": self.embedding_model
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            else:
                logger.error(f"Erro OpenAI API: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Timeout ao gerar embedding")
            return None
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            return None
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
        include_scores: bool = False
    ) -> Union[List[str], List[Dict[str, Union[str, float]]]]:
        """
        Busca documentos relevantes para uma query.
        
        Args:
            query: Pergunta ou texto de busca
            top_k: Número de resultados (sobrescreve o padrão)
            filter_dict: Filtros para aplicar na busca Pinecone
            include_scores: Se True, retorna dicts com texto e score
            
        Returns:
            Lista de textos relevantes ou lista de dicts com texto e score
        """
        logger.info(f"Nova busca: '{query[:100]}...'")
        
        # Gera embedding
        query_vector = self._get_embedding(query)
        if not query_vector:
            logger.warning("Falha ao gerar embedding, retornando lista vazia")
            return []
        
        # Define top_k
        k = top_k or self.top_k
        
        # Busca no Pinecone
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Processa resultados
            processed_results = []
            
            for match in results.get('matches', []):
                score = match.get('score', 0)
                
                # Aplica filtro de score mínimo
                if score < self.min_score:
                    continue
                
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')
                
                if text:
                    if include_scores:
                        processed_results.append({
                            'text': text,
                            'score': score,
                            'source': metadata.get('source_file', 'unknown'),
                            'chunk_id': metadata.get('chunk_id', 'unknown')
                        })
                    else:
                        processed_results.append(text)
            
            logger.info(f"Retornando {len(processed_results)} resultados relevantes")
            return processed_results
            
        except Exception as e:
            logger.error(f"Erro na busca Pinecone: {e}")
            return []
    
    def health_check(self) -> Dict[str, bool]:
        """
        Verifica a saúde das conexões.
        
        Returns:
            Dict indicando status de cada serviço
        """
        health = {
            'openai': False,
            'pinecone': False
        }
        
        # Testa OpenAI
        try:
            test_embedding = self._get_embedding("test")
            health['openai'] = test_embedding is not None
        except:
            pass
        
        # Testa Pinecone
        try:
            self.index.describe_index_stats()
            health['pinecone'] = True
        except:
            pass
        
        return health


# Instância global para uso direto (opcional)
_tool_instance = None

def get_tool_instance(**kwargs) -> VectorSearchTool:
    """
    Retorna uma instância singleton da ferramenta.
    
    Args:
        **kwargs: Argumentos para VectorSearchTool (usados apenas na primeira chamada)
        
    Returns:
        Instância de VectorSearchTool
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = VectorSearchTool(**kwargs)
    return _tool_instance


def search_vectorstore(
    query: str,
    top_k: int = 5,
    include_scores: bool = False,
    **kwargs
) -> Union[List[str], List[Dict[str, Union[str, float]]]]:
    """
    Função de busca vetorial compatível com CrewAI.
    
    Esta é a interface principal para agentes. Mantém compatibilidade
    com a assinatura original mas oferece funcionalidades extras.
    
    Args:
        query: Pergunta ou texto de busca
        top_k: Número de resultados a retornar
        include_scores: Se True, retorna dicts com texto e score
        **kwargs: Argumentos adicionais passados para VectorSearchTool
        
    Returns:
        Lista de textos relevantes ou lista de dicts com texto e score
    
    Examples:
        >>> # Uso básico
        >>> contexts = search_vectorstore("Quais são as normas de extensão?")
        >>> 
        >>> # Com scores
        >>> results = search_vectorstore("regimento interno", include_scores=True)
        >>> for r in results:
        ...     print(f"Score: {r['score']:.3f} - {r['text'][:50]}...")
    """
    try:
        tool = get_tool_instance(**kwargs)
        return tool.search(query, top_k=top_k, include_scores=include_scores)
    except Exception as e:
        logger.error(f"Erro crítico na busca: {e}")
        return []


# Alias para compatibilidade
search = search_vectorstore


# --- Bloco de teste e validação ---
if __name__ == '__main__':
    print("🔍 UFCSPA Vector Search Tool - Teste\n")
    
    # Verifica configuração
    try:
        tool = get_tool_instance()
        print("✅ Ferramenta inicializada com sucesso")
        
        # Health check
        health = tool.health_check()
        print(f"\n📊 Status dos serviços:")
        print(f"   OpenAI: {'✅' if health['openai'] else '❌'}")
        print(f"   Pinecone: {'✅' if health['pinecone'] else '❌'}")
        
        if all(health.values()):
            # Teste de busca
            print("\n🧪 Executando busca de teste...")
            
            queries = [
                "Quais são as normas para atividades de extensão universitária?",
                "Como funciona o conselho universitário CONSUN?",
                "Regimento interno da UFCSPA"
            ]
            
            for query in queries:
                print(f"\n❓ Query: {query}")
                
                # Busca com scores
                start_time = time.time()
                results = search_vectorstore(query, top_k=3, include_scores=True)
                elapsed = time.time() - start_time
                
                if results:
                    print(f"⏱️  Tempo: {elapsed:.2f}s")
                    for i, result in enumerate(results, 1):
                        print(f"\n📄 Resultado {i} (Score: {result['score']:.3f}):")
                        print(f"   Fonte: {result['source']}")
                        print(f"   Texto: {result['text'][:200]}...")
                else:
                    print("   ❌ Nenhum resultado encontrado")
                
                print("-" * 80)
        else:
            print("\n❌ Alguns serviços estão offline. Verifique as configurações.")
            
    except Exception as e:
        print(f"\n❌ Erro ao inicializar ferramenta: {e}")
        print("\n💡 Dicas:")
        print("1. Verifique se config.py existe com as chaves corretas")
        print("2. Ou defina as variáveis de ambiente:")
        print("   - OPENAI_API_KEY")
        print("   - PINECONE_API_KEY")
        print("   - PINECONE_INDEX_NAME")