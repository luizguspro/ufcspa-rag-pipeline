"""
Script de consulta com integração Llama para RAG completo.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Adiciona diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importa o handler do Llama
try:
    from query.llm_handler import get_llama_handler
    LLAMA_AVAILABLE = True
except:
    LLAMA_AVAILABLE = False

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class QueryEngineWithLlama:
    """Motor de consulta RAG com Llama integrado."""
    
    def __init__(self, index_dir: str = "faiss_index", k: int = 5):
        """Inicializa o motor de consulta.
        
        Args:
            index_dir: Diretório contendo o índice FAISS
            k: Número de chunks a recuperar (top-k)
        """
        self.index_dir = Path(index_dir)
        self.k = k
        self.llama_handler = None
        
        logger.info("Inicializando QueryEngine com Llama...")
        
        # Tenta carregar o índice FAISS
        self.has_faiss = self._try_load_faiss()
        
        # Carrega chunks
        self._load_chunks()
        
        # Inicializa Llama
        if LLAMA_AVAILABLE:
            logger.info("Inicializando Llama...")
            self.llama_handler = get_llama_handler()
            if self.llama_handler.available:
                logger.info("✅ Llama carregado com sucesso!")
            else:
                logger.warning("⚠️ Llama não disponível - usando modo fallback")
        else:
            logger.warning("⚠️ Módulo llm_handler não encontrado")
        
        logger.info(f"QueryEngine pronto! ({self.n_chunks} chunks disponíveis)")
    
    def _try_load_faiss(self):
        """Tenta carregar o índice FAISS."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            # Carrega informações do índice
            info_file = self.index_dir / "index_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                self.model_name = info['model_name']
                self.embedding_dim = info['embedding_dim']
                
                # Carrega modelo
                self.model = SentenceTransformer(self.model_name)
                
                # Carrega índice
                index_file = self.index_dir / "ufcspa.index"
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
                    logger.info(f"✅ Índice FAISS carregado: {self.index.ntotal} vetores")
                    return True
            
            logger.warning("Índice FAISS não encontrado - usando busca por palavras-chave")
            return False
            
        except ImportError:
            logger.warning("FAISS/Sentence-transformers não instalado - usando busca simples")
            return False
        except Exception as e:
            logger.error(f"Erro ao carregar FAISS: {e}")
            return False
    
    def _load_chunks(self):
        """Carrega os chunks de texto."""
        chunks_file = self.index_dir / "chunks.json"
        
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        else:
            # Fallback: cria chunks de exemplo
            logger.warning("Arquivo chunks.json não encontrado - usando dados de exemplo")
            self.chunks = [
                {
                    "chunk_id": 1,
                    "source_file": "estatuto_ufcspa.txt",
                    "text": "A UFCSPA tem como princípios fundamentais: excelência acadêmica em ensino, pesquisa e extensão; formação humanística e ética dos profissionais da saúde; compromisso social com a saúde pública; gestão democrática e participativa; autonomia universitária."
                },
                {
                    "chunk_id": 2,
                    "source_file": "normas_extensao.txt",
                    "text": "As atividades de extensão universitária são processos interdisciplinares que promovem a interação transformadora entre a Universidade e a sociedade. Incluem programas, projetos, cursos, eventos e prestação de serviços."
                },
                {
                    "chunk_id": 3,
                    "source_file": "regimento_interno.txt",
                    "text": "O Conselho Universitário (CONSUN) é o órgão máximo deliberativo da UFCSPA. A Reitoria é o órgão executivo central, sendo o Reitor a autoridade máxima executiva da universidade."
                }
            ]
        
        self.n_chunks = len(self.chunks)
    
    def query(self, question: str, k: Optional[int] = None) -> Dict:
        """Executa uma consulta RAG completa com Llama.
        
        Args:
            question: Pergunta do usuário
            k: Número de chunks a recuperar
            
        Returns:
            Dict contendo os resultados e resposta gerada
        """
        k = k or self.k
        logger.info(f"Processando consulta: '{question}' (k={k})")
        
        # Busca chunks relevantes
        if self.has_faiss:
            results = self._search_with_faiss(question, k)
        else:
            results = self._search_with_keywords(question, k)
        
        # Constrói contexto
        context = self._build_context(results)
        
        # Gera resposta com Llama
        if self.llama_handler and self.llama_handler.available:
            logger.info("Gerando resposta com Llama...")
            llm_response = self.llama_handler.generate_rag_response(context, question)
        else:
            llm_response = self._generate_fallback_response(context, question)
        
        return {
            'question': question,
            'results': results,
            'context': context,
            'response': llm_response,
            'k': k,
            'llm_used': bool(self.llama_handler and self.llama_handler.available)
        }
    
    def _search_with_faiss(self, question: str, k: int) -> List[Dict]:
        """Busca usando FAISS."""
        # Gera embedding da pergunta
        query_embedding = self.model.encode([question], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Busca no índice
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < self.n_chunks:
                chunk = self.chunks[idx]
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'source_file': chunk['source_file'],
                    'text': chunk['text'],
                    'score': float(score),
                    'method': 'faiss'
                })
        
        return results
    
    def _search_with_keywords(self, question: str, k: int) -> List[Dict]:
        """Busca por palavras-chave (fallback)."""
        import re
        
        # Extrai palavras-chave
        words = re.findall(r'\w+', question.lower())
        stopwords = {'o', 'a', 'de', 'do', 'da', 'que', 'e', 'para', 'com'}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Calcula scores
        results = []
        for chunk in self.chunks:
            text_lower = chunk['text'].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            
            if score > 0:
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'source_file': chunk['source_file'],
                    'text': chunk['text'],
                    'score': float(score),
                    'method': 'keywords'
                })
        
        # Ordena e retorna top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def _build_context(self, results: List[Dict]) -> str:
        """Constrói contexto para o LLM."""
        context_parts = []
        
        for i, result in enumerate(results):
            source = result['source_file']
            text = result['text']
            score = result['score']
            
            context_parts.append(f"[Documento {i+1}: {source} | Relevância: {score:.2f}]")
            context_parts.append(text)
            context_parts.append("")
        
        return '\n'.join(context_parts)
    
    def _generate_fallback_response(self, context: str, question: str) -> str:
        """Resposta quando Llama não está disponível."""
        return f"""[Resposta sem LLM - Contexto recuperado]

Com base nos documentos encontrados sobre '{question}':

{context[:500]}...

💡 Para respostas completas com IA:
1. Instale: pip install llama-cpp-python
2. Baixe um modelo Llama 2
3. Coloque na pasta 'models/'

O sistema continuará funcionando, mostrando o contexto relevante."""


def format_results_with_llama(query_result: Dict, verbose: bool = False):
    """Formata e imprime os resultados com resposta do Llama."""
    print("\n" + "=" * 80)
    print("🤖 SISTEMA RAG COM LLAMA - UFCSPA")
    print("=" * 80)
    
    print(f"\n📝 PERGUNTA: {query_result['question']}")
    
    print(f"\n🔍 BUSCA VETORIAL:")
    print(f"   - Chunks recuperados: {len(query_result['results'])}")
    print(f"   - Método: {query_result['results'][0]['method'] if query_result['results'] else 'N/A'}")
    
    if verbose and query_result['results']:
        print("\n📄 CHUNKS ENCONTRADOS:")
        for i, result in enumerate(query_result['results'], 1):
            print(f"\n   [{i}] {result['source_file']} (Score: {result['score']:.3f})")
            print(f"       {result['text'][:150]}...")
    
    print("\n" + "-" * 80)
    print("💬 RESPOSTA GERADA:")
    print("-" * 80)
    
    if query_result['llm_used']:
        print("✅ Usando Llama para gerar resposta:\n")
    else:
        print("⚠️ LLM não disponível - mostrando contexto:\n")
    
    print(query_result['response'])
    print("=" * 80)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Sistema RAG com Llama para normas UFCSPA"
    )
    
    parser.add_argument(
        'question',
        type=str,
        help='Pergunta sobre as normas da UFCSPA'
    )
    
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=5,
        help='Número de chunks a recuperar'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostra detalhes dos chunks'
    )
    
    parser.add_argument(
        '--download-model',
        choices=['tinyllama', 'llama2', 'mistral'],
        help='Baixa um modelo antes de executar'
    )
    
    args = parser.parse_args()
    
    # Baixa modelo se solicitado
    if args.download_model:
        from llm_handler import get_llama_handler
        handler = get_llama_handler()
        handler.download_model(args.download_model)
        return
    
    try:
        # Inicializa e executa consulta
        engine = QueryEngineWithLlama()
        result = engine.query(args.question, k=args.top_k)
        
        # Mostra resultados
        format_results_with_llama(result, verbose=args.verbose)
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()