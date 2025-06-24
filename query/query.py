"""
Script de consulta implementando RAG (Retrieval-Augmented Generation).

Este módulo permite fazer perguntas sobre as normas da UFCSPA usando
busca vetorial para recuperar contexto relevante dos documentos.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class QueryEngine:
    """Motor de consulta RAG para buscar informações nas normas da UFCSPA.
    
    Esta classe implementa o pipeline completo de RAG (Retrieval-Augmented Generation),
    desde a busca vetorial até a construção do contexto para geração de resposta.
    
    Attributes:
        index_dir: Diretório contendo o índice FAISS e metadados
        model: Modelo sentence-transformers para gerar embeddings
        index: Índice FAISS carregado
        chunks: Lista de chunks de texto com metadados
        k: Número de chunks a recuperar para cada consulta
    """
    
    def __init__(
        self,
        index_dir: str = "faiss_index",
        k: int = 5
    ):
        """Inicializa o motor de consulta.
        
        Args:
            index_dir: Diretório contendo o índice FAISS
            k: Número de chunks a recuperar (top-k)
        """
        self.index_dir = Path(index_dir)
        self.k = k
        
        logger.info("Inicializando QueryEngine...")
        
        # Carrega informações do índice
        self._load_index_info()
        
        # Carrega o modelo de embeddings
        logger.info(f"Carregando modelo: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.model.to('cpu')
        
        # Carrega o índice FAISS
        self._load_faiss_index()
        
        # Carrega os chunks
        self._load_chunks()
        
        logger.info(f"QueryEngine pronto! ({self.n_chunks} chunks disponíveis)")
    
    def _load_index_info(self):
        """Carrega as informações do índice."""
        info_file = self.index_dir / "index_info.json"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Arquivo de informações não encontrado: {info_file}")
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        self.model_name = info['model_name']
        self.embedding_dim = info['embedding_dim']
        self.n_vectors = info['n_vectors']
        
        logger.info(f"Informações do índice carregadas: {self.n_vectors} vetores, dimensão {self.embedding_dim}")
    
    def _load_faiss_index(self):
        """Carrega o índice FAISS."""
        index_file = self.index_dir / "ufcspa.index"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Índice FAISS não encontrado: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        
        if self.index.ntotal != self.n_vectors:
            logger.warning(f"Número de vetores no índice ({self.index.ntotal}) difere do esperado ({self.n_vectors})")
        
        logger.info(f"Índice FAISS carregado: {self.index.ntotal} vetores")
    
    def _load_chunks(self):
        """Carrega os chunks de texto."""
        chunks_file = self.index_dir / "chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Arquivo de chunks não encontrado: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        self.n_chunks = len(self.chunks)
        logger.info(f"Carregados {self.n_chunks} chunks")
    
    def query(self, question: str, k: Optional[int] = None, use_llm: bool = True) -> Dict:
        """Executa uma consulta RAG completa.
        
        Args:
            question: Pergunta do usuário
            k: Número de chunks a recuperar (usa self.k se não especificado)
            use_llm: Se True, usa Llama para gerar resposta
            
        Returns:
            Dict contendo os resultados da busca e a resposta gerada
        """
        if not question or not question.strip():
            raise ValueError("Pergunta não pode estar vazia")
        
        k = k or self.k
        logger.info(f"Processando consulta: '{question}' (k={k})")
        
        # Gera embedding da pergunta
        query_embedding = self._generate_query_embedding(question)
        
        # Busca chunks relevantes
        results = self._search_similar_chunks(query_embedding, k)
        
        # Constrói o contexto
        context = self._build_context(results)
        
        # Gera resposta com Llama se solicitado
        if use_llm:
            try:
                from .llama_model import get_llama_model
                llama = get_llama_model()
                
                # Verifica se o modelo está disponível
                model_info = llama.get_model_info()
                if model_info['status'] == 'ready':
                    prompt = llama.format_prompt(context, question)
                    answer = llama.generate(prompt)
                else:
                    answer = self._build_prompt(question, context, results)
                    answer += f"\n\n[Status do Llama: {model_info['status']} - {model_info.get('message', '')}]"
            except ImportError:
                answer = self._build_prompt(question, context, results)
                answer += "\n\n[Llama não disponível - Instale llama-cpp-python]"
        else:
            answer = self._build_prompt(question, context, results)
        
        return {
            'question': question,
            'results': results,
            'context': context,
            'answer': answer,
            'k': k
        }
    
    def _generate_query_embedding(self, question: str) -> np.ndarray:
        """Gera embedding para a pergunta do usuário.
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Array numpy com o embedding normalizado
        """
        # Gera embedding
        embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normaliza para usar com IndexFlatIP
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype('float32')
    
    def _search_similar_chunks(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Busca os k chunks mais similares à consulta.
        
        Args:
            query_embedding: Embedding da consulta
            k: Número de resultados a retornar
            
        Returns:
            Lista de dicionários com os chunks e seus scores
        """
        # Garante que k não exceda o número de chunks
        k = min(k, self.n_chunks)
        
        # Busca no índice FAISS
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepara os resultados
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= self.n_chunks:
                logger.warning(f"Índice inválido retornado: {idx}")
                continue
            
            chunk = self.chunks[idx]
            result = {
                'chunk_id': chunk['chunk_id'],
                'source_file': chunk['source_file'],
                'text': chunk['text'],
                'score': float(score),
                'index': int(idx)
            }
            results.append(result)
        
        logger.info(f"Encontrados {len(results)} chunks relevantes")
        return results
    
    def _build_context(self, results: List[Dict]) -> str:
        """Constrói o contexto concatenando os textos dos chunks.
        
        Args:
            results: Lista de resultados da busca
            
        Returns:
            String com o contexto concatenado
        """
        context_parts = []
        
        for i, result in enumerate(results):
            source = result['source_file']
            chunk_id = result['chunk_id']
            text = result['text']
            score = result['score']
            
            # Adiciona cabeçalho com informações do chunk
            header = f"[Documento: {source} | Chunk: {chunk_id} | Relevância: {score:.3f}]"
            context_parts.append(header)
            context_parts.append(text)
            
            # Adiciona separador entre chunks (exceto no último)
            if i < len(results) - 1:
                context_parts.append("---")
        
        return '\n'.join(context_parts)
    
    def _build_prompt(self, question: str, context: str, results: List[Dict]) -> str:
        """Constrói o prompt final para o LLM.
        
        Args:
            question: Pergunta original do usuário
            context: Contexto construído dos chunks
            results: Lista de resultados para estatísticas
            
        Returns:
            String com o prompt formatado
        """
        # Template do prompt
        prompt = f"""### SISTEMA DE CONSULTA - NORMAS UFCSPA ###

Você é um assistente especializado em normas e regulamentos da UFCSPA (Universidade Federal de Ciências da Saúde de Porto Alegre). Use o contexto dos documentos abaixo para responder à pergunta do usuário de forma precisa e completa.

### CONTEXTO DOS DOCUMENTOS ###
{context}

### PERGUNTA DO USUÁRIO ###
{question}

### INSTRUÇÕES ###
1. Baseie sua resposta APENAS nas informações presentes no contexto acima
2. Se a informação não estiver no contexto, informe claramente
3. Cite os documentos relevantes quando possível
4. Seja objetivo e direto na resposta

### Resposta Gerada por IA (Placeholder) ###
[NOTA: Em um sistema completo, aqui seria gerada uma resposta usando um LLM como GPT-4, Llama, ou similar]

Com base no contexto acima dos {len(results)} chunks mais relevantes encontrados, a resposta seria gerada aqui.

### METADADOS DA BUSCA ###
- Pergunta: "{question}"
- Chunks recuperados: {len(results)}
- Documentos únicos: {len(set(r['source_file'] for r in results))}
- Score médio de relevância: {np.mean([r['score'] for r in results]):.3f}
"""
        
        return prompt


def format_results(query_result: Dict, verbose: bool = False):
    """Formata e imprime os resultados da consulta.
    
    Args:
        query_result: Dicionário com os resultados da consulta
        verbose: Se True, mostra informações detalhadas
    """
    print("\n" + "=" * 80)
    print("RESULTADOS DA CONSULTA RAG - UFCSPA")
    print("=" * 80)
    
    print(f"\n📝 PERGUNTA: {query_result['question']}")
    
    print(f"\n🔍 BUSCA VETORIAL:")
    print(f"   - Chunks recuperados: {len(query_result['results'])}")
    
    if verbose:
        print("\n📄 CHUNKS ENCONTRADOS:")
        for i, result in enumerate(query_result['results'], 1):
            print(f"\n   [{i}] Documento: {result['source_file']}")
            print(f"       Chunk ID: {result['chunk_id']}")
            print(f"       Relevância: {result['score']:.3f}")
            print(f"       Preview: {result['text'][:150]}...")
    
    print("\n" + "-" * 80)
    print("💬 RESPOSTA GERADA:")
    print("-" * 80)
    
    # Mostra a resposta gerada ou o prompt
    if 'answer' in query_result:
        print(query_result['answer'])
    else:
        print(query_result.get('prompt', 'Nenhuma resposta gerada'))
    
    print("=" * 80)


def main():
    """Função principal do script de consulta."""
    parser = argparse.ArgumentParser(
        description="Sistema de consulta RAG para normas da UFCSPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python query.py "Qual o regimento para atividades de extensão?"
  python query.py "Como funciona o processo de matrícula?" -k 3
  python query.py "Normas para afastamento docente" -v
        """
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
        help='Número de chunks a recuperar (padrão: 5)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostra informações detalhadas sobre os chunks encontrados'
    )
    
    parser.add_argument(
        '--index-dir',
        type=str,
        default='faiss_index',
        help='Diretório contendo o índice FAISS (padrão: faiss_index)'
    )
    
    args = parser.parse_args()
    
    try:
        # Inicializa o motor de consulta
        engine = QueryEngine(
            index_dir=args.index_dir,
            k=args.top_k
        )
        
        # Executa a consulta
        result = engine.query(args.question, k=args.top_k)
        
        # Formata e exibe os resultados
        format_results(result, verbose=args.verbose)
        
    except FileNotFoundError as e:
        logger.error(f"Erro: {e}")
        logger.error("Execute primeiro o pipeline de ingestão para criar o índice FAISS")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro ao processar consulta: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()