"""
Script para gerar embeddings e criar índice FAISS para busca vetorial.

Este módulo lê os chunks de texto processados, gera embeddings usando
sentence-transformers e constrói um índice FAISS para busca eficiente.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Adiciona o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Gerador de embeddings e índice FAISS para busca vetorial.
    
    Esta classe gerencia a criação de embeddings de texto usando
    sentence-transformers e a construção de um índice FAISS para
    busca vetorial eficiente.
    
    Attributes:
        model_name: Nome do modelo sentence-transformers a usar
        model: Instância do modelo SentenceTransformer
        embedding_dim: Dimensão dos embeddings gerados
        batch_size: Tamanho do batch para geração de embeddings
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32
    ):
        """Inicializa o gerador de embeddings.
        
        Args:
            model_name: Nome do modelo sentence-transformers
            batch_size: Tamanho do batch para processamento
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        logger.info(f"Inicializando EmbeddingGenerator")
        logger.info(f"Modelo: {model_name}")
        
        # Tenta importar as bibliotecas necessárias
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.faiss = faiss
        except ImportError as e:
            logger.error(f"Erro ao importar bibliotecas necessárias: {e}")
            logger.error("Instale com: pip install sentence-transformers faiss-cpu")
            raise
        
        # Carrega o modelo
        logger.info("Carregando modelo de embeddings...")
        self.model = SentenceTransformer(model_name)
        
        # Obtém a dimensão dos embeddings
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Dimensão dos embeddings: {self.embedding_dim}")
        
        # Define para usar CPU (garante compatibilidade)
        self.model.to('cpu')
        logger.info("Modelo carregado e configurado para CPU")
    
    def process_chunks(
        self,
        input_file: str = None,
        index_dir: str = None
    ) -> Tuple[int, int]:
        """Processa chunks de texto e cria índice FAISS.
        
        Args:
            input_file: Caminho para o arquivo chunks.json
            index_dir: Diretório para salvar o índice FAISS
            
        Returns:
            Tupla (número de chunks processados, dimensão dos embeddings)
        """
        # Define caminhos padrão baseados no diretório do script
        base_dir = Path(__file__).parent.parent
        if input_file is None:
            input_file = str(base_dir / "data" / "processed" / "chunks.json")
        if index_dir is None:
            index_dir = str(base_dir / "faiss_index")
        
        # Carrega os chunks
        chunks = self._load_chunks(input_file)
        if not chunks:
            logger.error("Nenhum chunk encontrado para processar")
            return 0, 0
        
        logger.info(f"Carregados {len(chunks)} chunks para processar")
        
        # Extrai textos dos chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Gera embeddings
        logger.info("Gerando embeddings...")
        embeddings = self._generate_embeddings(texts)
        
        # Cria e treina índice FAISS
        logger.info("Criando índice FAISS...")
        index = self._create_faiss_index(embeddings)
        
        # Salva artefatos
        logger.info("Salvando artefatos...")
        self._save_artifacts(index, chunks, index_dir)
        
        logger.info(f"Processamento concluído com sucesso!")
        return len(chunks), self.embedding_dim
    
    def _load_chunks(self, input_file: str) -> List[Dict]:
        """Carrega chunks do arquivo JSON.
        
        Args:
            input_file: Caminho para o arquivo chunks.json
            
        Returns:
            Lista de chunks com metadados
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.error(f"Arquivo não encontrado: {input_path}")
            return []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logger.info(f"Carregados {len(chunks)} chunks de {input_path}")
            
            # Valida estrutura dos chunks
            required_fields = ['text', 'source_file', 'chunk_id']
            for i, chunk in enumerate(chunks):
                for field in required_fields:
                    if field not in chunk:
                        logger.warning(f"Chunk {i} sem campo '{field}'")
                        chunk[field] = "" if field == 'text' else f"unknown_{i}"
            
            # Remove chunks vazios
            chunks = [c for c in chunks if c.get('text', '').strip()]
            logger.info(f"{len(chunks)} chunks válidos após validação")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erro ao carregar chunks: {str(e)}")
            return []
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de textos para gerar embeddings
            
        Returns:
            Array numpy com embeddings (shape: [n_texts, embedding_dim])
        """
        embeddings = []
        
        # Processa em batches para eficiência
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(texts), desc="Gerando embeddings") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Gera embeddings para o batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size
                )
                
                embeddings.append(batch_embeddings)
                pbar.update(len(batch_texts))
        
        # Concatena todos os embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Normaliza embeddings para Inner Product funcionar como similaridade cosseno
        # IP(normalized_a, normalized_b) = cos_sim(a, b)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        all_embeddings = all_embeddings / norms
        
        logger.info(f"Gerados embeddings com shape: {all_embeddings.shape}")
        return all_embeddings
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Cria e popula um índice FAISS.
        
        Args:
            embeddings: Array de embeddings (shape: [n_texts, embedding_dim])
            
        Returns:
            Índice FAISS treinado
        """
        n_vectors, dim = embeddings.shape
        
        # Cria índice usando Inner Product (para similaridade cosseno com vetores normalizados)
        index = self.faiss.IndexFlatIP(dim)
        
        logger.info(f"Criado índice FAISS IndexFlatIP com dimensão {dim}")
        
        # Adiciona vetores ao índice
        logger.info(f"Adicionando {n_vectors} vetores ao índice...")
        index.add(embeddings.astype('float32'))
        
        # Verifica se o índice foi populado corretamente
        assert index.ntotal == n_vectors, f"Erro: índice tem {index.ntotal} vetores, esperado {n_vectors}"
        
        logger.info(f"Índice FAISS criado com {index.ntotal} vetores")
        
        return index
    
    def _save_artifacts(self, index, chunks: List[Dict], index_dir: str):
        """Salva o índice FAISS e metadados dos chunks.
        
        Args:
            index: Índice FAISS treinado
            chunks: Lista de chunks com metadados
            index_dir: Diretório para salvar os artefatos
        """
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Salva o índice FAISS
        index_file = index_path / "ufcspa.index"
        self.faiss.write_index(index, str(index_file))
        logger.info(f"Índice FAISS salvo em: {index_file}")
        
        # Salva os chunks (necessário para recuperar textos após busca)
        chunks_file = index_path / "chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadados dos chunks salvos em: {chunks_file}")
        
        # Salva informações sobre o índice
        index_info = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "n_vectors": index.ntotal,
            "index_type": "IndexFlatIP",
            "normalized": True,
            "chunks_file": "chunks.json"
        }
        
        info_file = index_path / "index_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(index_info, f, indent=2)
        logger.info(f"Informações do índice salvas em: {info_file}")


def validate_environment():
    """Valida se o ambiente está configurado corretamente."""
    try:
        # Testa importações
        import faiss
        import sentence_transformers
        logger.info("✓ Todas as dependências estão instaladas")
        
        # Verifica se o arquivo de chunks existe
        base_dir = Path(__file__).parent.parent
        chunks_path = base_dir / "data" / "processed" / "chunks.json"
        if not chunks_path.exists():
            logger.warning(f"⚠ Arquivo {chunks_path} não encontrado")
            logger.warning("  Execute primeiro os scripts de conversão e chunking")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Dependência não encontrada: {e}")
        logger.error("  Execute: pip install sentence-transformers faiss-cpu")
        return False


def main():
    """Função principal para executar o gerador de embeddings."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gera embeddings e cria índice FAISS para busca vetorial"
    )
    parser.add_argument(
        '--input',
        help='Arquivo JSON com chunks de texto'
    )
    parser.add_argument(
        '--output-dir',
        help='Diretório para salvar o índice FAISS'
    )
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Modelo sentence-transformers a usar'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch para geração de embeddings'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GERADOR DE EMBEDDINGS - UFCSPA")
    print("=" * 60)
    
    # Valida ambiente
    if not validate_environment():
        return
    
    # Executa geração de embeddings
    try:
        generator = EmbeddingGenerator(
            model_name=args.model,
            batch_size=args.batch_size
        )
        
        n_chunks, dim = generator.process_chunks(
            input_file=args.input,
            index_dir=args.output_dir
        )
        
        if n_chunks > 0:
            print("\n" + "=" * 60)
            print("RESUMO")
            print("=" * 60)
            print(f"✓ Chunks processados: {n_chunks}")
            print(f"✓ Dimensão dos embeddings: {dim}")
            print(f"✓ Índice salvo com sucesso!")
            print("\nPróximo passo: Execute o script de consulta (query)")
        else:
            print("\n✗ Nenhum chunk foi processado")
            
    except ImportError:
        print("\n❌ Erro: Bibliotecas necessárias não instaladas")
        print("Execute: pip install sentence-transformers faiss-cpu")
    except Exception as e:
        print(f"\n❌ Erro: {e}")


if __name__ == "__main__":
    main()