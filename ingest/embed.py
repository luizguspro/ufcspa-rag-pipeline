"""
Script para gerar embeddings usando OpenAI e armazenar no Pinecone.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import sys
import os

# CORREÇÃO: Importar requests ao invés de usar o cliente OpenAI
import requests
from pinecone import Pinecone
from tqdm import tqdm

# Adiciona o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PineconeEmbeddingPipeline:
    """Pipeline para gerar embeddings com OpenAI e armazenar no Pinecone."""
    
    def __init__(self, batch_size: int = 100):
        """Inicializa o pipeline com clientes OpenAI e Pinecone."""
        self.batch_size = batch_size
        self.openai_api_key = config.OPENAI_API_KEY
        
        # URL da API OpenAI
        self.openai_url = "https://api.openai.com/v1/embeddings"
        
        # Headers para requests
        self.headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info("Cliente OpenAI configurado via requests")
        
        # Inicializa Pinecone
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(f"Conectado ao índice Pinecone: {config.PINECONE_INDEX_NAME}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings usando a API OpenAI via requests."""
        try:
            data = {
                "input": texts,
                "model": "text-embedding-3-small"
            }
            
            response = requests.post(
                self.openai_url,
                headers=self.headers,
                json=data
            )
            
            if response.status_code != 200:
                logger.error(f"Erro na API OpenAI: {response.status_code}")
                logger.error(f"Resposta: {response.text}")
                return []
            
            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]
            return embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {str(e)}")
            return []
    
    def process_chunks(self, input_file: str = None) -> int:
        """Processa chunks de texto e armazena no Pinecone."""
        # Define caminho padrão
        base_dir = Path(__file__).parent.parent
        if input_file is None:
            input_file = str(base_dir / "data" / "processed" / "chunks.json")
        
        # Carrega chunks
        chunks = self._load_chunks(input_file)
        if not chunks:
            logger.error("Nenhum chunk encontrado para processar")
            return 0
        
        logger.info(f"Processando {len(chunks)} chunks")
        
        # Processa em batches
        total_processed = 0
        
        # Barra de progresso para o total
        with tqdm(total=len(chunks), desc="Total processado") as pbar_total:
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                success = self._process_batch(batch, start_idx=i)
                total_processed += success
                pbar_total.update(len(batch))
            
        logger.info(f"Total de chunks processados: {total_processed}")
        return total_processed
    
    def _load_chunks(self, input_file: str) -> List[Dict]:
        """Carrega chunks do arquivo JSON."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.error(f"Arquivo não encontrado: {input_path}")
            return []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Carregados {len(chunks)} chunks de {input_path}")
            return chunks
        except Exception as e:
            logger.error(f"Erro ao carregar chunks: {str(e)}")
            return []
    
    def _process_batch(self, batch: List[Dict], start_idx: int) -> int:
        """Processa um batch de chunks."""
        try:
            # Prepara textos e IDs
            texts = [chunk['text'] for chunk in batch]
            ids = [f"chunk_{start_idx + i}" for i in range(len(batch))]
            
            # Gera embeddings com OpenAI
            logger.info(f"Gerando embeddings para {len(texts)} textos...")
            
            # Processa em sub-batches menores se necessário
            all_embeddings = []
            for j in range(0, len(texts), 20):
                sub_batch = texts[j:j + 20]
                embeddings = self._get_embeddings(sub_batch)
                if not embeddings:
                    logger.error(f"Falha ao gerar embeddings para sub-batch {j}")
                    return 0
                all_embeddings.extend(embeddings)
            
            # Prepara dados para Pinecone
            vectors = []
            for i, embedding in enumerate(all_embeddings):
                vectors.append({
                    'id': ids[i],
                    'values': embedding,
                    'metadata': {
                        'text': batch[i]['text'][:1000],  # Limita texto para metadados
                        'source_file': batch[i].get('source_file', 'unknown'),
                        'chunk_id': batch[i].get('chunk_id', i),
                        'char_count': batch[i].get('char_count', len(batch[i]['text']))
                    }
                })
            
            # Faz upsert no Pinecone
            logger.info(f"Inserindo {len(vectors)} vetores no Pinecone...")
            self.index.upsert(vectors=vectors)
            
            return len(vectors)
            
        except Exception as e:
            logger.error(f"Erro ao processar batch: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gera embeddings com OpenAI e armazena no Pinecone"
    )
    parser.add_argument(
        '--input',
        help='Arquivo JSON com chunks de texto'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Tamanho do batch para processamento'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Executa teste com apenas 10 chunks'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PIPELINE DE EMBEDDINGS - PINECONE + OPENAI")
    print("=" * 60)
    
    try:
        # Modo teste
        if args.test:
            print("🧪 MODO TESTE - Processando apenas 10 chunks")
            # Cria arquivo de teste
            base_dir = Path(__file__).parent.parent
            chunks_file = base_dir / "data" / "processed" / "chunks.json"
            test_file = base_dir / "data" / "processed" / "chunks_test.json"
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks[:10], f, ensure_ascii=False, indent=2)
            
            args.input = str(test_file)
        
        # Executa pipeline
        pipeline = PineconeEmbeddingPipeline(batch_size=args.batch_size)
        total = pipeline.process_chunks(input_file=args.input)
        
        if total > 0:
            print(f"\n✅ Sucesso! {total} chunks indexados no Pinecone")
            print(f"📊 Acesse https://app.pinecone.io/ para ver os vetores")
        else:
            print("\n❌ Nenhum chunk foi processado")
            
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()