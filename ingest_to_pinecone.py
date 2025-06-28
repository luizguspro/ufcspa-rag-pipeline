import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

# --- Configuração do Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constantes ---
MODEL_DIMENSION = 384


class PineconeIngestionPipeline:
    """
    Pipeline completa para processar documentos de texto e fazer a ingestão
    dos seus vetores e metadados no Pinecone.
    """

    def __init__(self):
        """Inicializa o pipeline, carregando o modelo e conectando ao Pinecone."""
        logger.info("Inicializando pipeline de ingestão...")

        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device='cpu'
        )
        logger.info(f"Modelo de embeddings carregado: {config.EMBEDDING_MODEL}")

        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # --- BLOCO DE CÓDIGO CORRIGIDO (VERSÃO FINAL) ---
        if config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            logger.warning(f"Índice '{config.PINECONE_INDEX_NAME}' não encontrado. Criando um novo...")
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=MODEL_DIMENSION,
                metric='cosine',
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            logger.info(f"Índice '{config.PINECONE_INDEX_NAME}' criado com sucesso com dimensão {MODEL_DIMENSION}.")
        else:
            logger.info(f"Índice '{config.PINECONE_INDEX_NAME}' já existe.")

        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(f"Conectado ao índice Pinecone: {config.PINECONE_INDEX_NAME}")
        # --- FIM DO BLOCO CORRIGIDO ---

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

        self.processed_hashes = set()

    def clean_text(self, text: str) -> str:
        """Aplica uma série de regras para limpar e normalizar o texto bruto."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(filter(None, lines)).strip()

    def process_and_embed_file(self, filepath: Path) -> List[Dict]:
        """Lê, limpa, divide em chunks, e gera embeddings para um único arquivo."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Erro ao ler arquivo {filepath}: {e}")
            return []

        cleaned_text = self.clean_text(text)
        
        text_hash = hashlib.md5(cleaned_text.encode()).hexdigest()
        if not cleaned_text or text_hash in self.processed_hashes:
            logger.warning(f"Arquivo vazio ou duplicado (conteúdo já processado): {filepath.name}")
            return []
        self.processed_hashes.add(text_hash)

        chunks = self.text_splitter.split_text(cleaned_text)
        logger.info(f"Arquivo '{filepath.name}' dividido em {len(chunks)} chunks.")
        
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]

        if not valid_chunks:
            return []

        embeddings = self.embedding_model.encode(valid_chunks, show_progress_bar=False).tolist()

        vectors_to_upload = []
        for i, (chunk, embedding) in enumerate(zip(valid_chunks, embeddings)):
            vector_id = f"{filepath.stem}_chunk_{i}"
            vectors_to_upload.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'text': chunk,
                    'source': filepath.name,
                }
            })
        
        return vectors_to_upload

    def upload_to_pinecone(self, vectors: List[Dict]):
        """Faz o upload dos vetores para o Pinecone em lotes (batches)."""
        if not vectors:
            return

        logger.info(f"Iniciando upload de {len(vectors)} vetores...")
        for i in tqdm(range(0, len(vectors), config.UPLOAD_BATCH_SIZE), desc="Enviando para Pinecone"):
            batch = vectors[i:i + config.UPLOAD_BATCH_SIZE]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Erro no upload do batch: {e}")
        logger.info("Upload em lotes concluído!")

    def run(self, file_path: str = None):
        """
        Executa o pipeline completo. Processa um único arquivo se `file_path`
        for fornecido, ou todos os arquivos .txt no diretório de dados padrão.
        """
        if file_path:
            target_files = [Path(file_path)]
            if not target_files[0].exists():
                logger.error(f"Arquivo de entrada não encontrado: {file_path}")
                return
        else:
            data_dir = Path(config.DATA_DIRECTORY)
            if not data_dir.exists():
                logger.error(f"Diretório de dados não encontrado: {data_dir}")
                return
            target_files = list(data_dir.glob("*.txt"))
            logger.info(f"Encontrados {len(target_files)} arquivos .txt em '{data_dir}'.")

        if not target_files:
            logger.warning("Nenhum arquivo para processar.")
            return

        total_vectors_created = 0
        for filepath in tqdm(target_files, desc="Processando arquivos", unit="file"):
            vectors = self.process_and_embed_file(filepath)
            if vectors:
                self.upload_to_pinecone(vectors)
                total_vectors_created += len(vectors)
        
        logger.info("--- Processamento Concluído ---")
        logger.info(f"Total de arquivos processados: {len(self.processed_hashes)}")
        logger.info(f"Total de vetores criados e enviados: {total_vectors_created}")


def main():
    """Função principal que lida com os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Pipeline de ingestão de dados para o Pinecone.")
    parser.add_argument(
        '--file',
        type=str,
        help='Caminho opcional para um único arquivo a ser processado.'
    )
    args = parser.parse_args()

    if not all([config.PINECONE_API_KEY, config.PINECONE_ENVIRONMENT, config.PINECONE_INDEX_NAME]):
        logger.error("Uma ou mais configurações do Pinecone não foram encontradas no arquivo .env!")
        return
    
    try:
        pipeline = PineconeIngestionPipeline()
        pipeline.run(file_path=args.file)
    except Exception as e:
        logger.critical(f"Erro fatal no pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()