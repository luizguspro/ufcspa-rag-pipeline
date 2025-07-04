import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm

import config

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnrichedPineconeIngestion:
    """Pipeline para processar textos, enriquecê-los com IA e ingeri-los no Pinecone."""
    
    def __init__(self):
        logger.info("Inicializando pipeline de ingestão enriquecida para OpenAI...")
        
        # Clientes de API (usaremos o OpenAI para embeddings e metadados)
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        logger.info(f"Modelo de embeddings a ser usado: {config.EMBEDDING_MODEL}")
        logger.info(f"Dimensão esperada: {config.EMBEDDING_DIMENSION}")
        
        # Verificação e Criação do Índice
        if config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            logger.warning(f"Índice '{config.PINECONE_INDEX_NAME}' não encontrado. Criando um novo...")
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=int(config.EMBEDDING_DIMENSION),
                metric='cosine',
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            logger.info(f"Índice '{config.PINECONE_INDEX_NAME}' criado com sucesso.")
        
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(f"Conectado ao índice Pinecone: {config.PINECONE_INDEX_NAME}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(config.CHUNK_SIZE),
            chunk_overlap=int(config.CHUNK_OVERLAP)
        )

    def clean_text(self, text: str) -> str:
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def generate_metadata(self, chunk: str) -> Dict:
        prompt = f"""Com base no seguinte texto, gere uma pergunta provável que ele responderia, uma resposta concisa, e uma lista de até 3 tags de palavras-chave. Retorne o resultado estritamente no seguinte formato JSON: {{"question": "...", "answer": "...", "tags": ["...", "..."]}}.
Texto: \"\"\"{chunk[:1500]}\"\"\""""
        try:
            response = self.openai_client.chat.completions.create(
                model=config.GENERATIVE_MODEL_FOR_METADATA,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Não foi possível gerar metadados. Erro: {e}")
            return {}
            
    def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Gera embeddings para uma lista de textos usando o modelo da OpenAI especificado no config."""
        try:
            response = self.openai_client.embeddings.create(
                input=chunks,
                model=config.EMBEDDING_MODEL # Agora usa o modelo do config (ex: text-embedding-3-small)
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings com OpenAI: {e}")
            return []

    def process_file_and_get_vectors(self, filepath: Path) -> List[Dict]:
        logger.info(f"Processando arquivo: {filepath.name}")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        chunks = self.text_splitter.split_text(self.clean_text(text))
        valid_chunks = [chunk for chunk in chunks if len(chunk) > 50]
        if not valid_chunks: return []

        # Gera embeddings usando a função correta da OpenAI
        embeddings = self.get_embeddings(valid_chunks)
        if not embeddings or len(embeddings) != len(valid_chunks):
            logger.error(f"Falha ao gerar embeddings para o arquivo {filepath.name}. Pulando.")
            return []
        
        vectors_to_upload = []
        for i, (chunk, embedding) in enumerate(tqdm(zip(valid_chunks, embeddings), total=len(valid_chunks), desc=f"Enriquecendo {filepath.name}")):
            enriched_meta = self.generate_metadata(chunk)
            vectors_to_upload.append({
                'id': f"{filepath.stem}_chunk_{i}",
                'values': embedding,
                'metadata': {
                    'text': chunk, 'source': filepath.name,
                    'question': enriched_meta.get('question', ''),
                    'answer': enriched_meta.get('answer', ''),
                    'tags': enriched_meta.get('tags', [])
                }
            })
        return vectors_to_upload

    def upload_to_pinecone(self, vectors: List[Dict]):
        if not vectors: return
        for i in tqdm(range(0, len(vectors), config.UPLOAD_BATCH_SIZE), desc="Enviando para Pinecone"):
            batch = vectors[i:i + config.UPLOAD_BATCH_SIZE]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Erro no upload do batch: {e}")

    def run(self, file_path: str = None):
        if file_path:
            target_files = [Path(file_path)]
        else:
            data_dir = Path(config.DATA_DIRECTORY)
            target_files = list(data_dir.glob("*.txt"))
        
        if not target_files:
            logger.warning("Nenhum arquivo para processar.")
            return
        
        for filepath in target_files:
            vectors = self.process_file_and_get_vectors(filepath)
            self.upload_to_pinecone(vectors)
        logger.info("--- Processamento de todos os arquivos concluído! ---")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de ingestão enriquecida para Pinecone")
    parser.add_argument('--file', type=str, help='Caminho opcional para um único arquivo a ser processado.')
    args = parser.parse_args()

    # ... (validação de chaves e try/except principal continuam os mesmos) ...
    if not all([config.OPENAI_API_KEY, config.PINECONE_API_KEY, config.PINECONE_ENVIRONMENT]):
        logger.critical("ERRO: Verifique as chaves de API no seu arquivo .env!")
        return
    
    try:
        pipeline = EnrichedPineconeIngestion()
        pipeline.run(file_path=args.file)
    except Exception as e:
        logger.critical(f"Erro fatal no pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()