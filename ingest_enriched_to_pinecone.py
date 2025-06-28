import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
MODEL_DIMENSION = 384

class EnrichedPineconeIngestion:
    """Pipeline para processar textos, enriquecê-los com IA e ingeri-los no Pinecone."""
    
    def __init__(self):
        logger.info("Inicializando pipeline de ingestão enriquecida...")
        
        # Clientes de API
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Modelo de Embeddings
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu')
        logger.info(f"Modelo de embeddings carregado: {config.EMBEDDING_MODEL}")
        
        # --- CORREÇÃO: VERIFICAÇÃO E CRIAÇÃO DO ÍNDICE ---
        if config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            logger.warning(f"Índice '{config.PINECONE_INDEX_NAME}' não encontrado. Criando um novo...")
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=MODEL_DIMENSION,
                metric='cosine',
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            logger.info(f"Índice '{config.PINECONE_INDEX_NAME}' criado com sucesso.")
        else:
            logger.info(f"Índice '{config.PINECONE_INDEX_NAME}' já existe.")
        
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(f"Conectado ao índice Pinecone: {config.PINECONE_INDEX_NAME}")
        # --- FIM DA CORREÇÃO ---
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    def clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto, incluindo a remoção de quebras de linha."""
        text = text.replace('\n', ' ')  # Troca quebras de linha por espaço
        text = re.sub(r'\s+', ' ', text)  # Normaliza múltiplos espaços
        return text.strip()

    def generate_metadata(self, chunk: str) -> Dict:
        """Gera metadados (pergunta, resposta, tags) para um chunk usando OpenAI."""
        prompt = f"""Com base no seguinte texto, gere uma pergunta provável que ele responderia, uma resposta concisa, e uma lista de até 3 tags de palavras-chave. Retorne o resultado estritamente no seguinte formato JSON: {{"question": "...", "answer": "...", "tags": ["...", "..."]}}.
Texto: \"\"\"{chunk[:1500]}\"\"\""""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=config.GENERATIVE_MODEL_FOR_METADATA,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Não foi possível gerar metadados para o chunk. Erro: {e}")
            return {}

    def process_file_and_get_vectors(self, filepath: Path) -> List[Dict]:
        """Processa um único arquivo, gerando chunks, metadados e embeddings."""
        logger.info(f"Processando arquivo: {filepath.name}")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        chunks = self.text_splitter.split_text(cleaned_text)
        valid_chunks = [chunk for chunk in chunks if len(chunk) > 50]
        
        if not valid_chunks:
            logger.warning(f"Nenhum chunk válido encontrado em {filepath.name}")
            return []

        logger.info(f"Gerando embeddings para {len(valid_chunks)} chunks...")
        embeddings = self.embedding_model.encode(valid_chunks, show_progress_bar=True).tolist()
        
        vectors_to_upload = []
        logger.info(f"Gerando metadados enriquecidos para {len(valid_chunks)} chunks...")
        for i, (chunk, embedding) in enumerate(tqdm(zip(valid_chunks, embeddings), total=len(valid_chunks), desc="Enriquecendo Chunks")):
            enriched_meta = self.generate_metadata(chunk)
            
            vector_id = f"{filepath.stem}_chunk_{i}"
            
            final_metadata = {
                'text': chunk,
                'source': filepath.name,
                'question': enriched_meta.get('question', ''),
                'answer': enriched_meta.get('answer', ''),
                'tags': enriched_meta.get('tags', [])
            }
            
            vectors_to_upload.append({
                'id': vector_id,
                'values': embedding,
                'metadata': final_metadata
            })
        return vectors_to_upload

    def upload_to_pinecone(self, vectors: List[Dict]):
        """Faz upload dos vetores em lotes para o Pinecone."""
        if not vectors:
            return
        for i in tqdm(range(0, len(vectors), config.UPLOAD_BATCH_SIZE), desc="Enviando para Pinecone"):
            batch = vectors[i:i + config.UPLOAD_BATCH_SIZE]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Erro no upload do batch: {e}")

    def run(self, file_path: str = None):
        """Executa o pipeline de ingestão."""
        if file_path:
            target_files = [Path(file_path)]
            if not target_files[0].exists():
                logger.error(f"Arquivo de entrada não encontrado: {file_path}")
                return
        else:
            data_dir = Path(config.DATA_DIRECTORY)
            target_files = list(data_dir.glob("*.txt"))
            logger.info(f"Encontrados {len(target_files)} arquivos .txt em '{data_dir}'.")
        
        if not target_files:
            logger.warning("Nenhum arquivo para processar.")
            return
        
        logger.info(f"Iniciando processamento de {len(target_files)} arquivo(s)...")
        for filepath in target_files:
            vectors = self.process_file_and_get_vectors(filepath)
            self.upload_to_pinecone(vectors)
        logger.info("--- Processamento de todos os arquivos concluído! ---")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de ingestão enriquecida para Pinecone")
    parser.add_argument('--file', type=str, help='Caminho opcional para um único arquivo a ser processado.')
    args = parser.parse_args()

    if not all([config.OPENAI_API_KEY, config.PINECONE_API_KEY, config.PINECONE_ENVIRONMENT]):
        logger.critical("ERRO: Verifique se todas as chaves de API (OPENAI, PINECONE) estão no seu arquivo .env!")
        return
    
    try:
        pipeline = EnrichedPineconeIngestion()
        pipeline.run(file_path=args.file)
    except Exception as e:
        logger.critical(f"Erro fatal no pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()