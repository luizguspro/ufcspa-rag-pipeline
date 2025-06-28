"""
Script para criar todos os arquivos necessários
"""

import os

def create_file(filename, content):
    """Cria um arquivo com o conteúdo especificado."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Criado: {filename}")

# Conteúdo do ingest_final.py
ingest_final_content = '''import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import requests
from pinecone import Pinecone
from tqdm import tqdm
import config

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        """Inicializa o processador de documentos com clientes OpenAI e Pinecone."""
        try:
            self.openai_api_key = config.OPENAI_API_KEY
            self.headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Nova sintaxe do Pinecone
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.pinecone_index = self.pc.Index(config.PINECONE_INDEX_NAME)
            logger.info("Processador de documentos inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar processador: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Limpa o texto removendo artefatos indesejados."""
        # Remove caracteres de controle exceto newlines
        text = ''.join(char for char in text if char == '\\n' or ord(char) >= 32)
        
        # Normaliza quebras de linha múltiplas
        text = re.sub(r'\\n{3,}', '\\n\\n', text)
        
        # Remove espaços múltiplos
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove espaços no início e fim de linhas
        lines = text.split('\\n')
        lines = [line.strip() for line in lines]
        text = '\\n'.join(lines)
        
        # Remove linhas vazias consecutivas
        text = re.sub(r'\\n\\s*\\n', '\\n\\n', text)
        
        # Remove hífens de quebra de linha
        text = re.sub(r'(\\w+)-\\n(\\w+)', r'\\1\\2', text)
        
        # Remove espaços antes de pontuação
        text = re.sub(r'\\s+([.,;:!?])', r'\\1', text)
        
        # Garante espaço após pontuação
        text = re.sub(r'([.,;:!?])(?=[A-Za-zÀ-ÿ])', r'\\1 ', text)
        
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Divide o texto em chunks menores com sobreposição."""
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Tenta encontrar um ponto de quebra natural (fim de frase)
            if end < len(text):
                for delimiter in ['. ', '.\\n', '! ', '!\\n', '? ', '?\\n', '\\n\\n']:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter > start + chunk_size // 2:
                        end = last_delimiter + len(delimiter) - 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks

    def generate_metadata(self, chunk: str) -> Dict[str, any]:
        """Gera metadados enriquecidos para um chunk usando GPT."""
        prompt = f"""Analise o seguinte trecho de documento normativo e gere:
1. Uma pergunta que seria respondida por este trecho
2. Uma resposta resumida baseada no conteúdo
3. Até 3 tags descritivas relevantes

Trecho:
{chunk[:800]}

Responda em formato JSON:
{{"question": "...", "answer": "...", "tags": ["tag1", "tag2", "tag3"]}}"""

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": config.GENERATIVE_MODEL_FOR_METADATA,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Remove possíveis marcadores de código
                content = content.replace('```json', '').replace('```', '').strip()
                metadata = json.loads(content)
                
                # Validação básica
                metadata['question'] = metadata.get('question', 'Sem pergunta gerada')[:200]
                metadata['answer'] = metadata.get('answer', 'Sem resposta gerada')[:300]
                metadata['tags'] = metadata.get('tags', [])[:3]
                
                return metadata
            else:
                logger.warning(f"Erro na API: {response.status_code} - {response.text}")
                raise Exception("Erro na API")
                
        except Exception as e:
            logger.warning(f"Erro ao gerar metadados: {e}")
            return {
                'question': 'Sem pergunta gerada',
                'answer': 'Sem resposta gerada',
                'tags': []
            }

    def get_embedding(self, text: str) -> List[float]:
        """Gera embedding para um texto usando OpenAI."""
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
                logger.error(f"Erro na API: {response.status_code} - {response.text}")
                raise Exception("Erro ao gerar embedding")
                
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            raise

    def process_file(self, filepath: Path) -> int:
        """Processa um arquivo completo."""
        logger.info(f"Processando arquivo: {filepath}")
        
        try:
            # Lê o arquivo
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Limpa o texto
            text = self.clean_text(text)
            
            if not text.strip():
                logger.warning(f"Arquivo vazio após limpeza: {filepath}")
                return 0
            
            # Divide em chunks
            chunks = self.chunk_text(text)
            logger.info(f"Arquivo dividido em {len(chunks)} chunks")
            
            # Processa cada chunk
            vectors_to_upsert = []
            
            for i, chunk in enumerate(tqdm(chunks, desc="Processando chunks")):
                try:
                    # Gera metadados
                    metadata = self.generate_metadata(chunk)
                    
                    # Adiciona metadados fixos
                    metadata['text'] = chunk
                    metadata['source_file'] = filepath.name
                    metadata['chunk_id'] = i + 1
                    
                    # Gera embedding
                    embedding = self.get_embedding(chunk)
                    
                    # Prepara para upsert
                    vector_id = f"{filepath.stem}_chunk_{i+1}"
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': metadata
                    })
                    
                    # Faz upsert em lotes
                    if len(vectors_to_upsert) >= 10:
                        self.pinecone_index.upsert(vectors=vectors_to_upsert)
                        vectors_to_upsert = []
                        
                except Exception as e:
                    logger.error(f"Erro ao processar chunk {i+1}: {e}")
                    continue
            
            # Upsert dos vetores restantes
            if vectors_to_upsert:
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
            
            logger.info(f"Processamento concluído: {len(chunks)} chunks indexados")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Erro ao processar arquivo {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def process_directory(self, directory: Path) -> Dict[str, int]:
        """Processa todos os arquivos de texto em um diretório."""
        stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'failed_files': 0
        }
        
        # Busca arquivos de texto
        text_files = list(directory.glob('*.txt'))
        
        if not text_files:
            logger.warning(f"Nenhum arquivo .txt encontrado em {directory}")
            return stats
        
        logger.info(f"Encontrados {len(text_files)} arquivos para processar")
        
        for filepath in text_files:
            chunks = self.process_file(filepath)
            if chunks > 0:
                stats['files_processed'] += 1
                stats['total_chunks'] += chunks
            else:
                stats['failed_files'] += 1
        
        return stats


def main():
    """Função principal do script de ingestão."""
    parser = argparse.ArgumentParser(description="Pipeline de ingestão de documentos")
    parser.add_argument(
        '--file',
        type=str,
        help='Caminho para processar um único arquivo (opcional)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='data/processed',
        help='Diretório com arquivos para processar (padrão: data/processed)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PIPELINE DE INGESTÃO - VERSÃO CORRIGIDA")
    print("=" * 60)
    
    # Verifica configuração
    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY não configurada!")
        print("\\n❌ Configure a chave da OpenAI no arquivo .env")
        return
    
    if not config.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY não configurada!")
        print("\\n❌ Configure a chave do Pinecone no arquivo .env")
        return
    
    # Inicializa processador
    try:
        processor = DocumentProcessor()
    except Exception as e:
        logger.error(f"Erro ao inicializar: {e}")
        print("\\n💡 Dicas:")
        print("1. Verifique se as chaves de API estão corretas")
        print("2. Verifique se o índice Pinecone existe")
        print("3. Para Pinecone, não use mais o parâmetro 'environment'")
        return
    
    if args.file:
        # Processa arquivo único
        filepath = Path(args.file)
        if not filepath.exists():
            logger.error(f"Arquivo não encontrado: {filepath}")
            return
        
        chunks = processor.process_file(filepath)
        print(f"\\n✅ Processamento concluído: {chunks} chunks indexados")
    else:
        # Processa diretório
        directory = Path(args.dir)
        if not directory.exists():
            logger.error(f"Diretório não encontrado: {directory}")
            return
        
        stats = processor.process_directory(directory)
        
        print("\\n📊 Estatísticas do processamento:")
        print(f"   Arquivos processados: {stats['files_processed']}")
        print(f"   Total de chunks: {stats['total_chunks']}")
        print(f"   Arquivos com erro: {stats['failed_files']}")


if __name__ == "__main__":
    main()
'''

# Conteúdo do search_tool_final.py
search_tool_final_content = '''import requests
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
        
        print(f"\\n--- Contextos Encontrados para a pergunta: '{pergunta_exemplo}' ---\\n")
        if not contextos:
            print("Nenhum contexto foi encontrado. Verifique se o índice Pinecone está populado e se as chaves de API estão corretas.")
        else:
            for i, contexto in enumerate(contextos, 1):
                print(f"Resultado {i}:\\n{contexto}\\n---")
    except Exception as e:
        print(f"\\n❌ Erro ao executar busca: {e}")
        print("\\n💡 Verifique:")
        print("1. Se o arquivo .env está configurado corretamente")
        print("2. Se o índice Pinecone existe e tem o nome correto")
        print("3. Se as chaves de API estão válidas")
'''

def main():
    print("🔧 CRIANDO ARQUIVOS NECESSÁRIOS")
    print("=" * 50)
    
    # Cria os arquivos
    create_file("ingest_final.py", ingest_final_content)
    create_file("search_tool_final.py", search_tool_final_content)
    
    print("\n✅ Arquivos criados com sucesso!")
    print("\n🚀 Agora você pode executar:")
    print("   python ingest_final.py --file data/processed/0a1efcf5_Resultado_Consulta__Comunidade_2024_-_Discentes.txt")

if __name__ == "__main__":
    main()