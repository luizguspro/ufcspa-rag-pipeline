"""
Script para dividir textos em chunks para processamento por embeddings.

Este módulo lê arquivos de texto processados, limpa o conteúdo e divide
em chunks de tamanho apropriado para geração de embeddings e busca vetorial.
"""

import html
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TextChunker:
    """Divisor de textos em chunks para processamento.
    
    Esta classe gerencia a limpeza e divisão de textos em chunks
    de tamanho apropriado para embeddings, com sobreposição configurável.
    
    Attributes:
        input_dir: Diretório contendo os arquivos de texto
        output_dir: Diretório para salvar o arquivo JSON com chunks
        chunk_size: Tamanho máximo de cada chunk em tokens
        chunk_overlap: Sobreposição entre chunks em tokens
        method: Método de chunking ('langchain' ou 'tiktoken')
    """
    
    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        method: str = "langchain"
    ):
        """Inicializa o chunker de textos.
        
        Args:
            input_dir: Caminho para o diretório de entrada
            output_dir: Caminho para o diretório de saída
            chunk_size: Tamanho máximo de cada chunk
            chunk_overlap: Sobreposição entre chunks
            method: Método de chunking ('langchain' ou 'tiktoken')
        """
        # Define caminhos padrão baseados no diretório raiz do projeto
        base_dir = Path(__file__).parent.parent
        self.input_dir = Path(input_dir) if input_dir else base_dir / "data" / "processed"
        self.output_dir = Path(output_dir) if output_dir else base_dir / "data" / "processed"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method
        
        # Configura o tokenizer se usar método tiktoken
        if method == "tiktoken":
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Configura o splitter se usar método langchain
        if method == "langchain":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,  # Aproximação: 1 token ≈ 4 caracteres
                chunk_overlap=chunk_overlap * 4,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                is_separator_regex=False
            )
        
        logger.info(f"TextChunker inicializado")
        logger.info(f"Método: {method}")
        logger.info(f"Tamanho do chunk: {chunk_size} tokens")
        logger.info(f"Sobreposição: {chunk_overlap} tokens")
    
    def process_all_texts(self) -> dict:
        """Processa todos os arquivos de texto e gera chunks.
        
        Returns:
            Dict com estatísticas do processamento
        """
        txt_files = list(self.input_dir.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"Nenhum arquivo .txt encontrado em {self.input_dir}")
            return {"total_files": 0, "total_chunks": 0, "files_processed": 0, "files_skipped": 0}
        
        logger.info(f"Encontrados {len(txt_files)} arquivos de texto para processar")
        
        all_chunks = []
        stats = {
            "total_files": len(txt_files),
            "total_chunks": 0,
            "files_processed": 0,
            "files_skipped": 0
        }
        
        for txt_path in tqdm(txt_files, desc="Processando textos"):
            try:
                chunks = self.process_text_file(txt_path)
                if chunks:
                    all_chunks.extend(chunks)
                    stats["files_processed"] += 1
                else:
                    stats["files_skipped"] += 1
            except Exception as e:
                logger.error(f"Erro ao processar {txt_path.name}: {str(e)}")
                stats["files_skipped"] += 1
        
        stats["total_chunks"] = len(all_chunks)
        
        # Salva todos os chunks em um arquivo JSON
        if all_chunks:
            output_path = self.output_dir / "chunks.json"
            self._save_chunks(all_chunks, output_path)
            logger.info(f"Chunks salvos em: {output_path}")
        
        logger.info(f"Processamento concluído: {stats}")
        return stats
    
    def process_text_file(self, txt_path: Path) -> List[Dict[str, any]]:
        """Processa um único arquivo de texto e gera chunks.
        
        Args:
            txt_path: Caminho para o arquivo de texto
            
        Returns:
            Lista de chunks com metadados
        """
        logger.info(f"Processando: {txt_path.name}")
        
        try:
            # Lê o arquivo de texto
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Verifica se há conteúdo
            if not text or len(text.strip()) == 0:
                logger.warning(f"Arquivo vazio: {txt_path.name}")
                return []
            
            # Limpa o texto
            text = self._clean_text(text)
            
            # Divide em chunks
            if self.method == "langchain":
                chunks = self._chunk_with_langchain(text)
            else:  # tiktoken
                chunks = self._chunk_with_tiktoken(text)
            
            # Adiciona metadados aos chunks
            chunks_with_metadata = []
            for i, chunk_text in enumerate(chunks):
                chunk_data = {
                    'source_file': txt_path.name,
                    'chunk_id': i + 1,
                    'text': chunk_text.strip(),
                    'char_count': len(chunk_text),
                    'token_count': self._count_tokens(chunk_text)
                }
                chunks_with_metadata.append(chunk_data)
            
            logger.info(f"Gerados {len(chunks_with_metadata)} chunks de {txt_path.name}")
            return chunks_with_metadata
            
        except Exception as e:
            logger.error(f"Erro ao processar {txt_path.name}: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto antes do chunking.
        
        Args:
            text: Texto bruto
            
        Returns:
            Texto limpo e normalizado
        """
        # Decodifica entidades HTML
        text = html.unescape(text)
        
        # Remove caracteres de controle (exceto newlines e tabs)
        text = ''.join(
            char for char in text 
            if char in ['\n', '\t'] or ord(char) >= 32
        )
        
        # Normaliza espaços em branco
        text = re.sub(r'[ \t]+', ' ', text)  # Múltiplos espaços/tabs -> um espaço
        text = re.sub(r'\n[ \t]+', '\n', text)  # Remove espaços no início de linhas
        text = re.sub(r'[ \t]+\n', '\n', text)  # Remove espaços no fim de linhas
        
        # Remove quebras de linha excessivas
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove linhas vazias que contêm apenas espaços
        lines = text.split('\n')
        lines = [line for line in lines if line.strip() or line == '']
        text = '\n'.join(lines)
        
        # Corrige pontuação com espaços extras
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])(?=[A-Za-zÀ-ÿ])', r'\1 ', text)
        
        # Remove caracteres especiais problemáticos
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normaliza aspas
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove múltiplos hífens
        text = re.sub(r'-{3,}', '---', text)
        
        # Remove espaços antes e depois de parênteses/colchetes
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        text = re.sub(r'\s*\[\s*', ' [', text)
        text = re.sub(r'\s*\]\s*', '] ', text)
        
        # Limpa espaços finais
        text = text.strip()
        
        return text
    
    def _chunk_with_langchain(self, text: str) -> List[str]:
        """Divide texto em chunks usando LangChain.
        
        Args:
            text: Texto a ser dividido
            
        Returns:
            Lista de chunks
        """
        return self.text_splitter.split_text(text)
    
    def _chunk_with_tiktoken(self, text: str) -> List[str]:
        """Divide texto em chunks usando tiktoken para contagem precisa de tokens.
        
        Args:
            text: Texto a ser dividido
            
        Returns:
            Lista de chunks
        """
        # Tokeniza o texto completo
        tokens = self.encoding.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Define o fim do chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Extrai os tokens do chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decodifica de volta para texto
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Tenta ajustar o chunk para terminar em uma sentença completa
            if end_idx < len(tokens):
                # Procura o último ponto final, exclamação ou interrogação
                last_sentence_end = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind('!'),
                    chunk_text.rfind('?'),
                    chunk_text.rfind('\n')
                )
                
                if last_sentence_end > len(chunk_text) * 0.5:  # Se encontrou perto do fim
                    chunk_text = chunk_text[:last_sentence_end + 1]
                    # Recalcula os tokens para o próximo início
                    actual_tokens = self.encoding.encode(chunk_text)
                    end_idx = start_idx + len(actual_tokens)
            
            chunks.append(chunk_text)
            
            # Move para o próximo chunk com sobreposição
            start_idx = end_idx - self.chunk_overlap
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Conta o número de tokens em um texto.
        
        Args:
            text: Texto para contar tokens
            
        Returns:
            Número de tokens
        """
        if self.method == "tiktoken" or hasattr(self, 'encoding'):
            return len(self.encoding.encode(text))
        else:
            # Estimativa aproximada: 1 token ≈ 4 caracteres
            return len(text) // 4
    
    def _save_chunks(self, chunks: List[Dict], output_path: Path) -> None:
        """Salva os chunks em um arquivo JSON.
        
        Args:
            chunks: Lista de chunks com metadados
            output_path: Caminho do arquivo de saída
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Salvos {len(chunks)} chunks em {output_path}")


def main():
    """Função principal para executar o chunker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Processa textos e divide em chunks")
    parser.add_argument(
        '--method', 
        choices=['langchain', 'tiktoken'], 
        default='langchain',
        help='Método de chunking a usar'
    )
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=1000,
        help='Tamanho máximo de cada chunk em tokens'
    )
    parser.add_argument(
        '--chunk-overlap', 
        type=int, 
        default=200,
        help='Sobreposição entre chunks em tokens'
    )
    
    args = parser.parse_args()
    
    chunker = TextChunker(
        method=args.method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    stats = chunker.process_all_texts()
    
    print("\n=== Resumo do Chunking ===")
    print(f"Total de arquivos: {stats['total_files']}")
    print(f"Arquivos processados: {stats['files_processed']}")
    print(f"Arquivos ignorados: {stats['files_skipped']}")
    print(f"Total de chunks gerados: {stats['total_chunks']}")
    if stats['files_processed'] > 0:
        print(f"Média de chunks por arquivo: {stats['total_chunks'] / stats['files_processed']:.1f}")


if __name__ == "__main__":
    main()