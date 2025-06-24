"""
Script para executar o pipeline completo de ingestão.

Este script executa sequencialmente a conversão de PDFs, chunking de textos
e geração de embeddings com índice FAISS.
"""

import logging
import os
import sys
from pathlib import Path

# Adiciona o diretório pai ao path para permitir imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Agora importa os módulos usando o caminho completo
from ingest.convert import PDFConverter
from ingest.chunk import TextChunker
from ingest.embed import EmbeddingGenerator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    pdf_dir: str = "data/raw",
    text_dir: str = "data/processed",
    index_dir: str = "faiss_index",
    chunk_method: str = "langchain",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """Executa o pipeline completo de ingestão.
    
    Args:
        pdf_dir: Diretório contendo os PDFs
        text_dir: Diretório para arquivos processados
        index_dir: Diretório para o índice FAISS
        chunk_method: Método de chunking ('langchain' ou 'tiktoken')
        chunk_size: Tamanho dos chunks em tokens
        chunk_overlap: Sobreposição entre chunks
        embedding_model: Modelo para gerar embeddings
    """
    print("=" * 60)
    print("PIPELINE DE INGESTÃO - UFCSPA")
    print("=" * 60)
    
    # Etapa 1: Conversão de PDFs
    print("\n[1/3] Convertendo PDFs para texto...")
    print("-" * 40)
    
    converter = PDFConverter(
        input_dir=pdf_dir,
        output_dir=text_dir
    )
    
    convert_stats = converter.convert_all_pdfs()
    
    if convert_stats['success'] == 0:
        logger.error("Nenhum PDF foi convertido com sucesso. Abortando pipeline.")
        return
    
    # Etapa 2: Chunking de textos
    print("\n[2/3] Dividindo textos em chunks...")
    print("-" * 40)
    
    chunker = TextChunker(
        input_dir=text_dir,
        output_dir=text_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        method=chunk_method
    )
    
    chunk_stats = chunker.process_all_texts()
    
    if chunk_stats['total_chunks'] == 0:
        logger.error("Nenhum chunk foi gerado. Abortando pipeline.")
        return
    
    # Etapa 3: Geração de embeddings e índice FAISS
    print("\n[3/3] Gerando embeddings e índice FAISS...")
    print("-" * 40)
    
    embedder = EmbeddingGenerator(
        model_name=embedding_model
    )
    
    n_chunks, dim = embedder.process_chunks(
        input_file=f"{text_dir}/chunks.json",
        index_dir=index_dir
    )
    
    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO DO PIPELINE")
    print("=" * 60)
    print(f"\nConversão de PDFs:")
    print(f"  - Total de PDFs: {convert_stats['total']}")
    print(f"  - Convertidos com sucesso: {convert_stats['success']}")
    print(f"  - Usaram OCR: {convert_stats['used_ocr']}")
    print(f"  - Falhas: {convert_stats['failed']}")
    
    print(f"\nChunking de Textos:")
    print(f"  - Arquivos processados: {chunk_stats['files_processed']}")
    print(f"  - Total de chunks: {chunk_stats['total_chunks']}")
    if chunk_stats['files_processed'] > 0:
        avg_chunks = chunk_stats['total_chunks'] / chunk_stats['files_processed']
        print(f"  - Média de chunks/arquivo: {avg_chunks:.1f}")
    
    print(f"\nGeração de Embeddings:")
    print(f"  - Chunks processados: {n_chunks}")
    print(f"  - Dimensão dos embeddings: {dim}")
    print(f"  - Modelo usado: {embedding_model}")
    
    print(f"\n✓ Pipeline concluído com sucesso!")
    print(f"✓ Índice FAISS salvo em: {Path(index_dir) / 'ufcspa.index'}")
    print(f"\nPróximo passo: Execute o sistema de consultas (query)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de ingestão de documentos"
    )
    parser.add_argument(
        '--pdf-dir',
        default='data/raw',
        help='Diretório contendo os PDFs'
    )
    parser.add_argument(
        '--text-dir',
        default='data/processed',
        help='Diretório para arquivos processados'
    )
    parser.add_argument(
        '--index-dir',
        default='faiss_index',
        help='Diretório para o índice FAISS'
    )
    parser.add_argument(
        '--chunk-method',
        choices=['langchain', 'tiktoken'],
        default='langchain',
        help='Método de chunking'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Tamanho dos chunks em tokens'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Sobreposição entre chunks em tokens'
    )
    parser.add_argument(
        '--embedding-model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Modelo para gerar embeddings'
    )
    
    args = parser.parse_args()
    
    run_full_pipeline(
        pdf_dir=args.pdf_dir,
        text_dir=args.text_dir,
        index_dir=args.index_dir,
        chunk_method=args.chunk_method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model
    )