"""
Módulo de ingestão para processamento de documentos.

Este módulo contém ferramentas para converter PDFs em texto
e dividir textos em chunks para processamento vetorial.
"""

from .convert import PDFConverter
from .chunk import TextChunker

__all__ = ['PDFConverter', 'TextChunker']