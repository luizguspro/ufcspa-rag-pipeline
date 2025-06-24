"""
Script para converter PDFs em arquivos de texto.

Este módulo processa PDFs da pasta data/raw/, extrai texto usando pdfminer.six
e, se necessário, usa OCR com Tesseract para PDFs baseados em imagem.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from PIL import Image
from tqdm import tqdm


# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PDFConverter:
    """Conversor de PDFs para texto com suporte a OCR.
    
    Esta classe gerencia a conversão de arquivos PDF em texto, usando
    pdfminer.six como método principal e OCR como fallback para PDFs
    baseados em imagem.
    
    Attributes:
        input_dir: Diretório contendo os PDFs originais
        output_dir: Diretório para salvar os arquivos de texto
        min_text_length: Comprimento mínimo de texto para considerar extração bem-sucedida
        ocr_lang: Idioma para OCR (padrão: português)
    """
    
    def __init__(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed",
        min_text_length: int = 100,
        ocr_lang: str = "por"
    ):
        """Inicializa o conversor de PDFs.
        
        Args:
            input_dir: Caminho para o diretório de entrada
            output_dir: Caminho para o diretório de saída
            min_text_length: Comprimento mínimo de texto para não usar OCR
            ocr_lang: Código do idioma para OCR
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_text_length = min_text_length
        self.ocr_lang = ocr_lang
        
        # Cria diretório de saída se não existir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDFConverter inicializado")
        logger.info(f"Diretório de entrada: {self.input_dir.absolute()}")
        logger.info(f"Diretório de saída: {self.output_dir.absolute()}")
    
    def convert_all_pdfs(self) -> dict:
        """Converte todos os PDFs do diretório de entrada.
        
        Returns:
            Dict com estatísticas da conversão
        """
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"Nenhum arquivo PDF encontrado em {self.input_dir}")
            return {"total": 0, "success": 0, "failed": 0}
        
        logger.info(f"Encontrados {len(pdf_files)} arquivos PDF para processar")
        
        stats = {
            "total": len(pdf_files),
            "success": 0,
            "failed": 0,
            "used_ocr": 0
        }
        
        for pdf_path in tqdm(pdf_files, desc="Convertendo PDFs"):
            try:
                success, used_ocr = self.convert_pdf(pdf_path)
                if success:
                    stats["success"] += 1
                    if used_ocr:
                        stats["used_ocr"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Erro ao processar {pdf_path.name}: {str(e)}")
                stats["failed"] += 1
        
        logger.info(f"Conversão concluída: {stats}")
        return stats
    
    def convert_pdf(self, pdf_path: Path) -> Tuple[bool, bool]:
        """Converte um único PDF para texto.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Tupla (sucesso, usou_ocr)
        """
        logger.info(f"Processando: {pdf_path.name}")
        
        try:
            # Primeiro, tenta extrair texto com pdfminer.six
            text = self._extract_text_pdfminer(pdf_path)
            used_ocr = False
            
            # Verifica se o texto extraído é suficiente
            if self._is_text_insufficient(text):
                logger.info(f"Texto insuficiente extraído de {pdf_path.name}, tentando OCR...")
                text = self._extract_text_ocr(pdf_path)
                used_ocr = True
            
            # Limpa o texto extraído
            text = self._clean_text(text)
            
            # Salva o texto
            if text and len(text.strip()) > 0:
                output_path = self._get_output_path(pdf_path)
                self._save_text(text, output_path)
                logger.info(f"Texto salvo em: {output_path.name} ({'OCR' if used_ocr else 'PDFMiner'})")
                return True, used_ocr
            else:
                logger.warning(f"Nenhum texto extraído de {pdf_path.name}")
                return False, used_ocr
                
        except Exception as e:
            logger.error(f"Erro ao converter {pdf_path.name}: {str(e)}")
            return False, False
    
    def _extract_text_pdfminer(self, pdf_path: Path) -> str:
        """Extrai texto usando pdfminer.six.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Texto extraído do PDF
        """
        try:
            # Configurações para melhor extração
            laparams = LAParams(
                line_overlap=0.5,
                char_margin=2.0,
                word_margin=0.1,
                boxes_flow=0.5,
                detect_vertical=True,
                all_texts=True
            )
            
            text = extract_text(
                pdf_path,
                laparams=laparams,
                codec='utf-8'
            )
            
            return text
            
        except Exception as e:
            logger.warning(f"Erro ao extrair texto com PDFMiner de {pdf_path.name}: {str(e)}")
            return ""
    
    def _extract_text_ocr(self, pdf_path: Path) -> str:
        """Extrai texto usando OCR (Tesseract).
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Texto extraído via OCR
        """
        try:
            # Converte PDF para imagens
            logger.info(f"Convertendo {pdf_path.name} para imagens...")
            images = convert_from_path(
                pdf_path,
                dpi=300,  # Alta resolução para melhor OCR
                fmt='jpeg',
                thread_count=2
            )
            
            # Extrai texto de cada página
            all_text = []
            for i, image in enumerate(images):
                logger.debug(f"Processando página {i+1}/{len(images)} com OCR...")
                
                # Pré-processamento da imagem para melhor OCR
                image = self._preprocess_image_for_ocr(image)
                
                # Extrai texto usando Tesseract
                page_text = pytesseract.image_to_string(
                    image,
                    lang=self.ocr_lang,
                    config='--oem 3 --psm 6'  # Modo de engine neural e segmentação de página uniforme
                )
                
                all_text.append(page_text)
            
            return '\n\n'.join(all_text)
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto via OCR de {pdf_path.name}: {str(e)}")
            return ""
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Pré-processa imagem para melhorar resultado do OCR.
        
        Args:
            image: Imagem PIL para processar
            
        Returns:
            Imagem processada
        """
        # Converte para escala de cinza se não estiver
        if image.mode != 'L':
            image = image.convert('L')
        
        # Aqui poderiam ser adicionados outros processamentos como:
        # - Binarização
        # - Remoção de ruído
        # - Correção de inclinação
        # Por enquanto, retornamos a imagem em escala de cinza
        
        return image
    
    def _is_text_insufficient(self, text: str) -> bool:
        """Verifica se o texto extraído é insuficiente.
        
        Args:
            text: Texto extraído
            
        Returns:
            True se o texto for insuficiente
        """
        if not text:
            return True
        
        # Remove espaços e quebras de linha para contagem
        clean_text = text.strip()
        
        # Verifica comprimento mínimo
        if len(clean_text) < self.min_text_length:
            return True
        
        # Verifica se tem muitos caracteres não-ASCII (pode indicar erro de encoding)
        non_ascii_ratio = sum(1 for c in clean_text if ord(c) > 127) / len(clean_text)
        if non_ascii_ratio > 0.5:
            return True
        
        # Verifica se tem palavras reconhecíveis (não apenas símbolos)
        words = re.findall(r'\w+', clean_text)
        if len(words) < 10:
            return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto extraído.
        
        Args:
            text: Texto bruto extraído
            
        Returns:
            Texto limpo e normalizado
        """
        if not text:
            return ""
        
        # Remove caracteres de controle exceto newlines e tabs
        text = ''.join(
            char for char in text 
            if char == '\n' or char == '\t' or 
            (ord(char) >= 32 and ord(char) <= 126) or 
            ord(char) >= 160
        )
        
        # Normaliza quebras de linha
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove quebras de linha excessivas
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove espaços múltiplos
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove espaços no início e fim de linhas
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove hífens de quebra de linha
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        return text.strip()
    
    def _get_output_path(self, pdf_path: Path) -> Path:
        """Gera o caminho de saída para o arquivo de texto.
        
        Args:
            pdf_path: Caminho do arquivo PDF original
            
        Returns:
            Caminho para o arquivo de texto de saída
        """
        # Mantém o mesmo nome, mas muda a extensão
        txt_filename = pdf_path.stem + '.txt'
        return self.output_dir / txt_filename
    
    def _save_text(self, text: str, output_path: Path) -> None:
        """Salva o texto em um arquivo.
        
        Args:
            text: Texto a ser salvo
            output_path: Caminho do arquivo de saída
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)


def main():
    """Função principal para executar o conversor."""
    converter = PDFConverter()
    stats = converter.convert_all_pdfs()
    
    print("\n=== Resumo da Conversão ===")
    print(f"Total de PDFs: {stats['total']}")
    print(f"Convertidos com sucesso: {stats['success']}")
    print(f"Falhas: {stats['failed']}")
    print(f"Usaram OCR: {stats['used_ocr']}")


if __name__ == "__main__":
    main()