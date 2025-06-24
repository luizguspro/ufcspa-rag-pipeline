"""
Spider Scrapy simplificado para coletar normas e documentos da UFCSPA.

Versão mais direta que funciona melhor para páginas específicas.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Generator
from urllib.parse import urljoin, urlparse

import scrapy
from scrapy import Request, Spider
from scrapy.http import Response


class UFCSPASpiderSimple(Spider):
    """Spider simplificado para coletar PDFs da UFCSPA."""
    
    name = "ufcspa_simple"
    allowed_domains = ["ufcspa.edu.br"]
    start_urls = ["https://ufcspa.edu.br/sobre-a-ufcspa/normas"]
    
    custom_settings = {
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 8,
        'DOWNLOAD_DELAY': 1,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdf_dir = Path("data/raw")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.visited_urls = set()
        
    def parse(self, response: Response) -> Generator:
        """Processa a página inicial e busca por links."""
        self.logger.info(f"Processando página: {response.url}")
        
        # Marca URL como visitada
        self.visited_urls.add(response.url)
        
        # Busca por links de PDFs diretos
        pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
        pdf_links.extend(response.css('a[href*=".pdf"]::attr(href)').getall())
        
        # Remove duplicatas
        pdf_links = list(set(pdf_links))
        
        self.logger.info(f"Encontrados {len(pdf_links)} links para PDFs")
        
        for pdf_link in pdf_links:
            pdf_url = urljoin(response.url, pdf_link)
            if self._is_valid_url(pdf_url):
                self.logger.info(f"Baixando PDF: {pdf_url}")
                yield Request(
                    url=pdf_url,
                    callback=self.save_pdf,
                    dont_filter=True,
                    meta={'source_page': response.url}
                )
        
        # Busca por links para outras páginas de normas
        all_links = response.css('a::attr(href)').getall()
        
        for link in all_links:
            url = urljoin(response.url, link)
            
            # Verifica se é uma URL válida e relevante
            if (self._is_valid_url(url) and 
                url not in self.visited_urls and
                self._is_relevant_url(url)):
                
                self.logger.info(f"Seguindo link: {url}")
                yield Request(
                    url=url,
                    callback=self.parse,
                    dont_filter=True
                )
    
    def save_pdf(self, response: Response) -> Dict:
        """Salva um arquivo PDF."""
        try:
            filename = self._generate_pdf_filename(response.url)
            filepath = self.pdf_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.body)
            
            self.logger.info(f"PDF salvo: {filename} ({len(response.body)} bytes)")
            
            return {
                'url': response.url,
                'filename': filename,
                'size': len(response.body),
                'source_page': response.meta.get('source_page', '')
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar PDF {response.url}: {str(e)}")
            return {'error': str(e)}
    
    def _is_valid_url(self, url: str) -> bool:
        """Verifica se a URL é válida e está no domínio permitido."""
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in self.allowed_domains)
        except:
            return False
    
    def _is_relevant_url(self, url: str) -> bool:
        """Verifica se a URL é relevante para normas."""
        relevant_keywords = [
            'norma', 'regimento', 'estatuto', 'resolucao', 
            'portaria', 'regulamento', 'legislacao', 'conselho',
            'deliberacao', 'instrucao', 'sobre-a-ufcspa'
        ]
        
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in relevant_keywords)
    
    def _generate_pdf_filename(self, url: str) -> str:
        """Gera nome único para o PDF."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        parsed = urlparse(url)
        original_name = os.path.basename(parsed.path)
        
        if original_name and original_name.endswith('.pdf'):
            safe_name = ''.join(c for c in original_name[:-4] if c.isalnum() or c in '-_')[:50]
            return f"{url_hash}_{safe_name}.pdf"
        else:
            return f"{url_hash}.pdf"