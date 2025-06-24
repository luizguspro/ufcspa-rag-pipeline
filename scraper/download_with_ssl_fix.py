"""
Script para baixar PDFs da UFCSPA com correção para problema de SSL.
"""

import hashlib
import logging
import ssl
import time
import urllib3
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Desabilita avisos de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class UFCSPADownloader:
    """Downloader de PDFs com tratamento de SSL."""
    
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configura sessão com SSL desabilitado (apenas para este site específico)
        self.session = requests.Session()
        self.session.verify = False  # Desabilita verificação SSL
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def download_pdf(self, url, filename=None):
        """Baixa um PDF de uma URL."""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            if filename is None:
                # Gera nome único
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                parsed = urlparse(url)
                original_name = Path(parsed.path).name
                filename = f"{url_hash}_{original_name}"
            
            filepath = self.output_dir / filename
            
            # Baixa o arquivo
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Baixado: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao baixar {url}: {e}")
            return False
    
    def find_pdfs_on_page(self, url):
        """Encontra links para PDFs em uma página."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_links = []
            
            # Busca todos os links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Verifica se é um PDF
                if href.lower().endswith('.pdf') or 'pdf' in href.lower():
                    # Converte para URL absoluta
                    absolute_url = urljoin(url, href)
                    pdf_links.append(absolute_url)
            
            # Remove duplicatas
            pdf_links = list(set(pdf_links))
            logger.info(f"Encontrados {len(pdf_links)} PDFs em {url}")
            
            return pdf_links
            
        except Exception as e:
            logger.error(f"Erro ao buscar PDFs em {url}: {e}")
            return []
    
    def crawl_site(self, start_urls):
        """Navega pelo site buscando PDFs."""
        visited = set()
        to_visit = list(start_urls)
        all_pdfs = []
        
        while to_visit:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            visited.add(url)
            logger.info(f"Visitando: {url}")
            
            # Busca PDFs na página
            pdfs = self.find_pdfs_on_page(url)
            all_pdfs.extend(pdfs)
            
            # Busca links para outras páginas (limitado ao domínio)
            try:
                response = self.session.get(url, timeout=30)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = urljoin(url, link['href'])
                    
                    # Verifica se está no domínio e é relevante
                    if ('ufcspa.edu.br' in href and 
                        any(term in href.lower() for term in ['norma', 'regimento', 'estatuto', 'resolucao']) and
                        href not in visited and 
                        href not in to_visit):
                        
                        to_visit.append(href)
                
            except Exception as e:
                logger.error(f"Erro ao buscar links em {url}: {e}")
            
            # Delay entre requisições
            time.sleep(1)
            
            # Limita a busca
            if len(visited) > 50:
                break
        
        return list(set(all_pdfs))


def main():
    """Função principal."""
    print("="*60)
    print("DOWNLOAD DE PDFs - UFCSPA (com correção SSL)")
    print("="*60)
    
    downloader = UFCSPADownloader()
    
    # URLs para começar a busca
    start_urls = [
        "https://ufcspa.edu.br/sobre-a-ufcspa/normas",
        "https://ufcspa.edu.br/institucional/normas",
        "https://www.ufcspa.edu.br/index.php/normas",
        "https://www.ufcspa.edu.br/sobre-a-ufcspa/documentos"
    ]
    
    print("\n🔍 Buscando PDFs no site...")
    
    # Tenta método 1: Crawling
    all_pdfs = downloader.crawl_site(start_urls)
    
    if not all_pdfs:
        print("\n⚠️  Nenhum PDF encontrado por crawling.")
        print("Tentando URLs diretas conhecidas...")
        
        # Método 2: URLs diretas (adicione URLs conhecidas aqui)
        known_pdfs = [
            # Adicione URLs diretas de PDFs aqui se conhecer
        ]
        all_pdfs.extend(known_pdfs)
    
    if all_pdfs:
        print(f"\n📄 Total de PDFs encontrados: {len(all_pdfs)}")
        print("\n📥 Baixando PDFs...")
        
        success = 0
        for pdf_url in all_pdfs:
            if downloader.download_pdf(pdf_url):
                success += 1
            time.sleep(0.5)  # Delay entre downloads
        
        print(f"\n✅ Download concluído: {success}/{len(all_pdfs)} PDFs baixados")
    else:
        print("\n❌ Nenhum PDF encontrado")
        print("\n💡 Alternativas:")
        print("1. Baixe PDFs manualmente e coloque em data/raw/")
        print("2. Use os dados de exemplo: python create_sample_data.py")


if __name__ == "__main__":
    main()