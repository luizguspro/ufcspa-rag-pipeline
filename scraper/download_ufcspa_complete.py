"""
Download completo de documentos da UFCSPA.
Corrige problemas de SSL e busca em múltiplas páginas.
"""

import hashlib
import json
import logging
import os
import re
import ssl
import time
import urllib3
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Desabilita avisos SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class UFCSPAScraper:
    """Scraper completo para UFCSPA com múltiplas estratégias."""
    
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sessão com SSL desabilitado
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8'
        })
        
        # URLs conhecidas da UFCSPA
        self.known_urls = [
            # Páginas principais de normas
            "https://www.ufcspa.edu.br/sobre-a-ufcspa/normas",
            "https://ufcspa.edu.br/sobre-a-ufcspa/normas",
            "https://www.ufcspa.edu.br/institucional/normas",
            "https://ufcspa.edu.br/institucional/normas",
            
            # Subpáginas específicas
            "https://www.ufcspa.edu.br/sobre-a-ufcspa/normas/regimentos",
            "https://www.ufcspa.edu.br/sobre-a-ufcspa/normas/estatuto",
            "https://www.ufcspa.edu.br/sobre-a-ufcspa/normas/resolucoes",
            "https://www.ufcspa.edu.br/sobre-a-ufcspa/normas/portarias",
            
            # Páginas de documentos
            "https://www.ufcspa.edu.br/sobre-a-ufcspa/documentos",
            "https://www.ufcspa.edu.br/institucional/documentos-institucionais",
            
            # Conselhos
            "https://www.ufcspa.edu.br/institucional/conselhos/consun",
            "https://www.ufcspa.edu.br/institucional/conselhos/consepe",
            
            # Páginas antigas (podem ter mudado)
            "https://www.ufcspa.edu.br/index.php/normas",
            "https://www.ufcspa.edu.br/ufcspa/normasedocs"
        ]
        
        self.visited_urls = set()
        self.found_pdfs = set()
    
    def scrape_all(self):
        """Executa scraping completo com múltiplas estratégias."""
        print("=" * 70)
        print("DOWNLOAD COMPLETO - DOCUMENTOS UFCSPA")
        print("=" * 70)
        
        all_pdfs = []
        
        # Estratégia 1: URLs conhecidas
        print("\n📍 Estratégia 1: Verificando URLs conhecidas...")
        for url in self.known_urls:
            pdfs = self._scrape_page(url)
            all_pdfs.extend(pdfs)
            time.sleep(0.5)
        
        # Estratégia 2: Busca por padrões
        print("\n🔍 Estratégia 2: Busca por padrões de URL...")
        pattern_urls = self._generate_url_patterns()
        for url in pattern_urls:
            if url not in self.visited_urls:
                pdfs = self._scrape_page(url)
                all_pdfs.extend(pdfs)
                time.sleep(0.5)
        
        # Estratégia 3: Google search (simulado)
        print("\n🔎 Estratégia 3: Busca avançada...")
        search_urls = self._search_for_documents()
        for url in search_urls:
            if url not in self.visited_urls:
                pdfs = self._scrape_page(url)
                all_pdfs.extend(pdfs)
                time.sleep(0.5)
        
        # Remove duplicatas
        unique_pdfs = list(set(all_pdfs))
        
        # Download dos PDFs
        if unique_pdfs:
            print(f"\n📊 Total de PDFs únicos encontrados: {len(unique_pdfs)}")
            self._download_all_pdfs(unique_pdfs)
        else:
            print("\n❌ Nenhum PDF encontrado")
            self._create_sample_data()
        
        # Salva relatório
        self._save_report(unique_pdfs)
    
    def _scrape_page(self, url: str) -> List[str]:
        """Scrape uma página específica."""
        if url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        logger.info(f"Visitando: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Status {response.status_code} para {url}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            pdfs = []
            
            # Busca links diretos para PDFs
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Verifica se é PDF
                if self._is_pdf_link(href):
                    pdf_url = urljoin(url, href)
                    if pdf_url not in self.found_pdfs:
                        self.found_pdfs.add(pdf_url)
                        pdfs.append(pdf_url)
                        logger.info(f"  📄 PDF encontrado: {link.get_text(strip=True)[:50]}")
            
            # Busca links para subpáginas relevantes
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                if self._is_relevant_link(href) and href not in self.visited_urls:
                    # Recursivamente busca PDFs
                    sub_pdfs = self._scrape_page(href)
                    pdfs.extend(sub_pdfs)
            
            return pdfs
            
        except Exception as e:
            logger.error(f"Erro ao acessar {url}: {e}")
            return []
    
    def _is_pdf_link(self, href: str) -> bool:
        """Verifica se é link para PDF."""
        href_lower = href.lower()
        return (
            href_lower.endswith('.pdf') or
            'pdf' in href_lower or
            '/download/' in href_lower or
            'arquivo' in href_lower or
            'documento' in href_lower
        )
    
    def _is_relevant_link(self, href: str) -> bool:
        """Verifica se é link relevante para normas."""
        if not href.startswith('http'):
            return False
        
        # Deve ser do domínio UFCSPA
        if 'ufcspa.edu.br' not in href:
            return False
        
        # Palavras-chave relevantes
        keywords = [
            'norma', 'regimento', 'estatuto', 'resolucao', 'portaria',
            'regulamento', 'legislacao', 'documento', 'conselho',
            'consepe', 'consun', 'deliberacao', 'instrucao'
        ]
        
        href_lower = href.lower()
        return any(kw in href_lower for kw in keywords)
    
    def _generate_url_patterns(self) -> List[str]:
        """Gera padrões de URL para testar."""
        patterns = []
        base_urls = [
            "https://www.ufcspa.edu.br",
            "https://ufcspa.edu.br"
        ]
        
        paths = [
            "/normas",
            "/documentos",
            "/institucional/normas",
            "/sobre-a-ufcspa/documentos",
            "/administrativo/normas",
            "/arquivos/normas",
            "/downloads/normas"
        ]
        
        for base in base_urls:
            for path in paths:
                patterns.append(base + path)
        
        return patterns
    
    def _search_for_documents(self) -> List[str]:
        """Busca avançada por documentos."""
        # Simula busca por documentos específicos
        search_terms = [
            "estatuto ufcspa pdf",
            "regimento interno ufcspa",
            "normas extensão ufcspa",
            "resolução consepe ufcspa",
            "portarias ufcspa"
        ]
        
        urls = []
        # Aqui poderia implementar busca real no Google ou DuckDuckGo
        # Por enquanto, retorna URLs conhecidas
        
        return urls
    
    def _download_all_pdfs(self, pdf_urls: List[str]):
        """Baixa todos os PDFs encontrados."""
        print(f"\n📥 Baixando {len(pdf_urls)} PDFs...")
        
        success = 0
        failed = 0
        
        for pdf_url in tqdm(pdf_urls, desc="Baixando PDFs"):
            if self._download_pdf(pdf_url):
                success += 1
            else:
                failed += 1
            time.sleep(0.5)
        
        print(f"\n✅ Downloads concluídos:")
        print(f"   - Sucesso: {success}")
        print(f"   - Falhas: {failed}")
    
    def _download_pdf(self, url: str) -> bool:
        """Baixa um PDF específico."""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Gera nome do arquivo
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = self._extract_filename(url, response.headers)
            if not filename:
                filename = f"{url_hash}.pdf"
            else:
                filename = f"{url_hash}_{filename}"
            
            filepath = self.output_dir / filename
            
            # Baixa o arquivo
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✓ Baixado: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Erro ao baixar {url}: {e}")
            return False
    
    def _extract_filename(self, url: str, headers: dict) -> str:
        """Extrai nome do arquivo da URL ou headers."""
        # Tenta do header Content-Disposition
        cd = headers.get('content-disposition')
        if cd:
            fname = re.findall('filename="(.+)"', cd)
            if fname:
                return fname[0]
        
        # Tenta da URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if filename and filename.endswith('.pdf'):
            return filename
        
        return None
    
    def _create_sample_data(self):
        """Cria dados de exemplo se não encontrar PDFs."""
        print("\n📝 Criando dados de exemplo para teste...")
        
        samples_dir = Path("data/processed")
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        samples = {
            "estatuto_ufcspa_exemplo.txt": """ESTATUTO DA UFCSPA (EXEMPLO)

A Universidade Federal de Ciências da Saúde de Porto Alegre - UFCSPA é uma instituição pública federal.

CAPÍTULO I - DOS PRINCÍPIOS
- Excelência acadêmica
- Formação humanística
- Compromisso social
- Gestão democrática""",

            "normas_extensao_exemplo.txt": """NORMAS DE EXTENSÃO (EXEMPLO)

As atividades de extensão promovem interação entre universidade e sociedade.
Modalidades: programas, projetos, cursos, eventos, prestação de serviços.""",

            "regimento_interno_exemplo.txt": """REGIMENTO INTERNO (EXEMPLO)

Órgãos colegiados: CONSUN, CONSEPE, Reitoria.
O Reitor é a autoridade executiva máxima."""
        }
        
        for filename, content in samples.items():
            filepath = samples_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✅ {len(samples)} arquivos de exemplo criados em data/processed/")
    
    def _save_report(self, pdfs: List[str]):
        """Salva relatório do scraping."""
        report = {
            "data": time.strftime("%Y-%m-%d %H:%M:%S"),
            "urls_visitadas": len(self.visited_urls),
            "pdfs_encontrados": len(pdfs),
            "pdfs_urls": pdfs,
            "urls_visitadas_lista": list(self.visited_urls)
        }
        
        report_file = self.output_dir / "scraping_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Relatório salvo em: {report_file}")


def main():
    """Função principal."""
    scraper = UFCSPAScraper()
    
    print("\n🚀 Iniciando download de documentos da UFCSPA...")
    print("⚠️  Nota: O site tem problema de SSL, mas o download continuará")
    
    try:
        scraper.scrape_all()
    except KeyboardInterrupt:
        print("\n⚠️  Download interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("\n💡 Criando dados de exemplo...")
        scraper._create_sample_data()


if __name__ == "__main__":
    main()