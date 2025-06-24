"""
Script auxiliar para executar o spider UFCSPA.
"""

import logging
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from spider import UFCSPASpider


def run_spider():
    """Executa o spider UFCSPA com as configurações do projeto."""
    
    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Carrega as configurações do projeto
    settings = get_project_settings()
    
    # Cria o processo do crawler
    process = CrawlerProcess(settings)
    
    # Adiciona o spider ao processo
    process.crawl(UFCSPASpider)
    
    # Inicia o crawling
    logging.info("Iniciando o spider UFCSPA...")
    process.start()
    
    logging.info("Spider finalizado!")


if __name__ == "__main__":
    run_spider()