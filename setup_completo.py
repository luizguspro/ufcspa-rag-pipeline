"""
Script de setup completo para o sistema RAG UFCSPA.
Corrige todos os problemas e configura o ambiente.
"""

import os
import subprocess
import sys
from pathlib import Path
import json


class ProjectSetup:
    """Classe para configurar o projeto completo."""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.errors = []
        self.warnings = []
    
    def print_header(self, text):
        """Imprime cabeçalho formatado."""
        print(f"\n{'='*70}")
        print(f"🔧 {text}")
        print(f"{'='*70}")
    
    def check_python_version(self):
        """Verifica versão do Python."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor} detectado")
            return True
        else:
            self.errors.append(f"Python 3.8+ necessário (atual: {version.major}.{version.minor})")
            return False
    
    def create_directory_structure(self):
        """Cria estrutura completa de diretórios."""
        directories = [
            "data/raw",
            "data/processed", 
            "faiss_index",
            "logs",
            "scraper",
            "ingest",
            "query",
            "tests"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✅ Estrutura de diretórios criada")
    
    def install_dependencies(self):
        """Instala dependências necessárias."""
        print("\n📦 Instalando dependências...")
        
        # Lista de pacotes essenciais
        packages = [
            "scrapy==2.11.2",
            "requests",
            "beautifulsoup4",
            "pdfminer.six",
            "pytesseract",
            "pdf2image",
            "Pillow",
            "sentence-transformers",
            "faiss-cpu",
            "langchain",
            "tiktoken",
            "tqdm",
            "numpy",
            "pandas"
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                    stdout=subprocess.DEVNULL)
                print(f"  ✅ {package}")
            except:
                self.warnings.append(f"Falha ao instalar {package}")
                print(f"  ⚠️  {package} - falhou")
    
    def create_sample_data(self):
        """Cria dados de exemplo para teste."""
        output_dir = Path("data/processed")
        
        samples = {
            "estatuto_ufcspa.txt": """ESTATUTO DA UNIVERSIDADE FEDERAL DE CIÊNCIAS DA SAÚDE DE PORTO ALEGRE

CAPÍTULO I - DA UNIVERSIDADE E SEUS FINS

Art. 1º A Universidade Federal de Ciências da Saúde de Porto Alegre - UFCSPA, criada pela Lei nº 11.641, de 13 de janeiro de 2008, é uma fundação pública, dotada de personalidade jurídica de direito público, com autonomia didático-científica, administrativa e de gestão financeira e patrimonial.

Art. 2º A UFCSPA tem sede e foro na cidade de Porto Alegre, Estado do Rio Grande do Sul.

Art. 3º A UFCSPA tem por finalidade promover o ensino, a pesquisa e a extensão, com ênfase na área da saúde.

CAPÍTULO II - DOS PRINCÍPIOS

Art. 4º A UFCSPA, no cumprimento de sua missão, orienta-se pelos seguintes princípios:
I - indissociabilidade entre ensino, pesquisa e extensão;
II - universalidade do conhecimento e educação;
III - interdisciplinaridade do conhecimento;
IV - excelência acadêmica;
V - democratização do acesso e permanência;
VI - gestão democrática e colegiada;
VII - valorização dos profissionais da educação;
VIII - compromisso social e ético.""",

            "normas_extensao.txt": """RESOLUÇÃO Nº 001/2023 - CONSEPE

Estabelece as normas para as atividades de extensão universitária no âmbito da UFCSPA.

Art. 1º As atividades de extensão universitária são processos interdisciplinares, educativos, culturais, científicos e políticos que promovem a interação transformadora entre a Universidade e outros setores da sociedade.

Art. 2º São consideradas atividades de extensão:
I - programas: conjunto articulado de projetos e outras ações de extensão;
II - projetos: ações processuais e contínuas de caráter educativo, social, cultural, científico ou tecnológico;
III - cursos e oficinas: ações pedagógicas de caráter teórico e/ou prático;
IV - eventos: ações que implicam na apresentação e/ou exibição pública;
V - prestação de serviços: realização de trabalho oferecido pela UFCSPA.

Art. 3º Todas as atividades de extensão devem ser registradas e aprovadas pela Pró-Reitoria de Extensão.""",

            "regimento_interno.txt": """REGIMENTO INTERNO DA UFCSPA

TÍTULO I - DAS DISPOSIÇÕES PRELIMINARES

Art. 1º O presente Regimento Interno disciplina a organização e o funcionamento da Universidade Federal de Ciências da Saúde de Porto Alegre - UFCSPA.

TÍTULO II - DA ESTRUTURA ORGANIZACIONAL

Art. 2º A administração superior da UFCSPA é exercida pelos seguintes órgãos:
I - Conselho Universitário (CONSUN);
II - Conselho de Ensino, Pesquisa e Extensão (CONSEPE);
III - Reitoria.

Art. 3º O Conselho Universitário é o órgão máximo deliberativo e normativo da Universidade."""
        }
        
        for filename, content in samples.items():
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✅ Criados {len(samples)} arquivos de exemplo")
    
    def fix_file_paths(self):
        """Corrige problemas de paths nos scripts."""
        # Cria __init__.py em todos os módulos
        for module in ["scraper", "ingest", "query"]:
            init_file = Path(module) / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
        
        print("✅ Arquivos __init__.py criados")
    
    def create_config_file(self):
        """Cria arquivo de configuração central."""
        config = {
            "project_name": "UFCSPA RAG System",
            "version": "1.0.0",
            "paths": {
                "data_raw": "data/raw",
                "data_processed": "data/processed",
                "faiss_index": "faiss_index",
                "logs": "logs"
            },
            "models": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384
            },
            "chunking": {
                "method": "langchain",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Arquivo de configuração criado")
    
    def run_pipeline_test(self):
        """Testa o pipeline básico."""
        print("\n🧪 Testando pipeline...")
        
        # Testa chunking
        try:
            result = subprocess.run(
                [sys.executable, "ingest/chunk.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("  ✅ Chunking funcionando")
            else:
                print("  ⚠️  Erro no chunking")
                self.warnings.append("Chunking com problemas")
        except:
            print("  ❌ Chunking falhou")
            self.errors.append("Chunking não funciona")
    
    def create_run_script(self):
        """Cria script para executar o sistema completo."""
        run_script = '''#!/usr/bin/env python
"""
Script para executar o sistema RAG UFCSPA completo.
"""

import subprocess
import sys
from pathlib import Path

def run_step(command, description):
    print(f"\\n{'='*60}")
    print(f"Executando: {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True)
    return result.returncode == 0

def main():
    steps = [
        ("python scraper/download_with_ssl_fix.py", "Download de PDFs"),
        ("python ingest/convert.py", "Conversão PDF → Texto"),
        ("python ingest/chunk.py", "Divisão em chunks"),
        ("python ingest/embed.py", "Geração de embeddings"),
        ("python query/interactive.py", "Interface de consulta")
    ]
    
    print("SISTEMA RAG UFCSPA - EXECUÇÃO COMPLETA")
    
    for command, description in steps[:-1]:
        if not run_step(command, description):
            print(f"\\n❌ Erro em: {description}")
            print("Verifique os logs e tente novamente")
            return
    
    # Última etapa - interface
    print("\\n✅ Pipeline concluído! Iniciando interface...")
    run_step(steps[-1][0], steps[-1][1])

if __name__ == "__main__":
    main()
'''
        
        with open("run_system.py", "w") as f:
            f.write(run_script)
        
        print("✅ Script run_system.py criado")
    
    def print_summary(self):
        """Imprime resumo da configuração."""
        print("\n" + "="*70)
        print("📊 RESUMO DA CONFIGURAÇÃO")
        print("="*70)
        
        if self.errors:
            print("\n❌ ERROS ENCONTRADOS:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️  AVISOS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors:
            print("\n✅ SISTEMA CONFIGURADO COM SUCESSO!")
            print("\n🚀 Para usar o sistema:")
            print("  1. Teste com dados de exemplo:")
            print("     python rag_demo.py")
            print("\n  2. Execute o pipeline completo:")
            print("     python run_system.py")
            print("\n  3. Use apenas a interface:")
            print("     python query/interactive.py")


def main():
    """Executa o setup completo."""
    print("SETUP COMPLETO - SISTEMA RAG UFCSPA")
    print("="*70)
    
    setup = ProjectSetup()
    
    # Executa todas as etapas
    setup.print_header("Verificando Python")
    setup.check_python_version()
    
    setup.print_header("Criando estrutura de diretórios")
    setup.create_directory_structure()
    
    setup.print_header("Instalando dependências")
    setup.install_dependencies()
    
    setup.print_header("Criando dados de exemplo")
    setup.create_sample_data()
    
    setup.print_header("Corrigindo arquivos")
    setup.fix_file_paths()
    
    setup.print_header("Criando configurações")
    setup.create_config_file()
    setup.create_run_script()
    
    setup.print_header("Testando sistema")
    setup.run_pipeline_test()
    
    # Resumo final
    setup.print_summary()


if __name__ == "__main__":
    main()