"""
Script mestre para executar o sistema RAG UFCSPA completo.
Executa todas as etapas com tratamento de erros.
"""

import os
import sys
import subprocess
from pathlib import Path
import time


class RAGPipeline:
    """Gerenciador do pipeline RAG completo."""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.steps_completed = []
        self.errors = []
    
    def print_header(self, text, emoji="🔧"):
        """Imprime cabeçalho formatado."""
        print(f"\n{'='*70}")
        print(f"{emoji} {text}")
        print(f"{'='*70}")
    
    def run_command(self, command, description, required=True):
        """Executa um comando e retorna sucesso/falha."""
        print(f"\n⏳ {description}...")
        
        try:
            # Executa o comando
            if isinstance(command, str):
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )
            
            if result.returncode == 0:
                print(f"✅ {description} - Concluído!")
                self.steps_completed.append(description)
                return True
            else:
                error_msg = f"{description} falhou"
                print(f"❌ {error_msg}")
                if result.stderr:
                    print(f"   Erro: {result.stderr[:200]}...")
                
                if required:
                    self.errors.append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"{description} - Exceção: {str(e)}"
            print(f"❌ {error_msg}")
            if required:
                self.errors.append(error_msg)
            return False
    
    def check_environment(self):
        """Verifica o ambiente."""
        self.print_header("Verificando Ambiente", "🔍")
        
        # Python
        print(f"Python: {sys.version}")
        
        # Diretórios
        dirs_needed = ["data/raw", "data/processed", "faiss_index"]
        for dir_path in dirs_needed:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print("✅ Diretórios criados/verificados")
        
        # Dependências críticas
        critical_packages = ["scrapy", "requests", "beautifulsoup4"]
        missing = []
        
        for package in critical_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"\n⚠️  Pacotes faltando: {', '.join(missing)}")
            print("Instalando...")
            self.run_command(
                f"{sys.executable} -m pip install {' '.join(missing)}",
                "Instalação de dependências"
            )
    
    def create_sample_data(self):
        """Cria dados de exemplo."""
        self.print_header("Criando Dados de Exemplo", "📝")
        
        output_dir = Path("data/processed")
        
        samples = {
            "estatuto_ufcspa.txt": """ESTATUTO DA UNIVERSIDADE FEDERAL DE CIÊNCIAS DA SAÚDE DE PORTO ALEGRE

CAPÍTULO I - DA UNIVERSIDADE E SEUS FINS

Art. 1º A UFCSPA é uma fundação pública federal de ensino superior, com autonomia didático-científica, administrativa e de gestão financeira e patrimonial.

Art. 2º A UFCSPA tem por finalidade promover o ensino, a pesquisa e a extensão, com ênfase na área da saúde.

CAPÍTULO II - DOS PRINCÍPIOS

Art. 3º São princípios da UFCSPA:
I - Excelência acadêmica
II - Formação humanística
III - Compromisso social
IV - Gestão democrática
V - Autonomia universitária""",

            "normas_extensao.txt": """NORMAS PARA ATIVIDADES DE EXTENSÃO

Art. 1º As atividades de extensão promovem a interação entre universidade e sociedade.

Art. 2º Modalidades:
I - Programas
II - Projetos  
III - Cursos
IV - Eventos
V - Prestação de serviços

Art. 3º Requisitos:
I - Relevância social
II - Articulação com ensino/pesquisa
III - Participação estudantil""",

            "regimento_interno.txt": """REGIMENTO INTERNO DA UFCSPA

Art. 1º A administração superior é exercida por:
I - Conselho Universitário (CONSUN)
II - Conselho de Ensino, Pesquisa e Extensão (CONSEPE)
III - Reitoria

Art. 2º O Reitor é a autoridade executiva máxima."""
        }
        
        created = 0
        for filename, content in samples.items():
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            created += 1
        
        print(f"✅ {created} arquivos de exemplo criados")
        return created > 0
    
    def run_pipeline(self):
        """Executa o pipeline completo."""
        self.print_header("PIPELINE RAG UFCSPA", "🚀")
        
        # 1. Verifica ambiente
        self.check_environment()
        
        # 2. Tenta baixar PDFs ou cria dados de exemplo
        self.print_header("Obtenção de Dados", "📥")
        
        # Verifica se já tem dados
        existing_pdfs = list(Path("data/raw").glob("*.pdf"))
        existing_txts = list(Path("data/processed").glob("*.txt"))
        
        if existing_pdfs:
            print(f"✅ {len(existing_pdfs)} PDFs já existentes")
        elif existing_txts:
            print(f"✅ {len(existing_txts)} textos já existentes")
        else:
            print("📄 Criando dados de exemplo...")
            self.create_sample_data()
        
        # 3. Chunking
        self.print_header("Processamento de Texto", "✂️")
        self.run_command(
            [sys.executable, "ingest/chunk.py"],
            "Divisão em chunks"
        )
        
        # 4. Embeddings (opcional)
        self.print_header("Geração de Embeddings", "🧮")
        
        # Verifica se as bibliotecas estão instaladas
        try:
            import sentence_transformers
            import faiss
            
            self.run_command(
                [sys.executable, "ingest/embed.py"],
                "Criação de índice vetorial",
                required=False
            )
        except ImportError:
            print("⚠️  Bibliotecas de embeddings não instaladas")
            print("   O sistema funcionará sem busca vetorial")
            print("   Para instalar: pip install sentence-transformers faiss-cpu")
        
        # 5. Resumo
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumo da execução."""
        self.print_header("RESUMO DA EXECUÇÃO", "📊")
        
        print("\n✅ ETAPAS CONCLUÍDAS:")
        for step in self.steps_completed:
            print(f"   • {step}")
        
        if self.errors:
            print("\n❌ ERROS ENCONTRADOS:")
            for error in self.errors:
                print(f"   • {error}")
        
        # Verifica arquivos criados
        chunks_file = Path("data/processed/chunks.json")
        if chunks_file.exists():
            print(f"\n✅ Sistema pronto para uso!")
            print("\n🚀 COMO USAR:")
            print("\n1. Sistema simplificado (sempre funciona):")
            print("   python rag_demo.py")
            
            if Path("faiss_index/ufcspa.index").exists():
                print("\n2. Sistema completo com embeddings:")
                print("   python query/interactive.py")
            
            print("\n3. Consulta rápida:")
            print('   python query/query.py "Sua pergunta aqui"')
        else:
            print("\n⚠️  Sistema parcialmente configurado")
            print("Use: python rag_demo.py (versão simplificada)")


def main():
    """Função principal."""
    pipeline = RAGPipeline()
    
    try:
        pipeline.run_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()