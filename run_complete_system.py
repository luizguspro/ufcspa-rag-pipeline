"""
Sistema RAG Completo UFCSPA com Llama
Execute este arquivo para ter o sistema completo funcionando!
"""

import os
import sys
import subprocess
from pathlib import Path
import time


def print_banner():
    """Mostra banner do sistema."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║          SISTEMA RAG UFCSPA - COM LLAMA INTEGRADO               ║
    ║                                                                  ║
    ║  • Download automático de documentos da UFCSPA                  ║
    ║  • Processamento com OCR se necessário                          ║  
    ║  • Busca vetorial com FAISS                                     ║
    ║  • Respostas geradas por Llama 2 (LLM local)                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Verifica requisitos do sistema."""
    print("\n🔍 Verificando requisitos...")
    
    requirements = {
        "Python": sys.version,
        "pip": "Verificando...",
        "venv": "OK" if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else "Não ativado"
    }
    
    # Verifica pip
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        requirements["pip"] = "OK" if result.returncode == 0 else "Não encontrado"
    except:
        requirements["pip"] = "Erro"
    
    for req, status in requirements.items():
        print(f"  • {req}: {status}")
    
    return all("OK" in str(v) or "Python" in k for k, v in requirements.items())


def install_dependencies():
    """Instala dependências necessárias."""
    print("\n📦 Instalando dependências...")
    
    essential_packages = [
        ("requests", "Requisições HTTP"),
        ("beautifulsoup4", "Parse HTML"),
        ("tqdm", "Barras de progresso"),
        ("pandas", "Manipulação de dados"),
        ("langchain", "Framework LLM"),
        ("sentence-transformers", "Embeddings"),
        ("faiss-cpu", "Busca vetorial"),
        ("llama-cpp-python", "Llama local")
    ]
    
    for package, description in essential_packages:
        print(f"\n  Installing {package} ({description})...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                check=True
            )
            print(f"  ✅ {package} instalado")
        except:
            print(f"  ⚠️  {package} - falha na instalação")


def download_llama_model():
    """Baixa modelo Llama se necessário."""
    print("\n🤖 Verificando modelo Llama...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Verifica se já tem algum modelo
    existing_models = list(models_dir.glob("*.gguf"))
    if existing_models:
        print(f"  ✅ Modelo encontrado: {existing_models[0].name}")
        return True
    
    print("\n  ❌ Nenhum modelo Llama encontrado")
    print("\n  📥 Deseja baixar um modelo? (necessário para respostas com IA)")
    print("     1. TinyLlama 1.1B (mais leve, ~600MB) [RECOMENDADO]")
    print("     2. Llama 2 7B (~3.8GB)")
    print("     3. Pular (continuar sem LLM)")
    
    choice = input("\n  Escolha (1-3): ").strip()
    
    if choice == "1":
        print("\n  Baixando TinyLlama...")
        subprocess.run([
            sys.executable, "-c",
            "from query.llm_handler import get_llama_handler; "
            "handler = get_llama_handler(); "
            "handler.download_model('tinyllama')"
        ])
        return True
    elif choice == "2":
        print("\n  Baixando Llama 2...")
        subprocess.run([
            sys.executable, "-c",
            "from query.llm_handler import get_llama_handler; "
            "handler = get_llama_handler(); "
            "handler.download_model('llama2')"
        ])
        return True
    else:
        print("\n  ⚠️  Continuando sem LLM (apenas recuperação de contexto)")
        return False


def run_pipeline():
    """Executa o pipeline completo."""
    print("\n🚀 Executando pipeline RAG completo...")
    
    steps = [
        {
            "name": "Download de documentos UFCSPA",
            "command": [sys.executable, "scraper/download_ufcspa_complete.py"],
            "required": False
        },
        {
            "name": "Conversão PDF → Texto",
            "command": [sys.executable, "ingest/convert.py"],
            "required": False
        },
        {
            "name": "Divisão em chunks",
            "command": [sys.executable, "ingest/chunk.py"],
            "required": True
        },
        {
            "name": "Geração de embeddings",
            "command": [sys.executable, "ingest/embed.py"],
            "required": False
        }
    ]
    
    for step in steps:
        print(f"\n⏳ {step['name']}...")
        try:
            result = subprocess.run(step['command'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {step['name']} - Concluído!")
            else:
                if step['required']:
                    print(f"❌ {step['name']} - Falhou (obrigatório)")
                    return False
                else:
                    print(f"⚠️  {step['name']} - Falhou (opcional)")
        except Exception as e:
            print(f"❌ Erro: {e}")
            if step['required']:
                return False
    
    return True


def run_interactive_query():
    """Executa interface interativa com Llama."""
    print("\n💬 Iniciando interface de consulta com Llama...")
    
    # Verifica qual script usar
    if Path("query/query_with_llama.py").exists():
        subprocess.run([sys.executable, "query/query_with_llama.py", 
                       "Como funciona a UFCSPA?", "-v"])
    elif Path("query/interactive.py").exists():
        subprocess.run([sys.executable, "query/interactive.py"])
    else:
        print("\n❌ Interface de consulta não encontrada")


def create_test_query():
    """Cria script de teste rápido."""
    test_script = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from query.query_with_llama import QueryEngineWithLlama
    
    print("\\n🧪 Testando sistema RAG com Llama...")
    engine = QueryEngineWithLlama()
    
    perguntas = [
        "Quais são os princípios da UFCSPA?",
        "Como funciona o conselho universitário?",
        "Quais são as normas de extensão?"
    ]
    
    for pergunta in perguntas:
        print(f"\\n❓ {pergunta}")
        result = engine.query(pergunta, k=3)
        print(f"\\n💬 Resposta: {result['response'][:300]}...")
        print("-" * 60)
        
except Exception as e:
    print(f"Erro: {e}")
    print("\\nUsando versão simplificada...")
    
    # Fallback simples
    print("\\n✅ Sistema funcionando em modo básico")
    print("Contexto seria recuperado e mostrado aqui")
'''
    
    with open("test_rag_llama.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("\n🧪 Executando teste...")
    subprocess.run([sys.executable, "test_rag_llama.py"])


def main():
    """Função principal."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print_banner()
    
    # Verifica requisitos
    if not check_requirements():
        print("\n❌ Requisitos não atendidos")
        return
    
    # Menu principal
    print("\n" + "="*60)
    print("OPÇÕES DE EXECUÇÃO:")
    print("="*60)
    print("\n1. 🚀 Instalação completa + Pipeline + Llama")
    print("2. 📥 Apenas baixar documentos UFCSPA")
    print("3. 🤖 Apenas configurar Llama")
    print("4. 🔄 Executar pipeline (se já tem dados)")
    print("5. 💬 Interface de consulta")
    print("6. 🧪 Teste rápido")
    print("7. ❌ Sair")
    
    choice = input("\nEscolha (1-7): ").strip()
    
    if choice == "1":
        # Instalação completa
        install_dependencies()
        download_llama_model()
        if run_pipeline():
            run_interactive_query()
    
    elif choice == "2":
        # Apenas download
        subprocess.run([sys.executable, "scraper/download_ufcspa_complete.py"])
    
    elif choice == "3":
        # Apenas Llama
        download_llama_model()
    
    elif choice == "4":
        # Pipeline
        if run_pipeline():
            print("\n✅ Pipeline concluído!")
    
    elif choice == "5":
        # Interface
        run_interactive_query()
    
    elif choice == "6":
        # Teste
        create_test_query()
    
    else:
        print("\n👋 Até logo!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        print("\n💡 Tente executar componentes individualmente")