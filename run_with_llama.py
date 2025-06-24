"""
Sistema RAG UFCSPA com Llama integrado.
Este script configura e executa o sistema completo com geração local.
"""

import os
import sys
import subprocess
from pathlib import Path
import time


def print_banner():
    """Mostra banner do sistema."""
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║           SISTEMA RAG UFCSPA - COM LLAMA LOCAL                 ║
    ║                                                                ║
    ║  🦙 Geração de respostas com IA rodando no seu computador     ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_llama_cpp():
    """Verifica se llama-cpp-python está instalado."""
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


def install_llama_cpp():
    """Instala llama-cpp-python."""
    print("\n📦 Instalando llama-cpp-python...")
    print("Isso pode demorar alguns minutos na primeira vez...")
    
    try:
        # Tenta instalação padrão
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ llama-cpp-python instalado com sucesso!")
            return True
        else:
            print("⚠️  Instalação padrão falhou, tentando build local...")
            
            # Tenta com flags específicas para Windows
            if sys.platform == "win32":
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--force-reinstall"],
                    capture_output=True,
                    text=True
                )
            
            return result.returncode == 0
            
    except Exception as e:
        print(f"❌ Erro na instalação: {e}")
        return False


def check_models():
    """Verifica se há modelos Llama disponíveis."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    gguf_files = list(models_dir.glob("*.gguf"))
    
    if gguf_files:
        print(f"\n✅ Modelos encontrados: {len(gguf_files)}")
        for model in gguf_files[:3]:  # Mostra até 3
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   • {model.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("\n⚠️  Nenhum modelo Llama encontrado!")
        return False


def download_model():
    """Baixa um modelo Llama."""
    print("\n" + "="*60)
    print("📥 DOWNLOAD DE MODELO LLAMA")
    print("="*60)
    
    # Verifica se o script de download existe
    if Path("download_llama_model.py").exists():
        subprocess.run([sys.executable, "download_llama_model.py"])
    else:
        print("\nBaixando TinyLlama (modelo mais leve)...")
        
        # Download direto simplificado
        try:
            import requests
            from tqdm import tqdm
            
            url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            filename = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            
            Path("models").mkdir(exist_ok=True)
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print("\n✅ Modelo baixado com sucesso!")
            
        except Exception as e:
            print(f"\n❌ Erro no download: {e}")
            print("\nAlternativas:")
            print("1. Baixe manualmente de: https://huggingface.co/TheBloke")
            print("2. Use o sistema sem Llama (modo fallback)")


def test_llama():
    """Testa se o Llama está funcionando."""
    print("\n🧪 Testando Llama...")
    
    try:
        from query.llama_model import get_llama_model
        
        llama = get_llama_model()
        info = llama.get_model_info()
        
        print(f"Status: {info['status']}")
        
        if info['status'] == 'ready':
            print(f"Modelo: {info['model_name']}")
            print("\n✅ Llama está pronto para uso!")
            return True
        else:
            print(f"Mensagem: {info.get('message', 'Modelo não carregado')}")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar Llama: {e}")
        return False


def run_system():
    """Executa o sistema RAG com Llama."""
    print("\n🚀 Iniciando sistema RAG com Llama...")
    
    # Verifica se tem dados
    if not Path("data/processed/chunks.json").exists():
        print("\n⚠️  Dados não encontrados! Criando dados de exemplo...")
        
        # Cria dados de exemplo
        subprocess.run([sys.executable, "create_sample_data.py"])
        subprocess.run([sys.executable, "ingest/chunk.py"])
    
    # Executa interface
    if Path("query/interactive.py").exists():
        print("\n✅ Sistema pronto! Iniciando interface...")
        subprocess.run([sys.executable, "query/interactive.py"])
    else:
        print("\n❌ Interface não encontrada!")
        print("Execute: python rag_demo.py")


def main():
    """Função principal."""
    print_banner()
    
    # 1. Verifica llama-cpp-python
    print("\n[1/4] Verificando llama-cpp-python...")
    if not check_llama_cpp():
        if not install_llama_cpp():
            print("\n⚠️  Continuando sem llama-cpp-python...")
            print("O sistema funcionará em modo fallback")
    else:
        print("✅ llama-cpp-python está instalado")
    
    # 2. Verifica modelos
    print("\n[2/4] Verificando modelos Llama...")
    if not check_models():
        response = input("\nDeseja baixar um modelo agora? (s/n): ").lower()
        if response == 's':
            download_model()
    
    # 3. Testa Llama
    print("\n[3/4] Configurando Llama...")
    llama_ok = test_llama()
    
    # 4. Executa sistema
    print("\n[4/4] Executando sistema...")
    
    if llama_ok:
        print("\n🎉 Sistema RAG com Llama local ativo!")
    else:
        print("\n⚠️  Sistema RAG em modo fallback (sem Llama)")
    
    time.sleep(2)
    run_system()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação cancelada!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("\nExecute o sistema básico: python rag_demo.py")