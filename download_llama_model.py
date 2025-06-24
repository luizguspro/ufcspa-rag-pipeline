"""
Script para baixar modelos Llama compatíveis com o sistema.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib


class LlamaDownloader:
    """Gerenciador de download de modelos Llama."""
    
    # Modelos recomendados (do menor para o maior)
    MODELS = {
        "tinyllama": {
            "name": "TinyLlama 1.1B Chat",
            "size": "665 MB",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "description": "Modelo ultra-leve, roda em qualquer PC"
        },
        "orca-mini": {
            "name": "Orca Mini 3B",
            "size": "1.9 GB",
            "url": "https://huggingface.co/TheBloke/orca_mini_3B-GGUF/resolve/main/orca-mini-3b.q4_0.gguf",
            "filename": "orca-mini-3b.q4_0.gguf",
            "description": "Modelo pequeno mas capaz"
        },
        "llama2-7b": {
            "name": "Llama 2 7B Chat",
            "size": "3.8 GB",
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            "filename": "llama-2-7b-chat.Q4_K_M.gguf",
            "description": "Modelo padrão Llama 2, ótimo equilíbrio"
        },
        "mistral-7b": {
            "name": "Mistral 7B Instruct",
            "size": "4.1 GB", 
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "description": "Modelo mais recente e eficiente"
        }
    }
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    def list_models(self):
        """Lista modelos disponíveis para download."""
        print("\n" + "="*70)
        print("MODELOS LLAMA DISPONÍVEIS")
        print("="*70)
        
        for key, model in self.MODELS.items():
            print(f"\n[{key}] {model['name']}")
            print(f"    Tamanho: {model['size']}")
            print(f"    Descrição: {model['description']}")
            
            # Verifica se já está baixado
            if (self.models_dir / model['filename']).exists():
                print(f"    Status: ✅ Já baixado")
            else:
                print(f"    Status: ⬇️  Disponível para download")
    
    def download_model(self, model_key: str):
        """Baixa um modelo específico."""
        if model_key not in self.MODELS:
            print(f"❌ Modelo '{model_key}' não encontrado!")
            return False
        
        model = self.MODELS[model_key]
        filepath = self.models_dir / model['filename']
        
        # Verifica se já existe
        if filepath.exists():
            print(f"✅ {model['name']} já está baixado!")
            return True
        
        print(f"\n📥 Baixando {model['name']} ({model['size']})...")
        print(f"URL: {model['url']}")
        
        try:
            # Faz o download com barra de progresso
            response = requests.get(model['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=model['filename']) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"\n✅ {model['name']} baixado com sucesso!")
            print(f"📁 Salvo em: {filepath}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Erro ao baixar: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove arquivo parcial
            return False
        except KeyboardInterrupt:
            print("\n⚠️  Download cancelado!")
            if filepath.exists():
                filepath.unlink()  # Remove arquivo parcial
            return False
    
    def check_installed_models(self):
        """Verifica quais modelos estão instalados."""
        installed = []
        
        for key, model in self.MODELS.items():
            filepath = self.models_dir / model['filename']
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                installed.append({
                    'key': key,
                    'name': model['name'],
                    'file': model['filename'],
                    'size': f"{size_mb:.1f} MB"
                })
        
        return installed
    
    def auto_download_smallest(self):
        """Baixa automaticamente o menor modelo disponível."""
        # TinyLlama é o menor
        if not (self.models_dir / self.MODELS['tinyllama']['filename']).exists():
            print("\n🤖 Baixando modelo TinyLlama automaticamente...")
            return self.download_model('tinyllama')
        return True


def main():
    """Interface principal do downloader."""
    downloader = LlamaDownloader()
    
    print("\n" + "="*70)
    print("🦙 GERENCIADOR DE MODELOS LLAMA - UFCSPA RAG")
    print("="*70)
    
    # Verifica modelos instalados
    installed = downloader.check_installed_models()
    
    if installed:
        print("\n✅ Modelos instalados:")
        for model in installed:
            print(f"   • {model['name']} ({model['size']})")
    else:
        print("\n⚠️  Nenhum modelo instalado!")
    
    # Menu
    while True:
        print("\n" + "-"*70)
        print("OPÇÕES:")
        print("-"*70)
        print("1. Listar todos os modelos disponíveis")
        print("2. Baixar TinyLlama (665 MB - Recomendado para teste)")
        print("3. Baixar Orca Mini 3B (1.9 GB)")
        print("4. Baixar Llama 2 7B (3.8 GB)")
        print("5. Baixar Mistral 7B (4.1 GB)")
        print("6. Download automático (menor modelo)")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            downloader.list_models()
        elif choice == "2":
            downloader.download_model("tinyllama")
        elif choice == "3":
            downloader.download_model("orca-mini")
        elif choice == "4":
            downloader.download_model("llama2-7b")
        elif choice == "5":
            downloader.download_model("mistral-7b")
        elif choice == "6":
            downloader.auto_download_smallest()
        else:
            print("❌ Opção inválida!")
    
    print("\n👋 Até logo!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operação cancelada!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")