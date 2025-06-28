"""
Script para testar a configuração antes de processar arquivos.
"""

import os
import sys
from pathlib import Path

def test_config():
    """Testa se a configuração está correta."""
    print("🔍 TESTE DE CONFIGURAÇÃO")
    print("=" * 50)
    
    # Verifica se .env existe
    if not Path(".env").exists():
        print("❌ Arquivo .env não encontrado!")
        print("\n📝 Crie um arquivo .env com:")
        print('OPENAI_API_KEY="sua-chave-aqui"')
        print('PINECONE_API_KEY="sua-chave-aqui"')
        print('PINECONE_INDEX_NAME="ufcspa-index"')
        return False
    
    # Importa config
    try:
        import config
        print("✅ Arquivo config.py importado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao importar config.py: {e}")
        return False
    
    # Verifica variáveis
    checks = {
        "OPENAI_API_KEY": config.OPENAI_API_KEY,
        "PINECONE_API_KEY": config.PINECONE_API_KEY,
        "PINECONE_INDEX_NAME": config.PINECONE_INDEX_NAME
    }
    
    all_ok = True
    for name, value in checks.items():
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"✅ {name}: {masked}")
        else:
            print(f"❌ {name}: Não configurado")
            all_ok = False
    
    # Testa imports
    print("\n📦 Testando dependências...")
    
    try:
        import requests
        print("✅ requests instalado")
    except:
        print("❌ requests não instalado")
        all_ok = False
    
    try:
        from pinecone import Pinecone
        print("✅ pinecone-client instalado")
        
        # Testa conexão Pinecone
        if config.PINECONE_API_KEY:
            try:
                pc = Pinecone(api_key=config.PINECONE_API_KEY)
                index = pc.Index(config.PINECONE_INDEX_NAME)
                stats = index.describe_index_stats()
                print(f"✅ Conectado ao Pinecone! Vetores no índice: {stats.get('total_vector_count', 0)}")
            except Exception as e:
                print(f"⚠️  Erro ao conectar ao Pinecone: {e}")
                print("   Verifique se o índice existe e o nome está correto")
    except:
        print("❌ pinecone-client não instalado")
        all_ok = False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv instalado")
    except:
        print("❌ python-dotenv não instalado")
        all_ok = False
    
    try:
        from tqdm import tqdm
        print("✅ tqdm instalado")
    except:
        print("❌ tqdm não instalado")
        all_ok = False
    
    # Testa OpenAI
    if config.OPENAI_API_KEY:
        print("\n🧪 Testando OpenAI API...")
        try:
            import requests
            response = requests.post(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                timeout=5
            )
            if response.status_code == 200:
                print("✅ OpenAI API funcionando!")
            else:
                print(f"⚠️  OpenAI API retornou status: {response.status_code}")
        except Exception as e:
            print(f"❌ Erro ao testar OpenAI: {e}")
    
    return all_ok

def main():
    if test_config():
        print("\n✅ CONFIGURAÇÃO OK! Você pode executar:")
        print("   python ingest_final.py --file data/processed/0a1efcf5_Resultado_Consulta__Comunidade_2024_-_Discentes.txt")
    else:
        print("\n❌ Corrija os problemas acima antes de continuar")
        print("\n💡 Para instalar dependências:")
        print("   python install_deps.py")

if __name__ == "__main__":
    main()