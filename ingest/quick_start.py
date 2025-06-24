"""
Script rápido para configurar e testar o sistema RAG com dados de exemplo.
Execute este script para ter o sistema funcionando rapidamente.
"""

import os
import sys
from pathlib import Path

# Adiciona o diretório ao path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def run_command(command, description):
    """Executa um comando Python."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    result = os.system(f"{sys.executable} {command}")
    
    if result == 0:
        print(f"✓ {description} - Concluído!")
        return True
    else:
        print(f"✗ {description} - Falhou!")
        return False


def main():
    """Executa o setup rápido do sistema."""
    print("QUICK START - Sistema RAG UFCSPA")
    print("="*60)
    
    steps = [
        ("create_simple_pdfs.py", "Criando arquivos de exemplo"),
        ("ingest/chunk.py", "Dividindo textos em chunks"),
        ("ingest/embed.py", "Gerando embeddings e índice FAISS"),
        ("test_system.py", "Testando o sistema")
    ]
    
    all_success = True
    
    for script, description in steps:
        if not run_command(script, description):
            all_success = False
            print(f"\n❌ Erro na etapa: {description}")
            print("Verifique as mensagens de erro acima.")
            
            # Se falhou no embed.py, pode ser que o arquivo não exista
            if "embed.py" in script:
                print("\n💡 Dica: O arquivo embed.py pode não ter sido criado.")
                print("Você pode pular esta etapa por enquanto e usar apenas a busca por texto.")
            
            break
    
    if all_success:
        print("\n" + "="*60)
        print("✅ SISTEMA CONFIGURADO COM SUCESSO!")
        print("="*60)
        print("\n🚀 Para usar o sistema:")
        print("  python query/interactive.py")
        print("\n📝 Exemplos de perguntas:")
        print('  - "Quais são as normas para atividades de extensão?"')
        print('  - "Como funciona o Conselho Universitário?"')
        print('  - "Quais são os princípios da UFCSPA?"')
    else:
        print("\n" + "="*60)
        print("⚠️  Setup incompleto")
        print("="*60)
        print("\nVocê ainda pode testar partes do sistema:")
        print("  - Verifique se os chunks foram criados: dir data\\processed")
        print("  - Tente executar cada script individualmente")


if __name__ == "__main__":
    main()