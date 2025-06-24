"""
🚀 SISTEMA RAG UFCSPA - COMECE AQUI!

Este é o ponto de entrada principal do sistema.
Execute este arquivo para ter tudo funcionando!
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║           SISTEMA RAG UFCSPA - COMPLETO COM LLAMA              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Este sistema implementa:                                      ║
║  • Download automático de normas da UFCSPA ✓                  ║
║  • Processamento de PDFs com OCR ✓                            ║
║  • Busca vetorial com embeddings ✓                            ║
║  • Respostas com Llama 2 (IA local) ✓                         ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🎯 O QUE VOCÊ QUER FAZER?\n")
    print("1. 🚀 Executar demonstração rápida (2 min)")
    print("2. 📦 Instalação completa + Demo (10 min)")
    print("3. 🤖 Baixar Llama + Testar IA (5 min)")
    print("4. 📥 Baixar documentos UFCSPA")
    print("5. 💬 Interface de perguntas")
    print("6. 📚 Ver documentação")
    print("7. ❌ Sair")
    
    choice = input("\nEscolha (1-7): ").strip()
    
    if choice == "1":
        # Demo rápida
        print("\n🚀 Executando demonstração...")
        demo_code = '''
print("\\n=== DEMONSTRAÇÃO SISTEMA RAG ===\\n")

# Dados de exemplo
docs = {
    "Estatuto": "A UFCSPA tem como princípios: excelência acadêmica, formação humanística, compromisso social e gestão democrática.",
    "Extensão": "As atividades de extensão incluem programas, projetos, cursos, eventos e prestação de serviços à comunidade.",
    "Regimento": "O Conselho Universitário (CONSUN) é o órgão máximo. O Reitor é a autoridade executiva."
}

perguntas = [
    "Quais os princípios da UFCSPA?",
    "O que são atividades de extensão?",
    "Qual o órgão máximo da universidade?"
]

import time
for i, pergunta in enumerate(perguntas, 1):
    print(f"\\n[{i}] Pergunta: {pergunta}")
    time.sleep(1)
    
    # Busca simples
    for titulo, conteudo in docs.items():
        palavras = pergunta.lower().split()
        if any(p in conteudo.lower() for p in palavras):
            print(f"\\n📄 Documento relevante: {titulo}")
            print(f"Conteúdo: {conteudo}")
            print("\\n💬 Resposta (com Llama seria mais elaborada):")
            print(f"Com base no {titulo}, {conteudo.split('.')[0].lower()}.")
            break

print("\\n✅ Demo concluída! O sistema completo usa IA para respostas melhores.")
'''
        exec(demo_code)
        
    elif choice == "2":
        # Instalação completa
        print("\n📦 Iniciando instalação completa...")
        if Path("run_complete_system.py").exists():
            subprocess.run([sys.executable, "run_complete_system.py"])
        else:
            print("Instalando dependências básicas...")
            subprocess.run([sys.executable, "-m", "pip", "install", "requests", "beautifulsoup4", "tqdm"])
            print("\n✅ Dependências instaladas!")
    
    elif choice == "3":
        # Llama
        print("\n🤖 Configurando Llama...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
            print("\n✅ Llama instalado! Baixe um modelo em:")
            print("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        except:
            print("⚠️ Erro ao instalar Llama")
    
    elif choice == "4":
        # Download UFCSPA
        if Path("scraper/download_ufcspa_complete.py").exists():
            subprocess.run([sys.executable, "scraper/download_ufcspa_complete.py"])
        else:
            print("⚠️ Script de download não encontrado")
    
    elif choice == "5":
        # Interface
        scripts = ["query/query_with_llama.py", "query/interactive.py", "rag_demo.py"]
        for script in scripts:
            if Path(script).exists():
                subprocess.run([sys.executable, script])
                break
        else:
            print("⚠️ Interface não encontrada")
    
    elif choice == "6":
        # Documentação
        print("""
📚 DOCUMENTAÇÃO RÁPIDA

1. ESTRUTURA DO PROJETO:
   scraper/     - Download de documentos
   ingest/      - Processamento (PDF→Texto→Chunks)
   query/       - Interface de perguntas
   models/      - Modelos Llama

2. ARQUIVOS IMPORTANTES:
   • llm_handler.py - Integração com Llama
   • embed.py - Criação de embeddings
   • download_ufcspa_complete.py - Download com SSL fix

3. COMO FUNCIONA:
   Pergunta → Busca documentos → Contexto → Llama → Resposta

4. PARA MAIS DETALHES:
   Veja SISTEMA_COMPLETO.md
        """)
    
    else:
        print("\n👋 Até logo!")
        return
    
    # Oferece continuar
    print("\n" + "-"*60)
    input("\nPressione Enter para voltar ao menu...")
    main()


if __name__ == "__main__":
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Até logo!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("\n💡 Tente as opções mais simples primeiro (1 ou 6)")