"""
SCRIPT PRINCIPAL - SISTEMA RAG UFCSPA
Execute este arquivo para usar o sistema!
"""

import os
import sys
import subprocess
from pathlib import Path


def clear_screen():
    """Limpa a tela."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Mostra banner do sistema."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║              SISTEMA RAG - NORMAS UFCSPA                   ║
    ║                                                            ║
    ║  Recuperação e Geração Aumentada de Informações           ║
    ║  Universidade Federal de Ciências da Saúde de Porto Alegre║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_and_run():
    """Verifica o ambiente e executa o sistema apropriado."""
    
    print("\n🔍 Verificando sistema...\n")
    
    # Verifica Llama
    llama_status = "❌ Não instalado"
    try:
        from query.llama_model import get_llama_model
        llama = get_llama_model()
        info = llama.get_model_info()
        if info['status'] == 'ready':
            llama_status = f"✅ Pronto ({info['model_name']})"
        elif info['status'] == 'not_found':
            llama_status = "⚠️  Sem modelo (baixe com download_llama_model.py)"
        else:
            llama_status = f"⚠️  {info['status']}"
    except:
        pass
    
    print(f"🦙 Llama: {llama_status}")
    
    # Verifica se os arquivos principais existem
    files_to_check = {
        "rag_demo.py": "Demo simplificada (sempre funciona)",
        "run_all.py": "Pipeline completo",
        "run_with_llama.py": "Sistema com Llama integrado",
        "query/interactive.py": "Interface interativa",
        "setup_complete.py": "Setup automático"
    }
    
    available_options = []
    
    for file, description in files_to_check.items():
        if Path(file).exists():
            available_options.append((file, description))
            print(f"✅ {description}: {file}")
        else:
            print(f"❌ {description}: {file} (não encontrado)")
    
    if not available_options:
        print("\n❌ Nenhum arquivo do sistema encontrado!")
        print("\n📝 Criando sistema mínimo...")
        create_minimal_system()
        return
    
    # Menu de opções
    print("\n" + "="*60)
    print("OPÇÕES DISPONÍVEIS:")
    print("="*60)
    
    print("\n1. 🚀 Demo Rápida (Recomendado para primeira vez)")
    print("2. 🦙 Sistema com Llama (IA local)")
    print("3. 🔧 Setup Completo + Execução")
    print("4. 📊 Pipeline Completo (se já configurado)")
    print("5. 💬 Interface Interativa (se disponível)")
    print("6. ❌ Sair")
    
    while True:
        choice = input("\nEscolha uma opção (1-6): ").strip()
        
        if choice == "1":
            # Tenta executar demo
            if Path("rag_demo.py").exists():
                print("\n🚀 Iniciando demo...")
                subprocess.run([sys.executable, "rag_demo.py"])
            else:
                print("\n📝 Criando e executando demo...")
                create_minimal_system()
            break
            
        elif choice == "2":
            # Sistema com Llama
            if Path("run_with_llama.py").exists():
                print("\n🦙 Iniciando sistema com Llama...")
                subprocess.run([sys.executable, "run_with_llama.py"])
            else:
                print("\n❌ Script do Llama não encontrado")
            break
            
        elif choice == "3":
            # Setup completo
            if Path("setup_complete.py").exists():
                print("\n🔧 Executando setup completo...")
                subprocess.run([sys.executable, "setup_complete.py"])
                print("\n✅ Setup concluído! Executando sistema...")
                if Path("run_all.py").exists():
                    subprocess.run([sys.executable, "run_all.py"])
            else:
                print("\n❌ Arquivo de setup não encontrado")
            break
            
        elif choice == "4":
            # Pipeline completo
            if Path("run_all.py").exists():
                print("\n📊 Executando pipeline completo...")
                subprocess.run([sys.executable, "run_all.py"])
            else:
                print("\n❌ Pipeline não encontrado")
            break
            
        elif choice == "5":
            # Interface interativa
            if Path("query/interactive.py").exists():
                print("\n💬 Iniciando interface interativa...")
                subprocess.run([sys.executable, "query/interactive.py"])
            else:
                print("\n❌ Interface não encontrada")
            break
            
        elif choice == "6":
            print("\n👋 Até logo!")
            break
            
        else:
            print("❌ Opção inválida! Escolha entre 1-6")


def create_minimal_system():
    """Cria um sistema mínimo funcional."""
    
    minimal_code = '''
import re

print("\\n" + "="*60)
print("SISTEMA RAG MINIMO - UFCSPA")
print("="*60)

# Dados de exemplo
docs = [
    {
        "titulo": "Estatuto UFCSPA",
        "conteudo": "A UFCSPA tem como princípios: excelência acadêmica, formação humanística, compromisso social, gestão democrática e autonomia universitária."
    },
    {
        "titulo": "Normas de Extensão",
        "conteudo": "As atividades de extensão incluem: programas, projetos, cursos, eventos e prestação de serviços. Devem ter relevância social e participação estudantil."
    },
    {
        "titulo": "Regimento Interno",
        "conteudo": "A administração é exercida pelo Conselho Universitário (CONSUN), CONSEPE e Reitoria. O Reitor é a autoridade executiva máxima."
    }
]

print("\\n✅ Sistema carregado com 3 documentos de exemplo")
print("\\n💡 Digite 'sair' para encerrar")

while True:
    pergunta = input("\\n❓ Sua pergunta: ").strip()
    
    if pergunta.lower() in ['sair', 'exit']:
        print("\\n👋 Até logo!")
        break
    
    if not pergunta:
        continue
    
    # Busca simples
    palavras = re.findall(r'\\w+', pergunta.lower())
    resultados = []
    
    for doc in docs:
        score = sum(1 for p in palavras if p in doc["conteudo"].lower())
        if score > 0:
            resultados.append((doc, score))
    
    if resultados:
        resultados.sort(key=lambda x: x[1], reverse=True)
        print(f"\\n📄 Encontrei {len(resultados)} documento(s) relevante(s):")
        for doc, score in resultados:
            print(f"\\n• {doc['titulo']} (relevância: {score})")
            print(f"  {doc['conteudo'][:100]}...")
    else:
        print("\\n❌ Nenhum documento relevante encontrado")
'''
    
    # Salva o código mínimo
    with open("sistema_minimo.py", "w", encoding="utf-8") as f:
        f.write(minimal_code)
    
    print("✅ Sistema mínimo criado!")
    
    # Executa
    subprocess.run([sys.executable, "sistema_minimo.py"])


def main():
    """Função principal."""
    try:
        clear_screen()
        print_banner()
        check_and_run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida")
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        print("\n💡 Tente executar diretamente:")
        print("   python rag_demo.py")
        print("   python run_all.py")


if __name__ == "__main__":
    main()