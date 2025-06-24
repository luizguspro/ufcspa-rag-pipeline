"""
Interface interativa para o sistema de consulta RAG da UFCSPA.

Este script fornece uma interface de linha de comando interativa
para fazer múltiplas consultas sem precisar reiniciar o sistema.
"""

import logging
import sys
from query import QueryEngine


# Configuração do logging para modo menos verboso
logging.basicConfig(level=logging.WARNING)


def print_banner():
    """Imprime o banner do sistema."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║         SISTEMA DE CONSULTA RAG - NORMAS UFCSPA              ║
    ║                                                               ║
    ║  Digite suas perguntas sobre as normas da universidade       ║
    ║  Comandos especiais:                                          ║
    ║    /help     - Mostra esta ajuda                            ║
    ║    /config   - Mostra configurações atuais                  ║
    ║    /k [num]  - Define número de resultados (ex: /k 3)       ║
    ║    /verbose  - Liga/desliga modo detalhado                  ║
    ║    /exit     - Sair do sistema                              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Imprime ajuda detalhada."""
    help_text = """
    🔍 COMO USAR O SISTEMA:
    
    1. Digite sua pergunta normalmente e pressione Enter
    2. O sistema buscará os documentos mais relevantes
    3. Um prompt será construído com o contexto encontrado
    
    📝 EXEMPLOS DE PERGUNTAS:
    - "Quais são as normas para atividades de extensão?"
    - "Como funciona o processo de matrícula?"
    - "Direitos e deveres dos docentes"
    - "Regimento interno sobre pesquisa"
    
    ⚙️ COMANDOS ESPECIAIS:
    /help     - Mostra esta mensagem de ajuda
    /config   - Exibe as configurações atuais
    /k 3      - Define busca para top 3 resultados
    /k 10     - Define busca para top 10 resultados  
    /verbose  - Ativa/desativa modo detalhado
    /clear    - Limpa a tela
    /exit     - Sair do programa
    """
    print(help_text)


def interactive_session():
    """Executa uma sessão interativa de consultas."""
    print_banner()
    
    # Configurações da sessão
    config = {
        'k': 5,
        'verbose': False
    }
    
    print("Inicializando sistema... ", end='', flush=True)
    try:
        engine = QueryEngine()
        print("✓ Pronto!\n")
    except Exception as e:
        print(f"\n❌ Erro ao inicializar: {e}")
        print("Certifique-se de que o índice FAISS foi criado.")
        return
    
    # Verifica status do Llama
    try:
        from query.llama_model import get_llama_model
        llama = get_llama_model()
        model_info = llama.get_model_info()
        
        if model_info['status'] == 'ready':
            print(f"🦙 Llama ativo: {model_info['model_name']}")
        else:
            print(f"⚠️  Llama: {model_info['status']}")
            if model_info['status'] == 'not_found':
                print("   Execute 'python download_llama_model.py' para baixar um modelo")
    except:
        print("⚠️  Módulo Llama não disponível")
    
    print("\nDigite /help para ver comandos disponíveis\n")
    
    while True:
        try:
            # Solicita entrada do usuário
            user_input = input("🤔 Sua pergunta: ").strip()
            
            if not user_input:
                continue
            
            # Processa comandos especiais
            if user_input.startswith('/'):
                command = user_input.lower().split()
                
                if command[0] == '/exit':
                    print("\n👋 Até logo!")
                    break
                
                elif command[0] == '/help':
                    print_help()
                    continue
                
                elif command[0] == '/config':
                    print(f"\n⚙️  Configurações atuais:")
                    print(f"   - Top-K: {config['k']}")
                    print(f"   - Modo verbose: {'Ativado' if config['verbose'] else 'Desativado'}\n")
                    continue
                
                elif command[0] == '/k' and len(command) > 1:
                    try:
                        new_k = int(command[1])
                        if 1 <= new_k <= 20:
                            config['k'] = new_k
                            print(f"✓ Top-K definido para {new_k}\n")
                        else:
                            print("❌ K deve estar entre 1 e 20\n")
                    except ValueError:
                        print("❌ Use /k seguido de um número (ex: /k 5)\n")
                    continue
                
                elif command[0] == '/verbose':
                    config['verbose'] = not config['verbose']
                    status = 'ativado' if config['verbose'] else 'desativado'
                    print(f"✓ Modo verbose {status}\n")
                    continue
                
                elif command[0] == '/clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_banner()
                    continue
                
                else:
                    print(f"❌ Comando desconhecido: {command[0]}")
                    print("Digite /help para ver comandos disponíveis\n")
                    continue
            
            # Processa a pergunta
            print("\n🔍 Buscando...", end='', flush=True)
            
            try:
                result = engine.query(user_input, k=config['k'])
                print(" ✓")
                
                # Mostra resultados
                print(f"\n📊 Encontrados {len(result['results'])} chunks relevantes")
                
                # Lista documentos únicos
                sources = list(set(r['source_file'] for r in result['results']))
                print(f"📄 De {len(sources)} documento(s):")
                for source in sources:
                    print(f"   • {source}")
                
                if config['verbose']:
                    # Modo detalhado: mostra informações de cada chunk
                    print("\n🔎 Detalhes dos chunks:")
                    for i, res in enumerate(result['results'], 1):
                        print(f"\n   [{i}] {res['source_file']} (Chunk {res['chunk_id']})")
                        print(f"       Score: {res['score']:.3f}")
                        print(f"       Preview: {res['text'][:150]}...")
                
                # Mostra resposta gerada ou contexto
                print("\n" + "-"*60)
                
                if 'answer' in result and result['answer']:
                    print("💬 RESPOSTA GERADA:")
                    print("-"*60)
                    print(result['answer'])
                else:
                    print("📝 CONTEXTO CONSTRUÍDO:")
                    print("-"*60)
                    context_preview = result['context'][:500]
                    if len(result['context']) > 500:
                        context_preview += "...\n[Contexto truncado para visualização]"
                    print(context_preview)
                    print("-"*60)
                    print("\n💡 Em um sistema completo, um LLM geraria a resposta aqui.")
                
                print()
                
            except Exception as e:
                print(f"\n❌ Erro ao processar consulta: {e}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Sessão interrompida. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro inesperado: {e}\n")


if __name__ == "__main__":
    interactive_session()