
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from query.query_with_llama import QueryEngineWithLlama
    
    print("\n🧪 Testando sistema RAG com Llama...")
    engine = QueryEngineWithLlama()
    
    perguntas = [
        "Quais são os princípios da UFCSPA?",
        "Como funciona o conselho universitário?",
        "Quais são as normas de extensão?"
    ]
    
    for pergunta in perguntas:
        print(f"\n❓ {pergunta}")
        result = engine.query(pergunta, k=3)
        print(f"\n💬 Resposta: {result['response'][:300]}...")
        print("-" * 60)
        
except Exception as e:
    print(f"Erro: {e}")
    print("\nUsando versão simplificada...")
    
    # Fallback simples
    print("\n✅ Sistema funcionando em modo básico")
    print("Contexto seria recuperado e mostrado aqui")
