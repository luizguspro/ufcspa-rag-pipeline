#!/usr/bin/env python3
"""
Simulador de uso em produção - Testa a ferramenta como o CrewAI usaria.

Execute este script para simular chamadas reais à ferramenta de busca.
"""

import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Any
import random

# Importa a ferramenta
from search_tool import search_vectorstore, VectorSearchTool


class CrewAISimulator:
    """Simula como o CrewAI chamaria a ferramenta."""
    
    def __init__(self):
        self.call_history = []
        self.start_time = time.time()
        
    def log_call(self, query: str, results: List[str], duration: float):
        """Registra uma chamada para análise."""
        self.call_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'results_count': len(results),
            'duration': duration,
            'success': len(results) > 0
        })
    
    def simulate_agent_call(self, query: str, agent_name: str = "Pesquisador"):
        """Simula uma chamada de um agente CrewAI."""
        print(f"\n{'='*60}")
        print(f"🤖 AGENTE: {agent_name}")
        print(f"{'='*60}")
        print(f"📝 Query: {query}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        # Marca tempo
        start = time.time()
        
        try:
            # FAZ A CHAMADA REAL - exatamente como CrewAI faria
            results = search_vectorstore(query)
            
            duration = time.time() - start
            self.log_call(query, results, duration)
            
            # Mostra resultados
            print(f"\n✅ Sucesso! {len(results)} resultados em {duration:.2f}s")
            
            if results:
                print("\n📄 RESULTADOS:")
                for i, text in enumerate(results, 1):
                    print(f"\n[Resultado {i}]")
                    # Mostra primeiros 300 caracteres
                    preview = text[:300] + "..." if len(text) > 300 else text
                    print(preview)
                    print(f"(Total: {len(text)} caracteres)")
            else:
                print("\n⚠️  Nenhum resultado encontrado")
                
            return results
            
        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            print(f"Tipo: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return []
    
    def show_statistics(self):
        """Mostra estatísticas das chamadas."""
        if not self.call_history:
            return
        
        print(f"\n\n{'='*60}")
        print("📊 ESTATÍSTICAS DA SESSÃO")
        print(f"{'='*60}")
        
        total_calls = len(self.call_history)
        successful_calls = sum(1 for c in self.call_history if c['success'])
        total_duration = sum(c['duration'] for c in self.call_history)
        avg_duration = total_duration / total_calls if total_calls > 0 else 0
        
        print(f"Total de chamadas: {total_calls}")
        print(f"Chamadas bem-sucedidas: {successful_calls} ({successful_calls/total_calls*100:.1f}%)")
        print(f"Tempo total: {total_duration:.2f}s")
        print(f"Tempo médio por chamada: {avg_duration:.2f}s")
        
        if successful_calls > 0:
            avg_results = sum(c['results_count'] for c in self.call_history if c['success']) / successful_calls
            print(f"Média de resultados por busca: {avg_results:.1f}")


def test_basic_functionality():
    """Testa funcionalidade básica da ferramenta."""
    print("\n🧪 TESTE 1: Funcionalidade Básica")
    print("-" * 60)
    
    sim = CrewAISimulator()
    
    # Teste simples
    results = sim.simulate_agent_call(
        "Quais são as normas de extensão da UFCSPA?",
        "Agente Teste Básico"
    )
    
    return len(results) > 0


def test_production_scenarios():
    """Testa cenários reais de produção."""
    print("\n\n🎯 TESTE 2: Cenários de Produção")
    print("=" * 60)
    
    sim = CrewAISimulator()
    
    # Cenários que o CrewAI usaria
    production_queries = [
        {
            'agent': 'Pesquisador de Normas',
            'query': 'Estatuto e regimento interno da UFCSPA'
        },
        {
            'agent': 'Analista de Políticas',
            'query': 'Normas sobre progressão docente e carreira'
        },
        {
            'agent': 'Assistente Acadêmico',
            'query': 'Regulamento do Conselho Universitário CONSUN'
        },
        {
            'agent': 'Pesquisador de Extensão',
            'query': 'Modalidades de atividades de extensão universitária'
        },
        {
            'agent': 'Analista de Resoluções',
            'query': 'Resoluções do CONSEPE sobre ensino'
        }
    ]
    
    # Simula múltiplas chamadas como em produção
    for scenario in production_queries:
        results = sim.simulate_agent_call(
            scenario['query'],
            scenario['agent']
        )
        time.sleep(0.5)  # Simula delay entre chamadas
    
    # Mostra estatísticas
    sim.show_statistics()


def test_stress_and_edge_cases():
    """Testa casos extremos e stress."""
    print("\n\n⚡ TESTE 3: Stress e Edge Cases")
    print("=" * 60)
    
    sim = CrewAISimulator()
    
    edge_cases = [
        # Query vazia
        "",
        # Query muito curta
        "UFCSPA",
        # Query muito longa
        "Preciso de informações detalhadas sobre todas as normas, regulamentos, portarias, resoluções, instruções normativas e demais documentos legais relacionados às atividades de pesquisa, ensino e extensão da Universidade Federal de Ciências da Saúde de Porto Alegre, incluindo mas não limitado a progressão funcional, concursos públicos, afastamentos, licenças e benefícios",
        # Query com caracteres especiais
        "Normas@UFCSPA#2024!",
        # Query em inglês (não deve retornar nada relevante)
        "What are the university regulations?",
        # Query específica demais
        "Artigo 42 parágrafo 3 do regimento interno",
    ]
    
    print("\n🔥 Testando casos extremos...")
    
    for query in edge_cases:
        print(f"\n{'='*60}")
        print(f"Edge case: {query[:50]}...")
        try:
            start = time.time()
            results = search_vectorstore(query)
            duration = time.time() - start
            print(f"✅ Tratado com sucesso em {duration:.2f}s - {len(results)} resultados")
        except Exception as e:
            print(f"❌ Erro (esperado?): {type(e).__name__}: {e}")


def test_performance():
    """Testa performance e consistência."""
    print("\n\n⚡ TESTE 4: Performance e Consistência")
    print("=" * 60)
    
    query_test = "Normas e regulamentos da UFCSPA"
    iterations = 5
    
    print(f"Executando {iterations} vezes a mesma query...")
    print(f"Query: '{query_test}'")
    print("-" * 60)
    
    times = []
    result_counts = []
    
    for i in range(iterations):
        start = time.time()
        results = search_vectorstore(query_test)
        duration = time.time() - start
        
        times.append(duration)
        result_counts.append(len(results))
        
        print(f"Iteração {i+1}: {duration:.3f}s - {len(results)} resultados")
    
    # Análise
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Análise de Performance:")
    print(f"Tempo médio: {avg_time:.3f}s")
    print(f"Tempo mínimo: {min_time:.3f}s")
    print(f"Tempo máximo: {max_time:.3f}s")
    print(f"Variação: {max_time - min_time:.3f}s")
    
    # Verifica consistência
    if len(set(result_counts)) == 1:
        print(f"✅ Resultados consistentes: sempre {result_counts[0]} resultados")
    else:
        print(f"⚠️  Resultados variaram: {result_counts}")


def test_advanced_features():
    """Testa funcionalidades avançadas."""
    print("\n\n🚀 TESTE 5: Funcionalidades Avançadas")
    print("=" * 60)
    
    try:
        # Testa com a classe diretamente
        tool = VectorSearchTool()
        
        # Health check
        print("\n🏥 Health Check:")
        health = tool.health_check()
        for service, status in health.items():
            print(f"   {service}: {'✅' if status else '❌'}")
        
        # Busca com scores
        print("\n📊 Busca com Scores:")
        results = tool.search(
            "Conselho Universitário",
            include_scores=True,
            top_k=3
        )
        
        for r in results:
            print(f"\nScore: {r['score']:.3f}")
            print(f"Fonte: {r['source']}")
            print(f"Texto: {r['text'][:150]}...")
        
    except Exception as e:
        print(f"❌ Erro nas funcionalidades avançadas: {e}")


def interactive_test():
    """Modo interativo para testar queries personalizadas."""
    print("\n\n💬 MODO INTERATIVO")
    print("=" * 60)
    print("Digite suas queries para testar (ou 'sair' para terminar)")
    print("-" * 60)
    
    sim = CrewAISimulator()
    
    while True:
        try:
            query = input("\n🔍 Sua query: ").strip()
            
            if query.lower() in ['sair', 'exit', 'quit']:
                break
            
            if not query:
                print("⚠️  Digite uma query válida")
                continue
            
            # Simula chamada
            sim.simulate_agent_call(query, "Usuário Teste")
            
        except KeyboardInterrupt:
            print("\n\n👋 Saindo...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    # Mostra estatísticas finais
    sim.show_statistics()


def main():
    """Executa todos os testes."""
    print("🚀 SIMULADOR DE PRODUÇÃO - UFCSPA Search Tool")
    print("=" * 70)
    print("Este script simula como o CrewAI usaria a ferramenta em produção")
    print("=" * 70)
    
    # Menu de opções
    print("\nEscolha o tipo de teste:")
    print("1. Teste Básico (rápido)")
    print("2. Cenários de Produção")
    print("3. Stress e Edge Cases") 
    print("4. Performance e Consistência")
    print("5. Funcionalidades Avançadas")
    print("6. Modo Interativo")
    print("7. Executar TODOS os testes")
    print("0. Sair")
    
    try:
        choice = input("\nOpção: ").strip()
        
        if choice == "1":
            test_basic_functionality()
        elif choice == "2":
            test_production_scenarios()
        elif choice == "3":
            test_stress_and_edge_cases()
        elif choice == "4":
            test_performance()
        elif choice == "5":
            test_advanced_features()
        elif choice == "6":
            interactive_test()
        elif choice == "7":
            # Executa todos
            if test_basic_functionality():
                test_production_scenarios()
                test_stress_and_edge_cases()
                test_performance()
                test_advanced_features()
            else:
                print("\n❌ Teste básico falhou. Verifique a configuração.")
        elif choice == "0":
            print("\n👋 Até logo!")
        else:
            print("\n⚠️  Opção inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()