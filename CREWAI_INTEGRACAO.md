# 🤖 Guia de Integração com CrewAI

Este guia mostra como integrar a ferramenta de busca UFCSPA com agentes CrewAI.

## 📋 Pré-requisitos

```bash
pip install crewai
pip install -r requirements.txt
```

## 🔧 Configuração Básica

### 1. Importe a ferramenta

```python
from crewai import Agent, Task, Crew, Tool
from search_tool import search_vectorstore
```

### 2. Crie a ferramenta CrewAI

```python
ufcspa_search_tool = Tool(
    name="UFCSPA Document Search",
    description="""Busca documentos normativos da UFCSPA. 
    Use quando precisar encontrar informações sobre:
    - Normas e regulamentos
    - Estatutos e regimentos
    - Resoluções do CONSUN/CONSEPE
    - Portarias e instruções normativas
    - Documentos institucionais""",
    func=search_vectorstore
)
```

## 💡 Exemplos de Uso

### Exemplo 1: Agente Pesquisador

```python
# Define o agente
pesquisador = Agent(
    role='Pesquisador de Normas UFCSPA',
    goal='Encontrar e analisar documentos normativos da universidade',
    backstory='Você é um especialista em legislação universitária...',
    tools=[ufcspa_search_tool],
    verbose=True
)

# Define uma tarefa
tarefa_pesquisa = Task(
    description='Encontre as normas sobre atividades de extensão na UFCSPA',
    expected_output='Resumo das principais normas de extensão',
    agent=pesquisador
)

# Cria e executa a crew
crew = Crew(
    agents=[pesquisador],
    tasks=[tarefa_pesquisa],
    verbose=True
)

resultado = crew.kickoff()
print(resultado)
```

### Exemplo 2: Múltiplos Agentes

```python
# Agente 1: Pesquisador
pesquisador = Agent(
    role='Pesquisador',
    goal='Encontrar documentos relevantes',
    tools=[ufcspa_search_tool],
    verbose=True
)

# Agente 2: Analista
analista = Agent(
    role='Analista de Normas',
    goal='Analisar e sintetizar informações normativas',
    verbose=True
)

# Tarefas encadeadas
task1 = Task(
    description='Encontre todas as normas sobre progressão docente',
    agent=pesquisador
)

task2 = Task(
    description='Analise as normas encontradas e crie um resumo executivo',
    agent=analista,
    context=[task1]  # Usa o output da task1
)

# Executa
crew = Crew(
    agents=[pesquisador, analista],
    tasks=[task1, task2]
)

resultado = crew.kickoff()
```

## 🎯 Casos de Uso Avançados

### 1. Ferramenta com Parâmetros Customizados

```python
def busca_customizada(query: str) -> str:
    """Busca com mais resultados e formatação especial."""
    from search_tool import VectorSearchTool
    
    tool = VectorSearchTool(top_k=10, min_score=0.7)
    results = tool.search(query, include_scores=True)
    
    # Formata os resultados
    output = f"Encontrados {len(results)} documentos:\n\n"
    for i, r in enumerate(results, 1):
        output += f"{i}. (Relevância: {r['score']:.2%})\n"
        output += f"   {r['text'][:200]}...\n\n"
    
    return output

# Cria ferramenta customizada
ufcspa_search_advanced = Tool(
    name="UFCSPA Advanced Search",
    description="Busca avançada com scores de relevância",
    func=busca_customizada
)
```

### 2. Integração com Cache

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def busca_com_cache(query: str) -> list:
    """Versão com cache para queries repetidas."""
    return search_vectorstore(query)

cached_search_tool = Tool(
    name="UFCSPA Cached Search",
    description="Busca com cache para melhor performance",
    func=busca_com_cache
)
```

### 3. Ferramenta com Filtros

```python
def busca_por_tipo(query: str, tipo_documento: str = "norma") -> str:
    """Busca filtrando por tipo de documento."""
    from search_tool import get_tool_instance
    
    tool = get_tool_instance()
    
    # Adiciona o tipo ao query para melhor relevância
    enhanced_query = f"{query} {tipo_documento}"
    
    results = tool.search(enhanced_query, top_k=5)
    
    # Filtra resultados por tipo
    filtered = [r for r in results if tipo_documento.lower() in r.lower()]
    
    return "\n\n".join(filtered) if filtered else "Nenhum documento encontrado."

# Diferentes ferramentas para diferentes tipos
normas_tool = Tool(
    name="Buscar Normas",
    func=lambda q: busca_por_tipo(q, "norma")
)

regimentos_tool = Tool(
    name="Buscar Regimentos", 
    func=lambda q: busca_por_tipo(q, "regimento")
)
```

## 🐛 Troubleshooting

### Problema: "No documents found"
```python
# Adicione logging para debug
import logging
logging.basicConfig(level=logging.INFO)

# Teste a ferramenta diretamente
result = search_vectorstore("teste")
print(f"Resultados diretos: {len(result)}")
```

### Problema: "API Key errors"
```python
# Verifique as configurações antes de criar agentes
from search_tool import get_tool_instance

try:
    tool = get_tool_instance()
    health = tool.health_check()
    print(f"OpenAI: {'✅' if health['openai'] else '❌'}")
    print(f"Pinecone: {'✅' if health['pinecone'] else '❌'}")
except Exception as e:
    print(f"Erro de configuração: {e}")
```

## 📊 Métricas e Monitoramento

```python
import time
from datetime import datetime

class MonitoredSearchTool:
    def __init__(self):
        self.queries = []
    
    def search_with_metrics(self, query: str) -> tuple:
        start_time = time.time()
        results = search_vectorstore(query)
        elapsed = time.time() - start_time
        
        # Registra métricas
        self.queries.append({
            'timestamp': datetime.now(),
            'query': query,
            'results_count': len(results),
            'response_time': elapsed
        })
        
        return results
    
    def get_metrics(self):
        if not self.queries:
            return "Nenhuma query registrada"
        
        avg_time = sum(q['response_time'] for q in self.queries) / len(self.queries)
        avg_results = sum(q['results_count'] for q in self.queries) / len(self.queries)
        
        return f"""
        Total de queries: {len(self.queries)}
        Tempo médio de resposta: {avg_time:.2f}s
        Média de resultados: {avg_results:.1f}
        """

# Uso
monitor = MonitoredSearchTool()
monitored_tool = Tool(
    name="UFCSPA Search (Monitored)",
    func=monitor.search_with_metrics
)
```

## 🚀 Melhores Práticas

1. **Seja específico nas descrições das ferramentas** - Ajuda o agente a escolher quando usar
2. **Use context em tasks** - Permite que agentes construam sobre resultados anteriores
3. **Configure verbosity** - Para debug e entendimento do processo
4. **Implemente retry logic** - Para lidar com falhas temporárias
5. **Cache queries comuns** - Melhora performance significativamente

## 📚 Recursos Adicionais

- [Documentação CrewAI](https://docs.crewai.com/)
- [Exemplos de Agentes](https://github.com/joaomdmoura/crewAI-examples)
- [API Reference](../README.md#-api-reference)

---

**💡 Dica:** Para um exemplo completo e funcional, veja o arquivo `examples/crewai_complete_example.py`