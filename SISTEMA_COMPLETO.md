# 🚀 SISTEMA RAG COMPLETO - UFCSPA + LLAMA

## ✅ O QUE ESTÁ IMPLEMENTADO

### 1. **Download Automático da UFCSPA**
- ✅ Script `download_ufcspa_complete.py` com correção SSL
- ✅ Múltiplas estratégias de busca
- ✅ Fallback para dados de exemplo

### 2. **Integração com Llama (LLM Local)**
- ✅ `llm_handler.py` - Gerenciador completo do Llama
- ✅ `query_with_llama.py` - Query engine com Llama integrado
- ✅ Download automático de modelos
- ✅ Suporte para TinyLlama, Llama 2, Mistral

### 3. **Pipeline Completo**
- ✅ Web scraping → PDF → Texto → Chunks → Embeddings → Query + Llama

## 🎯 COMO EXECUTAR

### Opção 1: Sistema Completo (RECOMENDADO)
```bash
cd ufcspa_pipeline
python run_complete_system.py
```
Escolha opção 1 para instalação completa.

### Opção 2: Componentes Separados

#### Baixar documentos UFCSPA:
```bash
python scraper/download_ufcspa_complete.py
```

#### Configurar Llama:
```bash
python -c "from query.llm_handler import get_llama_handler; h = get_llama_handler(); h.download_model('tinyllama')"
```

#### Executar consulta com Llama:
```bash
python query/query_with_llama.py "Quais são as normas de extensão?" -v
```

## 🤖 MODELOS LLAMA DISPONÍVEIS

### 1. **TinyLlama 1.1B** (Recomendado para teste)
- Tamanho: ~600MB
- Velocidade: Rápido
- Qualidade: Boa para português

### 2. **Llama 2 7B Chat**
- Tamanho: ~3.8GB
- Velocidade: Médio
- Qualidade: Excelente

### 3. **Mistral 7B**
- Tamanho: ~4.1GB
- Velocidade: Médio
- Qualidade: Muito boa

## 📁 ESTRUTURA ATUALIZADA

```
ufcspa_pipeline/
├── scraper/
│   ├── download_ufcspa_complete.py  # Download com SSL fix [NOVO]
│   └── ...
├── ingest/
│   ├── embed.py                     # Embeddings FAISS [CRIADO]
│   └── ...
├── query/
│   ├── llm_handler.py              # Gerenciador Llama [NOVO]
│   ├── query_with_llama.py        # Query + Llama [NOVO]
│   └── ...
├── models/                         # Modelos Llama [NOVO]
│   └── tinyllama-1.1b-chat.gguf
└── run_complete_system.py          # Sistema completo [NOVO]
```

## 🔧 CORREÇÕES APLICADAS

1. **SSL da UFCSPA**: ✅ Resolvido com verify=False
2. **Arquivo embed.py**: ✅ Criado completo
3. **Integração Llama**: ✅ Implementada
4. **Download automático**: ✅ Funcionando

## 💡 DEMONSTRAÇÃO RÁPIDA

```python
# test_demo.py
from query.query_with_llama import QueryEngineWithLlama

# Inicializa
engine = QueryEngineWithLlama()

# Faz pergunta
result = engine.query("Quais são os princípios da UFCSPA?")

# Mostra resposta do Llama
print("Resposta:", result['response'])
```

## 🎓 PARA A FACULDADE

### Destaques do Projeto:
1. **RAG Completo**: Retrieval + Generation funcionando
2. **LLM Local**: Não depende de APIs externas
3. **Web Scraping Robusto**: Com tratamento de SSL
4. **Pipeline Modular**: Cada componente independente
5. **Fallbacks**: Funciona mesmo sem alguns componentes

### Como Apresentar:
1. Execute `python run_complete_system.py`
2. Escolha opção 6 (Teste rápido)
3. Mostre o Llama respondendo perguntas
4. Explique a arquitetura RAG

## ⚠️ REQUISITOS

- Python 3.8+
- 4GB RAM mínimo (8GB recomendado)
- ~2GB espaço em disco para modelos
- Conexão internet para downloads

## 🐛 TROUBLESHOOTING

### "llama-cpp-python não instala"
```bash
# Windows
pip install llama-cpp-python --no-cache-dir

# Linux/Mac
CMAKE_ARGS="-DLLAMA_CUBLAS=off" pip install llama-cpp-python
```

### "SSL Error ao baixar da UFCSPA"
- Já está corrigido no código
- Use `download_ufcspa_complete.py`

### "Modelo Llama não carrega"
- Verifique se baixou o modelo .gguf
- Use TinyLlama para testes (mais leve)

## ✨ FUNCIONALIDADES EXTRAS

1. **Multi-modelo**: Suporta vários modelos Llama
2. **Busca híbrida**: FAISS + keywords
3. **OCR automático**: Para PDFs escaneados
4. **Cache de embeddings**: Reutiliza processamento
5. **Logs detalhados**: Para debug

---

**🎉 SISTEMA 100% COMPLETO E FUNCIONAL!**

Execute `python run_complete_system.py` e impressione todos! 🚀