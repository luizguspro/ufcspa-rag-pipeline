# Módulo de Ingestão - UFCSPA Pipeline

Este módulo é responsável por processar documentos PDF, convertê-los em texto, dividi-los em chunks e gerar embeddings para busca vetorial.

## Componentes

### 1. `convert.py` - Conversor PDF → Texto
- Extrai texto de PDFs usando `pdfminer.six`
- Usa OCR (Tesseract) como fallback para PDFs baseados em imagem
- Limpa e normaliza o texto extraído

### 2. `chunk.py` - Divisor de Textos
- Divide textos em chunks de tamanho apropriado
- Suporta dois métodos: LangChain e Tiktoken
- Limpa texto (remove HTML, normaliza espaços, etc.)

### 3. `embed.py` - Gerador de Embeddings
- Gera embeddings usando sentence-transformers
- Cria índice FAISS para busca vetorial eficiente
- Normaliza vetores para similaridade cosseno

### 4. `test_index.py` - Testador do Índice
- Valida o índice FAISS criado
- Permite fazer buscas de teste

## Como Usar

### Executar Pipeline Completo

```bash
cd ufcspa_pipeline
python ingest/run_pipeline.py
```

### Executar Etapas Individualmente

```bash
# 1. Converter PDFs para texto
python ingest/convert.py

# 2. Dividir textos em chunks
python ingest/chunk.py --method langchain

# 3. Gerar embeddings e índice FAISS
python ingest/embed.py

# 4. Testar o índice criado
python ingest/test_index.py
```

### Opções Avançadas

```bash
# Pipeline completo com configurações customizadas
python ingest/run_pipeline.py \
    --chunk-method tiktoken \
    --chunk-size 800 \
    --chunk-overlap 150 \
    --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Estrutura de Dados

### Entrada
- PDFs em `data/raw/`

### Saída
- Textos extraídos em `data/processed/*.txt`
- Chunks em `data/processed/chunks.json`
- Índice FAISS em `faiss_index/ufcspa.index`
- Metadados em `faiss_index/chunks.json`

## Modelos de Embedding Suportados

- `sentence-transformers/all-MiniLM-L6-v2` (padrão, 384 dimensões)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilíngue)
- `sentence-transformers/all-mpnet-base-v2` (mais preciso, 768 dimensões)

## Requisitos

- Python 3.8+
- Tesseract OCR instalado no sistema
- Dependências em `requirements.txt`

## Troubleshooting

### Erro de OCR
Se receber erro do Tesseract:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-por

# macOS
brew install tesseract tesseract-lang

# Windows
# Baixe de: https://github.com/UB-Mannheim/tesseract/wiki
```

### Memória Insuficiente
Para grandes volumes de dados, considere:
- Processar em lotes menores
- Usar modelo de embedding menor
- Aumentar RAM disponível