"""
Integração com Llama para geração de respostas.
Usa llama-cpp-python para rodar modelos localmente.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict
import os

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaHandler:
    """Gerenciador de LLM local usando Llama."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Inicializa o handler do Llama.
        
        Args:
            model_path: Caminho para o modelo .gguf
        """
        self.model = None
        self.model_path = model_path
        
        # Tenta importar llama-cpp-python
        try:
            from llama_cpp import Llama
            self.Llama = Llama
            self.available = True
        except ImportError:
            logger.warning("llama-cpp-python não instalado. LLM não disponível.")
            logger.info("Instale com: pip install llama-cpp-python")
            self.available = False
            return
        
        # Se não especificou modelo, tenta encontrar
        if not model_path:
            self.model_path = self._find_model()
        
        # Carrega o modelo se encontrou
        if self.model_path and Path(self.model_path).exists():
            self._load_model()
        else:
            logger.warning("Nenhum modelo Llama encontrado.")
            self.available = False
    
    def _find_model(self) -> Optional[str]:
        """Procura por modelos Llama no sistema."""
        # Diretórios comuns para modelos
        search_dirs = [
            Path("models"),
            Path("llama_models"),
            Path.home() / "models",
            Path.home() / ".cache" / "llama",
            Path(".")
        ]
        
        # Extensões de modelo suportadas
        model_extensions = ['.gguf', '.ggml', '.bin']
        
        # Modelos recomendados (em ordem de preferência)
        preferred_models = [
            'llama-2-7b-chat.Q4_K_M.gguf',
            'llama-2-7b-chat.gguf',
            'mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            'mistral-7b-instruct.gguf',
            'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
        ]
        
        # Procura modelos preferidos primeiro
        for dir_path in search_dirs:
            if not dir_path.exists():
                continue
                
            for model_name in preferred_models:
                model_path = dir_path / model_name
                if model_path.exists():
                    logger.info(f"Modelo encontrado: {model_path}")
                    return str(model_path)
        
        # Se não encontrou preferidos, procura qualquer modelo
        for dir_path in search_dirs:
            if not dir_path.exists():
                continue
                
            for ext in model_extensions:
                models = list(dir_path.glob(f"*{ext}"))
                if models:
                    logger.info(f"Modelo encontrado: {models[0]}")
                    return str(models[0])
        
        return None
    
    def _load_model(self):
        """Carrega o modelo Llama."""
        try:
            logger.info(f"Carregando modelo: {self.model_path}")
            logger.info("Isso pode demorar alguns minutos na primeira vez...")
            
            # Configurações otimizadas para performance
            self.model = self.Llama(
                model_path=self.model_path,
                n_ctx=2048,        # Contexto de 2048 tokens
                n_threads=os.cpu_count() // 2,  # Metade dos cores da CPU
                n_batch=512,       # Batch size
                verbose=False      # Menos output
            )
            
            logger.info("✅ Modelo carregado com sucesso!")
            self.available = True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.available = False
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Gera resposta usando o modelo Llama.
        
        Args:
            prompt: Prompt completo para o modelo
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para controle de criatividade
            
        Returns:
            Resposta gerada pelo modelo
        """
        if not self.available or not self.model:
            return self._fallback_response()
        
        try:
            # Gera resposta
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n\n"],  # Stop sequences
                echo=False  # Não repetir o prompt
            )
            
            # Extrai o texto da resposta
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
            
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            return self._fallback_response()
    
    def generate_rag_response(self, context: str, question: str) -> str:
        """Gera resposta RAG usando contexto e pergunta.
        
        Args:
            context: Contexto dos documentos recuperados
            question: Pergunta do usuário
            
        Returns:
            Resposta gerada
        """
        # Template otimizado para RAG
        prompt = f"""Você é um assistente especializado em normas e regulamentos da UFCSPA (Universidade Federal de Ciências da Saúde de Porto Alegre).

Use APENAS as informações fornecidas no contexto abaixo para responder à pergunta. Se a informação não estiver no contexto, diga claramente que não tem essa informação disponível.

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA BASEADA NO CONTEXTO:"""
        
        return self.generate(prompt, max_tokens=512, temperature=0.3)
    
    def _fallback_response(self) -> str:
        """Resposta padrão quando o LLM não está disponível."""
        return """[Sistema LLM não disponível]

Para ativar respostas automáticas com IA:

1. Instale o llama-cpp-python:
   pip install llama-cpp-python

2. Baixe um modelo Llama 2 (recomendado):
   - Llama 2 7B Chat: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
   - Mistral 7B: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
   - TinyLlama 1.1B (mais leve): https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

3. Coloque o arquivo .gguf na pasta 'models/'

O sistema continuará funcionando sem LLM, mostrando apenas o contexto recuperado."""
    
    def download_model(self, model_name: str = "tinyllama"):
        """Baixa um modelo recomendado.
        
        Args:
            model_name: Nome do modelo para baixar
        """
        model_urls = {
            "tinyllama": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "llama2": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            "mistral": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        }
        
        if model_name not in model_urls:
            logger.error(f"Modelo {model_name} não reconhecido")
            return
        
        url = model_urls[model_name]
        filename = url.split('/')[-1]
        
        # Cria diretório models
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        output_path = models_dir / filename
        
        if output_path.exists():
            logger.info(f"Modelo já existe: {output_path}")
            return
        
        logger.info(f"Baixando {model_name}...")
        logger.info(f"URL: {url}")
        logger.info("Isso pode demorar dependendo da sua conexão...")
        
        try:
            import requests
            from tqdm import tqdm
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"✅ Modelo baixado: {output_path}")
            
        except Exception as e:
            logger.error(f"Erro ao baixar modelo: {e}")


# Singleton para reutilizar o modelo carregado
_llama_instance = None

def get_llama_handler(model_path: Optional[str] = None) -> LlamaHandler:
    """Obtém instância única do handler Llama.
    
    Args:
        model_path: Caminho opcional para o modelo
        
    Returns:
        Instância do LlamaHandler
    """
    global _llama_instance
    
    if _llama_instance is None:
        _llama_instance = LlamaHandler(model_path)
    
    return _llama_instance


if __name__ == "__main__":
    # Teste do módulo
    print("Testando integração com Llama...")
    
    handler = get_llama_handler()
    
    if not handler.available:
        print("\n❌ LLM não disponível")
        print("\n💡 Deseja baixar um modelo? (s/n)")
        
        if input().lower() == 's':
            print("\nModelos disponíveis:")
            print("1. TinyLlama 1.1B (mais leve, ~600MB)")
            print("2. Llama 2 7B Chat (~3.8GB)")
            print("3. Mistral 7B (~4.1GB)")
            
            choice = input("\nEscolha (1-3): ")
            
            model_map = {"1": "tinyllama", "2": "llama2", "3": "mistral"}
            if choice in model_map:
                handler.download_model(model_map[choice])
    else:
        # Teste de geração
        context = "A UFCSPA tem como princípios: excelência acadêmica, formação humanística e compromisso social."
        question = "Quais são os princípios da universidade?"
        
        print("\n🧪 Testando geração...")
        response = handler.generate_rag_response(context, question)
        print(f"\n💬 Resposta: {response}")