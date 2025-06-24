"""
Integração com Llama para geração de respostas.
Usa llama-cpp-python para rodar modelos localmente.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict
import json

try:
    from llama_cpp import Llama
except ImportError:
    print("⚠️  llama-cpp-python não instalado!")
    print("Instale com: pip install llama-cpp-python")
    Llama = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaModel:
    """Gerenciador do modelo Llama local."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0
    ):
        """Inicializa o modelo Llama.
        
        Args:
            model_path: Caminho para o arquivo .gguf do modelo
            n_ctx: Tamanho do contexto (tokens)
            n_threads: Número de threads para CPU
            n_gpu_layers: Camadas para GPU (0 = apenas CPU)
        """
        self.model = None
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        
        # Define caminho padrão para modelos
        if not model_path:
            base_dir = Path(__file__).parent.parent
            models_dir = base_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Procura por modelos disponíveis
            self.model_path = self._find_model(models_dir)
        
        # Carrega o modelo se disponível
        if self.model_path and Path(self.model_path).exists():
            self._load_model()
        else:
            logger.warning("Nenhum modelo Llama encontrado!")
            logger.info("Baixe um modelo GGUF e coloque na pasta 'models/'")
    
    def _find_model(self, models_dir: Path) -> Optional[str]:
        """Procura por modelos GGUF disponíveis."""
        # Lista de modelos recomendados (do menor para o maior)
        recommended_models = [
            "llama-2-7b-chat.Q4_K_M.gguf",
            "llama-2-7b-chat.Q5_K_M.gguf", 
            "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "orca-mini-3b.gguf",
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ]
        
        # Procura por qualquer arquivo .gguf
        gguf_files = list(models_dir.glob("*.gguf"))
        
        if gguf_files:
            # Usa o primeiro encontrado
            return str(gguf_files[0])
        
        # Mostra instruções para baixar
        print("\n" + "="*60)
        print("📥 NENHUM MODELO LLAMA ENCONTRADO!")
        print("="*60)
        print("\nPara usar geração local, baixe um modelo:")
        print("\n1. Modelos pequenos (< 4GB):")
        print("   - TinyLlama: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        print("   - Orca Mini: https://huggingface.co/TheBloke/orca_mini_3B-GGUF")
        print("\n2. Modelos médios (4-8GB):")
        print("   - Llama 2 7B: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        print("   - Mistral 7B: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
        print(f"\n3. Coloque o arquivo .gguf em: {models_dir}")
        print("="*60)
        
        return None
    
    def _load_model(self):
        """Carrega o modelo Llama."""
        if not Llama:
            logger.error("llama-cpp-python não está instalado!")
            return
        
        try:
            logger.info(f"Carregando modelo: {Path(self.model_path).name}")
            logger.info("Isso pode demorar alguns minutos na primeira vez...")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            
            logger.info("✅ Modelo Llama carregado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.model = None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1
    ) -> str:
        """Gera resposta usando o modelo Llama.
        
        Args:
            prompt: Prompt completo para o modelo
            max_tokens: Número máximo de tokens a gerar
            temperature: Controla aleatoriedade (0-1)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repeat_penalty: Penalidade para repetição
            
        Returns:
            Resposta gerada pelo modelo
        """
        if not self.model:
            return self._fallback_response()
        
        try:
            # Gera resposta
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                echo=False
            )
            
            # Extrai o texto gerado
            generated_text = response['choices'][0]['text']
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Resposta fallback quando o modelo não está disponível."""
        return """[Resposta do Sistema - Modo Fallback]

Com base nos documentos encontrados, posso fornecer as seguintes informações:

• Os documentos relevantes foram identificados e estão disponíveis para consulta
• O contexto foi extraído com sucesso dos chunks de texto
• Para uma resposta mais detalhada, seria necessário um modelo de linguagem ativo

💡 Para ativar geração automática:
1. Baixe um modelo Llama GGUF
2. Coloque na pasta 'models/'
3. Reinicie o sistema"""
    
    def format_prompt(self, context: str, question: str) -> str:
        """Formata o prompt para o modelo Llama.
        
        Args:
            context: Contexto dos documentos
            question: Pergunta do usuário
            
        Returns:
            Prompt formatado
        """
        # Template otimizado para Llama 2
        prompt = f"""[INST] <<SYS>>
Você é um assistente especializado em normas e regulamentos da UFCSPA (Universidade Federal de Ciências da Saúde de Porto Alegre). 
Responda às perguntas baseando-se APENAS nas informações fornecidas no contexto.
Se a informação não estiver no contexto, diga claramente que não encontrou a informação nos documentos.
<</SYS>>

Contexto dos documentos:
{context}

Pergunta: {question}

Resposta: [/INST]"""
        
        return prompt
    
    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo carregado."""
        if not self.model_path:
            return {
                "status": "not_configured",
                "message": "Nenhum modelo configurado"
            }
        
        if not Path(self.model_path).exists():
            return {
                "status": "not_found",
                "model_path": self.model_path,
                "message": "Arquivo do modelo não encontrado"
            }
        
        if not self.model:
            return {
                "status": "not_loaded",
                "model_path": self.model_path,
                "message": "Modelo encontrado mas não carregado"
            }
        
        return {
            "status": "ready",
            "model_path": self.model_path,
            "model_name": Path(self.model_path).name,
            "context_size": self.n_ctx,
            "threads": self.n_threads,
            "gpu_layers": self.n_gpu_layers
        }


# Instância global (singleton)
_llama_instance = None


def get_llama_model() -> LlamaModel:
    """Retorna instância singleton do modelo Llama."""
    global _llama_instance
    if _llama_instance is None:
        _llama_instance = LlamaModel()
    return _llama_instance


if __name__ == "__main__":
    # Teste do módulo
    print("Testando integração com Llama...")
    
    model = get_llama_model()
    info = model.get_model_info()
    
    print(f"\nStatus do modelo: {info['status']}")
    
    if info['status'] == 'ready':
        print(f"Modelo: {info['model_name']}")
        
        # Teste de geração
        test_prompt = "Olá! Você está funcionando?"
        print(f"\nTeste: {test_prompt}")
        response = model.generate(test_prompt, max_tokens=50)
        print(f"Resposta: {response}")
    else:
        print(f"Mensagem: {info['message']}")