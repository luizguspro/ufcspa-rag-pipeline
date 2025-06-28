"""
search_tool_final.py - Versão Refatorada
Sistema de busca vetorial para documentos da UFCSPA com pipeline otimizada

Esta versão implementa:
1. Carregamento inteligente de documentos
2. Deduplicação de conteúdo
3. Chunking semântico que preserva contexto
4. Busca vetorial eficiente com FAISS
5. Preparação para re-ranking futuro
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorSearchTool:
    """
    Ferramenta de busca vetorial otimizada para documentos da UFCSPA.
    
    Esta classe gerencia todo o pipeline de preparação de dados e busca,
    incluindo carregamento, deduplicação, chunking inteligente e busca vetorial.
    """
    
    def __init__(
        self,
        documents_dir: str = "data/processed",
        vectorstore_dir: str = "vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_rebuild: bool = False
    ):
        """
        Inicializa a ferramenta de busca vetorial.
        
        Args:
            documents_dir: Diretório contendo os documentos de texto
            vectorstore_dir: Diretório para armazenar o banco vetorial
            embedding_model: Modelo de embeddings a usar
            chunk_size: Tamanho máximo de cada chunk em caracteres
            chunk_overlap: Sobreposição entre chunks consecutivos
            force_rebuild: Se True, reconstrói o banco vetorial mesmo se já existir
        """
        self.documents_dir = Path(documents_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Cria diretórios se não existirem
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializa o modelo de embeddings
        logger.info(f"Inicializando modelo de embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Carrega ou cria o banco vetorial
        self.vectorstore = self._load_or_create_vectorstore(force_rebuild)
        
        logger.info("VectorSearchTool inicializada com sucesso")
    
    def _load_documents(self) -> List[Document]:
        """
        Carrega todos os documentos de texto do diretório especificado.
        
        Returns:
            Lista de documentos LangChain
        """
        logger.info(f"Carregando documentos de: {self.documents_dir}")
        
        # Usa DirectoryLoader para carregar todos os arquivos .txt
        loader = DirectoryLoader(
            str(self.documents_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            show_progress=True
        )
        
        try:
            documents = loader.load()
            logger.info(f"Carregados {len(documents)} documentos")
            return documents
        except Exception as e:
            logger.error(f"Erro ao carregar documentos: {e}")
            return []
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Remove documentos duplicados baseado no hash do conteúdo.
        
        Args:
            documents: Lista de documentos originais
            
        Returns:
            Lista de documentos únicos
        """
        logger.info("Iniciando deduplicação de documentos...")
        
        seen_hashes = set()
        unique_documents = []
        duplicates_count = 0
        
        for doc in documents:
            # Cria hash do conteúdo do documento
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(doc)
            else:
                duplicates_count += 1
        
        logger.info(f"Removidos {duplicates_count} documentos duplicados")
        logger.info(f"Documentos únicos restantes: {len(unique_documents)}")
        
        return unique_documents
    
    def _smart_chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos em chunks usando estratégia inteligente que preserva contexto.
        
        Args:
            documents: Lista de documentos para dividir
            
        Returns:
            Lista de chunks (documentos menores)
        """
        logger.info("Iniciando chunking inteligente de documentos...")
        
        # Configura o splitter com hierarquia de separadores
        # Prioriza quebras por: parágrafos duplos > parágrafo único > linha > ponto > espaço
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Múltiplas quebras (seções)
                "\n\n",    # Parágrafos
                "\n",      # Linhas
                ". ",      # Sentenças
                "! ",      # Exclamações
                "? ",      # Perguntas
                "; ",      # Ponto e vírgula
                ", ",      # Vírgulas
                " ",       # Espaços
                ""         # Caracteres individuais (último recurso)
            ]
        )
        
        # Processa cada documento
        all_chunks = []
        for doc in tqdm(documents, desc="Dividindo documentos em chunks"):
            chunks = text_splitter.split_documents([doc])
            
            # Adiciona metadados úteis a cada chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source_file': doc.metadata.get('source', 'unknown')
                })
            
            all_chunks.extend(chunks)
        
        logger.info(f"Criados {len(all_chunks)} chunks a partir de {len(documents)} documentos")
        
        # Remove chunks muito pequenos (provavelmente só têm pontuação ou espaços)
        filtered_chunks = [
            chunk for chunk in all_chunks 
            if len(chunk.page_content.strip()) > 50
        ]
        
        if len(filtered_chunks) < len(all_chunks):
            removed = len(all_chunks) - len(filtered_chunks)
            logger.info(f"Removidos {removed} chunks muito pequenos")
        
        return filtered_chunks
    
    def _deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Remove chunks com conteúdo muito similar para evitar redundância.
        
        Args:
            chunks: Lista de chunks originais
            
        Returns:
            Lista de chunks únicos
        """
        logger.info("Deduplicando chunks similares...")
        
        seen_contents = set()
        unique_chunks = []
        duplicates_count = 0
        
        for chunk in chunks:
            # Normaliza o conteúdo para comparação
            # Remove espaços extras e converte para minúsculas
            normalized_content = ' '.join(chunk.page_content.lower().split())
            
            # Cria hash do conteúdo normalizado
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunks.append(chunk)
            else:
                duplicates_count += 1
        
        logger.info(f"Removidos {duplicates_count} chunks duplicados")
        logger.info(f"Chunks únicos finais: {len(unique_chunks)}")
        
        return unique_chunks
    
    def _create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """
        Cria um banco vetorial FAISS a partir dos chunks.
        
        Args:
            chunks: Lista de chunks para vetorizar
            
        Returns:
            Banco vetorial FAISS
        """
        logger.info("Criando banco vetorial FAISS...")
        
        # Cria o banco vetorial com barra de progresso
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        logger.info(f"Banco vetorial criado com {len(chunks)} documentos")
        
        return vectorstore
    
    def _save_vectorstore(self, vectorstore: FAISS):
        """
        Salva o banco vetorial no disco para uso futuro.
        
        Args:
            vectorstore: Banco vetorial a salvar
        """
        logger.info(f"Salvando banco vetorial em: {self.vectorstore_dir}")
        
        # Salva o banco vetorial
        vectorstore.save_local(str(self.vectorstore_dir))
        
        # Salva metadados adicionais
        metadata = {
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_chunks': len(vectorstore.docstore._dict)
        }
        
        metadata_path = self.vectorstore_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("Banco vetorial salvo com sucesso")
    
    def _load_vectorstore(self) -> Optional[FAISS]:
        """
        Carrega um banco vetorial existente do disco.
        
        Returns:
            Banco vetorial FAISS ou None se não existir
        """
        index_path = self.vectorstore_dir / "index.faiss"
        
        if not index_path.exists():
            logger.info("Banco vetorial não encontrado")
            return None
        
        logger.info("Carregando banco vetorial existente...")
        
        try:
            vectorstore = FAISS.load_local(
                str(self.vectorstore_dir),
                self.embeddings
            )
            
            # Carrega metadados
            metadata_path = self.vectorstore_dir / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Banco vetorial carregado: {metadata['total_chunks']} chunks")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Erro ao carregar banco vetorial: {e}")
            return None
    
    def _load_or_create_vectorstore(self, force_rebuild: bool = False) -> FAISS:
        """
        Carrega um banco vetorial existente ou cria um novo se necessário.
        
        Args:
            force_rebuild: Se True, sempre reconstrói o banco
            
        Returns:
            Banco vetorial FAISS
        """
        # Tenta carregar banco existente se não for forçado rebuild
        if not force_rebuild:
            vectorstore = self._load_vectorstore()
            if vectorstore is not None:
                return vectorstore
        
        # Cria novo banco vetorial
        logger.info("Construindo novo banco vetorial...")
        
        # Pipeline de preparação de dados
        documents = self._load_documents()
        if not documents:
            raise ValueError("Nenhum documento encontrado para processar")
        
        # Deduplicação em nível de documento
        documents = self._deduplicate_documents(documents)
        
        # Chunking inteligente
        chunks = self._smart_chunk_documents(documents)
        
        # Deduplicação em nível de chunk
        chunks = self._deduplicate_chunks(chunks)
        
        # Criação do banco vetorial
        vectorstore = self._create_vectorstore(chunks)
        
        # Salva para uso futuro
        self._save_vectorstore(vectorstore)
        
        return vectorstore
    
    def search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        filter_dict: Optional[Dict] = None
    ) -> List[str]:
        """
        Realiza busca vetorial por similaridade.
        
        Args:
            query: Pergunta ou consulta do usuário
            k: Número de resultados finais a retornar
            fetch_k: Número de candidatos iniciais para buscar (para re-ranking futuro)
            filter_dict: Filtros opcionais para aplicar na busca
            
        Returns:
            Lista com o conteúdo dos k chunks mais relevantes
        """
        logger.info(f"Realizando busca: '{query}'")
        
        try:
            # Busca inicial com mais candidatos para permitir re-ranking futuro
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=fetch_k,
                filter=filter_dict
            )
            
            # TODO: Implementar um Cross-Encoder para re-ranking dos resultados aqui
            # O Cross-Encoder receberia a query e cada documento candidato,
            # calcularia um score mais preciso e reordenaria os resultados.
            # Por enquanto, usamos apenas os top-k resultados da busca vetorial.
            
            # Extrai apenas os top-k resultados
            top_results = results[:k]
            
            # Log dos scores para debug
            for i, (doc, score) in enumerate(top_results):
                logger.debug(f"Resultado {i+1} - Score: {score:.4f} - "
                           f"Fonte: {doc.metadata.get('source', 'unknown')}")
            
            # Retorna apenas o conteúdo dos chunks
            chunks_content = [doc.page_content for doc, _ in top_results]
            
            logger.info(f"Retornando {len(chunks_content)} resultados relevantes")
            
            return chunks_content
            
        except Exception as e:
            logger.error(f"Erro durante a busca: {e}")
            return []
    
    def search_with_metadata(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[str, Dict, float]]:
        """
        Realiza busca retornando conteúdo, metadados e scores.
        
        Útil para debug e análise dos resultados.
        
        Args:
            query: Pergunta ou consulta do usuário
            k: Número de resultados finais a retornar
            fetch_k: Número de candidatos iniciais
            filter_dict: Filtros opcionais
            
        Returns:
            Lista de tuplas (conteúdo, metadados, score)
        """
        logger.info(f"Realizando busca com metadados: '{query}'")
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=fetch_k,
                filter=filter_dict
            )
            
            # TODO: Cross-Encoder para re-ranking aqui
            
            top_results = results[:k]
            
            # Prepara resultados com todas as informações
            detailed_results = [
                (doc.page_content, doc.metadata, score)
                for doc, score in top_results
            ]
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"Erro durante a busca: {e}")
            return []


# Instância global para compatibilidade com a interface anterior
_search_tool_instance = None


def get_search_tool(**kwargs) -> VectorSearchTool:
    """
    Retorna uma instância singleton da ferramenta de busca.
    
    Args:
        **kwargs: Argumentos para VectorSearchTool
        
    Returns:
        Instância de VectorSearchTool
    """
    global _search_tool_instance
    if _search_tool_instance is None:
        _search_tool_instance = VectorSearchTool(**kwargs)
    return _search_tool_instance


def search_vectorstore(query: str, k: int = 5) -> List[str]:
    """
    Interface compatível com a versão anterior do script.
    
    Args:
        query: Pergunta do usuário
        k: Número de resultados a retornar
        
    Returns:
        Lista com o conteúdo dos chunks mais relevantes
    """
    tool = get_search_tool()
    return tool.search(query, k=k)


# Bloco de teste e demonstração
if __name__ == '__main__':
    print("=" * 70)
    print("FERRAMENTA DE BUSCA VETORIAL - VERSÃO REFATORADA")
    print("=" * 70)
    
    # Pergunta de teste
    pergunta_exemplo = "Quais são as normas para atividades de extensão universitária?"
    
    print(f"\n🔍 Pergunta: {pergunta_exemplo}")
    print("-" * 70)
    
    try:
        # Inicializa a ferramenta (criará o banco vetorial se necessário)
        print("\n⚙️  Inicializando ferramenta de busca...")
        tool = VectorSearchTool(
            documents_dir="data/processed",
            force_rebuild=False  # Mude para True para reconstruir o banco
        )
        
        # Realiza a busca
        print("\n🔎 Realizando busca...")
        resultados = tool.search_with_metadata(pergunta_exemplo, k=5)
        
        if not resultados:
            print("\n❌ Nenhum resultado encontrado")
        else:
            print(f"\n✅ Encontrados {len(resultados)} resultados relevantes:\n")
            
            for i, (conteudo, metadata, score) in enumerate(resultados, 1):
                print(f"{'='*70}")
                print(f"RESULTADO {i}")
                print(f"Score: {score:.4f}")
                print(f"Fonte: {metadata.get('source', 'unknown')}")
                print(f"Chunk: {metadata.get('chunk_index', '?')} de {metadata.get('total_chunks', '?')}")
                print(f"-" * 70)
                print(f"{conteudo[:500]}..." if len(conteudo) > 500 else conteudo)
                print()
        
        # Demonstra uso simples para integração
        print("\n" + "="*70)
        print("USO SIMPLES PARA INTEGRAÇÃO:")
        print("="*70)
        
        contextos = search_vectorstore(pergunta_exemplo)
        print(f"\nContextos prontos para LLM: {len(contextos)} chunks")
        print("\nPrimeiro contexto (preview):")
        print(contextos[0][:200] + "..." if contextos and len(contextos[0]) > 200 else contextos[0] if contextos else "Nenhum")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()