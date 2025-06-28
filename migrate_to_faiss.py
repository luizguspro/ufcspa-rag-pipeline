"""
Script de migração: Pinecone → FAISS local
Converte dados do sistema antigo para o novo
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import pickle

try:
    from pinecone import Pinecone
    import config
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️  Pinecone não disponível - usando dados locais apenas")

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeToFAISSMigrator:
    """Migra dados do Pinecone para FAISS local."""
    
    def __init__(self, output_dir: str = "vectorstore"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializa embeddings (mesmo modelo da versão refatorada)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def export_from_pinecone(self, index_name: str, namespace: str = "") -> List[Dict]:
        """
        Exporta todos os dados do Pinecone.
        
        Args:
            index_name: Nome do índice Pinecone
            namespace: Namespace opcional
            
        Returns:
            Lista de documentos com metadados
        """
        if not PINECONE_AVAILABLE:
            logger.error("Pinecone não está instalado")
            return []
        
        logger.info(f"Conectando ao Pinecone index: {index_name}")
        
        try:
            # Conecta ao Pinecone
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            index = pc.Index(index_name)
            
            # Obtém estatísticas
            stats = index.describe_index_stats()
            total_vectors = stats['total_vector_count']
            logger.info(f"Total de vetores no Pinecone: {total_vectors}")
            
            # Exporta todos os dados (em batches se necessário)
            all_data = []
            
            # Pinecone tem limite de 10k por query
            # Vamos buscar IDs primeiro e depois os dados
            logger.info("Exportando dados do Pinecone...")
            
            # Esta é uma abordagem simplificada
            # Para índices grandes, seria necessário paginar
            results = index.query(
                vector=[0.0] * 384,  # Vetor dummy
                top_k=min(10000, total_vectors),
                include_metadata=True,
                namespace=namespace
            )
            
            for match in tqdm(results['matches'], desc="Processando vetores"):
                if 'metadata' in match:
                    doc_data = {
                        'id': match['id'],
                        'text': match['metadata'].get('text', ''),
                        'metadata': match['metadata']
                    }
                    all_data.append(doc_data)
            
            logger.info(f"Exportados {len(all_data)} documentos do Pinecone")
            return all_data
            
        except Exception as e:
            logger.error(f"Erro ao exportar do Pinecone: {e}")
            return []
    
    def load_from_json(self, json_path: str) -> List[Dict]:
        """
        Carrega dados de um arquivo JSON (backup manual).
        
        Args:
            json_path: Caminho para o arquivo JSON
            
        Returns:
            Lista de documentos
        """
        logger.info(f"Carregando dados de: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Carregados {len(data)} documentos do JSON")
            return data
            
        except Exception as e:
            logger.error(f"Erro ao carregar JSON: {e}")
            return []
    
    def convert_to_langchain_docs(self, data: List[Dict]) -> List[Document]:
        """
        Converte dados para documentos LangChain.
        
        Args:
            data: Lista de dados exportados
            
        Returns:
            Lista de Documents do LangChain
        """
        logger.info("Convertendo para formato LangChain...")
        
        documents = []
        
        for item in tqdm(data, desc="Convertendo documentos"):
            # Extrai texto
            text = item.get('text', '')
            if not text:
                continue
            
            # Prepara metadados
            metadata = item.get('metadata', {})
            
            # Remove o texto dos metadados (já está no page_content)
            if 'text' in metadata:
                del metadata['text']
            
            # Adiciona ID original se disponível
            if 'id' in item:
                metadata['original_id'] = item['id']
            
            # Cria documento LangChain
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            
            documents.append(doc)
        
        logger.info(f"Convertidos {len(documents)} documentos")
        return documents
    
    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove documentos duplicados."""
        logger.info("Removendo duplicatas...")
        
        seen = set()
        unique_docs = []
        
        for doc in documents:
            # Hash do conteúdo normalizado
            content_hash = hash(doc.page_content.strip().lower())
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        removed = len(documents) - len(unique_docs)
        if removed > 0:
            logger.info(f"Removidas {removed} duplicatas")
        
        return unique_docs
    
    def create_faiss_index(self, documents: List[Document]) -> FAISS:
        """
        Cria índice FAISS a partir dos documentos.
        
        Args:
            documents: Lista de documentos
            
        Returns:
            Índice FAISS
        """
        logger.info("Criando índice FAISS...")
        
        # Remove documentos vazios
        valid_docs = [doc for doc in documents if doc.page_content.strip()]
        logger.info(f"Documentos válidos: {len(valid_docs)}")
        
        # Cria o índice
        vectorstore = FAISS.from_documents(
            documents=valid_docs,
            embedding=self.embeddings
        )
        
        logger.info("Índice FAISS criado com sucesso")
        return vectorstore
    
    def save_faiss_index(self, vectorstore: FAISS, metadata: Dict = None):
        """Salva o índice FAISS no disco."""
        logger.info(f"Salvando índice em: {self.output_dir}")
        
        # Salva o índice
        vectorstore.save_local(str(self.output_dir))
        
        # Salva metadados
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'source': 'migrated_from_pinecone',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'total_documents': len(vectorstore.docstore._dict)
        })
        
        metadata_path = self.output_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("Índice salvo com sucesso")
    
    def migrate(self, source: str = "pinecone", **kwargs):
        """
        Executa a migração completa.
        
        Args:
            source: Fonte dos dados ("pinecone" ou "json")
            **kwargs: Argumentos específicos da fonte
        """
        print("🔄 MIGRAÇÃO: PINECONE → FAISS")
        print("=" * 60)
        
        # 1. Exporta/carrega dados
        if source == "pinecone":
            index_name = kwargs.get('index_name', config.PINECONE_INDEX_NAME)
            data = self.export_from_pinecone(index_name)
        elif source == "json":
            json_path = kwargs.get('json_path', 'pinecone_backup.json')
            data = self.load_from_json(json_path)
        else:
            raise ValueError(f"Fonte inválida: {source}")
        
        if not data:
            logger.error("Nenhum dado para migrar")
            return
        
        # 2. Converte para LangChain
        documents = self.convert_to_langchain_docs(data)
        
        # 3. Remove duplicatas
        documents = self.deduplicate_documents(documents)
        
        # 4. Cria índice FAISS
        vectorstore = self.create_faiss_index(documents)
        
        # 5. Salva no disco
        self.save_faiss_index(vectorstore, {'source_type': source})
        
        print("\n✅ MIGRAÇÃO CONCLUÍDA!")
        print(f"   Documentos migrados: {len(documents)}")
        print(f"   Índice salvo em: {self.output_dir}")
        print("\n🚀 Teste com: python search_tool_final.py")


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migração Pinecone → FAISS")
    parser.add_argument(
        '--source',
        choices=['pinecone', 'json'],
        default='pinecone',
        help='Fonte dos dados'
    )
    parser.add_argument(
        '--index-name',
        default=None,
        help='Nome do índice Pinecone (usa config.py se não especificado)'
    )
    parser.add_argument(
        '--json-path',
        default='pinecone_backup.json',
        help='Caminho do arquivo JSON de backup'
    )
    parser.add_argument(
        '--output-dir',
        default='vectorstore',
        help='Diretório de saída para o índice FAISS'
    )
    
    args = parser.parse_args()
    
    # Executa migração
    migrator = PineconeToFAISSMigrator(output_dir=args.output_dir)
    
    if args.source == 'pinecone':
        if not PINECONE_AVAILABLE:
            print("\n❌ Pinecone não está instalado")
            print("💡 Use --source json com um backup ou instale:")
            print("   pip install pinecone-client")
            return
        
        migrator.migrate(
            source='pinecone',
            index_name=args.index_name
        )
    else:
        migrator.migrate(
            source='json',
            json_path=args.json_path
        )


if __name__ == "__main__":
    main()