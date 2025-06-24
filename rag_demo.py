"""
Sistema RAG UFCSPA - Demonstração Funcional
Este arquivo contém tudo necessário para demonstrar o conceito.
"""

import re
from typing import List, Dict


class RAGDemo:
    """Sistema RAG completo em um único arquivo."""
    
    def __init__(self):
        # Documentos de exemplo hardcoded
        self.documents = [
            {
                "id": 1,
                "title": "Estatuto da UFCSPA - Capítulo I",
                "content": """ESTATUTO DA UFCSPA

A Universidade Federal de Ciências da Saúde de Porto Alegre (UFCSPA) é uma instituição pública federal de ensino superior, criada pela Lei nº 11.641, de 13 de janeiro de 2008.

CAPÍTULO I - DOS PRINCÍPIOS
Art. 1º - A UFCSPA tem como princípios fundamentais:
I - Excelência acadêmica em ensino, pesquisa e extensão
II - Formação humanística e ética dos profissionais da saúde
III - Compromisso social com a saúde pública
IV - Gestão democrática e participativa
V - Autonomia universitária

Art. 2º - A universidade promove a indissociabilidade entre ensino, pesquisa e extensão, com ênfase na área da saúde."""
            },
            {
                "id": 2,
                "title": "Estatuto da UFCSPA - Capítulo II",
                "content": """CAPÍTULO II - DA ORGANIZAÇÃO
Art. 3º - A estrutura organizacional compreende:
I - Conselho Universitário (CONSUN) - órgão máximo deliberativo
II - Conselho de Ensino, Pesquisa e Extensão (CONSEPE)
III - Reitoria - órgão executivo central
IV - Pró-Reitorias
V - Departamentos Acadêmicos

Art. 4º - O Conselho Universitário é responsável por estabelecer a política geral da instituição em matéria administrativa, econômico-financeira, patrimonial e disciplinar."""
            },
            {
                "id": 3,
                "title": "Normas para Atividades de Extensão",
                "content": """NORMAS PARA ATIVIDADES DE EXTENSÃO

Art. 1º - As atividades de extensão são ações que promovem a interação transformadora entre a universidade e a sociedade.

Art. 2º - Modalidades de extensão universitária:
I - Programas: conjunto articulado de projetos de caráter contínuo
II - Projetos: ações processuais com objetivos específicos
III - Cursos: ações pedagógicas de caráter teórico-prático
IV - Eventos: ações pontuais como congressos, seminários, oficinas
V - Prestação de serviços: realização de trabalho oferecido pela UFCSPA

Art. 3º - Requisitos para aprovação de atividades de extensão:
I - Relevância social e impacto na comunidade
II - Articulação com ensino e pesquisa
III - Envolvimento obrigatório de estudantes
IV - Impacto na formação acadêmica
V - Registro na Pró-Reitoria de Extensão e Cultura"""
            },
            {
                "id": 4,
                "title": "Regimento Interno - Administração",
                "content": """REGIMENTO INTERNO DA UFCSPA

TÍTULO I - DA NATUREZA E FINALIDADE
Art. 1º - A UFCSPA é uma autarquia federal vinculada ao Ministério da Educação, com sede em Porto Alegre.

TÍTULO II - DA ADMINISTRAÇÃO
Art. 2º - A administração superior é exercida pelos órgãos colegiados e pela Reitoria.
Art. 3º - O Reitor é a autoridade máxima executiva da universidade, auxiliado pelo Vice-Reitor.
Art. 4º - O mandato do Reitor e Vice-Reitor é de 4 anos, permitida uma recondução."""
            },
            {
                "id": 5,
                "title": "Regimento Interno - Ensino",
                "content": """TÍTULO III - DO ENSINO
Art. 5º - O ensino na UFCSPA é ministrado em três níveis:
I - Graduação: formação profissional em nível superior
II - Pós-Graduação Lato Sensu: especialização e aperfeiçoamento
III - Pós-Graduação Stricto Sensu: mestrado e doutorado

Art. 6º - Os cursos de graduação conferem diplomas aos concluintes e destinam-se à formação de profissionais da saúde.

Art. 7º - A pós-graduação tem como objetivo formar docentes, pesquisadores e profissionais especializados."""
            }
        ]
        
        print("✅ Sistema RAG carregado com 5 documentos")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Busca documentos relevantes."""
        query_lower = query.lower()
        words = re.findall(r'\w+', query_lower)
        
        # Remove palavras muito comuns
        stopwords = {'o', 'a', 'de', 'da', 'do', 'que', 'e', 'é', 'para', 'com', 'uma', 'um', 'no', 'na'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Calcula relevância
        results = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            title_lower = doc['title'].lower()
            
            score = 0
            matches = []
            
            for word in words:
                count_content = content_lower.count(word)
                count_title = title_lower.count(word)
                
                if count_content > 0 or count_title > 0:
                    matches.append(word)
                    score += count_content * 1
                    score += count_title * 3
            
            if score > 0:
                results.append({
                    'document': doc,
                    'score': score,
                    'matches': matches
                })
        
        # Ordena por relevância
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def query(self, question: str):
        """Processa uma pergunta e mostra os resultados."""
        print(f"\n{'='*70}")
        print(f"🔍 PERGUNTA: {question}")
        print(f"{'='*70}")
        
        # Busca documentos
        results = self.search(question, k=3)
        
        if not results:
            print("\n❌ Nenhum documento relevante encontrado.")
            print("💡 Tente usar outras palavras-chave.")
            return
        
        print(f"\n📚 Encontrados {len(results)} documentos relevantes:\n")
        
        # Mostra resultados
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result['score']
            matches = result['matches']
            
            print(f"[{i}] {doc['title']}")
            print(f"    📊 Relevância: {score} pontos")
            print(f"    🔤 Palavras encontradas: {', '.join(matches)}")
            print(f"    📄 Trecho: {doc['content'][:150]}...")
            print()
        
        # Constrói contexto
        print("\n" + "="*70)
        print("📝 CONTEXTO PARA RESPOSTA (seria enviado ao LLM):")
        print("="*70)
        
        for result in results:
            doc = result['document']
            print(f"\n--- {doc['title']} ---")
            print(doc['content'][:300] + "...")
        
        print("\n" + "="*70)
        print("💬 RESPOSTA (Placeholder - aqui entraria um LLM):")
        print("="*70)
        print("\nCom base nos documentos encontrados, posso informar que...")
        print("[Aqui o ChatGPT, Claude ou outro LLM geraria uma resposta")
        print("baseada no contexto acima]")
        print("="*70)


def main():
    """Interface principal."""
    print("\n" + "="*70)
    print("🎓 SISTEMA RAG - NORMAS UFCSPA")
    print("="*70)
    print("\nSistema de busca e recuperação de informações sobre normas")
    print("da Universidade Federal de Ciências da Saúde de Porto Alegre")
    print("\n💡 Digite 'sair' para encerrar")
    print("💡 Digite 'ajuda' para ver exemplos de perguntas")
    
    rag = RAGDemo()
    
    while True:
        print("\n" + "-"*70)
        query = input("\n❓ Sua pergunta: ").strip()
        
        if query.lower() in ['sair', 'exit', 'quit', 'q']:
            print("\n👋 Obrigado por usar o sistema!")
            break
        
        elif query.lower() in ['ajuda', 'help', '?']:
            print("\n📋 EXEMPLOS DE PERGUNTAS:")
            print("  • Quais são as normas de extensão?")
            print("  • Como funciona o conselho universitário?")
            print("  • Quais são os princípios da UFCSPA?")
            print("  • O que é o CONSEPE?")
            print("  • Como é a estrutura organizacional?")
            print("  • Quais são os níveis de ensino?")
            print("  • Qual o papel do reitor?")
            continue
        
        elif query:
            rag.query(query)
        else:
            print("⚠️  Digite uma pergunta ou 'ajuda' para exemplos")


if __name__ == "__main__":
    main()