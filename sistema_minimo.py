
import re

print("\n" + "="*60)
print("SISTEMA RAG MINIMO - UFCSPA")
print("="*60)

# Dados de exemplo
docs = [
    {
        "titulo": "Estatuto UFCSPA",
        "conteudo": "A UFCSPA tem como princípios: excelência acadêmica, formação humanística, compromisso social, gestão democrática e autonomia universitária."
    },
    {
        "titulo": "Normas de Extensão",
        "conteudo": "As atividades de extensão incluem: programas, projetos, cursos, eventos e prestação de serviços. Devem ter relevância social e participação estudantil."
    },
    {
        "titulo": "Regimento Interno",
        "conteudo": "A administração é exercida pelo Conselho Universitário (CONSUN), CONSEPE e Reitoria. O Reitor é a autoridade executiva máxima."
    }
]

print("\n✅ Sistema carregado com 3 documentos de exemplo")
print("\n💡 Digite 'sair' para encerrar")

while True:
    pergunta = input("\n❓ Sua pergunta: ").strip()
    
    if pergunta.lower() in ['sair', 'exit']:
        print("\n👋 Até logo!")
        break
    
    if not pergunta:
        continue
    
    # Busca simples
    palavras = re.findall(r'\w+', pergunta.lower())
    resultados = []
    
    for doc in docs:
        score = sum(1 for p in palavras if p in doc["conteudo"].lower())
        if score > 0:
            resultados.append((doc, score))
    
    if resultados:
        resultados.sort(key=lambda x: x[1], reverse=True)
        print(f"\n📄 Encontrei {len(resultados)} documento(s) relevante(s):")
        for doc, score in resultados:
            print(f"\n• {doc['titulo']} (relevância: {score})")
            print(f"  {doc['conteudo'][:100]}...")
    else:
        print("\n❌ Nenhum documento relevante encontrado")
