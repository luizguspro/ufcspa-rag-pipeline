# teste_rapido.py
print("\n=== TESTE DO SISTEMA RAG ===\n")

docs = {
    "Estatuto": "A UFCSPA tem como princípios: excelência acadêmica, formação humanística, compromisso social.",
    "Extensão": "As atividades de extensão incluem programas, projetos, cursos e eventos.",
    "Regimento": "O CONSUN é o órgão máximo. O Reitor é a autoridade executiva."
}

pergunta = input("Faça uma pergunta sobre a UFCSPA: ")

print(f"\n🔍 Buscando sobre: {pergunta}")
print("\n📄 Documentos encontrados:")

for titulo, texto in docs.items():
    if any(palavra in texto.lower() for palavra in pergunta.lower().split()):
        print(f"\n• {titulo}: {texto}")

print("\n✅ Sistema funcionando! O sistema completo tem muito mais recursos.")