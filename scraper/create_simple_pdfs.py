"""
Cria PDFs de exemplo simples para testar o sistema.
Versão que usa apenas bibliotecas padrão.
"""

from pathlib import Path
import datetime


def create_text_files():
    """Cria arquivos de texto de exemplo diretamente."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Texto 1: Estatuto
    text1 = """ESTATUTO DA UNIVERSIDADE FEDERAL DE CIÊNCIAS DA SAÚDE DE PORTO ALEGRE

CAPÍTULO I - DA UNIVERSIDADE E SEUS FINS

Art. 1º A Universidade Federal de Ciências da Saúde de Porto Alegre - UFCSPA, criada pela Lei nº 11.641, de 13 de janeiro de 2008, é uma fundação pública, dotada de personalidade jurídica de direito público, com autonomia didático-científica, administrativa e de gestão financeira e patrimonial.

Art. 2º A UFCSPA tem sede e foro na cidade de Porto Alegre, Estado do Rio Grande do Sul.

Art. 3º A UFCSPA tem por finalidade promover o ensino, a pesquisa e a extensão, com ênfase na área da saúde.

CAPÍTULO II - DOS PRINCÍPIOS

Art. 4º A UFCSPA, no cumprimento de sua missão, orienta-se pelos seguintes princípios:
I - indissociabilidade entre ensino, pesquisa e extensão;
II - universalidade do conhecimento e educação;
III - interdisciplinaridade do conhecimento;
IV - excelência acadêmica;
V - democratização do acesso e permanência;
VI - gestão democrática e colegiada;
VII - valorização dos profissionais da educação;
VIII - compromisso social e ético.

CAPÍTULO III - DA ORGANIZAÇÃO

Art. 5º A estrutura organizacional da UFCSPA compreende:
I - Órgãos Colegiados Superiores;
II - Reitoria;
III - Unidades Acadêmicas;
IV - Órgãos Suplementares.

Art. 6º São órgãos colegiados superiores:
I - Conselho Universitário (CONSUN);
II - Conselho de Ensino, Pesquisa e Extensão (CONSEPE).

Art. 7º A Reitoria é o órgão executivo central que coordena e superintende todas as atividades da Universidade."""

    # Texto 2: Regimento
    text2 = """REGIMENTO INTERNO DA UFCSPA

TÍTULO I - DAS DISPOSIÇÕES PRELIMINARES

Art. 1º O presente Regimento Interno disciplina a organização e o funcionamento da Universidade Federal de Ciências da Saúde de Porto Alegre - UFCSPA, nos termos de seu Estatuto e da legislação em vigor.

TÍTULO II - DA ESTRUTURA ORGANIZACIONAL

CAPÍTULO I - DOS ÓRGÃOS COLEGIADOS

Art. 2º São órgãos colegiados superiores da UFCSPA:
I - o Conselho Universitário (CONSUN);
II - o Conselho de Ensino, Pesquisa e Extensão (CONSEPE).

Art. 3º O Conselho Universitário é o órgão máximo deliberativo e normativo da Universidade, competindo-lhe estabelecer a política geral da instituição em matéria administrativa, econômico-financeira, patrimonial e disciplinar.

CAPÍTULO II - DA REITORIA

Art. 4º A Reitoria é o órgão executivo central que coordena e superintende todas as atividades da Universidade.

Art. 5º A Reitoria é exercida pelo Reitor, auxiliado pelo Vice-Reitor.

Art. 6º O Reitor e o Vice-Reitor serão nomeados pelo Presidente da República, escolhidos entre os docentes dos dois níveis mais elevados da carreira ou que possuam título de doutor.

TÍTULO III - DO ENSINO

Art. 7º O ensino na UFCSPA será ministrado nos seguintes níveis:
I - Graduação;
II - Pós-Graduação;
III - Extensão.

Art. 8º Os cursos de graduação destinam-se à formação profissional em nível superior, conferindo diplomas aos concluintes."""

    # Texto 3: Normas de Extensão
    text3 = """NORMAS PARA ATIVIDADES DE EXTENSÃO

RESOLUÇÃO Nº 001/2023 - CONSEPE

Estabelece as normas para as atividades de extensão universitária no âmbito da UFCSPA.

Art. 1º As atividades de extensão universitária são processos interdisciplinares, educativos, culturais, científicos e políticos que promovem a interação transformadora entre a Universidade e outros setores da sociedade.

Art. 2º São consideradas atividades de extensão:
I - programas;
II - projetos;
III - cursos e oficinas;
IV - eventos;
V - prestação de serviços.

Art. 3º As atividades de extensão devem:
I - estar vinculadas ao processo de formação de pessoas e de geração de conhecimento;
II - ter relação com a sociedade de forma a estabelecer troca de saberes;
III - produzir impacto na formação do estudante;
IV - produzir impacto social.

Art. 4º Todas as atividades de extensão devem ser registradas na Pró-Reitoria de Extensão e Cultura.

Art. 5º Os projetos de extensão devem conter:
I - título e área temática;
II - coordenador e equipe executora;
III - justificativa e objetivos;
IV - metodologia;
V - cronograma de execução;
VI - orçamento detalhado;
VII - resultados esperados.

Art. 6º A certificação das atividades de extensão será emitida pela Pró-Reitoria de Extensão, após a conclusão e aprovação do relatório final."""

    # Salva os arquivos
    files = [
        ("exemplo_estatuto_ufcspa.txt", text1),
        ("exemplo_regimento_interno.txt", text2),
        ("exemplo_normas_extensao.txt", text3)
    ]
    
    for filename, content in files:
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Criado: {filename}")
    
    print(f"\n✓ Arquivos de texto criados em {output_dir}")
    print("\nAgora você pode pular direto para o chunking:")
    print("  python ingest/chunk.py")


if __name__ == "__main__":
    create_text_files()