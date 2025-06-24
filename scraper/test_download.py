"""
Script de teste para verificar o site da UFCSPA e baixar um PDF de exemplo.
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time


def test_site():
    """Testa acesso ao site e busca links."""
    print("🔍 Testando acesso ao site da UFCSPA...")
    
    url = "https://ufcspa.edu.br/sobre-a-ufcspa/normas"
    
    try:
        # Faz requisição
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }, timeout=10)
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Tamanho da página: {len(response.text)} bytes")
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Busca todos os links
        all_links = soup.find_all('a', href=True)
        print(f"\n📋 Total de links encontrados: {len(all_links)}")
        
        # Busca links de PDFs
        pdf_links = [link for link in all_links if '.pdf' in link.get('href', '').lower()]
        print(f"📄 Links para PDFs: {len(pdf_links)}")
        
        # Mostra primeiros 5 links de PDF
        if pdf_links:
            print("\nPrimeiros PDFs encontrados:")
            for i, link in enumerate(pdf_links[:5]):
                href = link['href']
                text = link.get_text(strip=True)
                print(f"  [{i+1}] {text[:50]}...")
                print(f"      URL: {href}")
        
        # Busca links relevantes
        relevant_links = []
        keywords = ['norma', 'regimento', 'estatuto', 'resolucao', 'portaria']
        
        for link in all_links:
            href = link.get('href', '').lower()
            text = link.get_text(strip=True).lower()
            
            if any(kw in href or kw in text for kw in keywords):
                relevant_links.append(link)
        
        print(f"\n🎯 Links relevantes (normas, regimentos, etc): {len(relevant_links)}")
        
        # Mostra alguns links relevantes
        if relevant_links:
            print("\nLinks relevantes encontrados:")
            for i, link in enumerate(relevant_links[:5]):
                href = link['href']
                text = link.get_text(strip=True)
                print(f"  [{i+1}] {text[:50]}...")
                print(f"      URL: {href}")
        
        # Testa download de um PDF se encontrar
        if pdf_links:
            print("\n📥 Tentando baixar o primeiro PDF...")
            first_pdf = pdf_links[0]['href']
            
            # Constrói URL completa
            if first_pdf.startswith('http'):
                pdf_url = first_pdf
            elif first_pdf.startswith('/'):
                pdf_url = f"https://ufcspa.edu.br{first_pdf}"
            else:
                pdf_url = f"https://ufcspa.edu.br/{first_pdf}"
            
            print(f"URL do PDF: {pdf_url}")
            
            # Baixa o PDF
            pdf_response = requests.get(pdf_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }, timeout=30)
            
            if pdf_response.status_code == 200:
                # Salva o PDF
                output_dir = Path("data/raw")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                filename = "teste_" + Path(pdf_url).name
                filepath = output_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(pdf_response.content)
                
                print(f"✓ PDF baixado com sucesso: {filename}")
                print(f"✓ Tamanho: {len(pdf_response.content)} bytes")
                print(f"✓ Salvo em: {filepath}")
            else:
                print(f"✗ Erro ao baixar PDF: Status {pdf_response.status_code}")
        
    except Exception as e:
        print(f"✗ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_site()