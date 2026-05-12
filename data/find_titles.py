import pdfplumber
import re

pdf_path = "data/LFT.pdf"
titles = []

try:
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i > 150: # Check first 150 pages
                break
            text = page.extract_text()
            if text:
                # Look for lines starting with TÍTULO or TITULO
                for line in text.split('\n'):
                    if re.match(r'^(TÍTULO|TITULO)\s+[A-Z]+', line):
                        titles.append((i, line))
                        break # One title per page is enough for splitting
except Exception as e:
    print(f"Error: {e}")

for t in titles:
    print(t)
