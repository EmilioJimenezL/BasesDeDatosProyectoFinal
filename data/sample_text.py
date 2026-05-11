import glob
import random
import pdfplumber

def sample():
    pdfs = glob.glob("data/pdfs/*.pdf")
    chosen = random.sample(pdfs, 3)
    for p in chosen:
        with pdfplumber.open(p) as pdf:
            text = pdf.pages[0].extract_text()
            print(f"File: {p}")
            print(text[:300].replace('\n', ' '))
            print("-" * 50)

if __name__ == "__main__":
    sample()
