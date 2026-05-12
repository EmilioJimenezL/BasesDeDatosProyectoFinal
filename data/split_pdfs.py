from pypdf import PdfReader, PdfWriter
import os

pdf_path = "data/LFT.pdf"
out_dir = "data/pdfs"

os.makedirs(out_dir, exist_ok=True)

reader = PdfReader(pdf_path)
total_pages = len(reader.pages)

chunks = [
    (0, 5, "lft_01_titulo_primero.pdf"),
    (5, 20, "lft_02_titulo_segundo.pdf"),
    (20, 32, "lft_03_titulo_tercero.pdf"),
    (32, 52, "lft_04_titulo_cuarto.pdf"),
    (52, 54, "lft_05_titulo_quinto.pdf"),
    (54, 57, "lft_06_titulo_quinto_bis.pdf"),
    (57, 107, "lft_07_titulo_sexto.pdf"),
    (107, 136, "lft_08_titulo_septimo.pdf"),
    (136, 139, "lft_09_titulo_octavo.pdf"),
    (139, total_pages, "lft_10_titulo_noveno.pdf"),
]

for start, end, filename in chunks:
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "wb") as f:
        writer.write(f)
    print(f"Created {filename} with {end - start} pages.")
