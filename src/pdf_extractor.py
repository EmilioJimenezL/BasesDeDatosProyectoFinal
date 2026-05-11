"""PDF text extraction using pdfplumber.

Extracts raw text page-by-page from PDF files in data/pdfs/. Handles
encrypted PDFs (retries with blank password) and strips likely headers/footers.
Returns a dict mapping document name to full extracted text.
"""

import glob
import logging
import os

import pdfplumber

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> str:
    """Open a PDF and return concatenated text from all pages.

    - Retries with password='' if the PDF is encrypted.
    - Strips the first and last line of each page when they are < 40 chars
      (likely page numbers or headers).
    - Raises ValueError if extracted text is < 100 characters.
    """
    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception:
        # Retry with blank password for encrypted PDFs
        pdf = pdfplumber.open(pdf_path, password="")

    pages_text = []
    with pdf:
        for page in pdf.pages:
            raw = page.extract_text()
            if not raw:
                continue
            lines = raw.split("\n")

            # Strip header (first line) if short
            if lines and len(lines[0]) < 40:
                lines = lines[1:]
            # Strip footer (last line) if short
            if lines and len(lines[-1]) < 40:
                lines = lines[:-1]

            pages_text.append("\n".join(lines))

    full_text = "\n".join(pages_text)

    if len(full_text) < 100:
        raise ValueError(
            f"{os.path.basename(pdf_path)}: only {len(full_text)} characters extracted"
        )

    return full_text


def extract_all(pdf_dir: str) -> dict[str, str]:
    """Extract text from every *.pdf in *pdf_dir*.

    Returns a dict mapping filename stem (no extension) → full text.
    Logs a WARNING and skips any file that raises an exception.
    """
    results: dict[str, str] = {}
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    for path in pdf_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            results[stem] = extract_text(path)
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)

    return results
