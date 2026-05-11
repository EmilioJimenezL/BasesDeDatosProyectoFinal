import os
import glob
import pdfplumber

def main():
    pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    
    total = len(pdf_files)
    passed = 0
    failed = 0
    
    print("Verifying PDFs...")
    print("-" * 50)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                extracted_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
                
                char_count = len(extracted_text)
                first_200 = extracted_text[:200].replace('\n', ' ')
                
                status = "PASS" if char_count > 500 else "FAIL"
                if status == "PASS":
                    passed += 1
                else:
                    failed += 1
                
                print(f"File: {filename}")
                print(f"Pages: {num_pages}")
                print(f"Characters: {char_count}")
                print(f"Text snippet: {first_200}")
                print(f"Status: {status}")
                print("-" * 50)
                
        except Exception as e:
            print(f"File: {filename}")
            print(f"Error opening/reading PDF: {e}")
            print("Status: FAIL")
            print("-" * 50)
            failed += 1
            
    print(f"Summary: Total: {total}, Passed: {passed}, Failed: {failed}")

if __name__ == "__main__":
    main()
