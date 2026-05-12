import os
import glob
from datetime import date
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "docbase")

def main():
    pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = conn.cursor()
    
    # Try to add UNIQUE constraint on url if it doesn't exist so INSERT IGNORE works
    try:
        cursor.execute("ALTER TABLE DOCUMENT ADD UNIQUE (url(255))")
    except Exception:
        pass # Probably already exists or can't be added, we'll fall back to manual check
        
    inserted_count = 0
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        title = os.path.splitext(filename)[0]
        author = "Cámara de Diputados"
        doc_date = date.today().isoformat()
        
        # We will use manual check to ensure idempotency since we don't know the exact index status
        cursor.execute("SELECT id FROM DOCUMENT WHERE url = %s", (filename,))
        row = cursor.fetchone()
        
        if not row:
            cursor.execute(
                "INSERT IGNORE INTO DOCUMENT (url, title, author, doc_date) VALUES (%s, %s, %s, %s)",
                (filename, title, author, doc_date)
            )
            inserted_id = cursor.lastrowid
            print(f"Inserted document ID: {inserted_id}, Title: {title}")
            inserted_count += 1
        else:
            print(f"Document already exists ID: {row[0]}, Title: {title}")
            
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM DOCUMENT")
    total_docs = cursor.fetchone()[0]
    
    print(f"\nTotal documents in the DOCUMENT table: {total_docs}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
