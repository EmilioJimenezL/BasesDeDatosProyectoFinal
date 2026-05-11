import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "docbase")

def main():
    stopwords_path = os.path.join(os.path.dirname(__file__), "spanish_stopwords.txt")
    with open(stopwords_path, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    # Remove duplicates from the file list just in case
    words = list(set(words))
    
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM STOP_WORD")
    initial_count = cursor.fetchone()[0]
    
    insert_query = "INSERT IGNORE INTO STOP_WORD (word) VALUES (%s)"
    data = [(w,) for w in words]
    cursor.executemany(insert_query, data)
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM STOP_WORD")
    final_count = cursor.fetchone()[0]
    
    inserted = final_count - initial_count
    existed = len(words) - inserted
    
    print(f"Total words to insert: {len(words)}")
    print(f"Words inserted: {inserted}")
    print(f"Words already existed (or ignored): {existed}")
    print(f"Total rows in STOP_WORD table: {final_count}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
