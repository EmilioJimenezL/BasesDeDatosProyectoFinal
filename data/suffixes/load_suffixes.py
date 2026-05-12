import os
import csv
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "docbase")

def main():
    suffixes_path = os.path.join(os.path.dirname(__file__), "spanish_suffixes.txt")
    
    suffixes = []
    with open(suffixes_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            suffix = row[0].strip()
            # If the CSV has a comma but nothing after it, row might have 1 or 2 elements
            replacement = row[1].strip() if len(row) > 1 else ""
            if suffix:
                suffixes.append((suffix, replacement))
    
    # Remove duplicates
    seen = set()
    unique_suffixes = []
    for s, r in suffixes:
        if s not in seen:
            seen.add(s)
            unique_suffixes.append((s, r))
    
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM SUFFIX")
    initial_count = cursor.fetchone()[0]
    
    # SUFFIX doesn't have UNIQUE constraint on suffix column in schema.sql
    # Wait, let me check schema.sql:
    # id INT AUTO_INCREMENT PRIMARY KEY, suffix VARCHAR(50), replacement VARCHAR(50) DEFAULT ''
    # There is NO unique constraint. To make it idempotent, we'll check if it exists first.
    
    inserted = 0
    for s, r in unique_suffixes:
        cursor.execute("SELECT id FROM SUFFIX WHERE suffix = %s", (s,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO SUFFIX (suffix, replacement) VALUES (%s, %s)", (s, r))
            inserted += 1
            
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM SUFFIX")
    final_count = cursor.fetchone()[0]
    
    print(f"Total rows inserted: {inserted}")
    print(f"Total rows in SUFFIX table: {final_count}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
