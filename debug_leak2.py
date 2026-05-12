import os
import mysql.connector
from src.preprocessor import preprocess, load_stop_words, load_suffix_rules
from dotenv import load_dotenv

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT', 3306)),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASS'),
    database=os.getenv('DB_NAME')
)

sw = load_stop_words(conn)
rules = load_suffix_rules(conn)

print("'del' in sw:", "del" in sw)
print("'dél' in sw:", "dél" in sw)
print("'deles' stripped:", [t for t in ["deles"] if t not in sw])

# Find ALL terms in TERM table that are also in STOP_WORD table
cursor = conn.cursor()
cursor.execute("SELECT t.name FROM TERM t JOIN STOP_WORD sw ON sw.word = t.name")
leaked = [r[0] for r in cursor.fetchall()]

print("Leaked terms:", leaked)

# Now for each leaked term, print its existence in `sw`
for l in leaked:
    print(f"'{l}' in sw (Python):", l in sw)

conn.close()
