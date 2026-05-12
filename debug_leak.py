import os
import mysql.connector
from src.preprocessor import preprocess, load_stop_words, load_suffix_rules
from src.pdf_extractor import extract_all
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

# Let's find which words become "del", "fue", "les", "son", "sólo"
texts = extract_all("data/pdfs")

from src.preprocessor import tokenize, strip_suffix

for name, text in texts.items():
    tokens = tokenize(text)
    for t in tokens:
        if t in sw:
            continue
        stripped = strip_suffix(t, rules)
        if stripped in ["del", "fue", "les", "son", "sólo", "esta"]:
            print(f"Original word: '{t}' -> '{stripped}'")

conn.close()
