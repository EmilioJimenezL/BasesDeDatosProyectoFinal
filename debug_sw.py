import os
import mysql.connector
from src.preprocessor import load_stop_words, preprocess
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
print("del in sw?", "del" in sw)
print("del" in [w.strip() for w in sw])
print("repr of del in sw:", [repr(w) for w in sw if "del" in w])

# Also check "fue", "esta", "son"
for test in ["fue", "esta", "son", "les"]:
    print(f"{test} in sw?", test in sw)
    print(f"repr:", [repr(w) for w in sw if test in w])

conn.close()
