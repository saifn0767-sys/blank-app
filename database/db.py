from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / 'emotion_log.db'
CSV_PATH = DATA_DIR / 'emotion_log.csv'
EMOTIONS = ['anger','contempt','disgust','fear','happy','sad','surprise']

def connection():
    return sqlite3.connect(DB_PATH, timeout=15)

def initialize_db():
    with connection() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS emotion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            dominant_emotion TEXT,
            confidence REAL,
            anger REAL DEFAULT 0, contempt REAL DEFAULT 0, disgust REAL DEFAULT 0,
            fear REAL DEFAULT 0, happy REAL DEFAULT 0, sad REAL DEFAULT 0, surprise REAL DEFAULT 0
        )''')

def log_prediction(session_id, label, probs):
    vals = [float(probs.get(k, 0.0)) for k in EMOTIONS]
    with connection() as con:
        con.execute('''INSERT INTO emotion_log
        (session_id, dominant_emotion, confidence, anger, contempt, disgust, fear, happy, sad, surprise)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (str(session_id), str(label), float(probs.get(label, max(probs.values(), default=0))), *vals))

def read_logs(limit=5000):
    with connection() as con:
        return pd.read_sql_query('SELECT * FROM emotion_log ORDER BY id DESC LIMIT ?', con, params=(int(limit),))

def export_csv():
    df = read_logs(limit=1000000)
    df.to_csv(CSV_PATH, index=False)
    return CSV_PATH
