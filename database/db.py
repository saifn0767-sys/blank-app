"""Prediction log. Uses a cloud SQL database when DATABASE_URL is set
(env var or Streamlit secret), otherwise falls back to local SQLite."""
import os
import queue
import threading
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, func, inspect, select, text)
from sqlalchemy.exc import IntegrityError, ProgrammingError, OperationalError

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise']

_metadata = MetaData()
emotion_log = Table(
    'emotion_log', _metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('captured_at', DateTime, server_default=func.now()),
    Column('session_id', String(32)),
    Column('dominant_emotion', String(20)),
    Column('confidence', Float),
    *[Column(e, Float, default=0.0) for e in EMOTIONS],
)


def _database_url():
    url = os.environ.get('DATABASE_URL')
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get('DATABASE_URL')
        except Exception:
            url = None
    if not url:
        return f"sqlite:///{DATA_DIR / 'emotion_log.db'}"
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url


@lru_cache(maxsize=1)
def engine():
    return create_engine(_database_url(), pool_pre_ping=True, pool_recycle=300)


def is_cloud():
    return not _database_url().startswith('sqlite')


_init_lock = threading.Lock()
_initialized = False


def initialize_db():
    """Create the table once per process. A Postgres advisory lock stops several
    sessions from running CREATE TABLE at the same moment (UniqueViolation)."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        eng = engine()
        last_error = None
        for _ in range(4):
            try:
                if not inspect(eng).has_table('emotion_log'):
                    with eng.begin() as con:
                        if eng.dialect.name == 'postgresql':
                            con.execute(text('SELECT pg_advisory_xact_lock(727274)'))
                        _metadata.create_all(con, checkfirst=True)
                last_error = None
                break
            except (IntegrityError, ProgrammingError, OperationalError) as exc:
                last_error = exc
                time.sleep(0.7)
        if inspect(eng).has_table('emotion_log'):
            _initialized = True
        elif last_error is not None:
            raise last_error


# Writes go through a background thread so a slow network never stalls the video.
_queue = queue.Queue(maxsize=200)


def _worker():
    while True:
        row = _queue.get()
        try:
            with engine().begin() as con:
                con.execute(emotion_log.insert().values(**row))
        except Exception as exc:
            print('DB write failed:', exc)


threading.Thread(target=_worker, daemon=True).start()


def log_prediction(session_id, label, probs):
    row = {'session_id': str(session_id), 'dominant_emotion': str(label),
           'confidence': float(probs.get(label, max(probs.values(), default=0.0)))}
    row.update({e: float(probs.get(e, 0.0)) for e in EMOTIONS})
    try:
        _queue.put_nowait(row)
    except queue.Full:
        pass


def read_logs(limit=5000):
    query = select(emotion_log).order_by(emotion_log.c.id.desc()).limit(int(limit))
    with engine().connect() as con:
        return pd.read_sql_query(query, con)
