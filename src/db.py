"""Database connection and CRUD helpers for the docbase MySQL schema.

Provides a connection factory (using mysql-connector-python and .env credentials)
and low-level insert/select utilities for every table in schema.sql.
All credentials are loaded from .env via python-dotenv — nothing is hardcoded.
"""

import os
from contextlib import contextmanager

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "docbase")


def get_connection():
    """Return a mysql.connector connection with autocommit=False.

    Raises RuntimeError on connection failure (includes host/db but not password).
    """
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            autocommit=False,
            charset="utf8mb4",
            collation="utf8mb4_spanish_ci",
        )
        return conn
    except mysql.connector.Error as e:
        raise RuntimeError(
            f"Failed to connect to MySQL at {DB_HOST}:{DB_PORT}/{DB_NAME}: {e}"
        ) from e


def execute_query(conn, sql, params=None):
    """Execute a SELECT query and return a list of dicts."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, params or ())
    rows = cursor.fetchall()
    cursor.close()
    return rows


def execute_write(conn, sql, params=None):
    """Execute an INSERT/UPDATE/DELETE, commit, and return lastrowid."""
    cursor = conn.cursor()
    cursor.execute(sql, params or ())
    conn.commit()
    last_id = cursor.lastrowid
    cursor.close()
    return last_id


def execute_many(conn, sql, rows):
    """Execute a batch insert via executemany, commit, and return rowcount."""
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    count = cursor.rowcount
    cursor.close()
    return count


@contextmanager
def get_db():
    """Context manager that opens a connection, yields it, and closes on exit.

    Rolls back on exception, always closes the connection.
    """
    conn = get_connection()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
