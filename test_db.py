#!/usr/bin/env python3
"""
test_postgres_db.py

Simple script to verify Postgres connectivity for this project.

Usage:
  - Set DATABASE_URL (preferred) or DB_USER, DB_USER_PASSWORD, DB_HOST, DB_PORT, DB_NAME.
  - Run: python test_postgres_db.py

Exit codes:
  0 - success (connected and queried)
  1 - connection/query failed
  2 - missing configuration
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env file
from environ import Env
env = Env()
Env.read_env(Path(__file__).resolve().parent / '.env')

import psycopg2


def get_dsn() -> str | None:
    """Return a connection string (DSN). Prefer DATABASE_URL.
    If not present, build DSN from DB_* environment variables.
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER")
    password = os.getenv("DB_USER_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB")

    if not (user and password and db):
        return None

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def mask_dsn(dsn: str) -> str:
    """Return a version of DSN with password masked for safe logging."""
    try:
        # naive masking: replace :<password>@ with :*@
        before, after = dsn.split("@", 1)
        if ":" in before:
            userpart = before.split(":", 1)[0]
            return f"{userpart}:*@{after}"
    except Exception:
        pass
    return dsn


def main() -> None:
    dsn = get_dsn()
    if not dsn:
        print(
            "ERROR: No DATABASE_URL or DB_USER/DB_USER_PASSWORD/DB_NAME environment variables found.",
            file=sys.stderr,
        )
        sys.exit(2)

    timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))

    print("Testing Postgres connectivity...")
    print("Using:", mask_dsn(dsn))

    try:
        conn = psycopg2.connect(dsn, connect_timeout=timeout)
        cur = conn.cursor()
        cur.execute("SELECT version(), current_database(), current_user;")
        row = cur.fetchone()
        print("Connected successfully.")
        print("Postgres version:", row[0])
        print("Database:", row[1])
        print("User:", row[2])
        cur.close()
        conn.close()
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - simple CLI
        print("Connection failed:", exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()