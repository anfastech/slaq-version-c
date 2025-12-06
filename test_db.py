#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time

import django
from django.conf import settings
import psycopg2


# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'slaq_project.settings')
django.setup()


# CLI arguments
parser = argparse.ArgumentParser(description="Simple Postgres connection test")
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
parser.add_argument('--retry', type=int, default=1, help='Number of connection retries')
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format='%(levelname)s: %(message)s'
)


def get_dsn():
    """Return DATABASE_URL from Django settings."""
    try:
        dsn = settings.DATABASES['default']['OPTIONS'].get('dsn') \
            if 'OPTIONS' in settings.DATABASES['default'] else None

        if not dsn:
            dsn = settings.DATABASES['default'].get('CONN_MAX_AGE')  # placeholder fallback
        dsn = settings.DATABASES['default'].get('ENGINE')

        dsn = settings.DATABASES['default'].get('NAME')

    except Exception:
        dsn = None

    # Preferred method
    try:
        url = settings.DATABASE_URL
        if url:
            return url
    except Exception:
        pass

    # Standard Django style
    try:
        db = settings.DATABASES['default']
        user = db.get('USER')
        password = db.get('PASSWORD')
        host = db.get('HOST') or 'localhost'
        port = db.get('PORT') or '5432'
        name = db.get('NAME')

        if user and password and name:
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    except Exception:
        pass

    return None


def mask_dsn(dsn: str) -> str:
    """Hide password for logging."""
    try:
        prefix, rest = dsn.split('@', 1)
        if ':' in prefix:
            user = prefix.split(':', 1)[0]
            return f"{user}:*****@{rest}"
    except Exception:
        pass
    return dsn


def main():
    dsn = get_dsn()
    if not dsn:
        logging.error("Database URL is missing")
        sys.exit(2)

    logging.info("Checking Postgres connection")
    logging.info(f"Using DSN: {mask_dsn(dsn)}")

    for attempt in range(1, args.retry + 1):
        try:
            conn = psycopg2.connect(dsn)
            conn.close()
            logging.info("Connection ok")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Connection failed on attempt {attempt}: {e}")
            if attempt < args.retry:
                time.sleep(2)
            else:
                logging.error("Connection cannot be established")
                sys.exit(1)


if __name__ == '__main__':
    main()
