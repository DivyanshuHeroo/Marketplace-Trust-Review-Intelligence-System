"""Run the named SQL queries in sql/analytics.sql against the SQLite database.

Each query in the .sql file is preceded by a `-- name: <query_name>` marker. This
module parses those out so we can run them by name from Python / the dashboard:

    from src.etl.queries import run_query, list_queries
    df = run_query("delivery_vs_satisfaction")

Run as a script to execute *all* analytics queries and print a preview of each:

    python -m src.etl.queries
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache

import pandas as pd

from src.utils.config import db_path, resolve_path

SQL_FILE = resolve_path("sql/analytics.sql")
_NAME_RE = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


@lru_cache(maxsize=1)
def _parse_named_queries() -> dict[str, str]:
    """Split analytics.sql into {name: sql} using the `-- name:` markers."""
    text = SQL_FILE.read_text()
    matches = list(_NAME_RE.finditer(text))
    queries: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # strip trailing comment-only lines but keep the SQL statement
        queries[name] = body
    return queries


def list_queries() -> list[str]:
    return list(_parse_named_queries().keys())


def get_sql(name: str) -> str:
    queries = _parse_named_queries()
    if name not in queries:
        raise KeyError(f"Unknown query '{name}'. Available: {list(queries)}")
    return queries[name]


def run_query(name: str) -> pd.DataFrame:
    """Execute a named query and return the result as a DataFrame."""
    sql = get_sql(name)
    with sqlite3.connect(db_path()) as conn:
        return pd.read_sql_query(sql, conn)


def run_sql(sql: str) -> pd.DataFrame:
    """Execute an arbitrary SQL string (used by the dashboard's SQL explorer)."""
    with sqlite3.connect(db_path()) as conn:
        return pd.read_sql_query(sql, conn)


def main() -> None:
    print("=" * 64)
    print("OlistTrust — running analytical SQL queries")
    print("=" * 64)
    for name in list_queries():
        df = run_query(name)
        print(f"\n### {name}  ({len(df)} rows)")
        print(df.head(8).to_string(index=False))
    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
