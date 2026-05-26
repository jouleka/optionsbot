"""Engine + session factory. Always enables WAL mode on SQLite."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event


def create_engine_for_path(db_path: Path | str) -> Engine:
    """Create a SQLAlchemy engine for the given SQLite file, with WAL mode enabled."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine
