"""Alembic environment, sourcing schema and DB path from optionsbot.config + optionsbot.storage."""

from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context
from optionsbot.config import load_settings
from optionsbot.storage.schema import metadata as target_metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _db_url() -> str:
    # If the caller (e.g., tests) has set sqlalchemy.url on the alembic config,
    # honor it. Otherwise derive from optionsbot settings.
    override = config.get_main_option("sqlalchemy.url")
    if override:
        return override
    settings = load_settings()
    return f"sqlite:///{settings.storage.db_path}"


_SQLITE_URL_PREFIX = "sqlite:///"


def _ensure_sqlite_parent_dir(url: str) -> None:
    """For sqlite:/// URLs, create the parent directory so a fresh DB can be opened.

    Mirrors the directory-creation behavior of ``optionsbot.storage.db.create_engine_for_path``
    so a first-ever ``alembic upgrade head`` succeeds before any runtime code has had a
    chance to create the data directory.
    """
    if url.startswith(_SQLITE_URL_PREFIX):
        db_path = Path(url[len(_SQLITE_URL_PREFIX):])
        if db_path.parts:
            db_path.parent.mkdir(parents=True, exist_ok=True)


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _db_url()
    _ensure_sqlite_parent_dir(url)
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = url
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
