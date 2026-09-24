"""Database connection and a read-only PostgreSQL/pgvector check."""

from sqlalchemy import Engine, create_engine, text

from jp_doc_agent.config import Settings


def create_database_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000"},
    )


def check_database(engine: Engine) -> dict[str, str | float]:
    """Verify connectivity, the extension, and an actual vector operation."""
    with engine.connect() as connection:
        postgres_version = connection.execute(text("SHOW server_version")).scalar_one()
        vector_version = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        if vector_version is None:
            raise RuntimeError("当前数据库尚未启用 vector 扩展，请执行 docker/init.sql。")

        distance = connection.execute(
            text("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector")
        ).scalar_one()
        if distance != 1.0:
            raise RuntimeError("向量距离计算结果异常。")

    return {
        "postgresql": postgres_version,
        "pgvector": vector_version,
        "vector_distance": float(distance),
    }
