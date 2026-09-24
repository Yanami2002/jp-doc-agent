# 本地开发

所有命令均在仓库根目录执行。Python 在本机运行，PostgreSQL 在 Docker 中运行。

## 首次准备

需要 Git、uv 和已启动的 Docker Desktop。安装 Python 和项目依赖：

```bash
uv python install
uv sync --locked
```

首次克隆项目时，复制 `.env.example` 为 `.env`，将 `POSTGRES_PASSWORD` 改为随机密码。`.env` 已被 Git 忽略，不要提交密码。

```bash
docker compose up -d --wait
uv run alembic upgrade head
uv run jp-doc-agent check-db
```

检查成功会输出 `status: ok`、PostgreSQL 版本、pgvector 版本和向量距离 `1.0`。该命令只读取数据库，不创建业务表。

默认使用 `127.0.0.1:5433`，避免与本机已有的 PostgreSQL（5432）冲突；数据库和用户名均为 `jp_doc_agent`，使用独立的容器与数据卷。

## 常用命令

```bash
# 启动数据库并等待就绪
docker compose up -d --wait

# 检查数据库及向量运算
uv run jp-doc-agent check-db

# 查看数据库状态和日志
docker compose ps
docker compose logs --tail 50 db

# 检查代码和格式
uv run ruff check .
uv run ruff format --check .

# 暂停数据库，保留数据
docker compose stop
```

修改 `.env` 中的密码不会修改已有数据库的密码；初始化 SQL 也只在空数据卷首次启动时运行。需要重新启用扩展时，可以对当前数据库执行已有脚本：

```bash
docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < docker/init.sql
```

## 文件结构

```text
jp-doc-agent/
├── README.md                 # 项目需求与开发进度
├── pyproject.toml            # 项目依赖、命令入口、代码检查配置
├── uv.lock                   # 锁定依赖版本，提交到 Git
├── .python-version           # Python 版本
├── .env.example              # 可提交的配置模板
├── .env                      # 本机配置与密码，不提交
├── .gitignore                # 排除环境、密码、缓存和下载数据
├── compose.yaml              # PostgreSQL 容器、端口和持久化配置
├── alembic.ini               # 数据库迁移配置
├── migrations/              # 版本化的表结构变更
├── sources/                 # 公开 PDF 来源清单
├── tests/                   # 隔离运行的数据库集成测试
├── docker/
│   └── init.sql              # 首次启动时启用 pgvector
├── docs/
│   ├── development.md        # 本文：启动方式与文件说明
│   └── document-import.md    # 文档导入流程、命令与模块说明
└── src/
    └── jp_doc_agent/
        ├── __init__.py       # Python 包标识
        ├── config.py         # 读取配置，构建数据库连接地址
        ├── database.py       # 创建连接，检查数据库与向量运算
        ├── cli.py            # jp-doc-agent 命令入口
        ├── models.py         # 文档与页面的数据模型
        ├── benchmark.py      # 单独下载数据集原始评测标注
        └── ingestion/        # PDF 下载、解析和入库
```

`.venv/` 是 uv 自动生成的本地 Python 环境，不属于源码。

目前已安装 SQLAlchemy、psycopg、配置管理、httpx、pypdf[crypto]、pdfminer.six 和 Alembic，以及 Ruff、pytest 开发工具。FastAPI 和 LangGraph 在开发对应功能时加入。

文档导入已完成，运行方式和新增文件说明见 [文档导入](document-import.md)。下一步是页面正文切分和基础检索。
