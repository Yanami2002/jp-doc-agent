# ローカル開発ガイド

コマンドはリポジトリのルートで実行してください。Python はホスト側、PostgreSQL は Docker コンテナで動かします。モデル API は取り込み・チャンク分割の実行には必要ありません。

## 初回セットアップ

Git、uv、起動済みの Docker Desktop を用意します。

```bash
uv python install
uv sync --locked
cp .env.example .env
```

`.env` の `POSTGRES_PASSWORD` をランダムなパスワードに変更してから起動します。既に `.env` がある環境では上書きしないでください。

```bash
docker compose up -d --wait
uv run alembic upgrade head
uv run jp-doc-agent check-db
```

接続確認に成功すると `status: ok`、PostgreSQL と pgvector のバージョン、ベクトル距離 `1.0` が返ります。`check-db` は読み取り専用で、業務テーブルは作成しません。

既定の接続先は `127.0.0.1:5433`、DB 名・ユーザー名は `jp_doc_agent` です。独立したコンテナと永続化ボリュームを使います。

## 日常の開発

```bash
# DB を起動
docker compose up -d --wait

# 依存関係とスキーマを最新のリポジトリに合わせる
uv sync --locked
uv run alembic upgrade head

# ヘルプを確認
uv run jp-doc-agent --help

# PDF と評価データを取得
uv run jp-doc-agent import-documents
uv run jp-doc-agent fetch-benchmark

# 最大 300 Token・重複最大 30 Token でチャンクを生成して文書一覧を確認
uv run jp-doc-agent chunk-documents
uv run jp-doc-agent documents

# API キー設定後にベクトル化し、保存件数を確認
uv run jp-doc-agent embed-chunks
uv run jp-doc-agent embedding-status

# テスト・静的検査・スキーマの差分確認
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run alembic check

# コンテナの状態・ログ
docker compose ps
docker compose logs --tail 50 db

# データを保持したまま停止
docker compose stop
```

文書 ID は `documents` の出力で確認してください。ページは `page <文書ID> <ページ番号>`、チャンクは `chunks <文書ID> --page <ページ番号>` で参照できます。

## ファイル構成

```text
jp-doc-agent/
├── README.md                 # 概要、実装状況、技術選定、評価方針
├── pyproject.toml            # 依存関係、CLI、検査設定
├── uv.lock                   # 依存バージョンの固定
├── .python-version           # Python バージョン
├── .env.example              # 設定のテンプレート
├── compose.yaml              # PostgreSQL と永続化設定
├── alembic.ini
├── migrations/               # 文書・ページ・チャンク・ベクトルのスキーマ変更
├── sources/                  # 公開 PDF の取得元とハッシュ
├── docker/init.sql           # pgvector の有効化
├── tests/
│   ├── conftest.py           # 一時 DB schema と PDF fixture
│   ├── test_ingestion.py    # PDF 取り込みの検証
│   ├── test_chunking.py     # DB を使わない分割・原文位置の検証
│   ├── test_chunk_service.py # 保存・参照・再生成・CLI の検証
│   ├── test_chunk_migrations.py # 旧スキーマからの移行と復元
│   ├── test_embedding.py     # API 応答と SDK のリトライ
│   ├── test_embedding_service.py # ベクトル保存・再開・競合
│   └── test_embedding_migrations.py # ベクトルのスキーマ移行
├── docs/
│   ├── development.md        # 本ガイド
│   ├── document-import.md    # PDF 取り込み
│   ├── chunking.md           # 分割方法、保存形式、検証結果
│   └── embedding.md          # API 接続、ベクトル保存、再開
└── src/jp_doc_agent/
    ├── config.py             # .env から設定を読み込む
    ├── database.py           # DB 接続と pgvector の確認
    ├── cli.py                # CLI の引数と出力
    ├── models.py             # データモデルと DB 制約
    ├── benchmark.py          # 正解データを本文とは別に取得
    ├── chunking/
    │   ├── splitter.py       # Token 計数、本文分割、ページへの対応付け
    │   └── service.py        # チャンクの保存、再生成、出典付き参照
    ├── embedding/
    │   ├── encoder.py        # API 接続、入力と応答の検証
    │   └── service.py        # バッチ保存、重複スキップ、保存件数の確認
    └── ingestion/
        ├── download.py       # ダウンロード、形式・サイズ・ハッシュの確認
        ├── pdf.py            # ページ単位の日本語本文抽出
        └── service.py        # 取り込みとページ参照
```

`.env`、`.venv/`、`data/`、キャッシュは Git の管理対象外です。FastAPI と LangGraph は対応機能の実装時に追加します。

機能ごとの処理は `ingestion/`、`chunking/`、`embedding/` に置き、設定・接続・モデル・CLI は共通部分としてパッケージ直下に置きます。`chunking/splitter.py` は DB 接続や PDF の取得に依存せず、`chunking/service.py` がトランザクションと永続化を担当します。`migrations/versions/` は適用済み環境を更新するための履歴なので、古いファイルも保持します。

## データベースの注意点

`.env` のパスワードを書き換えても、作成済みの DB のパスワードは変更されません。初期化 SQL は空のボリュームから初めて起動したときにだけ実行されます。既存の DB で拡張を有効にする場合は次のコマンドを使います。

```bash
docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < docker/init.sql
```

Docker に接続できない場合は Docker Desktop の起動を確認してください。テストの DB fixture は一時 schema を作成・削除するため、その権限が必要です。

## Git の運用

機能開発は `feat/basic-rag` のようなブランチで行い、動作確認できる単位でコミットします。必要なテスト・説明を含めて GitHub に push し、機能が利用可能になったら PR を通して `main` に統合します。PR には変更点と確認結果を記載します。
