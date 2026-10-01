# 日本語本文のチャンク分割

登録済みのページ本文を、検索に使える小さなテキストの単位に分割して保存します。実装は `src/jp_doc_agent/chunking.py`、テーブルは `document_chunks` です。Embedding と検索は後続の開発で追加します。

## 実行方法

```bash
uv sync --locked
docker compose up -d --wait
uv run alembic upgrade head

# 登録済みの全文書を分割
uv run jp-doc-agent chunk-documents

# 1 文書だけ分割（ID は documents の出力に合わせる）
uv run jp-doc-agent chunk-documents --document-id 1

# 最大文字数と重複の目安を変更して再生成
uv run jp-doc-agent chunk-documents --document-id 1 --chunk-size 600 --chunk-overlap 80

# 分割結果、原文位置、取得元、使用設定を確認
uv run jp-doc-agent chunks 1 --page 1 --limit 5
uv run jp-doc-agent chunks 1 --limit 20 --offset 20
```

`chunks` の `limit` は 1〜100、既定値は 20 です。`total` はページで絞り込んだ後の全件数、`offset` は先頭からスキップする件数です。未分割の文書は `total: 0`、`chunking_config: null` で返します。白紙ページにもチャンクは生成しませんが、ページの記録は保持します。

## 分割の設定と理由

[LangChain の RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) を利用し、次の順で区切りを探します。

```python
["\n\n", "。", "！", "？", "\n", "、", " ", ""]
```

段落と日本語の文末を優先し、長すぎる部分にはより細かい区切りを使います。最後の空文字列は、区切りのない長文を文字単位で分割するためのものです。PDF の単一改行はレイアウト由来の場合があるため、句点より低い優先度にしています。これは文章構造を復元する処理ではなく、現在の抽出本文に適用する初期設定です。

| 設定 | 既定値 | 意味 |
| --- | --- | --- |
| `chunk_size` | 800 | 1 チャンクの最大文字数 |
| `chunk_overlap` | 100 | 隣接チャンクの重複文字数の目安 |
| 長さの計測 | Python の `len` | Unicode のコードポイント数。モデルの Token 数とは異なる |
| `keep_separator` | `end` | 区切り文字を直前のテキストに残す |
| `strip_whitespace` | `False` | 原文との対応のため、空白を自動削除しない |
| 分割範囲 | 1 ページ内 | 物理ページ番号を直接参照できるようにする |

重複は目安で、段落・文の境界によって実際の重複文字数は少なくなる場合があります。`chunk_size` は正の整数、`chunk_overlap` は 0 以上かつ `chunk_size` 未満です。800/100 は初期値であり、最適な設定として評価した値ではありません。

文字数による上限は Embedding モデルの Token 上限を保証しません。モデル選定後に対応する tokenizer で長さを確認します。日本語の形態素解析は、後続のキーワード検索で別途検討します。LangGraph は今後の Agent フローを担当し、分割処理は独立して実行できます。

## 原文への追跡

各チャンクに `document_id`、`page_number`、ページ内の 0 始まりの `chunk_index`、本文、`start_char`、`end_char` を保存します。文字範囲は保存済みページ本文に対する Python のインデックスで、開始を含み、終了は含みません。

```python
chunk.text == page.text[chunk.start_char : chunk.end_char]
```

保存前に上記の一致、最大長、位置の進行、空白以外の本文が抜け落ちていないことを検証します。分割ライブラリが返す開始位置を検証できなければ、保存を中止します。元の本文は変更しません。

チャンクからページへの複合外部キーで、存在しないページへの登録を防ぎます。ページ内の順序は一意で、文字範囲と本文の長さも DB 制約で確認します。タイトルと取得元は文書テーブルから参照し、解決できない字形の有無は `has_unmapped_characters` で返します。

## 再実行・設定変更・再解析

文書ごとに、分割アルゴリズム、ライブラリのバージョン、設定、解析器のバージョン、全ページ本文から署名を作ります。設定は `documents.chunking_config`、署名は `documents.chunking_signature` に保存します。

- 同じ署名では `duplicate` を返し、既存チャンクを再利用します。ID も変わりません。
- 設定または本文が変わると、その文書のチャンクを同一トランザクションで置き換えます。生成するチャンクの ID は変わります。
- 文書をロックしてから処理し、同じ文書の並行分割や PDF 再解析と競合する更新を直列化します。
- 保存が失敗すると、以前のチャンクと設定を保持します。
- PDF 再解析でページを置き換える際は、古いチャンクを削除して設定をリセットします。改めて `chunk-documents` を実行してください。
- 一括処理のトランザクション境界は文書単位です。途中で失敗した場合、それ以前に完了した文書は保持されます。エラーを修正して再実行すれば、同じ設定で完了済みの文書はスキップされます。

## 検証結果と現在の制約

既定設定で 5 文書・170 ページから 393 チャンクを生成しました。同じ処理を繰り返すと 5 件とも `duplicate` になり、件数は変わりません。チャンク分割の単体・DB 統合テストでは、日本語の句点、改行、反復文、Unicode、長文、白紙ページ、原文の一致、重複排除、設定変更、ロールバック、並行処理、再解析、CLI を確認しています。

```bash
uv run pytest -q tests/test_chunking.py
```

ページをまたぐ文章、見出しの階層、表の行・列関係、図の内容は今回の分割では復元しません。後続では、検索で見つけた小さなチャンクにページ本文や隣接ページを補い、必要な文脈を確保する方法を評価します。検索の改善は、分割結果の見た目だけで判断せず、証拠ページの取得率と回答結果で確認します。
