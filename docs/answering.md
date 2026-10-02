# 出典付きの基本 RAG 問答

1 回のベクトル検索と 1 回の回答生成を組み合わせ、日本語の結論に原文引用・文書名・取得元 URL・物理ページ番号を付けます。`answering/generator.py` は回答用の指示、`schema.py` は草案の構造、`service.py` は検索との連携・引用の照合を担当します。共用 API クライアントは `llm.py` に置き、[LangGraph による追加調査](agent.md) でも利用します。

## 準備と実行

[ベクトル検索](retrieval.md) が実行できる環境を使います。既存の `.env` の API キーを再利用し、次の項目は省略すると既定値になります。

```dotenv
OPENAI_ANSWER_MODEL=gpt-4.1-mini-2025-04-14
OPENAI_ANSWER_MAX_OUTPUT_TOKENS=2048
```

既定の GPT-4.1 mini は Responses API と構造化出力に対応しています。基準のモデル版を固定するためスナップショットを指定します。比較時は環境変数で変更でき、返却結果に実際のモデル名と要求したモデル名を記録します。別モデルには Responses API と JSON Schema の構造化出力の対応が必要です。利用可否はアカウントによって異なり、API エラー時に別モデルへ自動的に切り替えません。

仕様は OpenAI Docs の [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)、[Responses API によるテキスト生成](https://developers.openai.com/api/docs/guides/text)、[構造化出力](https://developers.openai.com/api/docs/guides/structured-outputs) を参照してください。

```bash
# 文書 ID は documents で確認（1 は例）
uv run jp-doc-agent ask "2024年度第2四半期の売上収益はいくらですか？" --document-id 1

# 全文書から上位 5 件を検索して回答
uv run jp-doc-agent ask "2021年1月1日時点の富士通の組織構成はどうなっていますか？" --top-k 5
```

`top-k` は 1〜100、既定値は 5 です。質問の検証と文書の指定は検索と共通です。出力 Token 上限は 256〜8,192、既定値は 2,048 です。既存の検索結果だけを文脈に使い、検索漏れを別の資料や外部知識で補いません。

## 回答と引用の契約

API には質問と検索した chunk ID・文書名・本文だけを JSON として送ります。資料内の命令は参照データとして扱うよう指示し、年度・日付・単位・条件・図表の対応関係を推測しないように指示します。評価の正解、証拠ページ、期待するキーワードは送信しません。

Structured Outputs の厳密な JSON Schema と Pydantic を使い、次の草案を受け取ります。

- `answered`: 1 件以上の結論があり、全ての結論に 1 件以上の chunk ID・連続した引用原文がある。不足情報は空配列。
- `insufficient_evidence`: 結論と引用は空配列。不足している情報を日本語で示す。

プログラムは、引用した chunk が今回の検索結果に含まれること、引用原文が本文に完全一致することを確認します。改行や空白の正規化、離れた原文の連結を許可しません。同じ chunk と同じ引用原文は一つの引用番号にまとめ、各結論に番号を付けます。草案に URL や引用番号を直接生成した場合は中止します。

URL、文書名、ページ番号、文字範囲は取得済みの検索結果から補います。引用がページをまたぐ場合は、引用が交差する全ページの原文位置を返します。引用が一つのページに収まる場合は、そのページだけを引用します。文字範囲は 0 始まり・終端を含まない Python 文字列の位置で、ページ結合の改行だけは原文に対応しません。

例として、ローカルの決算概要を対象にした実 API の確認では、次の回答を得ました。

> 2024年度第2四半期の売上収益は8,666億円です。 [1]

引用 `[1]` の原文が取得した本文に含まれることを検証し、文書名・取得元 URL・物理ページ 31 と対応する文字範囲を返しました。これは少数例の動作確認です。

## 出力・保存・失敗

CLI は JSON を返します。

| 項目 | 内容 |
| --- | --- |
| `question` / `status` / `answer` | 質問、日本語の回答、回答可能または根拠不足 |
| `statements` | 結論本文と検証済み引用番号の対応 |
| `citations` | 引用原文、chunk ID、文書情報、URL、全ページ・文字範囲 |
| `missing_information` | 取得資料では確認できない情報 |
| `model` / `requested_model` / `prompt_version` | 実行モデル、要求モデル、指示の版 |
| `max_output_tokens` / `elapsed_seconds` | 出力上限と検索・生成・検証の所要時間 |
| `usage` | 質問 Embedding と回答生成の使用 Token 数を別々に表示 |
| `retrieval` | Embedding 設定、検索範囲、未処理数、上位 K 件の ID・ページ・類似度 |

`ask` は DB を更新せず、質問・回答を DB に保存しません。必要な場合は出力を保存できます。

```bash
mkdir -p data/reports
uv run jp-doc-agent ask "2024年度第2四半期の売上収益はいくらですか？" > data/reports/answer.json
```

API の待機中は DB 接続を保持しません。回答生成中に再分割された場合は、検索で取得した時点の本文・出典を使って引用を照合します。その chunk ID は後から DB で参照できなくなる場合がありますが、返却結果の引用原文と文字範囲は保持します。Agent の回答・実行記録は JSON に保存できます。DB による永続的な回答履歴や文書版へのリンクは未実装です。

API 接続のタイムアウトは 1 回 30 秒、SDK の再試行上限は 2 回です。応答の未完了・生成拒否・不正な JSON・引用不一致では回答を表示せず終了コード 1 とします。API 本文・キー・不正な草案はエラーに表示しません。根拠不足は正常な回答結果なので終了コード 0 です。

`usage` は正常に検証した応答の Token 数です。SDK 内の再試行、API 応答後の検証失敗に伴う使用量、課金総額は表しません。Responses API には `store=False` を指定していますが、この設定はアカウント全体のデータ保持条件を変更するものではありません。

## 確認と制約

```bash
uv run pytest -q tests/test_answering.py tests/test_evaluation.py
uv run ruff check .
uv run ruff format --check .
uv run alembic check
```

API の模擬応答と一時 DB schema で、構造の検証、全結論への引用、未知 chunk の拒否、引用原文の一致、ページをまたぐ位置、重複引用、根拠不足、未完了・拒否・API エラー、回答中の再生成、CLI の表示とクライアントの終了を検証します。

引用検証は参照先と原文の一致を確認します。原文が結論を意味的に裏付けるか、数値の解釈・年月・単位が正しいかは、この検証だけでは保証しません。検索漏れ、OCR 未対応、表の読み順や字形の解析不良も回答に影響します。[少数の固定ケース](evaluation.md) で失敗を記録し、Agent と比較しています。
