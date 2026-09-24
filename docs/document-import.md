# 文档导入

当前已导入 Fujitsu RAG Hard Benchmark 中的 5 份日语 PDF，共 170 页。这是文档处理阶段，还没有分块、向量化和问答。

## 数据来源

[Fujitsu RAG Hard Benchmark](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark)提供原始 PDF、100 道问答和证据页码。当前选择其仓库中的 5 份 PDF，覆盖其中 16 题的全部证据文档。

来源和文件 SHA-256 固定在 `sources/fujitsu.json`，数据集标签为 `fujitsu-rag-hard`。下载链接绑定上游 Git commit；标题保留数据集原始文件名，方便匹配问答证据。

软件与 PDF 的许可不同：PDF 限基准评测使用，不上传 GitHub 或用于公开文档服务。来源、许可和当前子集题目 ID 见 [数据来源说明](../sources/README.md)。原始问答单独保存，不写入正文表。

## 怎么运行

在项目目录执行：

```bash
# 启动数据库，应用表结构迁移
docker compose up -d --wait
uv run alembic upgrade head

# 下载并导入默认清单的 5 份 PDF
uv run jp-doc-agent import-documents

# 获取原始问答文件与使用条款（不写入数据库）
uv run jp-doc-agent fetch-benchmark

# 查看文档 ID、标题、页数、数据集和文件哈希
uv run jp-doc-agent documents

# 查看文档 1 的第 1 个物理页面
uv run jp-doc-agent page 1 1
```

最后一条命令中的 ID 以 `documents` 输出为准。系统使用从 1 开始的 PDF 物理页码，不使用纸面印刷页码。

导入其他本地文件：

```bash
uv run jp-doc-agent import-pdf /完整路径/document.pdf --title "文档标题"
```

可加 `--source-url` 记录原始链接、`--dataset` 标记来源。省略时记录本地文件地址，并使用 `local` 标签。批量导入其他来源时，可按 `sources/fujitsu.json` 的结构建立清单，再通过 `--manifest 清单路径` 指定。

## 处理流程

```text
读取来源清单 / 本地文件
          ↓
下载或复制 PDF → 检查格式和体积 → 计算 SHA-256 → 保存原文件
          ↓
查询相同哈希是否已入库
     ↙                  ↘
已存在：返回 duplicate    新文件：按页提取正文
                               ↓
                    文档信息与全部页面一起入库
                               ↓
                      输出结果与批量导入报告
```

- **去重与重解析**：相同文件、相同解析器版本返回 `duplicate`。升级解析器后，再次导入返回 `updated`，原子替换页面正文并保留文档 ID。相同内容从另一来源导入时保留首次来源；URL 内容变化则形成新的文件哈希。
- **版本核对**：示例清单已经固定文件哈希；网站更新文件导致哈希不符时，报告失败，核对后再更新清单。未经固定的自定义清单允许导入新版本。
- **页码**：按 PDF 实际顺序从 1 编号。中间空白页保留为空文本，避免后面的引用页码错位。
- **事务**：文档信息和全部页面在同一事务中保存。任何页面写入失败，整份文档的数据库写入回滚；已下载文件保留，便于排查和重试。
- **失败处理**：HTTP 错误、非 PDF、损坏或无法提取正文等情况会记录原因，批量任务继续尝试下一份。报告位于 `data/reports/`；存在失败时命令返回非零退出码。数据库不可用等整体故障会直接终止。
- **当前边界**：单文件最多 50 MiB、500 页；支持带加密标记但无需打开密码的公开 PDF，需要密码的文件会被拒绝。`pypdf[crypto]` 检查文件和页数，pdfminer.six 提取正文，改善本数据集的日语字体编码问题。暂不支持 OCR，复杂版式和图表理解仍需后续改善。
- **字形缺失提示**：仍无法识别的字形保留为 `(cid:编号)`，不猜测原字。导入报告列出相关页面，`page` 命令返回 `has_unmapped_characters`。这些页面后续需要人工核对或 OCR，涉及缺失字形的答案不能直接作为可靠结果。

## 数据保存在哪里

| 位置 | 内容 |
| --- | --- |
| `data/pdfs/<sha256>.pdf` | 下载或复制后的原始文件，本机缓存 |
| `data/reports/import-<时间>.json` | 每次批量导入的成功、重复或失败记录 |
| `data/evaluation/fujitsu/` | 原始问答 YAML、官方条款和软件许可证；不进入知识库 |
| PostgreSQL `documents` 表 | 文档 ID、标题、来源及重定向地址、数据集、文件哈希、路径、页数、获取/入库时间、解析器版本 |
| PostgreSQL `document_pages` 表 | 文档 ID、物理页码、页面正文 |

向量和评测答案都没有写入这两张表。数据库文件路径指向当前机器，换机器时根据来源清单重新导入原文件。

## 新增代码的职责

| 文件 | 功能 |
| --- | --- |
| `sources/fujitsu.json` | 固定版本与文件哈希的数据集 PDF 来源清单 |
| `src/jp_doc_agent/benchmark.py` | 下载并验证原始问答文件及条款，独立保存评测材料 |
| `src/jp_doc_agent/models.py` | 定义文档与页面两张表及约束 |
| `src/jp_doc_agent/ingestion/download.py` | HTTP 下载、本地文件复制、格式与体积检查、文件哈希及原文件保存 |
| `src/jp_doc_agent/ingestion/pdf.py` | 校验 PDF、按页提取日语文字、检查页数和缺失字形 |
| `src/jp_doc_agent/ingestion/service.py` | 协调下载、解析、事务入库，以及文档列表和页面查询；后续 HTTP 接口可复用 |
| `src/jp_doc_agent/cli.py` | 接收命令参数、调用业务功能、显示结果和保存报告 |
| `alembic.ini`、`migrations/env.py` | 配置数据库迁移，复用现有 `.env` 连接设置 |
| `migrations/versions/0001_documents.py` | 创建文档表、页面表、唯一哈希与外键约束 |
| `migrations/script.py.mako` | 以后生成新迁移文件时使用的模板 |
| `tests/conftest.py` | 为测试创建临时 PostgreSQL schema，并在结束后清理 |
| `tests/test_ingestion.py` | 验证去重、页码、回滚、并发、下载失败、加密及损坏 PDF、哈希与解析器升级 |

`ingestion/__init__.py` 标记文档导入模块为 Python 包。文件按下载、解析、存储流程划分，方便分别排查问题。

## 验证

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run alembic check
```

14 项测试覆盖解析、下载和数据库行为；数据库测试在真实 PostgreSQL 的临时 schema 中运行，不读写应用业务表。测试使用的小型构造 PDF 仅用于边界情况验证；实际资料来自富士通公开评测集。

本机实测导入 5 份，重复运行返回 5 个 `duplicate`，数据库仍为 5 份文档、170 页。页面查询能够返回日语正文、来源和物理页码。16 道已覆盖的题目仍含视觉任务，不能直接认定当前纯文本解析已足够回答全部问题。
