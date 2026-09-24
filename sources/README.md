# 数据来源

当前数据来自 [Fujitsu RAG Hard Benchmark](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark)，由 FujitsuResearch 发布。

- 固定版本：`39fc62415ae5557f51d61ae64faf05494703afe5`。
- `fujitsu.json`：从该版本仓库直接下载的 5 份 PDF；`title` 保留原始文件名，用于与问答标注的 `rationales.file_name` 对应。
- 每份 PDF 固定 SHA-256，原文件和问答标注均保存在被 Git 忽略的 `data/`。
- 官方原始问答共 100 题，包含答案、证据文件、从 1 开始的页码和难度等标注。运行 `uv run jp-doc-agent fetch-benchmark` 获取，保持文件原样。

## 当前子集

5 份 PDF 共 170 页。原始 100 题中，以下 16 题的所有证据文档均包含在本子集中：

`64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 81`

这是按证据文件覆盖筛选的结果，不代表 16 题都适合纯文本 RAG。其中部分问题需要理解图表，后续应单独报告正文、表格和视觉任务结果。其他问题可能引用未导入的 PDF，不应直接用于这个子集的正确率统计。

## 使用范围

官方[使用条款](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark/blob/39fc62415ae5557f51d61ae64faf05494703afe5/TERMS_OF_USE.md)区分软件与数据：软件采用 Apache-2.0；仓库提供的 PDF 限于该基准的评测用途，禁止商业使用及重新分发等行为。

本项目将这些 PDF 用于本地基准评测开发，公开仓库只提交自己的代码、下载清单和说明，不上传数据或用它们提供公开文档服务。不能把仓库的软件许可证理解为 PDF 也采用 Apache-2.0。日后如需公开产品演示或商业部署，应使用另行获准的数据。
