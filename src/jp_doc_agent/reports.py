"""評価と HTTP 問答の JSON 実行記録を保存する。"""

import json
from pathlib import Path


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
