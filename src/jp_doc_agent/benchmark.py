"""Download unmodified benchmark annotations separately from the document corpus."""

import hashlib
from pathlib import Path

import httpx

REVISION = "39fc62415ae5557f51d61ae64faf05494703afe5"
BASE_URL = (
    f"https://raw.githubusercontent.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark/{REVISION}/"
)
FILES = {
    "dataset/FJ_KGQA_Hard.yaml": "37eeaf1ae9088a2042fe15c983a46fdd8d85a1b430a65dade8cf8ec41c276b55",
    "TERMS_OF_USE.md": "0a39b67295d6e20d869c7564cbfacaab647c6c354158a9ccccd40adad815a7f1",
    "LICENSE": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
}


def fetch_benchmark(data_dir: Path) -> dict:
    destination = data_dir / "evaluation" / "fujitsu"
    destination.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for source, expected_hash in FILES.items():
            target = destination / Path(source).name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash:
                continue
            response = client.get(BASE_URL + source)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != expected_hash:
                raise ValueError(f"评测文件哈希不符：{source}")
            target.write_bytes(response.content)
    return {"status": "ok", "revision": REVISION, "directory": str(destination)}
