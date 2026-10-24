"""Download the dense embedding model: sentence-transformers/all-MiniLM-L6-v2 as ONNX.

Why this copy: the usual source (Hugging Face) may be blocked, and we want a
CPU-only, torch-free encoder. Chroma publishes the same model exported to ONNX
on S3, with a published sha256; we verify that hash before extracting.

Output: data/models/all-MiniLM-L6-v2/{model.onnx, tokenizer.json, ...}
"""
import hashlib
import io
import tarfile
from pathlib import Path

import requests

URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
OUT = Path("data/models/all-MiniLM-L6-v2")


def main() -> None:
    if (OUT / "model.onnx").exists():
        print(f"already present: {OUT}")
        return
    data = requests.get(URL, timeout=120).content
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch: got {digest}")
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():  # flatten onnx/<file> -> OUT/<file>; ignore anything else
                (OUT / Path(member.name).name).write_bytes(tar.extractfile(member).read())
    print(f"ok ({len(data) / 1e6:.0f} MB, sha256 verified): {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
