from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

SUPPORTED = {".pdf", ".txt"}


def load_file(file_path) -> List[Document]:
    """Load a single PDF or TXT file. Page numbers are 1-based."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
        for d in docs:
            d.metadata["page"] = int(d.metadata.get("page", 0)) + 1  # 0-based -> 1-based
    elif ext == ".txt":
        docs = TextLoader(str(path), encoding="utf-8").load()
        for d in docs:
            d.metadata["page"] = 1
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for d in docs:
        d.metadata["source"] = path.name
    return docs


def list_files(directory) -> List[Path]:
    """Return all supported files in a folder, sorted by name."""
    return sorted(
        p for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
