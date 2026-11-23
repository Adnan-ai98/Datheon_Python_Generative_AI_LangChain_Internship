from langchain_core.tools import tool

from config import DATA_DIR

MAX_CHARS = 6000
ALLOWED = {".txt", ".md", ".csv", ".json", ".log"}


@tool
def read_file(filename: str) -> str:
    """Read a local text file from the data folder (txt, md, csv, json, log).
    Input is just the file name, e.g. 'sample.txt'. Pass 'list' to see files."""
    if filename.strip().lower() == "list":
        files = [p.name for p in DATA_DIR.iterdir() if p.is_file()]
        return "Files: " + ", ".join(files) if files else "data folder is empty."

    path = (DATA_DIR / filename).resolve()
    # Path traversal se bachao: file data/ ke andar hi honi chahiye
    if DATA_DIR.resolve() not in path.parents:
        return "Error: access outside the data folder is not allowed."
    if not path.is_file():
        return f"Error: '{filename}' not found. Use 'list' to see available files."
    if path.suffix.lower() not in ALLOWED:
        return f"Error: unsupported file type '{path.suffix}'."

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n...[truncated]"
    return text
