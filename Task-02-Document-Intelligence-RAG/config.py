import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True, encoding="utf-8-sig")

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Accepts either GEMINI_API_KEY or GOOGLE_API_KEY from the .env file
GOOGLE_API_KEY = (
    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
).strip().strip('"').strip("'")

# Models are tried in this order. If one fails, the next one is used automatically.
_DEFAULT_MODELS = "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.6-flash,gemini-3.1-flash-lite"
MODEL_CHAIN = [
    m.strip() for m in os.getenv("LLM_MODELS", _DEFAULT_MODELS).split(",") if m.strip()
]

# Seconds to wait for one model before falling back to the next one
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

# A model that fails/times out is skipped (tried last) for this many seconds
MODEL_COOLDOWN = int(os.getenv("MODEL_COOLDOWN", "300"))

# Low "thinking" makes Gemini 3.x answer much faster. Set to empty to disable.
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "low").strip()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

COLLECTION_NAME = "document_intelligence"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Chunks retrieved from EACH selected document, and a safety cap on the total sent to the model
TOP_K = 5
MAX_CHUNKS = 16

# Chunks with a relevance score below this are ignored (0.0 - 1.0)
MIN_RELEVANCE = 0.15
