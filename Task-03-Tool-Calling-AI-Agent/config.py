import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

DATA_DIR = BASE_DIR / "data"
MAX_STEPS = 10  # agent loop ki max iterations
SHOW_THOUGHTS = os.getenv("SHOW_THOUGHTS", "true").lower() == "true"

SYSTEM_PROMPT = """You are a research assistant with three tools:
- web_search: real-time web search (news, facts, current data)
- calculator: exact math evaluation
- read_file: read local files from the data folder

Rules:
1. REASONING FIRST: every time you decide to call a tool, you MUST also write one short
   plain-text sentence in the same turn explaining what you need and why
   (example: "I need the file contents first, so I will read sample.txt.").
   Never call a tool without writing this sentence.
2. Break compound questions into steps and use the tools one step at a time.
3. Never do arithmetic in your head. Always use the calculator.
4. Never guess current facts. Use web_search.
5. If web_search returns SEARCH_UNAVAILABLE or fails twice, stop searching and tell the user
   honestly; never invent numbers. If web sources disagree, mention the range and say which one you used.
6. When you have everything, give a short final answer in PLAIN TEXT.
   No LaTeX, no $...$ math, no markdown tables."""
