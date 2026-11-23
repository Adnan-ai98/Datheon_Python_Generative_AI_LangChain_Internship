"""Show exactly what retrieval returns, without calling the LLM.

Usage (from the project root, with the venv active):
    python debug_retrieval.py "How long did the big models train?" attention_is_all_you_need.pdf
    python debug_retrieval.py "How long did the big models train?" gpt3.pdf attention_is_all_you_need.pdf
    python debug_retrieval.py "How long did the big models train?"        (all indexed documents)
"""
import sys

import config
from src.vector_store import list_indexed_sources, search

if len(sys.argv) < 2:
    sys.exit(__doc__)

question = sys.argv[1]
docs = sys.argv[2:] or None

print("Indexed documents:", sorted(list_indexed_sources()))
print("Selected         :", docs or "ALL")
print("Question         :", question)
print(f"MIN_RELEVANCE    : {config.MIN_RELEVANCE}\n")

results = search(question, docs, config.TOP_K)
if not results:
    print("No chunks returned. Is the document name spelled exactly as in the sidebar?")

for doc, score in results:
    flag = "USED" if score >= config.MIN_RELEVANCE else "below threshold"
    print(f"[{score:.0%}] {flag:15} {doc.metadata.get('source')} p.{doc.metadata.get('page')}")
    print("      " + doc.page_content[:150].replace("\n", " ") + "...\n")
