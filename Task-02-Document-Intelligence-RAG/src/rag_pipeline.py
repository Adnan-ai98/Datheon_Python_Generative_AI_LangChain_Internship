import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

import config
from src.document_loader import list_files, load_file
from src.text_splitter import split_documents
from src.vector_store import add_chunks, delete_source, list_indexed_sources, search

NOT_FOUND = "I could not find this information in the selected documents."

ANSWER_PROMPT = """You are a strict document question-answering assistant.
The context below is split into numbered chunks like [1], [2], ...

Rules:
1. Use ONLY facts that are explicitly stated in the context. Never guess, infer, calculate, or add outside knowledge.
2. The context may contain chunks from several different documents. Ignore chunks that are not relevant to the question and answer from the relevant ones. If the question names a specific paper or document (for example "GPT-3"), use only chunks from that document.
3. Answer only what was asked. Do not add unrelated facts. Start directly with the answer (no "Based on the context").
4. Copy numbers, names and values exactly as they appear in the context.
5. Write symbols in plain text, for example d_model, d_k, sqrt(d_k).
6. If the context does not contain the answer, reply exactly: "{not_found}" and nothing else.
7. Otherwise, end your reply with one final line in this format: SOURCES: 1, 3
   (list only the numbers of the chunks you actually used).

Context:
{context}

Question: {question}

Answer:"""

CONDENSE_PROMPT = """Given the conversation below and a follow-up question, rewrite the
follow-up as a single standalone question that can be understood without the conversation.
If it is already standalone, return it unchanged. Return ONLY the question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

# Words that suggest a question depends on earlier messages
_REFERENCE_WORDS = {
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "he", "she", "his", "her", "above", "previous", "same", "former", "latter",
    "also", "another",
}
_SOURCES_RE = re.compile(r"^\s*SOURCES?\s*:\s*([\d,\s]+)\s*$", re.IGNORECASE | re.MULTILINE)


def _to_text(message) -> str:
    """Extract plain text from a model response (string or list of content blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _looks_like_follow_up(question: str) -> bool:
    q = question.lower().strip()
    words = set(re.findall(r"[a-z']+", q))
    return bool(words & _REFERENCE_WORDS) or q.startswith(("and ", "what about", "how about"))


class RAGPipeline:
    def __init__(self):
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("API key not found. Set GEMINI_API_KEY in the .env file.")
        self._clients: Dict[tuple, ChatGoogleGenerativeAI] = {}
        self._cooldown: Dict[str, float] = {}  # model -> time until which it is skipped
        self.notes: List[str] = []  # what happened during the last call (for the UI)

    # ---------- LLM with timeout, cooldown and automatic fallback ----------
    def _client(self, model: str, thinking: bool) -> ChatGoogleGenerativeAI:
        key = (model, thinking)
        if key not in self._clients:
            kwargs = dict(
                model=model,
                temperature=0,
                google_api_key=config.GOOGLE_API_KEY,
                max_retries=0,
                timeout=config.REQUEST_TIMEOUT,
            )
            if thinking and config.THINKING_LEVEL:
                kwargs["thinking_level"] = config.THINKING_LEVEL
            self._clients[key] = ChatGoogleGenerativeAI(**kwargs)
        return self._clients[key]

    def _generate(self, prompt: str, models: Optional[List[str]] = None):
        """Try models in order (recently failed ones last). Returns (text, model)."""
        now = time.time()
        models = list(models or config.MODEL_CHAIN)
        healthy = [m for m in models if self._cooldown.get(m, 0) <= now]
        order = healthy + [m for m in models if m not in healthy]

        errors = []
        for model in order:
            for thinking in ([True, False] if config.THINKING_LEVEL else [False]):
                began = time.perf_counter()
                try:
                    response = self._client(model, thinking).invoke(prompt)
                    return _to_text(response).strip(), model
                except Exception as e:
                    msg = str(e)
                    took = time.perf_counter() - began
                    errors.append(f"{model}: {msg[:200]}")
                    if "thinking" in msg.lower():
                        continue  # retry same model without the thinking setting
                    self._cooldown[model] = time.time() + config.MODEL_COOLDOWN
                    self.notes.append(f"{model} failed after {took:.0f}s")
                    break
        raise RuntimeError("All models failed:\n" + "\n".join(errors))

    # ---------- Documents ----------
    def list_documents(self) -> List[str]:
        """Documents saved on disk (they persist between runs)."""
        return [p.name for p in list_files(config.DATA_DIR)]

    def ingest(self, file_paths: List[str]) -> int:
        """Load -> split -> embed -> store. Re-uploading a file replaces its old chunks."""
        total = 0
        for fp in file_paths:
            delete_source(Path(fp).name)
            total += add_chunks(split_documents(load_file(fp)))
        return total

    def sync(self) -> int:
        """Index any saved file that is not in the database yet."""
        indexed = list_indexed_sources()
        missing = [str(p) for p in list_files(config.DATA_DIR) if p.name not in indexed]
        return self.ingest(missing) if missing else 0

    def delete_document(self, name: str) -> None:
        delete_source(name)
        path = config.DATA_DIR / name
        if path.exists():
            path.unlink()

    # ---------- Question answering ----------
    def _standalone_question(self, question: str, history: Optional[List[Dict]]) -> str:
        """Rewrite follow-up questions (it/this/that...) so retrieval works."""
        if not history or not _looks_like_follow_up(question):
            return question
        lines = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
            for m in history[-4:]
        ]
        prompt = CONDENSE_PROMPT.format(history="\n".join(lines), question=question)
        try:
            text, _ = self._generate(prompt)
            return text or question
        except RuntimeError:
            return question

    def ask(
        self,
        question: str,
        sources: Optional[List[str]] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Answer strictly from the selected documents and cite only the chunks used."""
        start = time.perf_counter()
        self.notes = []
        standalone = self._standalone_question(question, history)

        raw = search(standalone, sources, config.TOP_K)
        hits = [(doc, score) for doc, score in raw if score >= config.MIN_RELEVANCE]
        debug = {
            "selected": list(sources or []),
            "query": standalone,
            "chunks": [
                {
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                    "score": score,
                    "used": score >= config.MIN_RELEVANCE,
                    "snippet": d.page_content[:120].replace("\n", " "),
                }
                for d, score in raw
            ],
        }
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "model": None,
                    "seconds": None, "notes": self.notes, "debug": debug}

        context = "\n\n".join(
            f"[{i}] (file: {d.metadata.get('source')}, page {d.metadata.get('page')})\n{d.page_content}"
            for i, (d, _) in enumerate(hits, start=1)
        )
        prompt = ANSWER_PROMPT.format(
            not_found=NOT_FOUND, context=context, question=standalone
        )
        answer, model = self._generate(prompt)
        seconds = time.perf_counter() - start

        if NOT_FOUND in answer:
            return {"answer": NOT_FOUND, "sources": [], "model": model,
                    "seconds": seconds, "notes": self.notes, "debug": debug}

        # Keep only the chunks the model says it used
        used = list(range(1, len(hits) + 1))
        matches = _SOURCES_RE.findall(answer)
        if matches:
            nums = [int(n) for n in re.findall(r"\d+", matches[-1])]
            picked = [n for n in nums if 1 <= n <= len(hits)]
            if picked:
                used = sorted(set(picked))
        answer = _SOURCES_RE.sub("", answer).strip()

        sources_out = [
            {
                "source": hits[i - 1][0].metadata.get("source"),
                "page": hits[i - 1][0].metadata.get("page"),
                "score": hits[i - 1][1],
                "snippet": hits[i - 1][0].page_content[:300].replace("\n", " "),
            }
            for i in used
        ]
        return {"answer": answer, "sources": sources_out, "model": model,
                "seconds": seconds, "notes": self.notes, "debug": debug}
