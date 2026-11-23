import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.embeddings import get_embeddings
from src.hybrid import BM25, fuse


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """One shared, persistent ChromaDB store (saved in the chroma_db folder)."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def _make_id(doc: Document) -> str:
    raw = f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{doc.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ---------- in-memory index (vectors + keyword index) per document ----------
@dataclass
class _DocIndex:
    docs: List[Document]
    vecs: np.ndarray  # L2-normalized embeddings, shape (n_chunks, dim)
    bm25: BM25


_index_cache: Dict[str, _DocIndex] = {}


def _load_index(name: str) -> Optional[_DocIndex]:
    if name in _index_cache:
        return _index_cache[name]
    res = get_vector_store().get(
        where={"source": name}, include=["documents", "metadatas", "embeddings"]
    )
    texts = res.get("documents")
    metas = res.get("metadatas")
    emb = res.get("embeddings")
    if texts is None or len(texts) == 0 or emb is None or len(emb) == 0:
        return None
    vecs = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas)]
    index = _DocIndex(docs=docs, vecs=vecs / norms, bm25=BM25(list(texts)))
    _index_cache[name] = index
    return index


# ---------- writing ----------
def add_chunks(chunks: List[Document]) -> int:
    """Store chunks in ChromaDB. Identical chunks are not added twice."""
    store = get_vector_store()
    ids = [_make_id(c) for c in chunks]
    unique = {i: c for i, c in zip(ids, chunks)}
    if unique:
        store.add_documents(list(unique.values()), ids=list(unique.keys()))
    _index_cache.clear()
    return len(unique)


def list_indexed_sources() -> Set[str]:
    """Names of all documents that currently have chunks in the database."""
    data = get_vector_store().get(include=["metadatas"])
    return {m.get("source", "?") for m in data["metadatas"]}


def delete_source(name: str) -> None:
    """Remove every chunk that belongs to one document."""
    store = get_vector_store()
    ids = store.get(where={"source": name})["ids"]
    if ids:
        store.delete(ids=ids)
    _index_cache.clear()


# ---------- searching ----------
def _hybrid_search(query: str, names: List[str], quota: int) -> List[Tuple[Document, float]]:
    """Vector similarity + BM25 keyword match, fused, for each document separately."""
    q = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm:
        q = q / norm

    merged: List[Tuple[Document, float]] = []
    for name in names:
        index = _load_index(name)
        if index is None:
            continue
        cosine = index.vecs @ q
        keyword = index.bm25.scores(query)
        for i in fuse(cosine, keyword, quota):
            merged.append((index.docs[i], float(max(0.0, min(1.0, cosine[i])))))
    return merged


def _vector_search(query: str, names: List[str], quota: int) -> List[Tuple[Document, float]]:
    """Plain vector search (fallback). Chroma returns squared L2 distance on normalized
    embeddings, so cosine similarity = 1 - distance / 2."""
    store = get_vector_store()
    merged: List[Tuple[Document, float]] = []
    for name in names:
        results = store.similarity_search_with_score(query, k=quota, filter={"source": name})
        merged.extend(
            (doc, max(0.0, min(1.0, 1.0 - dist / 2.0))) for doc, dist in results
        )
    return merged


def search(
    query: str, sources: Optional[List[str]] = None, k: int = config.TOP_K
) -> List[Tuple[Document, float]]:
    """Hybrid search over EACH selected document separately.

    Every selected document contributes its own best chunks, so a large document can
    never crowd out a small one. Returns (chunk, relevance) pairs sorted by relevance,
    where relevance is the cosine similarity (0 - 1) between question and chunk.
    """
    names = list(sources) if sources else sorted(list_indexed_sources())
    if not names:
        return []

    # chunks per document (shrinks only when many documents are selected)
    quota = min(k, max(2, config.MAX_CHUNKS // len(names)))

    try:
        merged = _hybrid_search(query, names, quota)
    except Exception:
        merged = []
    if not merged:  # fall back to plain vector search
        merged = _vector_search(query, names, quota)

    merged.sort(key=lambda pair: pair[1], reverse=True)
    return merged
