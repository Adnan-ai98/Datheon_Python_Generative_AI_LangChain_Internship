"""Keyword (BM25) scoring and rank fusion, used together with vector search.

Vector search finds chunks with a similar *meaning*, but it can miss chunks that
contain the exact words of a factual question (for example "175 billion parameters").
BM25 finds exact-word matches. Fusing both rankings gives the best of both.
"""
import math
import re
from collections import Counter
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "do", "does", "did", "have", "has", "had",
    "what", "which", "who", "when", "where", "why", "how", "many", "much",
    "and", "or", "that", "this", "it", "its", "we", "our",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(c.values()) for c in self.tf], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(texts) else 1.0
        self.df: Counter = Counter()
        for c in self.tf:
            self.df.update(c.keys())

    def scores(self, query: str) -> np.ndarray:
        n = len(self.tf)
        out = np.zeros(n, dtype=np.float32)
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out


def fuse(vec_scores: np.ndarray, kw_scores: np.ndarray, top: int,
         pool: int = 30, rrf_k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion of the vector ranking and the keyword ranking."""
    fused: dict = {}
    for rank, i in enumerate(np.argsort(-vec_scores)[:pool]):
        fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank)
    for rank, i in enumerate(np.argsort(-kw_scores)[:pool]):
        if kw_scores[i] <= 0:
            break
        fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused, key=fused.get, reverse=True)[:top]
