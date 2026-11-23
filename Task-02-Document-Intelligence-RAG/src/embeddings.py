from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local (free) embedding model - no API key required."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
