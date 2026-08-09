from .core import VectorDB

__all__ = ["VectorDB"]
__version__ = "0.1.0"

# TextEmbedder is not imported here by default because fastembed is an
# optional dependency — import it explicitly when needed:
#   from vectordb.embeddings import TextEmbedder
