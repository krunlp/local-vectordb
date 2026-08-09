"""
Optional text-embedding helper. Requires the `fastembed` package, which
uses ONNX Runtime rather than PyTorch — much smaller install, fast CPU
inference, good default choice for a local vector DB.

Install: pip install fastembed

Note: the first time you construct a TextEmbedder with a given model name,
fastembed downloads the model weights, which requires outbound internet
access to Hugging Face / the fastembed model registry. After that first
run, the model is cached locally (~/.cache/fastembed by default) and no
network access is needed.
"""
from typing import List, Optional

import numpy as np


class TextEmbedder:
    # bge-small-en-v1.5 is a strong, fast, small (384-dim) default.
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Optional[str] = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "TextEmbedder requires the `fastembed` package. "
                "Install it with: pip install fastembed"
            ) from e

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.model_name = model_name
        # Infer dim once via a throwaway encode call.
        probe = next(self._model.embed(["dimension probe"]))
        self.dim = len(probe)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings -> (n, dim) float32 array, L2-normalized."""
        vectors = list(self._model.embed(texts))
        arr = np.asarray(vectors, dtype=np.float32)
        return arr
