"""A local, production-style reference implementation of a multimodal RAG platform."""

from .agent import RagWorkflow
from .index import HybridIndex
from .models import Answer, Chunk, Document, SearchResult

__all__ = ["Answer", "Chunk", "Document", "HybridIndex", "RagWorkflow", "SearchResult"]
__version__ = "0.1.0"
