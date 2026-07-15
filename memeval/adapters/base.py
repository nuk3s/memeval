from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class MemoryAdapter(ABC):
    """The contract a memory system has to meet to be evaluated.

    An item is a dict: {"content": str, "id": Optional[str], "tags": Optional[list],
    "context": Optional[str]}. A recall result is a dict: {"text": str, "score": float}.

    retain and recall are required. consolidate, supersede, prepare, and config are
    optional; the eval uses them if present and skips the parts that need them if not.
    """

    name = "base"
    status = "template"  # "tested" | "template" | "test-only"

    def config(self):
        """Connection settings for run_meta. Values pass through events.redact(),
        but never return anything you would mind seeing masked in a log."""
        return {}

    @abstractmethod
    def retain(self, items: List[Dict]) -> None:
        """Store a batch of items. Should return only once they are persisted."""

    @abstractmethod
    def recall(self, query: str, k: int = 10) -> List[Dict]:
        """Return up to k results, best first, as {"text", "score"} dicts."""

    def consolidate(self) -> None:
        """Trigger any read-time or batch consolidation. No-op if unsupported."""
        return

    def supersede(self, doc_id: str, content: str, tags: Optional[list] = None) -> None:
        """Replace the document with id doc_id. Raise if the system can't do it."""
        raise NotImplementedError(f"{self.name} does not support supersede")

    def prepare(self) -> None:
        """Optional setup before an eval run (e.g. create a scratch namespace)."""
        return
