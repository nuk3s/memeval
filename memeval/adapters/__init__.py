from .base import MemoryAdapter
from .example_rest import ExampleRestAdapter
from .hindsight import HindsightAdapter
from .letta import LettaAdapter
from .mem0 import Mem0Adapter
from .zep import ZepAdapter

REGISTRY = {
    "hindsight": HindsightAdapter,
    "mem0": Mem0Adapter,
    "letta": LettaAdapter,
    "zep": ZepAdapter,
    "example-rest": ExampleRestAdapter,
}

__all__ = ["MemoryAdapter", "REGISTRY"]
