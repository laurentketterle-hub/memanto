"""Adapter auto-discovery and registration."""

from .memanto_adapter import MemantoAdapter
from .mem0_adapter import Mem0Adapter
from .zep_adapter import ZepAdapter
from .letta_adapter import LettaAdapter
from .langmem_adapter import LangMemAdapter

__all__ = [
    "MemantoAdapter",
    "Mem0Adapter",
    "ZepAdapter",
    "LettaAdapter",
    "LangMemAdapter",
]

ADAPTER_REGISTRY: dict[str, type] = {
    "memanto": MemantoAdapter,
    "mem0": Mem0Adapter,
    "zep": ZepAdapter,
    "letta": LettaAdapter,
    "langmem": LangMemAdapter,
}
