from .._lib import store as _store  # pyright: ignore[reportMissingModuleSource]


class MemoryStore(_store.MemoryStore):
    """A fully in-memory implementation of ObjectStore.

    Create a new in-memory store::

        store = MemoryStore()
    """

    def __init__(self) -> None:
        return super().__init__()
