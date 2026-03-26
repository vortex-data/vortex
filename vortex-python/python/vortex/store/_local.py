from pathlib import Path
from typing import Self

from typing_extensions import override

from .._lib import store as _store  # pyright: ignore[reportMissingModuleSource]


class LocalStore(_store.LocalStore):
    """An ObjectStore interface to local filesystem storage.

    Create a local store with an optional directory prefix::

        from pathlib import Path

        store = LocalStore()
        store = LocalStore(prefix="/path/to/directory")
        store = LocalStore(prefix=Path("."))
    """

    def __init__(
        self,
        prefix: str | Path | None = None,
        *,
        automatic_cleanup: bool = False,
        mkdir: bool = False,
    ) -> None:
        """Create a new LocalStore.

        Args:
            prefix: Use the specified prefix applied to all paths. Defaults to ``None``.

        Keyword Args:
            automatic_cleanup: if ``True``, enables automatic cleanup of empty directories
                when deleting files. Defaults to False.
            mkdir: if ``True`` and ``prefix`` is not ``None``, the directory at ``prefix`` will
                attempt to be created. Note that this root directory will not be cleaned
                up, even if ``automatic_cleanup`` is ``True``.

        """
        return super().__init__(prefix, automatic_cleanup=automatic_cleanup, mkdir=mkdir)

    @classmethod
    @override
    def from_url(
        cls,
        url: str,
        *,
        automatic_cleanup: bool = False,
        mkdir: bool = False,
    ) -> Self:
        """Construct a new LocalStore from a ``file://`` URL.

        **Examples:**

        Construct a new store pointing to the root of your filesystem::

            url = "file:///"
            store = LocalStore.from_url(url)

        Construct a new store with a directory prefix::

            url = "file:///Users/kyle/"
            store = LocalStore.from_url(url)

        """
        return super(cls).from_url(url, automatic_cleanup=automatic_cleanup, mkdir=mkdir)

    @override
    def __eq__(self, value: object, /) -> bool:
        return super().__eq__(value)

    @override
    def __getnewargs_ex__(self) -> tuple[tuple[()], dict[str, object]]:
        return super().__getnewargs_ex__()

    @property
    @override
    def prefix(self) -> Path | None:
        """Get the prefix applied to all operations in this store, if any."""
        return super().prefix
