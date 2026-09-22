# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 Development Seed

from typing import Self

from .._lib import store as _store


class MemoryStore(_store.MemoryStore):
    """A fully in-memory implementation of ObjectStore.

    Create a new in-memory store::

        store = MemoryStore()
    """

    def __new__(cls) -> Self:
        return super().__new__(cls)
