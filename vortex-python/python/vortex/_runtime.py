# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
import threading
from collections.abc import Generator
from contextlib import contextmanager

from ._lib.runtime import set_worker_threads as _set_worker_threads
from ._lib.runtime import worker_threads as _worker_threads


class _UnthreadedLeaseManager:
    """Reference-counted manager for temporary ``use_threads=False`` worker overrides.

    ``use_threads=True`` leaves the configured worker pool untouched. Only
    ``use_threads=False`` mutates global state: the 0->1 lease transition saves
    the current worker count and sets it to ``0``, and the 1->0 transition
    restores the saved value.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._unthreaded_leases = 0
        self._saved_workers: int | None = None

    def acquire(self) -> int:
        with self._lock:
            if self._unthreaded_leases == 0:
                self._saved_workers = _worker_threads()
                try:
                    _set_worker_threads(0)
                except BaseException:
                    self._saved_workers = None
                    raise
            self._unthreaded_leases += 1
            return self._generation

    def release(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._unthreaded_leases -= 1
            if self._unthreaded_leases == 0:
                saved = self._saved_workers
                try:
                    if saved is not None:
                        _set_worker_threads(saved)
                finally:
                    self._saved_workers = None

    def reset_after_fork(self) -> None:
        saved = self._saved_workers
        self._lock = threading.RLock()
        self._generation += 1
        self._unthreaded_leases = 0
        self._saved_workers = None
        if saved is not None:
            _set_worker_threads(saved)


_UNTHREADED_LEASES = _UnthreadedLeaseManager()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_UNTHREADED_LEASES.reset_after_fork)


@contextmanager
def _temporary_worker_threads(use_threads: bool) -> Generator[None]:
    if use_threads:
        yield
        return

    generation = _UNTHREADED_LEASES.acquire()
    try:
        yield
    finally:
        _UNTHREADED_LEASES.release(generation)
