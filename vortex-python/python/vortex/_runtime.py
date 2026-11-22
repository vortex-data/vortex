# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import os
import threading
from collections.abc import Generator
from contextlib import contextmanager

from ._lib.runtime import set_worker_threads as _set_worker_threads
from ._lib.runtime import worker_threads as _worker_threads


class _WorkerThreadLeaseManager:
    """Reference-counted leases for temporary Vortex worker-pool overrides.

    Overlapping readers share a single process-wide worker pool. When any lease
    requests ``use_threads=True``, threaded execution takes precedence over
    concurrent ``use_threads=False`` leases. The original baseline (or the most
    recent explicit user update made while leases were active) is restored only
    after the last lease exits. No lock is ever held across a generator yield.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._busy = False
        self._pending_releases: list[tuple[bool, int]] = []
        self._generation = 0
        self._threaded_leases = 0
        self._unthreaded_leases = 0
        self._baseline_workers: int | None = None
        self._last_applied_workers: int | None = None
        self._active_policy: bool | None = None

    def _sync_external_baseline_locked(self) -> None:
        if self._last_applied_workers is None:
            return
        current = _worker_threads()
        if current != self._last_applied_workers:
            self._baseline_workers = current
            self._last_applied_workers = None

    def _effective_policy_locked(self) -> bool | None:
        if self._threaded_leases > 0:
            return True
        if self._unthreaded_leases > 0:
            return False
        return None

    def _reconcile_policy_locked(self) -> None:
        desired = self._effective_policy_locked()
        if desired is None:
            baseline = self._baseline_workers
            self._baseline_workers = None
            self._last_applied_workers = None
            self._active_policy = None
            if baseline is not None:
                _set_worker_threads(baseline)
            return

        if self._baseline_workers is None:
            self._baseline_workers = _worker_threads()

        if desired != self._active_policy or self._last_applied_workers is None:
            _set_worker_threads(None if desired else 0)
            self._last_applied_workers = _worker_threads()
            self._active_policy = desired

    def _apply_release_locked(self, use_threads: bool, generation: int) -> None:
        if generation != self._generation:
            return

        self._sync_external_baseline_locked()

        if use_threads:
            self._threaded_leases -= 1
        else:
            self._unthreaded_leases -= 1

        self._reconcile_policy_locked()

    def _drain_pending_releases_locked(self) -> None:
        while self._pending_releases:
            queued = self._pending_releases
            self._pending_releases = []
            changed = False
            for use_threads, generation in queued:
                if generation != self._generation:
                    continue
                if use_threads:
                    self._threaded_leases -= 1
                else:
                    self._unthreaded_leases -= 1
                changed = True
            if changed:
                self._reconcile_policy_locked()

    def _finish_locked(self) -> None:
        while True:
            self._drain_pending_releases_locked()
            self._busy = False
            if not self._pending_releases:
                return
            self._busy = True

    def acquire(self, use_threads: bool) -> int:
        with self._lock:
            self._busy = True
            try:
                if self._threaded_leases == 0 and self._unthreaded_leases == 0:
                    self._baseline_workers = _worker_threads()
                    self._last_applied_workers = None
                    self._active_policy = None
                else:
                    self._sync_external_baseline_locked()

                if use_threads:
                    self._threaded_leases += 1
                else:
                    self._unthreaded_leases += 1

                try:
                    self._reconcile_policy_locked()
                except BaseException:
                    if use_threads:
                        self._threaded_leases -= 1
                    else:
                        self._unthreaded_leases -= 1
                    raise

                return self._generation
            finally:
                self._finish_locked()

    def release(self, use_threads: bool, generation: int) -> None:
        with self._lock:
            if self._busy:
                self._pending_releases.append((use_threads, generation))
                return
            self._busy = True
            try:
                self._apply_release_locked(use_threads, generation)
            finally:
                self._finish_locked()

    def get_user_worker_threads(self) -> int:
        with self._lock:
            self._busy = True
            try:
                if self._threaded_leases > 0 or self._unthreaded_leases > 0:
                    self._sync_external_baseline_locked()
                    if self._baseline_workers is not None:
                        return self._baseline_workers
                return _worker_threads()
            finally:
                self._finish_locked()

    def set_user_worker_threads(self, n: int | None = None) -> None:
        with self._lock:
            self._busy = True
            try:
                _set_worker_threads(n)
                if self._threaded_leases > 0 or self._unthreaded_leases > 0:
                    self._baseline_workers = _worker_threads()
                    self._last_applied_workers = None
            finally:
                self._finish_locked()

    def reset_after_fork(self) -> None:
        baseline = self._baseline_workers
        self._lock = threading.RLock()
        self._busy = False
        self._pending_releases = []
        self._generation += 1
        self._threaded_leases = 0
        self._unthreaded_leases = 0
        self._baseline_workers = None
        self._last_applied_workers = None
        self._active_policy = None
        if baseline is not None:
            _set_worker_threads(baseline)


_WORKER_THREAD_LEASES = _WorkerThreadLeaseManager()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_WORKER_THREAD_LEASES.reset_after_fork)


def worker_threads() -> int:
    """Return the configured number of background worker threads driving Vortex futures.

    While a dataset reader holds a temporary worker-thread override, returns the
    baseline count that will be restored once active readers finish so save/restore
    patterns around ``set_worker_threads`` preserve the caller's baseline.
    """
    return _WORKER_THREAD_LEASES.get_user_worker_threads()


def set_worker_threads(n: int | None = None) -> None:
    """Set the number of background worker threads driving Vortex futures.

    If ``n`` is ``None``, resets the pool to ``available_parallelism() - 1``.
    When called while a dataset reader holds a temporary worker-thread override,
    the requested count also becomes the baseline restored once active readers
    finish.
    """
    _WORKER_THREAD_LEASES.set_user_worker_threads(n)


@contextmanager
def _temporary_worker_threads(use_threads: bool) -> Generator[None]:
    generation = _WORKER_THREAD_LEASES.acquire(use_threads)
    try:
        yield
    finally:
        _WORKER_THREAD_LEASES.release(use_threads, generation)
