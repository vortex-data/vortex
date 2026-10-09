# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests that ArrayIterator scan paths honour pending Python signals.

Regression tests for https://github.com/vortex-data/vortex/issues/1459:
long scans run with the GIL released, so the Rust code must call
``py.check_signals()`` at chunk boundaries. Otherwise a Ctrl+C
(``SIGINT``) appears to hang until the whole scan finishes.

Strategy: queue a real ``SIGINT`` for this process, then drive each scan
entry point (``next()``, ``read_all()`` and the ``to_arrow`` record-batch
reader). With the default ``SIGINT`` handler the pending signal must
surface as ``KeyboardInterrupt`` *from* the scan call, and a counting
Python iterator must prove the scan aborted early instead of consuming
every chunk first.
"""

import os
import signal
import threading
import time

import pyarrow as pa
import pytest

import vortex as vx

N_CHUNKS = 10


def _raise_sigint() -> None:
    if hasattr(signal, "raise_signal"):
        signal.raise_signal(signal.SIGINT)
    else:  # pragma: no cover - very old Pythons
        os.kill(os.getpid(), signal.SIGINT)


def _counting_chunks(dtype, n_chunks: int, consumed: list[int]):
    chunk = vx.array(pa.array([1, 2, 3]))

    def gen():
        for _ in range(n_chunks):
            consumed.append(1)
            yield chunk

    assert chunk.dtype == dtype
    return gen()


@pytest.fixture
def int_dtype():
    return vx.array(pa.array([1, 2, 3])).dtype


def _install_default_sigint():
    old = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return old


def test_next_raises_on_pending_sigint(int_dtype) -> None:
    consumed: list[int] = []
    it = vx.ArrayIterator.from_iter(int_dtype, _counting_chunks(int_dtype, N_CHUNKS, consumed))
    old = _install_default_sigint()
    try:
        _raise_sigint()
        with pytest.raises(KeyboardInterrupt):
            next(it)
        # The signal must be observed before the scan makes progress.
        assert len(consumed) <= 1
    finally:
        signal.signal(signal.SIGINT, old)


def test_read_all_aborts_early_on_pending_sigint(int_dtype) -> None:
    consumed: list[int] = []
    it = vx.ArrayIterator.from_iter(int_dtype, _counting_chunks(int_dtype, N_CHUNKS, consumed))
    old = _install_default_sigint()
    try:
        _raise_sigint()
        with pytest.raises(KeyboardInterrupt):
            it.read_all()
        # Without per-chunk `check_signals`, the whole scan would run to
        # completion inside a single detached section and consume everything.
        assert len(consumed) < N_CHUNKS
    finally:
        signal.signal(signal.SIGINT, old)


def test_read_all_interrupted_mid_scan(int_dtype) -> None:
    """A SIGINT arriving mid-scan aborts `read_all` promptly."""
    consumed: list[int] = []
    chunk = vx.array(pa.array([1, 2, 3]))
    assert chunk.dtype == int_dtype

    def slow_gen():
        for _ in range(200):
            consumed.append(1)
            time.sleep(0.01)
            yield chunk

    it = vx.ArrayIterator.from_iter(int_dtype, slow_gen())
    old = _install_default_sigint()
    try:

        def _interrupt() -> None:
            time.sleep(0.1)
            _raise_sigint()

        t = threading.Thread(target=_interrupt, daemon=True)
        start = time.monotonic()
        t.start()
        with pytest.raises(KeyboardInterrupt):
            it.read_all()
        elapsed = time.monotonic() - start
        # Full scan would take ~2s; an interruptible scan aborts far sooner
        # and without consuming every chunk.
        assert elapsed < 1.5
        assert len(consumed) < 200
        t.join(timeout=5)
    finally:
        signal.signal(signal.SIGINT, old)


def test_to_arrow_reader_aborts_on_pending_sigint() -> None:
    """The `to_arrow` record-batch reader checks signals per chunk."""
    chunk = vx.array([{"x": 1}, {"x": 2}])
    consumed: list[int] = []

    def gen():
        for _ in range(N_CHUNKS):
            consumed.append(1)
            yield chunk

    it = vx.ArrayIterator.from_iter(chunk.dtype, gen())
    old = _install_default_sigint()
    try:
        _raise_sigint()
        reader = it.to_arrow()
        # The signal check is mapped through VortexError into an Arrow
        # external error, so accept either surface form. The early-abort
        # assertion below is the real proof the signal was honoured.
        with pytest.raises(Exception) as exc_info:  # noqa: BLE001, PT011
            reader.read_all()
        message = str(exc_info.value).lower()
        assert isinstance(exc_info.value, KeyboardInterrupt) or "interrupt" in message
        assert len(consumed) < N_CHUNKS
    finally:
        signal.signal(signal.SIGINT, old)
