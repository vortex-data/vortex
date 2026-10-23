# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests that ArrayIterator scan paths honour pending Python signals.

Regression tests for https://github.com/vortex-data/vortex/issues/1459:
long scans run with the GIL released, so the Rust code must call
``py.check_signals()`` at chunk boundaries. Otherwise a Ctrl+C
(``SIGINT``) appears to hang until the whole scan finishes.

Strategy: drive each scan entry point with a counting (and, where useful,
slow) Python iterator while a background thread delivers a real ``SIGINT``
mid-scan. The scan must surface it promptly as ``KeyboardInterrupt``
without consuming every chunk first.
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
    # Make the first chunk slow so a SIGINT delivered from a background
    # thread arrives while the scan is in flight.
    chunk = vx.array(pa.array([1, 2, 3]))
    assert chunk.dtype == int_dtype

    def slow_gen():
        consumed.append(1)
        time.sleep(0.2)
        yield chunk

    it = vx.ArrayIterator.from_iter(int_dtype, slow_gen())
    old = _install_default_sigint()
    try:

        def _interrupt() -> None:
            time.sleep(0.05)
            _raise_sigint()

        t = threading.Thread(target=_interrupt, daemon=True)
        t.start()
        start = time.monotonic()
        with pytest.raises(Exception) as exc_info:  # noqa: BLE001, PT011
            next(it)
        message = str(exc_info.value)
        assert isinstance(exc_info.value, KeyboardInterrupt) or "interrupt" in message.lower() or "keyboardinterrupt" in message.lower()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
        t.join(timeout=5)
    finally:
        signal.signal(signal.SIGINT, old)


def test_read_all_aborts_early_on_pending_sigint(int_dtype) -> None:
    consumed: list[int] = []
    chunk = vx.array(pa.array([1, 2, 3]))
    assert chunk.dtype == int_dtype

    def slow_gen():
        for _ in range(N_CHUNKS):
            consumed.append(1)
            time.sleep(0.05)
            yield chunk

    it = vx.ArrayIterator.from_iter(int_dtype, slow_gen())
    old = _install_default_sigint()
    try:

        def _interrupt() -> None:
            time.sleep(0.02)
            _raise_sigint()

        t = threading.Thread(target=_interrupt, daemon=True)
        start = time.monotonic()
        t.start()
        with pytest.raises(Exception) as exc_info:  # noqa: BLE001, PT011
            it.read_all()
        message = str(exc_info.value).lower()
        assert isinstance(exc_info.value, KeyboardInterrupt) or "interrupt" in message or "keyboardinterrupt" in message
        assert len(consumed) < N_CHUNKS
        t.join(timeout=5)
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
        with pytest.raises(Exception) as exc_info:  # noqa: BLE001, PT011
            it.read_all()
        message = str(exc_info.value)
        assert isinstance(exc_info.value, KeyboardInterrupt) or "interrupt" in message.lower() or "keyboardinterrupt" in message.lower()
        elapsed = time.monotonic() - start
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
            time.sleep(0.02)
            yield chunk

    it = vx.ArrayIterator.from_iter(chunk.dtype, gen())
    old = _install_default_sigint()
    try:

        def _interrupt() -> None:
            time.sleep(0.05)
            _raise_sigint()

        t = threading.Thread(target=_interrupt, daemon=True)
        t.start()
        reader = it.to_arrow()

        with pytest.raises(Exception) as exc_info:  # noqa: BLE001, PT011
            reader.read_all()
        message = str(exc_info.value).lower()
        assert isinstance(exc_info.value, KeyboardInterrupt) or "interrupt" in message
        assert len(consumed) < N_CHUNKS
        t.join(timeout=5)
    finally:
        signal.signal(signal.SIGINT, old)
