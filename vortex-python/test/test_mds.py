# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from collections.abc import Iterator
from typing_extensions import override
from streaming import StreamingDataset
from typing import Any, Iterable, final
import logging
import math
import os
import pytest
import random
import string
import time
import torch
from torch.utils.data import DataLoader
import vortex.mds as vxmds

log = logging.getLogger(__name__)


def _random_text(min_len: int = 50, max_len: int = 1000) -> str:
    L = random.randint(min_len, max_len)
    alphabet = string.ascii_letters + string.digits + " .,;:!?-()'\"/\\\n"
    return "".join(random.choice(alphabet) for _ in range(L))


def _finewebish_sample(i: int) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    # A minimal FineWeb-like schema: https://huggingface.co/datasets/HuggingFaceFW/fineweb
    return {
        "id": i,
        "url": f"https://example.com/page/{i}",
        "language": "en",
        "timestamp": 1_700_000_000 + i,  # fake epoch seconds
        "text": _random_text(),
    }


def _write_split(path: str, n: int, max_shard_rows: int = 2048) -> None:
    """Write `n` samples to an MDS directory using the VortexWriter."""

    with vxmds.VortexWriter(out=path, max_shard_rows=max_shard_rows) as writer:
        for i in range(n):
            writer.write(_finewebish_sample(i))


def test_write_and_stream_train_val(tmpdir_factory: pytest.TempPathFactory):
    # Small train/val datasets
    root = tmpdir_factory.mktemp("data")
    _write_split(str(root / "train"), n=500, max_shard_rows=128)
    _write_split(str(root / "val"), n=120, max_shard_rows=64)

    # Read back via StreamingDataset
    train = StreamingDataset(local=str(root), split="train")
    val = StreamingDataset(local=str(root), split="val")

    # Basic sanity checks
    assert len(train) == 500
    assert len(val) == 120

    first = train[0]  # pyright: ignore[reportAny]
    assert "text" in first and isinstance(first["text"], str)
    assert "url" in first
    assert "id" in first


def test_toy_training_loop(tmpdir_factory: pytest.TempPathFactory):
    root = tmpdir_factory.mktemp("data")

    # Create a modest training corpus
    _write_split(str(root / "train"), n=1000, max_shard_rows=256)
    train = StreamingDataset(local=str(root), split="train", batch_size=16)

    # Tiny character-level classifier that predicts a bucket of text length (for demonstration)
    buckets = [128, 256, 512, 1024]

    def bucketize(L: int) -> int:
        for i, b in enumerate(buckets):
            if L <= b:
                return i
        return len(buckets) - 1

    vocab = {ch: i + 1 for i, ch in enumerate(string.printable)}  # 0 is padding
    vocab_size = len(vocab) + 1

    @final
    class Model(torch.nn.Module):
        def __init__(self, vocab_size: int, hidden: int, num_classes: int):
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.emb = torch.nn.Embedding(vocab_size, 16, padding_idx=0)
            self.rnn = torch.nn.GRU(16, hidden, batch_first=True)
            self.out = torch.nn.Linear(hidden, num_classes)

        @override
        def forward(self, x: Any):  # pyright: ignore[reportAny, reportExplicitAny]
            # x: [B, T]
            e = self.emb(x)  # pyright: ignore[reportAny]
            _, h = self.rnn(e)  # h: [1, B, H]  # pyright: ignore[reportAny]
            return self.out(h.squeeze(0))  # pyright: ignore[reportAny]

    def collate(batch: Iterable[dict[str, Any]]):  # pyright: ignore[reportExplicitAny]
        # Convert strings to fixed-length integer tensors (very small max_len to keep test fast)
        max_len = 256
        xs: list[list[int]] = []
        ys: list[int] = []
        for ex in batch:
            s: str = ex["text"]  # pyright: ignore[reportAny]
            ids = [vocab.get(ch, 0) for ch in s[:max_len]]
            if len(ids) < max_len:
                ids.extend([0] * (max_len - len(ids)))
            xs.append(ids)
            ys.append(bucketize(len(s)))
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(vocab_size=vocab_size, hidden=32, num_classes=len(buckets)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # One tiny "epoch"
    _ = model.train()
    losses: list[float] = []

    for x, y in DataLoader(train, batch_size=16, collate_fn=collate):  # pyright: ignore[reportAny]
        x = x.to(device)  # pyright: ignore[reportAny]
        y = y.to(device)  # pyright: ignore[reportAny]
        logits = model(x)  # pyright: ignore[reportAny]
        loss = loss_fn(logits, y)  # pyright: ignore[reportAny]
        opt.zero_grad()
        loss.backward()  # pyright: ignore[reportAny]
        opt.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        losses.append(loss.item())  # pyright: ignore[reportAny]

    assert math.isfinite(sum(losses)) and len(losses) > 0


def test_throughput_benchmark(tmpdir_factory: pytest.TempPathFactory):
    """A basic throughput benchmark for FineWeb-shaped samples."""
    root = tmpdir_factory.mktemp("data") / "mds_vortex_bench"
    os.makedirs(root, exist_ok=True)

    _write_split(str(root / "train"), n=10_000)

    ds = StreamingDataset(local=str(root), split="train", batch_size=16)
    dl = DataLoader(ds, batch_size=16)  # pyright: ignore[reportUnknownVariableType]

    assert len(ds) == 10_000

    start = time.perf_counter()
    count = 0
    for sample in dl:  # pyright: ignore[reportAny]
        assert "text" in sample
        count += len(sample["text"])  # pyright: ignore[reportAny]
    elapsed = time.perf_counter() - start
    sps = count / elapsed if elapsed > 0 else float("inf")
    log.info(f"[Vortex/MDS] Loaded {count} samples in {elapsed:.3f}s → {sps:.0f} samples/sec")

    assert sps > 500, f"Throughput too low: {sps:.1f} samples/sec"
