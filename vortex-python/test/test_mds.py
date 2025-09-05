# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from collections.abc import Iterator
import os
import random
import string
import logging
import math
import queue
import threading
import time
from typing import Any, Callable
import pytest
import vortex as vx
import vortex.mds
import torch
from datasets import IterableDataset, IterableDatasetDict  # pyright: ignore[reportMissingTypeStubs]
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_constant_schedule_with_warmup


log = logging.getLogger(__name__)


def mds_dataset(
    train_data_dir: str, validation_data_dir: str, batch_size: int = 1000, shuffle_buffer: int = 10_000
) -> IterableDatasetDict:
    kwargs = {
        "shuffle": True,
        "batch_size": batch_size,
        "num_canonical_nodes": 1,
        "shuffle_block_size": shuffle_buffer,
        "predownload": 2 * batch_size,
        "cache_limit": "512mb",
    }

    assert train_data_dir.endswith("/"), "data_dir must end with /"
    assert validation_data_dir.endswith("/"), "data_dir must end with /"

    def create_train_generator() -> Iterator[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        dataset = StreamingDataset(local=train_data_dir, **kwargs)  # pyright: ignore[reportArgumentType]
        yield from dataset

    def create_validation_generator() -> Iterator[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        dataset = StreamingDataset(local=validation_data_dir, **kwargs)  # pyright: ignore[reportArgumentType]
        yield from dataset

    return IterableDatasetDict(
        {
            "train": IterableDataset.from_generator(create_train_generator),  # pyright: ignore[reportUnknownMemberType]
            "validation": IterableDataset.from_generator(create_validation_generator),  # pyright: ignore[reportUnknownMemberType]
        }
    )


def prefetching_map(
    dataset: IterableDataset,
    function: Callable[..., Any],  # pyright: ignore[reportExplicitAny]
    batched: bool = False,
    batch_size: int = 1000,
    prefetch_factor: int = 2,
    remove_columns: list[str] | None = None,
) -> IterableDataset:
    q = queue.Queue(maxsize=prefetch_factor * batch_size)  # pyright: ignore[reportUnknownVariableType]

    sentinel = object()

    def _producer():
        mapped_dataset = dataset.map(  # pyright: ignore[reportUnknownMemberType]
            function=function,
            batched=batched,
            batch_size=batch_size,
            remove_columns=remove_columns,
        )
        try:
            for item in mapped_dataset:  # pyright: ignore[reportUnknownVariableType]
                q.put(item)  # pyright: ignore[reportUnknownMemberType]
        finally:
            q.put(sentinel)  # pyright: ignore[reportUnknownMemberType]

    def _consumer_generator():  # pyright: ignore[reportUnknownParameterType]
        while True:
            item = q.get()  # pyright: ignore[reportUnknownVariableType]
            if item is sentinel:
                break
            yield item

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    return IterableDataset.from_generator(_consumer_generator)  # pyright: ignore[reportUnknownMemberType]


def calculate_perplexity(model, data_iter, device: str, max_batches: int = 5) -> tuple[float, float]:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportMissingParameterType]
    total_loss: int = 0
    total_tokens: int = 0
    batches_processed: int = 0

    with torch.no_grad():
        for batch in data_iter:  # pyright: ignore[reportUnknownVariableType]
            if batches_processed >= max_batches:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)  # pyright: ignore[reportUnknownVariableType]

            # Calculate loss only for non-padded tokens
            shift_labels = input_ids[..., 1:].contiguous()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            shift_logits = outputs.logits[..., :-1, :].contiguous()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            shift_attention_mask = attention_mask[..., 1:].contiguous()  # pyright: ignore[reportUnknownVariableType, reportUnknownVariableType, reportUnknownMemberType]

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # pyright: ignore[reportAny, reportUnknownMemberType]
            losses = losses.view(shift_labels.shape)  # pyright: ignore[reportAny, reportUnknownMemberType]

            # Mask out padded tokens
            losses = losses * shift_attention_mask  # pyright: ignore[reportUnknownVariableType]
            total_loss += losses.sum().item()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            total_tokens += shift_attention_mask.sum().item()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            batches_processed += 1

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss: float = total_loss / total_tokens  # pyright: ignore[reportUnknownVariableType]
    perplexity: float = math.exp(min(avg_loss, 50.0))  # pyright: ignore[reportUnknownArgumentType]
    return perplexity, avg_loss  # pyright: ignore[reportUnknownVariableType]


_ALPHABET = string.ascii_letters + string.digits


def rand_str() -> str:
    return "".join(random.choices(_ALPHABET, k=1000))


def generate_vortex_mds_dataset(tmpdir_factory: pytest.TempPathFactory) -> str:
    folder = tmpdir_factory.mktemp("data")
    with vortex.mds.VortexWriter(out=str(folder), max_shard_rows=100) as out:
        for x in range(1000):
            out.write({"text": rand_str(), "id": x})
    return str(folder) + "/"


def test_mds(tmpdir_factory: pytest.TempPathFactory):
    log.info("Generating train data.")
    train_data_dir = generate_vortex_mds_dataset(tmpdir_factory)
    log.info("Generating validation data.")
    validation_data_dir = generate_vortex_mds_dataset(tmpdir_factory)
    log.info("Finished generating data.")

    max_length = 256
    batch_size = 16
    learning_rate = 2.5e-4
    warmup_steps = 100

    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    # Hack, but the data doesn't really matter anyway.
    tokenizer.pad_token = tokenizer.eos_token  # pyright: ignore[reportUnknownMemberType]
    config = AutoConfig.from_pretrained("gpt2")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 128

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    model = AutoModelForCausalLM.from_config(config)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    model = torch.compile(model, mode="max-autotune")  # pyright: ignore[reportUnknownMemberType, reportUnknownMemberType, reportUnknownArgumentType]

    def init_weights(module):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # pyright: ignore[reportUnusedCallResult]
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)  # pyright: ignore[reportUnusedCallResult]
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # pyright: ignore[reportUnusedCallResult]

    model.apply(init_weights)  # pyright: ignore[reportAny, reportFunctionMemberAccess]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():  # Mac M1/M2
        device = "mps"
    else:
        device = "cpu"

    model.to(device)  # pyright: ignore[reportAny, reportFunctionMemberAccess]

    datasets = mds_dataset(train_data_dir, validation_data_dir)

    def tokenize_fn(batch):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return tokenizer(  # pyright: ignore[reportUnknownVariableType]
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
            stride=0,
        )

    train = DataLoader(  # pyright: ignore[reportUnknownVariableType]
        prefetching_map(  # pyright: ignore[reportArgumentType]
            datasets["train"],
            tokenize_fn,  # pyright: ignore[reportUnknownArgumentType]
            batched=True,
            batch_size=batch_size,
            remove_columns=datasets["train"].column_names,
        ),
        batch_size=batch_size,
        pin_memory=True,
    )

    train_iter = iter(train)  # pyright: ignore[reportUnknownArgumentType]
    next(train_iter)

    validate = DataLoader(  # pyright: ignore[reportUnknownVariableType]
        prefetching_map(  # pyright: ignore[reportArgumentType]
            datasets["validation"],
            tokenize_fn,  # pyright: ignore[reportUnknownArgumentType]
            batched=True,
            batch_size=batch_size,
            remove_columns=datasets["validation"].column_names,
        ),
        batch_size=batch_size,
        pin_memory=True,
    )
    # Warm up validation iterator
    validate_iter = iter(validate)  # pyright: ignore[reportUnknownArgumentType]
    next(validate_iter)

    optimizer = torch.optim.AdamW(
        model.parameters(),  # pyright: ignore[reportAny, reportFunctionMemberAccess]
        lr=learning_rate,
    )

    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    scaler = torch.GradScaler()

    model.train()  # pyright: ignore[reportAny, reportFunctionMemberAccess]
    total_tokens = 0
    start = time.time()

    for epoch in range(1):
        log.info(f"Starting epoch {epoch}")

        for batch_i, batch in enumerate(train_iter):  # pyright: ignore[reportAny]
            input_ids = batch["input_ids"].to(device)  # pyright: ignore[reportAny]
            attention_mask = batch["attention_mask"].to(device)  # pyright: ignore[reportAny]

            with torch.autocast(device):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)  # pyright: ignore[reportUnknownVariableType]
                loss = outputs.loss  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

            if torch.isnan(loss) or torch.isinf(loss):  # pyright: ignore[reportUnknownArgumentType, reportUnknownArgumentType]
                raise ValueError(f"Invalid loss value: {loss.item()}")  # pyright: ignore[reportUnknownMemberType]

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # pyright: ignore[reportUnusedCallResult, reportAny, reportFunctionMemberAccess]

            scaler.step(optimizer)  # pyright: ignore[reportUnusedCallResult]
            scaler.update()
            scheduler.step()

            total_tokens += input_ids.numel()  # pyright: ignore[reportAny]

            if batch_i % 10 == 0:
                elapsed = time.time() - start
                current_lr = scheduler.get_last_lr()[0]

                model.eval()  # pyright: ignore[reportAny, reportFunctionMemberAccess]
                perplexity, avg_loss = calculate_perplexity(model, validate_iter, device, max_batches=5)
                log.info(
                    f"Step {batch_i}: lr={current_lr:.2e}, tokens/sec={total_tokens / elapsed:.0f}, "  # pyright: ignore[reportImplicitStringConcatenation]
                    f"perplexity={perplexity:.2f}, val_loss={avg_loss:.4f}"
                )
                if log.isEnabledFor(logging.DEBUG):
                    with torch.no_grad():
                        input_text = "SpiralDB is a company that"
                        inputs = tokenizer(input_text, return_tensors="pt")  # pyright: ignore[reportUnknownVariableType]
                        input_ids = inputs.input_ids.to(device)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                        attention_mask = inputs.attention_mask.to(device)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                        outputs = model.generate(  # pyright: ignore[reportAny, reportFunctionMemberAccess]
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=50,
                            do_sample=True,
                            temperature=0.8,
                            pad_token_id=tokenizer.pad_token_id,  # pyright: ignore[reportUnknownMemberType]
                            eos_token_id=tokenizer.eos_token_id,  # pyright: ignore[reportUnknownMemberType]
                        )
                        log.debug(tokenizer.decode(outputs[0], skip_special_tokens=True))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                model.train()  # pyright: ignore[reportAny, reportFunctionMemberAccess]
                start = time.time()
                total_tokens = 0
