# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# pyright: reportAny=false
# pyright: reportMissingTypeStubs=false
# pyright: reportMissingTypeArgument=false
# pyright: reportPrivateUsage=false
# pyright: reportUnannotatedClassAttribute=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import copy
import glob
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast, final

import pyarrow as pa
from typing_extensions import override

import vortex as vx
from vortex.expr import Expr, and_

try:
    import datasets as hf_datasets
    from datasets.iterable_dataset import (
        FormattingConfig,
        _BaseExamplesIterable,
        get_format_type_from_alias,
    )
    from datasets.table import InMemoryTable
except ImportError as e:  # pragma: no cover - exercised only without optional deps.
    raise ImportError("Install vortex-data[hf] to use vortex.datasets.") from e


_DEFAULT_SPLIT = "train"
_DEFAULT_DATA_FILES = "**/*.vortex"
_ITERABLE_DATASET_HAS_SHUFFLING = "shuffling" in inspect.signature(hf_datasets.IterableDataset).parameters


def load_dataset(
    path: str | Path,
    *,
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
    split: str | Sequence[str] | None = None,
    streaming: bool = True,
    columns: Sequence[str] | None = None,
    filter: Expr | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    cache_dir: str | Path | None = None,
    keep_in_memory: bool = False,
    num_proc: int | None = None,
    revision: str | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
) -> (
    VortexIterableDataset
    | hf_datasets.Dataset
    | hf_datasets.IterableDatasetDict
    | hf_datasets.DatasetDict
    | list[VortexIterableDataset | hf_datasets.Dataset]
):
    """Load Vortex files as Hugging Face Datasets objects.

    Unlike ``datasets.load_dataset``, this defaults to ``streaming=True``. The streaming path
    keeps Vortex in charge of reading and pushes column selection, Vortex expressions, and row
    limits into each Vortex scan before examples are yielded to Hugging Face Datasets transforms.
    Pass ``streaming=False`` to eagerly materialize an in-memory ``datasets.Dataset``.
    """

    split_to_files = _resolve_data_files(
        path,
        data_files=data_files,
        split=split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    def build_one(split_name: str):
        files = split_to_files[split_name]
        if streaming:
            return VortexIterableDataset(
                files,
                columns=columns,
                filter=filter,
                limit=limit,
                batch_size=batch_size,
                split=split_name,
            )
        return _materialize_dataset(
            files,
            columns=columns,
            filter=filter,
            limit=limit,
            batch_size=batch_size,
            split=split_name,
            cache_dir=None if cache_dir is None else str(cache_dir),
            keep_in_memory=keep_in_memory,
            num_proc=num_proc,
        )

    if split is None:
        if streaming:
            return hf_datasets.IterableDatasetDict(
                (name, cast(hf_datasets.IterableDataset, build_one(name))) for name in split_to_files
            )
        return hf_datasets.DatasetDict((name, cast(hf_datasets.Dataset, build_one(name))) for name in split_to_files)

    if isinstance(split, str):
        if split not in split_to_files:
            raise ValueError(f"Unknown split {split!r}. Available splits: {list(split_to_files)}")
        return build_one(split)

    unknown = [split_name for split_name in split if split_name not in split_to_files]
    if unknown:
        raise ValueError(f"Unknown splits {unknown}. Available splits: {list(split_to_files)}")
    return [build_one(split_name) for split_name in split]


@final
class VortexIterableDataset(hf_datasets.IterableDataset):
    """A Hugging Face IterableDataset backed by Vortex scans."""

    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        columns: Sequence[str] | None = None,
        filter: Expr | None = None,
        limit: int | None = None,
        batch_size: int | None = None,
        split: str = _DEFAULT_SPLIT,
        formatting: FormattingConfig | None = None,
        shuffling: object | None = None,
        distributed: object | None = None,
        token_per_repo_id: dict[str, bool | str | None] | None = None,
    ):
        file_names = tuple(str(file) for file in files)
        if not file_names:
            raise ValueError("VortexIterableDataset requires at least one Vortex file")

        self._vortex_files = file_names
        self._vortex_columns = _normalize_columns(columns)
        self._vortex_filter = filter
        self._vortex_limit = limit
        self._vortex_batch_size = batch_size

        features = _features_for_files(file_names, self._vortex_columns)
        info = hf_datasets.DatasetInfo(features=features)
        ex_iterable = _VortexExamplesIterable(
            file_names,
            columns=self._vortex_columns,
            filter=filter,
            limit=limit,
            batch_size=batch_size,
            features=features,
        )
        if _ITERABLE_DATASET_HAS_SHUFFLING:
            super().__init__(
                ex_iterable=ex_iterable,
                info=info,
                split=hf_datasets.Split(split),
                formatting=formatting,
                shuffling=shuffling,  # pyright: ignore[reportCallIssue]
                distributed=distributed,
                token_per_repo_id=token_per_repo_id,
            )
        else:
            super().__init__(
                ex_iterable=ex_iterable,
                info=info,
                split=hf_datasets.Split(split),
                formatting=formatting,
                distributed=distributed,  # pyright: ignore[reportArgumentType]
                token_per_repo_id=token_per_repo_id,
            )

    @override
    def select_columns(self, column_names: str | list[str]) -> VortexIterableDataset:
        if isinstance(column_names, str):
            column_names = [column_names]
        current_columns = self._current_column_names()
        available = set(current_columns)
        missing = set(column_names) - available
        if missing:
            raise ValueError(
                f"Column name {list(missing)} not in the dataset. Columns in the dataset: {current_columns}."
            )
        return self._with_pushdown(columns=column_names)

    @override
    def remove_columns(self, column_names: str | list[str]) -> VortexIterableDataset:
        if isinstance(column_names, str):
            column_names = [column_names]
        current_columns = self._current_column_names()
        remove = set(column_names)
        missing = remove - set(current_columns)
        if missing:
            raise ValueError(
                f"Column name {list(missing)} not in the dataset. Columns in the dataset: {current_columns}."
            )
        return self._with_pushdown(columns=[column for column in current_columns if column not in remove])

    @override
    def take(self, n: int) -> VortexIterableDataset | hf_datasets.IterableDataset:
        if getattr(self, "_shuffling", None) is not None or self._distributed is not None:
            return super().take(n)
        limit = n if self._vortex_limit is None else min(self._vortex_limit, n)
        return self._with_pushdown(limit=limit)

    @override
    def filter(
        self,
        function: Callable[..., object] | Expr | None = None,
        with_indices: bool = False,
        input_columns: str | list[str] | None = None,
        batched: bool = False,
        batch_size: int | None = 1000,
        fn_kwargs: dict[str, object] | None = None,
    ) -> VortexIterableDataset | hf_datasets.IterableDataset:
        if (
            isinstance(function, Expr)
            and not with_indices
            and input_columns is None
            and not batched
            and fn_kwargs is None
        ):
            # Vortex applies the predicate before the scan limit, so folding a filter into a
            # dataset that already carries a pushed-down limit (e.g. from take()) would filter the
            # whole file and then limit, rather than limiting first and then filtering. That
            # silently reorders the operations, so refuse it. An Expr also cannot be delegated to
            # the base filter(), which expects a callable.
            if self._vortex_limit is not None:
                raise ValueError(
                    "Cannot push a filter expression down after a row limit (e.g. take()); filter before limiting."
                )
            row_filter = function if self._vortex_filter is None else and_(self._vortex_filter, function)
            return self._with_pushdown(filter=row_filter)
        return super().filter(
            function=cast(Callable[..., object] | None, function),
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            fn_kwargs=fn_kwargs,
        )

    @override
    def with_format(self, type: str | None = None) -> VortexIterableDataset:
        return self._with_pushdown(formatting=FormattingConfig(format_type=get_format_type_from_alias(type)))

    def _with_pushdown(
        self,
        *,
        columns: Sequence[str] | None | object = ...,
        filter: Expr | None | object = ...,
        limit: int | None | object = ...,
        formatting: FormattingConfig | None | object = ...,
    ) -> VortexIterableDataset:
        return VortexIterableDataset(
            self._vortex_files,
            columns=self._vortex_columns if columns is ... else columns,  # pyright: ignore[reportArgumentType]
            filter=self._vortex_filter if filter is ... else filter,  # pyright: ignore[reportArgumentType]
            limit=self._vortex_limit if limit is ... else limit,  # pyright: ignore[reportArgumentType]
            batch_size=self._vortex_batch_size,
            split=str(self._split),
            formatting=self._formatting if formatting is ... else formatting,  # pyright: ignore[reportArgumentType]
            shuffling=copy.deepcopy(getattr(self, "_shuffling", None)),
            distributed=copy.deepcopy(self._distributed),
            token_per_repo_id=self._token_per_repo_id,
        )

    def _current_column_names(self) -> list[str]:
        column_names = self.column_names
        if column_names is not None:
            return list(column_names)
        features = self.features
        if features is None:
            return []
        return list(features)


class _VortexExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        files: Sequence[str],
        *,
        columns: Sequence[str] | None,
        filter: Expr | None,
        limit: int | None,
        batch_size: int | None,
        features: hf_datasets.Features,
    ):
        super().__init__()
        self.files: tuple[str, ...] = tuple(files)
        self.columns: tuple[str, ...] | None = _normalize_columns(columns)
        self.filter: Expr | None = filter
        self.limit: int | None = limit
        self.batch_size: int | None = batch_size
        self._features: hf_datasets.Features = features

    @property
    @override
    def iter_arrow(self):
        return self._iter_arrow

    @property
    @override
    def is_typed(self) -> bool:
        return True

    @property
    @override
    def features(self) -> hf_datasets.Features:
        return self._features

    @property
    @override
    def num_shards(self) -> int:
        return max(1, len(self.files))

    @override
    def _init_state_dict(self) -> dict[str, int | str]:
        self._state_dict = {
            "file_idx": 0,
            "file_row_idx": 0,
            "num_yielded": 0,
            "type": self.__class__.__name__,
        }
        return self._state_dict

    @override
    def __iter__(self):
        # Every row in a batch shares that batch's key. Hugging Face discards these keys when
        # iterating an IterableDataset (the formatted path rebatches to single rows first), so a
        # per-batch key is sufficient here.
        for key, table in self._iter_arrow():
            for row in table.to_pylist():
                yield key, row

    def _iter_arrow(self):
        state = self._state()
        start_file_idx = state["file_idx"] if state is not None else 0
        start_file_row_idx = state["file_row_idx"] if state is not None else 0
        yielded = state["num_yielded"] if state is not None else 0

        for file_idx, file_name in enumerate(self.files[start_file_idx:], start=start_file_idx):
            file_row_idx = 0
            # On resume we re-scan the resume file from row 0 and skip the rows already yielded
            # (below), so the scan must read those skipped rows in addition to the rows still owed
            # against the limit; otherwise the limit cuts the scan short and we under-read.
            skip = start_file_row_idx if file_idx == start_file_idx else 0
            for table in _scan_file_as_tables(
                file_name,
                columns=self.columns,
                filter=self.filter,
                limit=None if self.limit is None else self.limit - yielded + skip,
                batch_size=self.batch_size,
            ):
                if self.limit is not None and yielded >= self.limit:
                    return

                if file_idx == start_file_idx and file_row_idx + len(table) <= start_file_row_idx:
                    file_row_idx += len(table)
                    continue

                if file_idx == start_file_idx and file_row_idx < start_file_row_idx:
                    offset = start_file_row_idx - file_row_idx
                    table = table.slice(offset)
                    file_row_idx = start_file_row_idx

                if self.limit is not None and yielded + len(table) > self.limit:
                    table = table.slice(0, self.limit - yielded)

                if len(table) == 0:
                    continue

                yielded += len(table)
                file_row_idx += len(table)
                state = self._state()
                if state is not None:
                    state["file_row_idx"] = file_row_idx
                    state["num_yielded"] = yielded
                yield f"{file_idx}:{file_row_idx - len(table)}", table

            state = self._state()
            if state is not None:
                state["file_idx"] = file_idx + 1
                state["file_row_idx"] = 0

    @override
    def shuffle_data_sources(self, generator) -> _VortexExamplesIterable:  # pyright: ignore[reportMissingParameterType]
        indices = generator.permutation(len(self.files))
        return self._with_files(tuple(self.files[int(idx)] for idx in indices))

    @override
    def shard_data_sources(self, num_shards: int, index: int, contiguous: bool = True) -> _VortexExamplesIterable:
        shard_indices = self.split_shard_indices_by_worker(num_shards, index, contiguous=contiguous)
        return self._with_files(tuple(self.files[i] for i in shard_indices))

    def _with_files(self, files: Sequence[str]) -> _VortexExamplesIterable:
        return _VortexExamplesIterable(
            files,
            columns=self.columns,
            filter=self.filter,
            limit=self.limit,
            batch_size=self.batch_size,
            features=self._features,
        )

    def _state(self) -> dict[str, int] | None:
        if isinstance(self._state_dict, dict):
            return cast(dict[str, int], self._state_dict)
        return None


def _materialize_dataset(
    files: Sequence[str],
    *,
    columns: Sequence[str] | None,
    filter: Expr | None,
    limit: int | None,
    batch_size: int | None,
    split: str,
    cache_dir: str | None,
    keep_in_memory: bool,
    num_proc: int | None,
) -> hf_datasets.Dataset:
    features = _features_for_files(files, _normalize_columns(columns))
    if filter is not None:
        # vortex.Expr cannot be pickled, so it cannot pass through Dataset.from_generator's
        # gen_kwargs (which Hugging Face hashes for the cache fingerprint). Read the filtered rows
        # in-process and build an in-memory dataset from the resulting Arrow tables instead;
        # cache_dir, keep_in_memory, and num_proc do not apply to this path.
        return _materialize_filtered(
            files,
            columns=columns,
            filter=filter,
            limit=limit,
            batch_size=batch_size,
            split=split,
            features=features,
        )
    gen_kwargs = {
        "files": list(files),
        # Keep `columns` a tuple (never a list): Hugging Face shards `num_proc` work across the
        # list-valued gen_kwargs entries, so `files` must be the only list. See
        # datasets.utils.sharding._number_of_shards_in_gen_kwargs.
        "columns": _normalize_columns(columns),
        "filter": filter,
        "limit": limit,
        "batch_size": batch_size,
    }
    # `cache_dir=None` is the Hugging Face default, so a single call covers both cases; the stub
    # mistypes the parameter as `str`, hence the ignore.
    return hf_datasets.Dataset.from_generator(
        _generate_rows,
        features=features,
        cache_dir=cache_dir,  # pyright: ignore[reportArgumentType]
        keep_in_memory=keep_in_memory,
        gen_kwargs=gen_kwargs,
        # A global row limit cannot be divided across processes without overshooting, so force
        # single-process generation whenever a limit is set.
        num_proc=None if limit is not None else num_proc,
        split=hf_datasets.Split(split),
    )


def _materialize_filtered(
    files: Sequence[str],
    *,
    columns: Sequence[str] | None,
    filter: Expr | None,
    limit: int | None,
    batch_size: int | None,
    split: str,
    features: hf_datasets.Features,
) -> hf_datasets.Dataset:
    tables: list[pa.Table] = []
    yielded = 0
    for file_name in files:
        if limit is not None and yielded >= limit:
            break
        for table in _scan_file_as_tables(
            file_name,
            columns=_normalize_columns(columns),
            filter=filter,
            limit=None if limit is None else limit - yielded,
            batch_size=batch_size,
        ):
            if limit is not None and yielded + len(table) > limit:
                table = table.slice(0, limit - yielded)
            if len(table) == 0:
                continue
            tables.append(table)
            yielded += len(table)
            if limit is not None and yielded >= limit:
                break
    combined = pa.concat_tables(tables) if tables else features.arrow_schema.empty_table()
    return hf_datasets.Dataset(
        InMemoryTable(combined),
        info=hf_datasets.DatasetInfo(features=features),
        split=hf_datasets.Split(split),
    )


def _generate_rows(
    files: Sequence[str],
    columns: Sequence[str] | None,
    filter: Expr | None,
    limit: int | None,
    batch_size: int | None,
):
    yielded = 0
    for file_name in files:
        remaining = None if limit is None else limit - yielded
        if remaining is not None and remaining <= 0:
            return
        for table in _scan_file_as_tables(
            file_name,
            columns=columns,
            filter=filter,
            limit=remaining,
            batch_size=batch_size,
        ):
            for row in table.to_pylist():
                yielded += 1
                yield row
                if limit is not None and yielded >= limit:
                    return


def _scan_file_as_tables(
    file_name: str,
    *,
    columns: Sequence[str] | None,
    filter: Expr | None,
    limit: int | None,
    batch_size: int | None,
) -> Iterable[pa.Table]:
    projection = None if columns is None else list(columns)
    # Vortex cannot push a filter and a limit into the same scan. When both are set we scan with
    # only the filter and let the caller enforce the limit while consuming the lazy reader, which
    # still stops early once enough rows have been yielded.
    scan_limit = None if filter is not None else limit
    reader = vx.open(file_name).to_arrow(projection=projection, expr=filter, limit=scan_limit, batch_size=batch_size)
    for batch in reader:
        yield _to_hf_compatible_table(pa.Table.from_batches([batch], schema=reader.schema))


def _features_for_files(files: Sequence[str], columns: Sequence[str] | None) -> hf_datasets.Features:
    # Assumes every file shares the schema of the first; mixed-schema datasets would mis-type
    # because the features are derived from files[0] alone.
    schema = vx.open(files[0]).dtype.to_arrow_schema()
    if columns is not None:
        schema = pa.schema([schema.field(column) for column in columns])
    schema = _hf_compatible_schema(schema)
    return hf_datasets.Features.from_arrow_schema(schema)


def _to_hf_compatible_table(table: pa.Table) -> pa.Table:
    schema = _hf_compatible_schema(table.schema)
    if table.schema.equals(schema, check_metadata=False):
        return table
    return table.cast(schema)


def _hf_compatible_schema(schema: pa.Schema) -> pa.Schema:
    metadata = cast(dict[bytes | str, bytes | str] | None, schema.metadata)
    return pa.schema([_hf_compatible_field(field) for field in schema], metadata=metadata)


def _hf_compatible_field(field: pa.Field) -> pa.Field:
    return pa.field(
        field.name,
        _hf_compatible_type(field.type),
        nullable=field.nullable,
        metadata=field.metadata,
    )


def _hf_compatible_type(dtype: pa.DataType) -> pa.DataType:
    if pa.types.is_string_view(dtype):
        return pa.string()
    if pa.types.is_binary_view(dtype):
        return pa.binary()
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        value_field = _hf_compatible_field(dtype.value_field)
        if pa.types.is_large_list(dtype):
            return pa.large_list(value_field)
        if pa.types.is_fixed_size_list(dtype):
            return pa.list_(value_field, dtype.list_size)
        return pa.list_(value_field)
    if pa.types.is_struct(dtype):
        return pa.struct([_hf_compatible_field(field) for field in dtype])
    return dtype


def _normalize_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    return tuple(columns)


def _resolve_data_files(
    path: str | Path,
    *,
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None,
    split: str | Sequence[str] | None,
    revision: str | None,
    token: bool | str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> dict[str, list[str]]:
    normalized = _normalize_data_files(data_files, split=split)
    path_str = str(path)
    path_obj = Path(path)

    if path_obj.exists() or _looks_like_local_path(path_str):
        if not path_obj.exists() and data_files is None:
            split_name = split if isinstance(split, str) else _DEFAULT_SPLIT
            return _resolve_local_files(Path("."), {split_name: [path_str]})
        return _resolve_local_files(path_obj, normalized)

    from huggingface_hub import snapshot_download

    allow_patterns = sorted({pattern for patterns in normalized.values() for pattern in patterns})
    snapshot_dir = Path(
        snapshot_download(
            path_str,
            repo_type="dataset",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
    )
    return _resolve_local_files(snapshot_dir, normalized)


def _normalize_data_files(
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None,
    *,
    split: str | Sequence[str] | None,
) -> dict[str, list[str]]:
    if isinstance(data_files, Mapping):
        return {str(name): _as_list(patterns) for name, patterns in data_files.items()}
    if split is not None and not isinstance(split, str):
        raise ValueError(f"Requesting multiple splits {list(split)!r} requires a `data_files` mapping; pass one.")
    split_name = split if isinstance(split, str) else _DEFAULT_SPLIT
    return {split_name: _as_list(data_files or _DEFAULT_DATA_FILES)}


def _resolve_local_files(base: Path, split_to_patterns: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {}
    for split_name, patterns in split_to_patterns.items():
        files: list[str] = []
        for pattern in patterns:
            pattern_path = Path(pattern)
            if pattern_path.is_absolute():
                matches = sorted(glob.glob(str(pattern_path), recursive=True))
            elif base.is_file() and pattern == _DEFAULT_DATA_FILES:
                matches = [str(base)]
            else:
                matches = sorted(glob.glob(str(base / pattern), recursive=True))
            files.extend(match for match in matches if Path(match).is_file())
        if not files:
            raise FileNotFoundError(f"No Vortex files matched split {split_name!r} patterns {list(patterns)!r}")
        resolved[split_name] = files
    return resolved


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return [_DEFAULT_DATA_FILES]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _looks_like_local_path(path: str) -> bool:
    return path.startswith((".", "/", "~")) or any(char in path for char in "*?[]")


__all__ = ["VortexIterableDataset", "load_dataset"]
