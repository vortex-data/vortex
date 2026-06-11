# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""A Hugging Face ``datasets`` builder for Vortex files.

This module is registered with the ``datasets`` library by
:func:`vortex.hf.register_datasets`, after which ``.vortex`` files can be loaded with
``datasets.load_dataset``. It is structured like the packaged ``parquet`` builder that
ships with ``datasets``.

Supported ``load_dataset`` keyword arguments (via :class:`VortexConfig`):

- ``columns``: project a subset of columns (pushed down to the Vortex scan).
- ``filters``: a predicate pushed down to the Vortex scan, either a
  :class:`vortex.expr.Expr` or parquet-style DNF tuples such as
  ``[("age", ">", 35)]`` (AND) or ``[[("x", "==", 1)], [("x", "==", 5)]]`` (OR of ANDs).
- ``limit``: maximum number of rows to read across all files (after filtering).
- ``indices``: explicit row indices to read, global across the split's files in
  listed order. Indices are deduplicated and rows are returned in ascending order.
- ``batch_size``: rows per generated Arrow batch.
- ``features``: explicit :class:`datasets.Features` instead of schema inference.
- ``on_bad_files``: ``"error"`` (default), ``"warn"``, or ``"skip"`` — what to do
  when a file cannot be opened as a Vortex file.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import datasets
import pyarrow as pa
from datasets.table import table_cast

from ..arrays import array as _vx_array
from ..expr import Expr, column, not_
from ..file import VortexFile
from ..file import open as _open_vortex

logger = logging.getLogger(__name__)

_ON_BAD_FILES = ("error", "warn", "skip")

try:
    from datasets.builder import Key

    _key: Callable[[int, int], object] = Key
except ImportError:
    # `datasets` < 5.0 uses plain string keys.
    def _string_key(file_idx: int, batch_idx: int) -> str:
        return f"{file_idx}_{batch_idx}"

    _key = _string_key

try:
    from datasets.builder import (
        _CountableBuilderMixin,  # pyright: ignore[reportPrivateUsage, reportAssignmentType]
    )
except ImportError:
    # `datasets` < 5.0 has no countable-builder support; fall back to a no-op base.
    class _CountableBuilderMixin:  # pyright: ignore[reportRedeclaration]
        pass


#: Parquet-style filter condition: a ``(column, op, value)`` tuple. A list of
#: conditions is an AND group; a list of such lists is an OR of AND groups (DNF).
FilterTuple = tuple[str, str, Any]


def _condition_to_expr(condition: FilterTuple) -> Expr:
    if not (isinstance(condition, tuple) and len(condition) == 3):
        raise ValueError(f"Each filter condition must be a (column, op, value) tuple, got {condition!r}")
    col, op, value = condition
    field = column(col)
    if op in ("==", "="):
        return field == value
    if op == "!=":
        return field != value
    if op == "<":
        return field < value
    if op == "<=":
        return field <= value
    if op == ">":
        return field > value
    if op == ">=":
        return field >= value
    if op in ("in", "not in"):
        values = list(value)  # pyright: ignore[reportArgumentType]
        if not values:
            raise ValueError(f"{op!r} filter on column {col!r} requires at least one value")
        expr = field == values[0]
        for v in values[1:]:
            expr = expr | (field == v)
        return not_(expr) if op == "not in" else expr
    raise ValueError(f"Unsupported filter operator {op!r} in condition {condition!r}")


def filters_to_expr(filters: Expr | list[FilterTuple] | list[list[FilterTuple]]) -> Expr:
    """Convert parquet-style DNF filter tuples (or a ready-made expression) into a
    :class:`vortex.expr.Expr`."""
    if isinstance(filters, Expr):
        return filters
    if not isinstance(filters, list) or not filters:
        raise ValueError(
            "filters must be a vortex.expr.Expr, a list of (column, op, value) tuples, "
            f"or a list of lists of tuples, got {filters!r}"
        )
    groups: list[list[FilterTuple]] = [filters] if isinstance(filters[0], tuple) else filters  # pyright: ignore[reportAssignmentType]
    group_exprs: list[Expr] = []
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError(f"Each filter group must be a non-empty list of tuples, got {group!r}")
        expr = _condition_to_expr(group[0])
        for condition in group[1:]:
            expr = expr & _condition_to_expr(condition)
        group_exprs.append(expr)
    result = group_exprs[0]
    for expr in group_exprs[1:]:
        result = result | expr
    return result


def _without_view_types(schema: pa.Schema) -> pa.Schema:
    """Map Arrow view types to their non-view equivalents.

    ``datasets`` features do not understand ``string_view``/``binary_view``, which
    Vortex produces for variable-length data.
    """

    def convert(dtype: pa.DataType) -> pa.DataType:
        if dtype == pa.string_view():
            return pa.string()
        if dtype == pa.binary_view():
            return pa.binary()
        if isinstance(dtype, pa.StructType):
            return pa.struct([field.with_type(convert(field.type)) for field in dtype.fields])
        if isinstance(dtype, (pa.ListType, pa.ListViewType)):
            return pa.list_(convert(dtype.value_type))
        if isinstance(dtype, pa.LargeListType):
            return pa.large_list(convert(dtype.value_type))
        if isinstance(dtype, pa.FixedSizeListType):
            return pa.list_(convert(dtype.value_type), dtype.list_size)
        return dtype

    return pa.schema(
        [field.with_type(convert(field.type)) for field in schema],
        metadata=schema.metadata,  # pyright: ignore[reportArgumentType]
    )


@dataclass
class VortexConfig(datasets.BuilderConfig):
    """BuilderConfig for the Vortex file format."""

    batch_size: int | None = None
    columns: list[str] | None = None
    features: datasets.Features | None = None
    filters: Expr | list[FilterTuple] | list[list[FilterTuple]] | None = None
    limit: int | None = None
    indices: list[int] | None = None
    on_bad_files: str = "error"

    def __post_init__(self):
        super().__post_init__()

    def create_config_id(self, config_kwargs: dict, *args, **kwargs) -> str:  # pyright: ignore[reportMissingParameterType,reportMissingTypeArgument]
        # `datasets` hashes non-default config kwargs with pickle to build the cache
        # fingerprint, and vortex Expr objects are not picklable. Hash their stable
        # string form instead (DNF tuple filters are lists and hash as-is).
        filters = config_kwargs.get("filters")
        if filters is not None and not isinstance(filters, list):
            config_kwargs = {**config_kwargs, "filters": str(filters)}
        return super().create_config_id(config_kwargs, *args, **kwargs)


class Vortex(datasets.ArrowBasedBuilder, _CountableBuilderMixin):
    """A ``datasets`` builder that reads ``.vortex`` files."""

    BUILDER_CONFIG_CLASS = VortexConfig
    config: VortexConfig

    def _info(self) -> datasets.DatasetInfo:
        if (
            self.config.columns is not None
            and self.config.features is not None
            and set(self.config.columns) != set(self.config.features)
        ):
            raise ValueError(
                "The columns and features argument must contain the same columns, but got "
                f"{self.config.columns} and {self.config.features}"
            )
        if self.config.on_bad_files not in _ON_BAD_FILES:
            raise ValueError(f"on_bad_files must be one of {_ON_BAD_FILES}, got {self.config.on_bad_files!r}")
        if self.config.indices is not None:
            if self.config.filters is not None or self.config.limit is not None:
                raise ValueError("indices cannot be combined with filters or limit")
            if any(i < 0 for i in self.config.indices):
                raise ValueError("indices must be non-negative")
        if self.config.filters is not None:
            # Validate eagerly so malformed filters fail at load time, not mid-scan.
            filters_to_expr(self.config.filters)
        return datasets.DatasetInfo(features=self.config.features)

    def _open_or_handle_bad_file(self, file: str) -> VortexFile | None:
        """Open ``file``, applying the ``on_bad_files`` policy on failure."""
        try:
            return _open_vortex(file)
        except Exception:
            if self.config.on_bad_files == "error":
                raise
            if self.config.on_bad_files == "warn":
                logger.warning("Skipping file that could not be opened as Vortex: %s", file)
            return None

    def _split_generators(self, dl_manager) -> list[datasets.SplitGenerator]:  # pyright: ignore[reportMissingParameterType]
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        if hasattr(dl_manager, "download_config"):
            dl_manager.download_config.extract_on_the_fly = True
        downloaded = dl_manager.download(self.config.data_files)
        data_files = downloaded if isinstance(downloaded, dict) else {"train": downloaded}
        splits: list[datasets.SplitGenerator] = []
        for split_name, raw_files in data_files.items():
            if isinstance(raw_files, str):
                raw_files = [raw_files]
            # Infer features from the first readable file if not explicitly specified.
            if self.info.features is None:
                for first_file in itertools.chain.from_iterable(dl_manager.iter_files(file) for file in raw_files):
                    vxf = self._open_or_handle_bad_file(first_file)
                    if vxf is None:
                        continue
                    schema = vxf.dtype.to_arrow_schema()
                    self.info.features = datasets.Features.from_arrow_schema(_without_view_types(schema))
                    break
            files = [dl_manager.iter_files(file) for file in raw_files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        if self.config.columns is not None and set(self.config.columns) != set(self.info.features or {}):
            self.info.features = datasets.Features(
                {col: feat for col, feat in (self.info.features or {}).items() if col in self.config.columns}
            )
        return splits

    def _generate_tables(self, files: list[Iterable[str]]) -> Iterator[tuple[object, pa.Table]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        target_schema = self.info.features.arrow_schema if self.info.features is not None else None
        expr = filters_to_expr(self.config.filters) if self.config.filters is not None else None
        indices = sorted(set(self.config.indices)) if self.config.indices is not None else None
        remaining = self.config.limit
        row_offset = 0
        # `datasets` requires the shard ids in yielded keys to be dense (a file that
        # yields no tables must not leave a gap), so count yielding files, not files.
        out_shard_id = -1

        for file in itertools.chain.from_iterable(files):
            if remaining is not None and remaining <= 0:
                break
            vxf = self._open_or_handle_bad_file(file)
            if vxf is None:
                continue

            if indices is not None:
                row_count = len(vxf)
                local = [i - row_offset for i in indices if row_offset <= i < row_offset + row_count]
                row_offset += row_count
                if not local:
                    continue
                reader = vxf.scan(
                    projection=self.config.columns,
                    indices=_vx_array(local),
                    batch_size=self.config.batch_size,
                ).to_arrow()
            else:
                # Vortex scans cannot combine a filter with a pushed-down limit, so
                # with both set the filter is pushed down and the limit enforced here.
                reader = vxf.to_arrow(
                    projection=self.config.columns,
                    expr=expr,
                    limit=remaining if expr is None else None,
                    batch_size=self.config.batch_size,
                )

            started = False
            for batch_idx, batch in enumerate(reader):
                table = pa.Table.from_batches([batch])
                if target_schema is not None:
                    table = table_cast(table, target_schema)
                if remaining is not None:
                    if table.num_rows > remaining:
                        table = table.slice(0, remaining)
                    remaining -= table.num_rows
                if not started:
                    out_shard_id += 1
                    started = True
                yield _key(out_shard_id, batch_idx), table
                if remaining is not None and remaining <= 0:
                    break

        if indices is not None and indices and indices[-1] >= row_offset:
            raise IndexError(f"indices contain values >= total row count {row_offset}: {indices[-1]}")

    def _generate_num_examples(self, files: list[Iterable[str]]) -> Iterator[int]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Yield per-file row counts from file footers, without scanning any data.

        Counting is only supported for plain scans: with ``filters``, ``limit``, or
        ``indices`` the number of rows cannot be derived from footer metadata alone.
        """
        if self.config.filters is not None or self.config.limit is not None or self.config.indices is not None:
            raise NotImplementedError("Counting examples is not supported with filters, limit, or indices")
        for file in itertools.chain.from_iterable(files):
            vxf = self._open_or_handle_bad_file(file)
            yield len(vxf) if vxf is not None else 0
