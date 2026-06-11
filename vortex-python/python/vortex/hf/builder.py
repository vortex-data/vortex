# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""A Hugging Face ``datasets`` builder for Vortex files.

This module is registered with the ``datasets`` library by
:func:`vortex.hf.register_datasets`, after which ``.vortex`` files can be loaded with
``datasets.load_dataset``. It is structured like the packaged ``parquet`` builder that
ships with ``datasets``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import datasets
import pyarrow as pa
from datasets.table import table_cast

from ..file import open as _open_vortex

try:
    from datasets.builder import Key

    _key: Callable[[int, int], object] = Key
except ImportError:
    # `datasets` < 5.0 uses plain string keys.
    def _string_key(file_idx: int, batch_idx: int) -> str:
        return f"{file_idx}_{batch_idx}"

    _key = _string_key


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

    def __post_init__(self):
        super().__post_init__()


class Vortex(datasets.ArrowBasedBuilder):
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
        return datasets.DatasetInfo(features=self.config.features)

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
            # Infer features from the first file if not explicitly specified.
            if self.info.features is None:
                for first_file in itertools.chain.from_iterable(dl_manager.iter_files(file) for file in raw_files):
                    schema = _open_vortex(first_file).dtype.to_arrow_schema()
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
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            vxf = _open_vortex(file)
            reader = vxf.to_arrow(projection=self.config.columns, batch_size=self.config.batch_size)
            for batch_idx, batch in enumerate(reader):
                table = pa.Table.from_batches([batch])
                if target_schema is not None:
                    table = table_cast(table, target_schema)
                yield _key(file_idx, batch_idx), table
