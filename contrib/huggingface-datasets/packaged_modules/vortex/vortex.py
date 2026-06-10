# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# NOTE: strip the two SPDX lines above when copying this file into
# huggingface/datasets — upstream files carry no license headers.

"""A ``datasets`` builder for the Vortex columnar file format.

Destined for ``src/datasets/packaged_modules/vortex/vortex.py`` in the
``huggingface/datasets`` repository. Mirrors the structure of the packaged
``parquet`` and ``lance`` builders.

This file is the *upstream* variant of ``vortex.hf.builder`` from the
``vortex-data`` package. The differences are intentional and minimal:

- ``vortex`` (the ``vortex-data`` PyPI package) is an *optional* dependency of
  ``datasets``, so it is imported lazily inside builder methods rather than at
  module import time (the same pattern the ``lance`` builder uses).
- A ``token`` config option is forwarded to ``vortex.hf.open`` for ``hf://``
  URIs so gated/private repositories work with an explicit token, matching the
  ``lance`` builder's config surface.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import datasets
import pyarrow as pa
from datasets.table import table_cast

try:
    from datasets.builder import Key

    _key: Callable[[int, int], object] = Key
except ImportError:
    # `datasets` < 5.0 uses plain string keys. When upstreaming against a
    # pinned `datasets` version this shim can be collapsed to whichever
    # branch applies.
    def _string_key(file_idx: int, batch_idx: int) -> str:
        return f"{file_idx}_{batch_idx}"

    _key = _string_key


def _open_vortex(file: str, token: str | None = None):
    """Lazily open a Vortex file from a local path or URL.

    ``vortex.open`` natively understands local paths, ``s3://``/``gs://``/
    ``abfss://``/``https://`` object-store URLs, and ``hf://`` Hub URLs (using
    the ambient ``HF_TOKEN`` / ``huggingface_hub`` token cache). An explicit
    ``token`` is honoured for ``hf://`` URIs.
    """
    try:
        import vortex
    except ImportError as err:
        raise ImportError(
            "Loading .vortex files requires the vortex-data package: `pip install vortex-data`"
        ) from err
    if token is not None and file.startswith("hf://"):
        import vortex.hf

        return vortex.hf.open(file, token=token)
    return vortex.open(file)


def _without_view_types(schema: pa.Schema) -> pa.Schema:
    """Map Arrow view types to their non-view equivalents.

    ``datasets`` features do not understand ``string_view``/``binary_view``,
    which Vortex produces for variable-length data.
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
        metadata=schema.metadata,
    )


@dataclass
class VortexConfig(datasets.BuilderConfig):
    """BuilderConfig for the Vortex file format."""

    batch_size: int | None = None
    columns: list[str] | None = None
    features: datasets.Features | None = None
    token: str | None = None

    def __post_init__(self):
        super().__post_init__()


class Vortex(datasets.ArrowBasedBuilder):
    """A ``datasets`` builder that reads ``.vortex`` files.

    Vortex is a single-file columnar format: there are no sidecar metadata
    files, so no ``METADATA_FILE_NAMES``/``METADATA_EXTENSIONS`` are declared.
    Scans are lazy — in streaming mode only the file footer plus the segments
    backing the requested columns/batches are fetched via ranged HTTP reads.
    """

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

    def _split_generators(self, dl_manager) -> list[datasets.SplitGenerator]:
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
                    schema = _open_vortex(first_file, token=self.config.token).dtype.to_arrow_schema()
                    self.info.features = datasets.Features.from_arrow_schema(_without_view_types(schema))
                    break
            files = [dl_manager.iter_files(file) for file in raw_files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        if self.config.columns is not None and set(self.config.columns) != set(self.info.features or {}):
            self.info.features = datasets.Features(
                {col: feat for col, feat in (self.info.features or {}).items() if col in self.config.columns}
            )
        return splits

    def _generate_tables(self, files: list[Iterable[str]]) -> Iterator[tuple[object, pa.Table]]:
        target_schema = self.info.features.arrow_schema if self.info.features is not None else None
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            vxf = _open_vortex(file, token=self.config.token)
            reader = vxf.to_arrow(projection=self.config.columns, batch_size=self.config.batch_size)
            for batch_idx, batch in enumerate(reader):
                table = pa.Table.from_batches([batch])
                if target_schema is not None:
                    table = table_cast(table, target_schema)
                yield _key(file_idx, batch_idx), table
