# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import itertools
import posixpath
from collections.abc import Iterable
from typing import final
from urllib.parse import quote

import pyarrow as pa
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.planner.plan_write_op import WRITE_UUID_KWARG_NAME
from ray.data._internal.util import RetryingPyFileSystem
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_datasink import BlockBasedFileDatasink
from typing_extensions import override

from .. import io as vx_io
from .datasource import _url_scheme


@final
class VortexDatasink(BlockBasedFileDatasink):
    """Write a Ray Dataset as one Vortex file per write task."""

    def __init__(self, path: str) -> None:
        super().__init__(path, file_format="vortex")

    @override
    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> None:
        blocks = iter(blocks)
        try:
            first = next(blocks)
        except StopIteration:
            return

        first_table = BlockAccessor.for_block(first).to_arrow()
        tables = itertools.chain(
            (first_table,),
            (BlockAccessor.for_block(block).to_arrow() for block in blocks),
        )
        batches = (batch for table in tables for batch in table.to_batches())
        reader = pa.RecordBatchReader.from_batches(first_table.schema, batches)

        write_uuid = ctx.kwargs[WRITE_UUID_KWARG_NAME]
        if hasattr(self.filename_provider, "get_filename_for_task"):
            filename = self.filename_provider.get_filename_for_task(write_uuid, ctx.task_idx)
        else:
            # Ray 2.48 only provides the block-based filename hook.
            filename = self.filename_provider.get_filename_for_block(first_table, write_uuid, ctx.task_idx, 0)

        write_path = posixpath.join(self.path, filename)
        filesystem = self.filesystem
        if isinstance(filesystem, RetryingPyFileSystem):
            filesystem = filesystem.unwrap()
        scheme = _url_scheme(filesystem)
        if scheme is not None:
            write_path = f"{scheme}://{quote(write_path)}"
        vx_io.write(reader, write_path)
