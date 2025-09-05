# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
from copy import deepcopy
from typing import Any, TypedDict, final
from typing_extensions import override

import vortex as vx

from streaming.base.format.base.reader import FileInfo, JointReader
from streaming.base.format.base.writer import JointWriter


class MdsDatasetParameters(TypedDict):
    version: int
    format: str
    dirname: str
    split: str | None
    raw_data: FileInfo
    zip_data: FileInfo | None
    compression: str | None
    hashes: list[str]
    samples: int
    size_limit: int | str | None


@final
class VortexReader(JointReader):
    """Teaches :class:`~streaming.StreamingDataset` how to read Vortex files.

    Examples
    --------

    Given a folder of files, `/path/to/dataset/train`, written by :class:`.VortexWriter`, construct
    a :class:`~torch.utils.data.DataLoader`:

    .. code-block:: Python

        import vortex.mds as vxmds

        vxmds.register_vortex_with_mds()

        ds = StreamingDataset(local="/path/to/dataset", split="train", batch_size=16)
        dl = DataLoader(ds, batch_size=16)

    """

    # TODO(DK): There is no good way to copy an args & kwargs signature:
    # https://github.com/python/typing/discussions/1079
    def __init__(
        self,
        dirname: str,
        split: str | None,
        compression: str | None,
        hashes: list[str],
        raw_data: FileInfo,
        samples: int,
        size_limit: int | str | None,
        zip_data: FileInfo | None,
    ) -> None:
        super().__init__(
            dirname,
            split,
            compression,
            hashes,
            raw_data,
            samples,
            size_limit,
            zip_data,
        )
        self._scan: vx.RepeatedScan | None = None

    @classmethod
    def from_json(cls, dirname: str, split: str | None, obj: MdsDatasetParameters):
        from streaming.base.format.base.reader import FileInfo

        args = deepcopy(obj)

        args_version = args["version"]
        if args_version != 2:
            raise ValueError(f"Unsupported streaming data version: {args_version}. Expected version 2.")
        _ = args.pop("version")

        args_format = args["format"]
        if args_format != "vortex":
            raise ValueError(f"Unsupported data format: {args_format}. Expected to be `vortex`.")
        _ = args.pop("format")

        args["dirname"] = dirname
        args["split"] = split
        args["raw_data"] = FileInfo(**args["raw_data"])  # pyright: ignore[reportCallIssue]
        args["zip_data"] = None  # Vortex files aren't compressed.

        return cls(**args)  # pyright: ignore[reportCallIssue]

    @override
    def evict(self) -> int:
        """Remove all files belonging to this shard."""
        self._scan = None  # Clean up the scan handle first.
        return super().evict()

    @override
    def decode_sample(self, data: bytes) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        # NOTE(marko): Annoying but abstractions are wrong.
        raise NotImplementedError("`decode_sample` should NOT be called")

    @override
    def get_sample_data(self, idx: int) -> bytes:
        # NOTE(marko): Annoying but abstractions are wrong.
        raise NotImplementedError("`get_sample_data` should NOT be called")

    @override
    def get_item(self, idx: int) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        if self._scan is None:
            filename = os.path.join(self.dirname, self.split, self.raw_data.basename)
            self._scan = vx.open(filename, without_segment_cache=True).to_repeated_scan()
        item = self._scan.scalar_at(idx).as_py()
        assert isinstance(item, dict)
        return item


@final
class VortexWriter(JointWriter):
    """Write samples as a sequence of Vortex files.

    Warnings
    --------

    Vortex does not support the `size_limit` parameter.

    Parameters
    ----------
    out : str
        Output dataset directory to save shard files.

    max_shard_rows : int
        Maximum number of samples per shard.

    Examples
    --------

    Given a folder of files, `/path/to/dataset/train`, written by :class:`.VortexWriter`, construct
    a :class:`~torch.utils.data.DataLoader`:

    .. code-block:: Python

        import vortex.mds as vxmds

        with vxmds.VortexWriter(out="/path/to/dataset/train", max_shard_rows=128) as writer:
            for i in range(1000):
                sample = {"text": "Hello world!", "id": i, "other": "metadata"}
                writer.write(sample)

    """

    format = "vortex"

    def __init__(self, *, out: str | tuple[str, str], max_shard_rows: int) -> None:
        super().__init__(out=out, size_limit=None)
        self._max_shard_rows = max_shard_rows

    @override
    def encode_sample(self, sample: dict[str, Any]) -> bytes:  # pyright: ignore[reportExplicitAny]
        raise NotImplementedError("`encode_sample` should NOT be called")

    @override
    def encode_joint_shard(self) -> bytes:
        raise NotImplementedError("`encode_joint_shard` should NOT be called")

    @override
    def flush_shard(self) -> None:
        # Never zip, `zip_data_basename` is always None as writer doesn't support `compression`.
        raw_data_basename, _zip_data_basename = self._name_next_shard()

        array = vx.array(self.new_samples)
        raw_data = self._process_array(array, raw_data_basename)
        obj: MdsDatasetParameters = {  # pyright: ignore[reportAssignmentType]
            "samples": len(self.new_samples),
            "raw_data": raw_data,
            "zip_data": None,
            **self.get_config(),
        }
        self.shards.append(obj)  # pyright: ignore[reportUnknownMemberType]

        # Execute the task if there is no exception in any of the async threads.
        future = self.executor.submit(self.cloud_writer.upload_file, raw_data_basename)  # pyright: ignore[reportAny]
        future.add_done_callback(self.exception_callback)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    def _process_array(self, array: vx.Array, raw_basename: str) -> FileInfo:
        filename = os.path.join(self.local, raw_basename)  # pyright: ignore[reportAny]
        vx.io.write(array, filename)
        with open(filename, "rb") as in_:
            raw_data = in_.read()
            return self._hash(raw_data, raw_basename)  # pyright: ignore[reportReturnType]

    @override
    def write(self, sample: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
        if self.event.is_set():
            # Shutdown the executor and cancel all the pending futures
            # due to exception in one of the threads.
            self.cancel_future_jobs()
            raise Exception("One of the threads failed. Check other traceback for more details.")

        if len(self.new_samples) + 1 > self._max_shard_rows:
            self.flush_shard()
            self._reset_cache()

        self.new_samples.append(sample)  # pyright: ignore[reportArgumentType]
