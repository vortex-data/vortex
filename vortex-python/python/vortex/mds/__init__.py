# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Support for Vortex within a `Mosaic Data Streaming <https://github.com/mosaicml/streaming>`__ StreamingDataset.

This module depends on the optional `mosaicml-streaming` dependency. You must install that package
and also explicitly import this module before use:

.. code-block:: python

    import vortex.mds as vxmds

    vxmds.register_vortex_with_mds()

    with vxmds.VortexWriter(out="/path/to/dataset/train", max_shard_rows=128) as writer:
        for i in range(1000):
            sample = {"text": "Hello world!", "id": i, "other": "metadata"}
            writer.write(sample)

    ds = StreamingDataset(local="/path/to/dataset", split="train", batch_size=16)
    dl = DataLoader(ds, batch_size=16)


:class:`.VortexReader` implements the :class:`~streaming.base.format.Reader` interface and enables
usage of Vortex files as MDS shards. :class:`VortexWriter` implements the corresponding
:class:`~streaming.MDSWriter` interface.

"""

from .format import VortexReader, VortexWriter


def register_vortex_with_mds():
    """Register the Vortex format with Mosaic Data Streaming.

    You must call this method before you create a Vortex :class:`~streaming.StreamingDataset`; otherwise, you
    will receive errors like the following:

    .. code-block:: python

        >       cls = _readers[obj['format']]
                      ^^^^^^^^^^^^^^^^^^^^^^^
        E       KeyError: 'vortex'

    """
    from streaming.base.format import _readers as _registry  # pyright: ignore[reportPrivateUsage]

    # NOTE(marko): There is no good way to register a format currently.
    if VortexWriter.format not in _registry:
        _registry[VortexWriter.format] = VortexReader  # pyright: ignore[reportArgumentType]


__all__ = ["VortexReader", "VortexWriter"]
