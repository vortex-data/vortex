# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""This module permits reading Vortex files with a [Mosaic Data Streaming](https://github.com/mosaicml/streaming) StreamingDataset.

This module depends on the optional `mosaicml-streaming` dependency. You must install that package
and also explicitly import this module before use:

```
import vortex as vx
import vortex.mds

```

:class:`VortexReader` implements the :class:`streaming.base.format.Reader` interface and enables
usage of Vortex files as MDS shards. :class:`VortexWriter` implements the corresponding
:class:`streaming.base.format.Writer` interface.

"""

from streaming.base.format import _readers as _registry  # pyright: ignore[reportPrivateUsage]

from .format import VortexReader, VortexWriter

# NOTE(marko): There is no good way to register a format currently.
if VortexWriter.format not in _registry:
    _registry[VortexWriter.format] = VortexReader  # pyright: ignore[reportArgumentType]

__all__ = ["VortexReader", "VortexWriter"]
