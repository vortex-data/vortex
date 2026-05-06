# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from vortex._lib.runtime import (  # pyright: ignore[reportMissingModuleSource]
    set_worker_threads,
    set_worker_threads_to_available_parallelism,
    worker_count,
)

__all__ = [
    "set_worker_threads",
    "set_worker_threads_to_available_parallelism",
    "worker_count",
]
