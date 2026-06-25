# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest
import vortex_cuda

import vortex


def test_debug_array_metadata_dtype_reads_base_vortex_array():
    array = vortex.Array.from_range(range(0, 3))

    assert vortex_cuda._debug_array_metadata_dtype(array) == str(array.dtype)


def test_metadata_bridge_reports_arrays_that_need_buffer_handoff():
    array = vortex.array([1, 2, 3])

    with pytest.raises(RuntimeError, match="metadata-only bridge.*buffers"):
        _ = vortex_cuda._debug_array_metadata_dtype(array)


def test_export_device_array_returns_capsules_or_clean_cuda_error():
    array = vortex.Array.from_range(range(0, 3))

    if not vortex_cuda.cuda_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            _ = vortex_cuda.export_device_array(array)
        return

    schema, device_array = vortex_cuda.export_device_array(array)
    assert type(schema).__name__ == "PyCapsule"
    assert type(device_array).__name__ == "PyCapsule"
