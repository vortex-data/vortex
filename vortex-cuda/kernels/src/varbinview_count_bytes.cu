// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "varbinview.cuh"

// Count the total number of bytes needed to store the string data
// stored in the varbinview array.
// NOTE: this works only if the data has no nulls.
extern "C" __global__ void varbinview_count_bytes(
) {
}