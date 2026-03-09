// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// In-place posterization of a uint8 buffer.
//
// Quantizes each value to one of `levels` evenly spaced steps.
// For example, with levels=4 the output values are {0, 85, 170, 255}.
//
// Formula: out[i] = round((floor(in[i] * levels / 256) + 0.5) * 256 / levels)
// Simplified with integer math:
//   bucket = in[i] * levels / 256
//   out[i] = (bucket * 255 + (levels - 1) / 2) / (levels - 1)

#include <stdint.h>

extern "C" __global__ void posterize(
    uint8_t* __restrict__ data,
    uint32_t len,
    uint32_t levels)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    uint32_t v = data[i];
    // Quantize: which bucket does this value fall in? (0..levels-1)
    uint32_t bucket = v * levels / 256;
    // Clamp bucket to levels-1 (in case v == 255)
    if (bucket >= levels) bucket = levels - 1;
    // Map bucket back to 0..255 range evenly
    uint32_t out = bucket * 255 / (levels - 1);
    data[i] = static_cast<uint8_t>(out);
}
