// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

// FastLanes ordering array
__constant__ int FL_ORDER[] = {0, 4, 2, 6, 1, 5, 3, 7};

// FastLanes organises every 1024-element vector into a transposed layout
// of FL_LANES columns × (1024 / FL_LANES) rows. Each column is a "lane"
// that can be processed independently of every other lane, which is what
// makes all FastLanes encodings (FFOR, DELTA, RLE, ALP, …) fully
// data-parallel. One CUDA thread or one CPU SIMD lane handles one
// FastLanes lane.
//
// Paper: https://ir.cwi.nl/pub/35881/35881.pdf
// Repo:  https://github.com/cwida/FastLanes

/// FastLanes chunk size in elements.
constexpr uint32_t FL_CHUNK = 1024;

/// Number of FastLanes lanes for element type T (1024 / bit-width).
template <typename T>
constexpr uint32_t FL_LANES = FL_CHUNK / (sizeof(T) * 8);

// Compute the index in the FastLanes layout
#define INDEX(row, lane) (FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane)

// Create a mask with 'width' bits set
#define MASK(T, width) (((T)1 << width) - 1)
