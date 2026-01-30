// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

// FastLanes ordering array
__device__ int FL_ORDER[] = {0, 4, 2, 6, 1, 5, 3, 7};

// Compute the index in the FastLanes layout
#define INDEX(row, lane) (FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane)

// Create a mask with 'width' bits set
#define MASK(T, width) (((T)1 << width) - 1)
