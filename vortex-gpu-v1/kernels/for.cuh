// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Frame-of-Reference kernel declarations

#ifndef FOR_CUH
#define FOR_CUH

#include <stdint.h>

// Device function template (callable from other kernels)
template<typename ValueT>
__device__ __forceinline__ void for_device(
    ValueT *__restrict values_in_out,
    ValueT reference,
    int thread_idx
);

// Kernel functions (callable from host)
extern "C" __global__ void for_vu8(uint8_t *__restrict values, uint8_t reference);
extern "C" __global__ void for_vu16(uint16_t *__restrict values, uint16_t reference);
extern "C" __global__ void for_vu32(uint32_t *__restrict values, uint32_t reference);
extern "C" __global__ void for_vu64(uint64_t *__restrict values, uint64_t reference);

extern "C" __global__ void for_vi8(int8_t *__restrict values, int8_t reference);
extern "C" __global__ void for_vi16(int16_t *__restrict values, int16_t reference);
extern "C" __global__ void for_vi32(int32_t *__restrict values, int32_t reference);
extern "C" __global__ void for_vi64(int64_t *__restrict values, int64_t reference);

#endif // FOR_CUH
