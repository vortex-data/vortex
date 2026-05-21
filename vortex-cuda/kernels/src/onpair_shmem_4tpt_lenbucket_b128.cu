// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Length-bucket dict at 128-thread block granularity (4 warps). The B200
// granularity sweep showed 128-thread blocks are the Blackwell lever; the base
// lenbucket kernel was only ever measured at 512 threads, the same granularity
// that made split8read look "dead" before it won at 128. This re-tests the
// smaller-working-set lenbucket layout at the granularity that matters.
#define WARPS_PER_BLOCK_MAX 4u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 8)
#define onpair_shmem_4tpt_lenbucket onpair_shmem_4tpt_lenbucket_b128
#include "onpair_shmem_4tpt_lenbucket.cu"
