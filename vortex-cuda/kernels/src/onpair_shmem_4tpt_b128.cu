// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B: finer block granularity (128-thread blocks, 4 warps), 50% occ at 64 regs.
#define WARPS_PER_BLOCK_MAX 4u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 8)
#define onpair_shmem_4tpt onpair_shmem_4tpt_b128
#include "onpair_shmem_4tpt.cu"
