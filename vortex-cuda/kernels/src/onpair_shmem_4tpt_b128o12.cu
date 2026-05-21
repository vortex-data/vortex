// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B: finest granularity (128-thread) + force 12 blocks/SM -> ~42 regs -> 75% occ.
#define WARPS_PER_BLOCK_MAX 4u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 12)
#define onpair_shmem_4tpt onpair_shmem_4tpt_b128o12
#include "onpair_shmem_4tpt.cu"
