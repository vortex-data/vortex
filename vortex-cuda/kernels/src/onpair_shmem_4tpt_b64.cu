// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B+: finest granularity (64-thread/2-warp blocks), 16 blocks/SM, 50% occ.
#define WARPS_PER_BLOCK_MAX 2u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(64, 16)
#define onpair_shmem_4tpt onpair_shmem_4tpt_b64
#include "onpair_shmem_4tpt.cu"
