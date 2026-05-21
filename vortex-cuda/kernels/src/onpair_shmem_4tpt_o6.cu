// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B: 256-thread blocks, force 6 blocks/SM -> ptxas cuts regs (~42) -> 75% occ.
#define WARPS_PER_BLOCK_MAX 8u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 6)
#define onpair_shmem_4tpt onpair_shmem_4tpt_o6
#include "onpair_shmem_4tpt.cu"
