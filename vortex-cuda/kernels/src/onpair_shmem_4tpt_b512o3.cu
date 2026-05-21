// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B: 512-thread blocks, force 3 blocks/SM -> ptxas cuts regs (~42) -> 75% occ.
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 3)
#define onpair_shmem_4tpt onpair_shmem_4tpt_b512o3
#include "onpair_shmem_4tpt.cu"
