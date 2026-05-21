// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B+: 64-thread blocks + force 24 blocks/SM -> ~42 regs -> 75% occ.
#define WARPS_PER_BLOCK_MAX 2u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(64, 24)
#define onpair_shmem_4tpt onpair_shmem_4tpt_b64o24
#include "onpair_shmem_4tpt.cu"
