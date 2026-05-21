// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B"+: split8read (8B reads, request-reducing) at finest granularity + 75% occ.
#define WARPS_PER_BLOCK_MAX 4u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 12)
#define onpair_shmem_4tpt_split8read onpair_shmem_4tpt_split8read_b128o12
#include "onpair_shmem_4tpt_split8read.cu"
