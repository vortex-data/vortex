// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Variable-width dict at the B200 granularity lever (128-thread blocks, 75% occ).
#define WARPS_PER_BLOCK_MAX 4u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 12)
#define onpair_shmem_4tpt_vwidth onpair_shmem_4tpt_vwidth_b128
#include "onpair_shmem_4tpt_vwidth.cu"
