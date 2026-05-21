// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Track B": split8read at finer granularity (256-thread blocks, 4 blocks/SM).
#define WARPS_PER_BLOCK_MAX 8u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#define onpair_shmem_4tpt_split8read onpair_shmem_4tpt_split8read_occ
#include "onpair_shmem_4tpt_split8read.cu"
