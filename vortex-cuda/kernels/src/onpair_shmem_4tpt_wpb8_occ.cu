// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#define WARPS_PER_BLOCK_MAX 8u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#define onpair_shmem_4tpt onpair_shmem_4tpt_wpb8_occ
#include "onpair_shmem_4tpt.cu"
