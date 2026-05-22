// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// 8tpt at the B200 granularity lever (128-thread blocks, 4 warps). 50% occ via
// `__launch_bounds__(128, 8)` → 56 reg budget, giving 8tpt's heavier register
// footprint headroom before spilling (75%/40-reg would spill). Tests whether
// more per-thread amortisation/MLP helps once the block size is already optimal.
#define WARPS_PER_BLOCK_MAX 4u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 8)
#define onpair_shmem_8tpt onpair_shmem_8tpt_b128
#include "onpair_shmem_8tpt.cu"
