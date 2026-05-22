// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// split4read (4B reads from the 16 KB dict_s4) at the B200 granularity lever
// (128-thread blocks, 75% occ). split4read was only ever measured at 512-thread
// — the same trap that hid split8read's +26% before it was retested at 128. On
// the shortest-token columns (fineweb/wikipedia bits12, mean ~4.2 B, ~77% of
// tokens <= 4 B) the 4 B read further halves the request width vs split8read and
// the dict array is 2x smaller (16 KB vs 32 KB), so this is the natural next
// step of the request-narrowing + granularity mechanism that wins on Blackwell.
#define WARPS_PER_BLOCK_MAX 4u
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 12)
#define onpair_shmem_4tpt_split4read onpair_shmem_4tpt_split4read_b128o12
#include "onpair_shmem_4tpt_split4read.cu"
