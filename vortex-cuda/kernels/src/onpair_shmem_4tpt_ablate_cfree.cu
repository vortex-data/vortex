// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Ablation proxy: emit with conflict-free addressing (timing only). If much
// faster than full, the emit is bank-conflict bound (swizzle fixable); if ~same
// as full, it is store-count/throughput bound (needs shuffle/fewer stores).
#define ABLATE_EMIT_CFREE
#define ONPAIR_ABLATE_NAME onpair_shmem_4tpt_ablate_cfree
#include "onpair_shmem_4tpt_ablate.cu"
