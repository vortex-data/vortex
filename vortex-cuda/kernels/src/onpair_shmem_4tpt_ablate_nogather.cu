// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Ablation proxy: full 4tpt/b128o12 decode minus the gather stage (timing only).
#define ABLATE_GATHER
#define ONPAIR_ABLATE_NAME onpair_shmem_4tpt_ablate_nogather
#include "onpair_shmem_4tpt_ablate.cu"
