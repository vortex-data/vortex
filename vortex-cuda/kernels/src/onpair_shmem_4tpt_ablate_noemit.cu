// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Ablation proxy: full 4tpt/b128o12 decode minus the emit stage (timing only).
#define ABLATE_EMIT
#define ONPAIR_ABLATE_NAME onpair_shmem_4tpt_ablate_noemit
#include "onpair_shmem_4tpt_ablate.cu"
