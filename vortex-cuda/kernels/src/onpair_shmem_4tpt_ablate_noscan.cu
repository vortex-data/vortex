// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Ablation proxy: full 4tpt/b128o12 decode minus the scan stage (timing only).
#define ABLATE_SCAN
#define ONPAIR_ABLATE_NAME onpair_shmem_4tpt_ablate_noscan
#include "onpair_shmem_4tpt_ablate.cu"
