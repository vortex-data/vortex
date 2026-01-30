// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::patches::Patches;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;

#[derive(Debug)]
pub struct PatchesExecutor;

pub(crate) async fn execute_patches<ValuesT: NativePType, IndicesT: NativePType>(
    patches: Patches,
    array: Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    let len = array.len();
    let values = array.into_primitive();

    todo!()

    // Based on the typed indices and values instead...we can apply those
    // launch_cuda_kernel!(
    //     execution_ctx: ctx,
    //     module: "patches",
    //     ptypes: &[ValuesT::PTYPE, IndicesT::PTYPE],
    //     launch_args: [],
    //     event_recording: CU_EVENT_DISABLE_TIMING,
    //     array_len:
    // )
}
