// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaStream;
use itertools::all;
use vortex_alp::{ALPFloat, ALPVTable, match_each_alp_float_ptype};
use vortex_array::{Array, ArrayRef};
use vortex_buffer::Buffer;
use vortex_dtype::match_each_native_ptype;
use vortex_error::{VortexUnwrap, vortex_err};
use vortex_fastlanes::{BitPackedVTable, FoRVTable};

use crate::jit::arrays::{alp, bitpack, for_};
use crate::jit::{GPUPipelineJIT, ScalarGPUPipelineJITNode, StepIdAllocator};

pub fn handle_array(
    a: &ArrayRef,
    stream: &Arc<CudaStream>,
    allocator: &mut StepIdAllocator,
) -> Box<dyn GPUPipelineJIT> {
    if let Some(alp) = a.as_opt::<ALPVTable>() {
        return alp::new_jit(alp, stream, allocator);
    }
    if let Some(bp) = a.as_opt::<BitPackedVTable>() {
        return bitpack::new_jit(bp, stream, allocator);
    };

    if let Some(for_) = a.as_opt::<FoRVTable>() {
        return for_::new_jit(for_, stream, allocator);
    }

    todo!("unimplemented jit for {}", a.encoding_id())
}
