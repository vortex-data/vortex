// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaStream;
use vortex_alp::ALPVTable;
use vortex_array::{Array, ArrayRef};
use vortex_fastlanes::{BitPackedVTable, FoRVTable};

use crate::jit::arrays::{alp, bitpack, for_};
use crate::jit::{GPUPipelineJIT, StepIdAllocator};

pub fn new_jit_array(
    a: &ArrayRef,
    stream: &Arc<CudaStream>,
    output_array: String,
) -> Box<dyn GPUPipelineJIT> {
    handle_array(a, stream, &mut StepIdAllocator::default(), output_array)
}

pub fn handle_array(
    a: &ArrayRef,
    stream: &Arc<CudaStream>,
    allocator: &mut StepIdAllocator,
    output_array: String,
) -> Box<dyn GPUPipelineJIT> {
    if let Some(alp) = a.as_opt::<ALPVTable>() {
        return alp::new_jit(alp, stream, allocator, output_array);
    }
    if let Some(bp) = a.as_opt::<BitPackedVTable>() {
        return bitpack::new_jit(bp, stream, allocator, output_array);
    };

    if let Some(for_) = a.as_opt::<FoRVTable>() {
        return for_::new_jit(for_, stream, allocator, output_array);
    }

    todo!("unimplemented jit for {}", a.encoding_id())
}
