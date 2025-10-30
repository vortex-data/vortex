// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::jit::arrays::{alp, bitpack, for_};
use crate::jit::{
    AlpEncodingTree, BitPackedEncodingTree, EncodingTreeRef, FoREncodingTree, GPUPipelineJIT,
    StepIdAllocator,
};

pub fn new_jit_array<'a>(
    a: &'a EncodingTreeRef,
    stream: &Arc<CudaStream>,
    output_array: String,
) -> Box<dyn GPUPipelineJIT + 'a> {
    handle_array(a, stream, &mut StepIdAllocator::default(), output_array)
}

pub fn handle_array<'a>(
    a: &'a EncodingTreeRef,
    stream: &Arc<CudaStream>,
    allocator: &mut StepIdAllocator,
    output_array: String,
) -> Box<dyn GPUPipelineJIT + 'a> {
    if let Some(alp) = a.as_any().downcast_ref::<AlpEncodingTree>() {
        return alp::new_jit(alp, stream, allocator, output_array);
    }
    if let Some(bp) = a.as_any().downcast_ref::<BitPackedEncodingTree>() {
        return bitpack::new_jit(bp, stream, allocator, output_array);
    };

    if let Some(for_) = a.as_any().downcast_ref::<FoREncodingTree>() {
        return for_::new_jit(for_, stream, allocator, output_array);
    }

    todo!("unimplemented jit for kernel ?")
}
