// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

use fastlanes::BitPacking;
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Element, Kernel, KernelContext, N};
use vortex_buffer::Buffer;
use vortex_dtype::PhysicalPType;
use vortex_error::VortexResult;

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
#[derive(Clone)]
pub struct BitPackedKernel<T: PhysicalPType<Physical: BitPacking>> {
    width: usize,
    packed_stride: usize,
    buffer: Buffer<<T as PhysicalPType>::Physical>,
}

impl<T: PhysicalPType<Physical: BitPacking>> BitPackedKernel<T> {
    pub fn new(
        width: usize,
        packed_stride: usize,
        buffer: Buffer<<T as PhysicalPType>::Physical>,
    ) -> Self {
        Self {
            width,
            packed_stride,
            buffer,
        }
    }
}

impl<T> Kernel for BitPackedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn step(
        &self,
        _ctx: &KernelContext,
        chunk_idx: usize,
        _selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        assert_eq!(
            N % 1024,
            0,
            "BitPackedKernel assumes N is a multiple of 1024"
        );

        // We re-interpret the output view as the unsigned bitpacked type.
        out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = out.as_array_mut::<<T as PhysicalPType>::Physical>();

        let packed_offset = ((chunk_idx * N) / 1024) * self.packed_stride;
        let packed = &self.buffer.as_slice()[packed_offset..];

        // We compute the number of FastLanes vectors for this chunk.
        let nvecs = min(N / 1024, packed.len() / self.packed_stride);

        for i in 0..nvecs {
            // TODO(ngates): decide if the selection mask is sufficiently sparse to warrant
            //  unpacking only the selected elements.
            unsafe {
                BitPacking::unchecked_unpack(
                    self.width,
                    &packed[(i * self.packed_stride)..][..self.packed_stride],
                    &mut elements[(i * 1024)..],
                );
            }
        }

        out.reinterpret_as::<T>();

        Ok(())
    }
}
