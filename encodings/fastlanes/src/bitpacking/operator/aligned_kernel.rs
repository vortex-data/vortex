// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
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
    packed_offset: usize,
}

impl<T: PhysicalPType<Physical: BitPacking>> BitPackedKernel<T> {
    pub fn new(
        width: usize,
        packed_stride: usize,
        buffer: Buffer<<T as PhysicalPType>::Physical>,
        packed_offset: usize,
    ) -> Self {
        Self {
            width,
            packed_stride,
            buffer,
            packed_offset,
        }
    }
}

impl<T> Kernel for BitPackedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn step(&mut self, _ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        // We re-interpret the output view as the unsigned bitpacked type.
        out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = out.as_slice_mut::<<T as PhysicalPType>::Physical>();
        let packed = &self.buffer.as_slice()[self.packed_offset..];

        // We compute the number of FastLanes vectors that we have remaining.
        let nvecs = (N / 1024).min(packed.len() / self.packed_stride);

        // We short-circuit full unpacking logic if the mask is sufficiently sparse.
        for i in 0..nvecs {
            unsafe {
                BitPacking::unchecked_unpack(
                    self.width,
                    &packed[(i * self.packed_stride)..][..self.packed_stride],
                    &mut elements[(i * 1024)..],
                );
            }
        }
        self.packed_offset += nvecs * self.packed_stride;

        out.reinterpret_as::<T>();

        Ok(())
    }
}
