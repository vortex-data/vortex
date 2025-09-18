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

impl<T> BitPackedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    pub fn new(
        width: usize,
        packed_stride: usize,
        buffer: Buffer<<T as PhysicalPType>::Physical>,
        packed_offset: usize,
    ) -> Self {
        BitPackedKernel::<T> {
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
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (N / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &KernelContext,
        out: &mut ViewMut,
        physical_out: &mut ViewMut,
    ) -> VortexResult<()> {
        // We re-interpret the output view as the unsigned bitpacked type.
        physical_out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = physical_out.as_slice_mut::<<T as PhysicalPType>::Physical>();
        let packed = &self.buffer.as_slice()[self.packed_offset..];

        // We compute the number of FastLanes vectors that we have remaining.
        let nvecs = (N / 1024).min(packed.len() / self.packed_stride);

        // We short-circuit full unpacking logic if the mask is sufficiently sparse.
        if out.true_count() > 8 {
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

            // Set the selection to the given mask, which is a bit array of length N.
            physical_out.flatten::<<T as PhysicalPType>::Physical>(&out);
        } else {
            let mut offset = 0;
            out.iter_ones(|idx| {
                let chunk_idx = idx / 1024;
                let bit_idx = idx % 1024;
                // SAFETY: we verify the bounds of the vector during construction.
                unsafe {
                    *elements.get_unchecked_mut(offset) = BitPacking::unchecked_unpack_single(
                        self.width,
                        &packed[(chunk_idx * self.packed_stride)..][..self.packed_stride],
                        bit_idx,
                    );
                }
                offset += 1;
            });

            self.packed_offset += nvecs * self.packed_stride;
        }
        physical_out.reinterpret_as::<T>();

        Ok(())
    }
}
