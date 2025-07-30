// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::pipeline::{Pipeline, SupportsPipeline};
use crate::vector::view::View;
use fastlanes::BitPacking;
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;
use vortex_fastlanes::BitPackedArray;
use vortex_mask::Mask;

impl SupportsPipeline for BitPackedArray {
    fn pipeline(&self) -> Box<dyn Pipeline> {
        // TODO(ngates): support 1024 bit offsets?
        assert_eq!(
            self.offset(),
            0,
            "BitPackedArray does not support offset exports"
        );

        match_each_unsigned_integer_ptype!(self.ptype(), |T| {
            // Create a BitPackedPipeline for the specific type T
            Box::new(BitPackedPipeline::<T>::new(
                Buffer::<T>::from_byte_buffer(self.packed().clone()),
                self.bit_width() as usize,
            ))
        })
    }
}

/// A pipeline for exporting BitPacked arrays into a vector stream.
struct BitPackedPipeline<T> {
    packed: Buffer<T>,
    width: usize,
    packed_offset: usize,
    packed_stride: usize,
}

impl<T: NativePType + BitPacking> BitPackedPipeline<T> {
    fn new(packed: Buffer<T>, width: usize) -> Self {
        let packed_stride = 1024 * width / T::PTYPE.bit_width();
        Self {
            packed,
            width,
            packed_offset: 0,
            packed_stride,
        }
    }
}

impl<T: NativePType + BitPacking> Pipeline for BitPackedPipeline<T> {
    fn next<'v>(&mut self, mask: &Mask, out: &'v mut View<'v>) -> VortexResult<()> {
        debug_assert_eq!(out.capacity(), 2048);
        match mask {
            Mask::AllTrue(_) => {
                let mut view = out.as_primitive::<T>();
                // FIXME(ngates): allow larger than necessary slices, instead of exact size
                unsafe {
                    BitPacking::unchecked_unpack(
                        self.width,
                        &self.packed.as_slice()[self.packed_offset..][..self.packed_stride],
                        &mut view.as_mut()[0..2048],
                    )
                }
                Ok(())
            }
            Mask::AllFalse(_) => {
                todo!()
            }
            Mask::Values(_) => {
                todo!()
            }
        }
    }
}
