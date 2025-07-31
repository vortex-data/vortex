// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::pipeline::{Pipeline, SupportsPipeline};
use crate::experiment::vector::{Selection, Vector};
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
    fn next<'v>(&mut self, mask: &Mask, out: &'v mut Vector<'v>) -> VortexResult<()> {
        if mask.true_count() < 16 {
            // TODO(ngates): I think we found it was <= 8 elements where unpack_single is faster
            //  than unpacking the whole chunk... Given we do two chunks, that's ~16 elements?
        }

        // TODO(ngates): deal with our own nulls. We basically take the validity array and
        //  create a pipeline to export it into a BitVector, which we construct by re-wrapping
        //  the output vector's validity.

        // Otherwise, we unconditionally unpack two chunks of 1024 elements each into the
        // output vector, and simply return the mask we were given.
        let mut view = out.as_primitive::<T>();
        unsafe {
            BitPacking::unchecked_unpack(
                self.width,
                &self.packed.as_slice()[self.packed_offset..][..self.packed_stride],
                &mut view.as_mut()[0..1024],
            );
            BitPacking::unchecked_unpack(
                self.width,
                &self.packed.as_slice()[self.packed_offset + self.packed_stride..]
                    [..self.packed_stride],
                &mut view.as_mut()[1024..2048],
            );
        }

        self.packed_offset += 2 * self.packed_stride;

        // Set the selection to the given mask, which is a bit array of length N.
        out.set_selection(Selection::Mask(mask.clone()));

        Ok(())
    }
}
