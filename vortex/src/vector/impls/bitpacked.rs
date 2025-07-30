// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::pipeline::{Pipeline, SupportsPipeline};
use crate::vector::vector::Vector;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::unpack_iter::BitPacked;
use vortex_mask::Mask;

impl SupportsPipeline for BitPackedArray {
    fn pipeline(&self) -> Box<dyn Pipeline> {
        match_each_integer_ptype!(self.ptype(), |T| {
            // Create a BitPackedPipeline for the specific type T
            Box::new(BitPackedPipeline::<T> {})
        })
    }
}

/// A pipeline for exporting BitPacked arrays into a vector stream.
struct BitPackedPipeline<T: BitPacked> {}

impl<T: BitPacked> Pipeline for BitPackedPipeline<T> {
    fn next(&mut self, mask: &Mask, out: &mut Vector) -> VortexResult<()> {
        todo!()
    }
}
