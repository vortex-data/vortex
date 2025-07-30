// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::pipeline::{Pipeline, SupportsPipeline};
use crate::vector::view::View;
use bitvec::vec::BitVec;
use std::ops::Deref;
use vortex_array::Array;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// A trait for exporting arrays into canonical primitive form.
struct PrimitiveExport<T: NativePType> {
    offset: usize,
    values: BufferMut<T>,
    validity: BitVec<u64>,
    pipeline: Box<dyn Pipeline>,
}

pub fn export_primitive<P: Deref<Target = dyn Array> + SupportsPipeline>(
    array: P,
) -> VortexResult<PrimitiveArray> {
    match_each_native_ptype!(array.dtype().as_ptype(), |T| {
        PrimitiveExport::<T>::new(array.pipeline(), array.len()).collect()
    })
}

impl<T: NativePType> PrimitiveExport<T> {
    pub fn new(pipeline: Box<dyn Pipeline>, len: usize) -> Self {
        let mut values = BufferMut::with_capacity(len);
        unsafe { values.set_len(len) };

        let mut validity = BitVec::<u64>::with_capacity(len);
        unsafe { validity.set_len(len) };

        Self {
            offset: 0,
            values,
            validity,
            pipeline,
        }
    }

    pub fn collect(mut self) -> VortexResult<PrimitiveArray> {
        // Iterate over the values in chunks of 2048
        for chunk in self.values.as_mut_slice().chunks_mut(2048) {
            let len = chunk.len();
            let mut view = View::new_primitive::<T>(chunk);
            self.pipeline.next(&Mask::AllTrue(len), &mut view)?;
        }
        Ok(PrimitiveArray::new(
            self.values.freeze(),
            Validity::AllValid,
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::IntoArray;
    use crate::buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;

    #[test]
    fn test_bitpacked() -> VortexResult<()> {
        let array = BitPackedArray::encode(&buffer![4u32; 100000].into_array(), 3)?;
        let exporter = export_primitive(array);

        Ok(())
    }
}
