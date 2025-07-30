// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::pipeline::{Pipeline, SupportsPipeline};
use vortex_array::Array;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_mask::Mask;
use crate::vector::vector::Vector;

/// A trait for exporting arrays into canonical primitive form.
struct PrimitiveExport<T: NativePType> {
    offset: usize,
    values: BufferMut<T>,
    // TODO(ngates): we really need BitBufferMut
    validity: BufferMut<u64>,
    pipeline: Box<dyn Pipeline>,
}

impl<T: NativePType> PrimitiveExport<T> {
    pub fn new(pipeline: Box<dyn Pipeline>, len: usize) -> Self {
        let mut values = BufferMut::with_capacity(len);
        unsafe { values.set_len(len) };

        let mut validity = BufferMut::with_capacity((len + 63) / 64);
        unsafe { validity.set_len((len + 63) / 64) };

        Self {
            offset: 0,
            values,
            validity,
            pipeline,
        }
    }

    pub fn collect(self) -> PrimitiveArray {
        // Iterate over the values in chunks of 2048
        let mut vector = Vector {}

        self.pipeline.next(&Mask::AllTrue(2046))
    }

    pub fn collect_array<P: Array + SupportsPipeline>(array: P) -> PrimitiveArray {
        let pipeline = array.pipeline();
        match_each_native_ptype!(array.dtype().as_ptype(), |T| {
            PrimitiveExport::<T>::new(pipeline, array.len()).collect()
        })
    }
}

#[cfg(test)]
mod test {
    use crate::vector::export::PrimitiveExport;
    use crate::vector::pipeline::SupportsPipeline;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;

    #[test]
    fn test_bitpacked() -> VortexResult<()> {
        let array = BitPackedArray::encode(buffer![4; 100000].into_array(), 3)?;

        let exporter = PrimitiveExport::collect_array(array);

        Ok(())
    }
}
