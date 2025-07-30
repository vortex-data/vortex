// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::pipeline::{Pipeline, SupportsPipeline};
use crate::vector::{N, Vector};
use bitvec::access::BitSafeU64;
use bitvec::order::Msb0;
use bitvec::vec::BitVec;
use std::ops::Deref;
use vortex_array::Array;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::Mask::Values;

/// A trait for exporting arrays into canonical primitive form.
struct PrimitiveExport<T: NativePType> {
    len: usize,
    values: BufferMut<T>,
    validity: BitVec<BitSafeU64, Msb0>,
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
        // We round up to the next multiple of N to ensure that we can export the entire array
        // directly in chunks of N elements, we slice back down to the actual length later.
        let capacity = len.next_multiple_of(N);

        let mut values = BufferMut::with_capacity(capacity);
        unsafe { values.set_len(capacity) };

        let mut validity = BitVec::with_capacity(capacity);
        unsafe { validity.set_len(capacity) };

        Self {
            len,
            values,
            validity,
            pipeline,
        }
    }

    pub fn collect(mut self) -> VortexResult<PrimitiveArray> {
        // Iterate over the values in chunks of 2048
        let elements = self.values.as_mut_slice().chunks_mut(N);
        let validity = self.validity.chunks_exact_mut(N);
        for (e_chunk, v_chunk) in elements.zip(validity) {
            let mut view = Vector::new_primitive::<T>(e_chunk, v_chunk);
            self.pipeline.next(&Mask::AllTrue(N), &mut view)?;
        }

        // Set the length of the values and validity buffers to the actual length
        unsafe { self.values.set_len(self.len) };
        unsafe { self.validity.set_len(self.len) };

        // FIXME(ngates): we should better support BitVec, or a vortex-buffer equivalent.
        //  For now, we just ignore it.
        let validity = Validity::AllValid;

        Ok(PrimitiveArray::new(self.values.freeze(), validity))
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
        let exported = export_primitive(array)?;
        assert_eq!(exported.as_slice::<u32>().len(), 100000);
        assert_eq!(exported.as_slice::<u32>(), &[4; 100000]);
        Ok(())
    }
}
