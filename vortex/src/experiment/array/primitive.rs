// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::array::Array;
use crate::experiment::encodings::BindContext;
use crate::experiment::mask::{BitMask, BitMaskView, BitVector, BitVectorMaskExt};
use crate::experiment::vector::{N, Vector};
use arrow_array::BooleanArray;
use arrow_buffer::BooleanBuffer;
use bitvec::array::BitArray;
use bitvec::bitarr;
use bitvec::order::Msb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use std::task::Poll;
use vortex_array::arrays::{BoolArray, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_buffer::{BufferMut, ByteBufferMut};
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_err, vortex_panic};
use vortex_mask::Mask;
use vortex_mask::Mask::Values;

/// Utility for exporting an encoding into a canonical boolean array.
pub(super) fn export_primitive(
    array: &Array,
    mask: &BitSlice<u64, Msb0>,
) -> VortexResult<PrimitiveArray> {
    let ptype = array.dtype.as_ptype();
    match_each_native_ptype!(ptype, |T| { export_primitive_impl::<T>(array, mask) })
}

/// Export into  a primitive array using the given selection mask.
fn export_primitive_impl<T: NativePType>(
    array: &Array,
    mask: &BitSlice<u64, Msb0>,
) -> VortexResult<PrimitiveArray> {
    debug_assert!(mask.count_ones() <= array.len);

    // Create a pipeline for the array.
    let mut pipeline = array.encoding.bind(&BindContext {
        len: array.len,
        dtype: &array.dtype,
        stats: Some(&array.stats_set),
    })?;

    // Take the array length and round it up to the next multiple of N.
    let capacity = array.len().next_multiple_of(N);

    // Create the output bit vector.
    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };
    let elements_slice = elements.as_mut_slice();

    // Optionally create a validity vector if the array has a validity mask.
    let mut validity = BitVec::<u64, Msb0>::with_capacity(capacity);
    unsafe { validity.set_len(capacity) };
    let validity_slice = validity.as_mut_bitslice();

    // Iterate the given mask in chunks of N.
    let mut offset = 0;
    for m in mask.chunks_exact(N) {
        let m = BitVector::try_from(m).expect("Mask chunks should be valid BitVector");
        let mut view = Vector::new_primitive::<T>(&mut elements_slice[offset..][..N], None);
        match pipeline.step(&(), BitMask::Some(m), BitMask::All, &mut view) {
            Poll::Ready(result) => result?,
            Poll::Pending => {
                vortex_panic!("Array pipelines cannot yield pending");
            }
        }
        view.flatten();
        offset += view.len();
    }

    // Set the length of the values and validity buffers to the actual length
    unsafe { elements.set_len(offset) };
    unsafe { validity.set_len(offset) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        if array.dtype().is_nullable() {
            Validity::from(BooleanBuffer::from_iter(validity.into_iter()))
        } else {
            Validity::NonNullable
        },
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use super::*;
    use crate::IntoArray;
    use crate::buffer::buffer;
    use crate::experiment::buffers::{BufferId, ByteBufferHandle};
    use crate::experiment::encodings::bitpacked::BitPackedEncoding;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;
    use vortex_utils::aliases::hash_map::HashMap;

    #[test]
    fn test_bitpacked() -> VortexResult<()> {
        let old_array = BitPackedArray::encode(&buffer![4u32; 100000].into_array(), 3)?;
        let buffer = ByteBufferHandle::new(old_array.packed().clone());
        let encoding = BitPackedEncoding::new(old_array.bit_width() as usize, buffer);

        let array = Array {
            len: old_array.len(),
            dtype: old_array.dtype().clone(),
            stats_set: old_array.statistics().to_owned(),
            encoding: Box::new(encoding),
        };

        let mask = BitVec::repeat(true, array.len());
        let exported = export_primitive(&array, &mask)?;
        assert_eq!(exported.len(), 100000);
        assert_eq!(exported.as_slice::<u32>().len(), 100000);
        assert_eq!(exported.as_slice::<u32>(), &[4; 100000]);
        Ok(())
    }
}
