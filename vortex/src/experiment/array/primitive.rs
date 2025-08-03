// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::array::Array;
use crate::experiment::encodings::BindContext;
use crate::experiment::mask::{BitMask, BitVectorMaskExt};
use crate::experiment::vector::{N, Vector};
use arrow_array::BooleanArray;
use arrow_buffer::BooleanBuffer;
use bitvec::order::Msb0;
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
pub(super) fn export_primitive(array: &Array) -> VortexResult<PrimitiveArray> {
    let ptype = array.dtype.as_ptype();
    match_each_native_ptype!(ptype, |T| { export_primitive_impl::<T>(array) })
}

fn export_primitive_impl<T: NativePType>(array: &Array) -> VortexResult<PrimitiveArray> {
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

    // Optionally create a validity vector if the array has a validity mask.
    let mut validity = array.dtype.is_nullable().then(|| {
        let mut v = BitVec::<u64, Msb0>::with_capacity(capacity);
        unsafe { v.set_len(capacity) };
        v
    });

    let elements_iter = elements.chunks_exact_mut(N);

    // FIXME(ngates): should we set the selection mask for the final chunk?
    if let Some(validity) = validity.as_mut() {
        let validity_iter = unsafe { validity.iter_vector_chunks() };

        for (e, v) in elements_iter.zip(validity_iter) {
            let mut view = Vector::new_primitive::<T>(e, Some(v));
            match pipeline.step(&array.buffers, &BitMask::All, &BitMask::All, &mut view) {
                Poll::Ready(result) => result?,
                Poll::Pending => {
                    vortex_panic!("Array pipelines cannot yield pending");
                }
            }
        }
    } else {
        for e in elements_iter {
            let mut view = Vector::new_primitive::<T>(e, None);
            match pipeline.step(&array.buffers, &BitMask::All, &BitMask::All, &mut view) {
                Poll::Ready(result) => result?,
                Poll::Pending => {
                    vortex_panic!("Array pipelines cannot yield pending");
                }
            }
        }
    }

    // Set the length of the values and validity buffers to the actual length
    unsafe { elements.set_len(array.len) };
    if let Some(validity) = validity.as_mut() {
        unsafe { validity.set_len(array.len) };
    }

    Ok(PrimitiveArray::new(
        elements.freeze(),
        validity
            .map(|v| Validity::from(BooleanBuffer::from_iter(v.into_iter())))
            .unwrap_or_else(|| Validity::NonNullable),
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use super::*;
    use crate::IntoArray;
    use crate::buffer::buffer;
    use crate::experiment::buffers::BufferId;
    use crate::experiment::encodings::bitpacked::BitPackedEncoding;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;
    use vortex_utils::aliases::hash_map::HashMap;

    #[test]
    fn test_bitpacked() -> VortexResult<()> {
        let mut buffers = HashMap::new();

        let old_array = BitPackedArray::encode(&buffer![4u32; 100000].into_array(), 3)?;
        let buffer_id = BufferId::new();
        buffers.insert(buffer_id, old_array.packed().clone());
        let encoding = BitPackedEncoding::new(old_array.bit_width() as usize, buffer_id);

        let array = Array {
            len: old_array.len(),
            dtype: old_array.dtype().clone(),
            stats_set: old_array.statistics().to_owned(),
            encoding: Box::new(encoding),
            buffers,
        };

        let exported = export_primitive(&array)?;
        assert_eq!(exported.len(), 100000);
        assert_eq!(exported.as_slice::<u32>().len(), 100000);
        assert_eq!(exported.as_slice::<u32>(), &[4; 100000]);
        Ok(())
    }
}
