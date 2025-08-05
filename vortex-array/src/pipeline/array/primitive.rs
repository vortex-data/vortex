// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveArray;
use crate::pipeline::N;
use crate::pipeline::array::Array;
use crate::pipeline::bits::{BitView, BooleanBufferChunksIter};
use crate::pipeline::encodings::BindContext;
use crate::pipeline::view::{Canonical, ViewMut};
use crate::validity::Validity;
use std::task::Poll;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;

/// Utility for exporting an encoding into a canonical boolean array.
pub(super) fn export_primitive(array: &Array, mask: &Mask) -> VortexResult<PrimitiveArray> {
    let ptype = array.dtype.as_ptype();
    match_each_native_ptype!(ptype, |T| { export_primitive_impl::<T>(array, mask) })
}

/// Export into  a primitive array using the given selection mask.
fn export_primitive_impl<T: Canonical<Element = T> + NativePType>(
    array: &Array,
    mask: &Mask,
) -> VortexResult<PrimitiveArray> {
    debug_assert!(mask.true_count() <= array.len);

    // Create a pipeline for the array.
    let mut pipeline = array.encoding.bind(&BindContext {
        len: array.len,
        dtype: &array.dtype,
        stats: Some(&array.stats_set),
    })?;

    // Take the array length and round it up to the next multiple of N.
    // We add an extra N to ensure we have enough space for the last chunk.
    let capacity = array.len().next_multiple_of(N) + N;

    // Create the output bit vector.
    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };
    let elements_slice = elements.as_mut_slice();

    // Iterate the given mask in chunks of N.
    // TODO(ngates): deal with AllTrue / AllFalse masks?
    let mut offset = 0;
    let boolean_buffer = mask.to_boolean_buffer();
    let chunks = BooleanBufferChunksIter::new(&boolean_buffer);
    for chunk in chunks {
        let mask_view = BitView::new(&chunk);
        let mut view = ViewMut::new::<T>(&mut elements_slice[offset..][..N], None);
        match pipeline.step(&(), mask_view, &mut view) {
            Poll::Ready(result) => result?,
            Poll::Pending => {
                vortex_panic!("Array pipelines cannot yield pending");
            }
        }
        view.flatten::<T>();
        offset += view.len();
    }

    // TODO(ngates): deal with chunks remainder

    // Set the length of the values and validity buffers to the actual length
    unsafe { elements.set_len(offset) };
    // unsafe { validity.set_len(offset) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        if array.dtype().is_nullable() {
            // Validity::from(BooleanBuffer::from_iter(validity.into_iter()))
            todo!()
        } else {
            Validity::NonNullable
        },
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::IntoArray;
    use crate::pipeline::buffers::BufferHandle;
    use crate::pipeline::encodings::bitpacked::BitPackedEncoding;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;

    #[test]
    fn test_bitpacked() -> VortexResult<()> {
        let old_array = BitPackedArray::encode(&buffer![4u32; 100000].into_array(), 3)?;
        let buffer = BufferHandle::new(old_array.packed().clone());
        let encoding = BitPackedEncoding::new(old_array.bit_width() as usize, buffer);

        let array = Array {
            len: old_array.len(),
            dtype: old_array.dtype().clone(),
            stats_set: old_array.statistics().to_owned(),
            encoding: Box::new(encoding),
        };

        let exported = export_primitive(&array, &Mask::new_true(array.len))?;
        assert_eq!(exported.len(), 100000);
        assert_eq!(exported.as_slice::<u32>().len(), 100000);
        assert_eq!(exported.as_slice::<u32>(), &[4; 100000]);
        Ok(())
    }
}
