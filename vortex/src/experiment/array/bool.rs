// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::array::Array;
use crate::experiment::encodings::BindContext;
use crate::experiment::mask::BitVector;
use crate::experiment::view_mut::ViewMut;
use arrow_buffer::BooleanBuffer;
use bitvec::order::Msb0;
use bitvec::vec::BitVec;
use std::task::Poll;
use vortex_array::arrays::BoolArray;
use vortex_array::validity::Validity;
use vortex_error::{VortexResult, vortex_panic};

/// Utility for exporting an encoding into a canonical boolean array.
pub(super) fn export_bool(array: &Array) -> VortexResult<BoolArray> {
    // Create a pipeline for the array.
    let mut pipeline = array.encoding.bind(&BindContext {
        len: array.len,
        dtype: &array.dtype,
        stats: Some(&array.stats_set),
    })?;

    // Take the array length and round it up to the next multiple of N.
    let capacity = array.len().next_multiple_of(N);

    // Create the output bit vector.
    let mut bits = BitVec::<u64, Msb0>::with_capacity(capacity);
    unsafe { bits.set_len(capacity) };

    // Optionally create a validity vector if the array has a validity mask.
    let mut validity = array.dtype.is_nullable().then(|| {
        let mut v = BitVec::<u64, Msb0>::with_capacity(capacity);
        unsafe { v.set_len(capacity) };
        v
    });

    let bits_iter = unsafe { bits.iter_vector_chunks() };

    // FIXME(ngates): should we set the selection mask for the final chunk?
    if let Some(validity) = validity.as_mut() {
        let validity_iter = unsafe { validity.iter_vector_chunks() };

        for (e, v) in bits_iter.zip(validity_iter) {
            let mut view = ViewMut::new_bool(e, Some(v));
            match pipeline.step(&(), BitVector::full(), &mut view) {
                Poll::Ready(result) => result?,
                Poll::Pending => {
                    vortex_panic!("Array pipelines cannot yield pending");
                }
            }
        }
    } else {
        for e in bits_iter {
            let mut view = ViewMut::new_bool(e, None);
            match pipeline.step(&(), BitMask::All, BitMask::All, &mut view) {
                Poll::Ready(result) => result?,
                Poll::Pending => {
                    vortex_panic!("Array pipelines cannot yield pending");
                }
            }
        }
    }

    // Set the length of the values and validity buffers to the actual length
    unsafe { bits.set_len(array.len) };
    if let Some(validity) = validity.as_mut() {
        unsafe { validity.set_len(array.len) };
    }

    Ok(BoolArray::new(
        BooleanBuffer::from_iter(bits.into_iter()),
        validity
            .map(|v| Validity::from(BooleanBuffer::from_iter(v.into_iter())))
            .unwrap_or_else(|| Validity::NonNullable),
    ))
}
