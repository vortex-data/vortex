// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveArray;
use crate::pipeline::bits::{BitVector, BitView, BitViewMut};
use crate::pipeline::types::Element;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{N, Pipeline, PipelineExt};
use crate::validity::Validity;
use crate::{Array, Canonical};
use bitvec::order::Msb0;
use bitvec::vec::BitVec;
use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

pub fn export_canonical(array: &dyn Array, mask: &Mask) -> VortexResult<Canonical> {
    if mask.all_false() {
        return Ok(Canonical::empty(array.dtype()));
    }

    let pipeline = array.to_pipeline()?;
    match array.dtype() {
        DType::Primitive(ptype, Nullability::NonNullable) => {
            if mask.all_true() {
                match_each_native_ptype!(ptype, |T| {
                    export_primitive_nonnull::<T>(array.len(), pipeline).map(Canonical::Primitive)
                })
            } else {
                match_each_native_ptype!(ptype, |T| {
                    export_primitive_nonnull_masked::<T>(mask, pipeline).map(Canonical::Primitive)
                })
            }
        }
        _ => vortex_bail!("Expected a primitive array, got: {}", array.dtype()),
    }
}

fn export_primitive_nonnull<T: Element + NativePType>(
    len: usize,
    mut pipeline: Box<dyn Pipeline>,
) -> VortexResult<PrimitiveArray> {
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mut remaining = len;
    while remaining >= N {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        pipeline.step_now(&(), BitView::all_true(), &mut elements_view)?;
        remaining -= N;
    }

    if remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let mask = BitVector::true_until(remaining);
        pipeline.step_now(&(), mask.as_view(), &mut elements_view)?;
    }

    unsafe { elements.set_len(len) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

fn export_primitive_nonnull_masked<T: Element + NativePType>(
    mask: &Mask,
    mut pipeline: Box<dyn Pipeline>,
) -> VortexResult<PrimitiveArray> {
    let len = mask.len();
    let capacity = mask.true_count().next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mask_buffer = mask.to_boolean_buffer();
    let bit_chunks = mask_buffer.bit_chunks();
    let mut bit_chunks_iter = bit_chunks.iter_padded();

    let mut mask = [0u64; N / 64];
    let mut mask_view = BitViewMut::new(&mut mask);

    let mut offset = 0;
    let mut remaining = len;
    while remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[offset..][..N], None);

        mask_view.clear();
        mask_view.fill_with_words(&mut bit_chunks_iter);

        pipeline.step_now(&(), mask_view.as_view(), &mut elements_view)?;

        // Flatten the elements in place.
        elements_view.flatten::<T>();
        offset += elements_view.len();

        remaining = remaining.saturating_sub(N);
    }

    unsafe { elements.set_len(offset) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}
