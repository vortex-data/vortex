// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::Canonical;
use crate::arrays::{BoolArray, PrimitiveArray};
use crate::pipeline::bits::{BitVector, BitView, BitViewMut};
use crate::pipeline::operators::Operator;
use crate::pipeline::query::Pipeline;
use crate::pipeline::types::Element;
use crate::pipeline::vector::Vector;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelExt, PIPELINE_STEP_COUNT};
use crate::validity::Validity;

pub fn export_canonical_pipeline(
    dtype: &DType,
    len: usize,
    pipeline: &mut dyn Kernel,
    mask: &Mask,
) -> VortexResult<Canonical> {
    match dtype {
        DType::Bool(Nullability::NonNullable) => {
            export_bool_nonnull_masked(mask, pipeline).map(Canonical::Bool)
        }
        DType::Primitive(ptype, Nullability::NonNullable) => {
            if mask.all_true() {
                match_each_native_ptype!(ptype, |T| {
                    export_primitive_nonnull::<T>(len, pipeline).map(Canonical::Primitive)
                })
            } else {
                match_each_native_ptype!(ptype, |T| {
                    export_primitive_nonnull_masked::<T>(mask, pipeline).map(Canonical::Primitive)
                })
            }
        }
        _ => vortex_bail!("Expected a primitive array, got: {}", dtype),
    }
}

pub fn export_canonical_pipeline_expr_offset(
    dtype: &DType,
    offset: usize,
    len: usize,
    expression: &dyn Operator,
    mask: &Mask,
) -> VortexResult<Canonical> {
    let mut pipeline = Pipeline::new(expression)?;
    pipeline.seek(offset)?;
    export_canonical_pipeline(dtype, len, &mut pipeline, mask)
}

pub fn export_canonical_pipeline_expr(
    dtype: &DType,
    len: usize,
    expression: &dyn Operator,
    mask: &Mask,
) -> VortexResult<Canonical> {
    let mut pipeline = Pipeline::new(expression)?;
    export_canonical_pipeline(dtype, len, &mut pipeline, mask)
}

fn export_primitive_nonnull<T: Element + NativePType>(
    len: usize,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let capacity = len.next_multiple_of(PIPELINE_STEP_COUNT) + PIPELINE_STEP_COUNT;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mut remaining = len;
    while remaining >= PIPELINE_STEP_COUNT {
        let mut elements_view = ViewMut::new(
            &mut elements[len - remaining..][..PIPELINE_STEP_COUNT],
            None,
        );
        pipeline.step_now(&(), BitView::all_true(), &mut elements_view)?;
        remaining -= PIPELINE_STEP_COUNT;
    }

    if remaining > 0 {
        println!("remaining: {}", remaining);
        let mut elements_view = ViewMut::new(
            &mut elements[len - remaining..][..PIPELINE_STEP_COUNT],
            None,
        );
        let mask = BitVector::true_until(remaining);
        println!("mask: {:?}", mask);
        pipeline.step_now(&(), mask.as_view(), &mut elements_view)?;
    }

    println!("elements: {:?}", elements);
    unsafe { elements.set_len(len) };
    println!("elements: {:?}", elements);

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

fn export_primitive_nonnull_masked<T: Element + NativePType>(
    mask: &Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let len = mask.len();
    let capacity = mask.true_count().next_multiple_of(PIPELINE_STEP_COUNT) + PIPELINE_STEP_COUNT;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mask_buffer = mask.to_boolean_buffer();
    let bit_chunks = mask_buffer.bit_chunks();
    let mut bit_chunks_iter = bit_chunks.iter_padded();

    let mut mask = [0u64; PIPELINE_STEP_COUNT / 64];
    let mut mask_view = BitViewMut::new(&mut mask);

    let mut offset = 0;
    let mut remaining = len;
    while remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[offset..][..PIPELINE_STEP_COUNT], None);

        mask_view.clear();
        mask_view.fill_with_words(&mut bit_chunks_iter);

        pipeline.step_now(&(), mask_view.as_view(), &mut elements_view)?;
        offset += mask_view.true_count();

        remaining = remaining.saturating_sub(PIPELINE_STEP_COUNT);
    }

    println!("elements: {:?}", elements);
    unsafe { elements.set_len(offset) };
    println!("elements: {:?}", elements);

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

fn export_bool_nonnull_masked(mask: &Mask, pipeline: &mut dyn Kernel) -> VortexResult<BoolArray> {
    let len = mask.len();
    let true_count = mask.true_count();

    let mut elements_buffer = Vector::new::<bool>();
    let mut elements_buffer_mut = elements_buffer.as_view_mut();

    let mask_buffer = mask.to_boolean_buffer();
    let bit_chunks = mask_buffer.bit_chunks();
    let mut bit_chunks_iter = bit_chunks.iter_padded();

    let mut mask = [0u64; PIPELINE_STEP_COUNT / 64];
    let mut mask_view = BitViewMut::new(&mut mask);

    // Fast path: collect all bools first, then use collect_bool for optimal packing
    let mut all_bools: Vec<bool> = Vec::with_capacity(true_count);
    let mut remaining = len;

    while remaining > 0 {
        mask_view.clear();
        mask_view.fill_with_words(&mut bit_chunks_iter);

        // Handle partial iteration on the last chunk
        let current_len = remaining.min(PIPELINE_STEP_COUNT);
        if current_len < PIPELINE_STEP_COUNT {
            mask_view.intersect_prefix(current_len);
        }

        pipeline.step_now(&(), mask_view.as_view(), &mut elements_buffer_mut)?;

        // Collect bools efficiently with unsafe for better performance
        let bool_slice = elements_buffer_mut.as_slice::<bool>();
        let count = mask_view.true_count();

        // Unsafe version to avoid bounds checking in hot path
        let old_len = all_bools.len();
        unsafe {
            all_bools.set_len(old_len + count);
            std::ptr::copy_nonoverlapping(
                bool_slice.as_ptr(),
                all_bools.as_mut_ptr().add(old_len),
                count,
            );
        }

        remaining = remaining.saturating_sub(PIPELINE_STEP_COUNT);
    }

    // Use collect_bool for optimal bit packing - avoid closure overhead
    let values = arrow_buffer::BooleanBuffer::collect_bool(all_bools.len(), |idx| unsafe {
        *all_bools.get_unchecked(idx)
    });

    Ok(BoolArray::new(values, Validity::NonNullable))
}
