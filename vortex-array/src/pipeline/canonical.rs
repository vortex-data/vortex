// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::{BooleanBuffer, MutableBuffer};
use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::Canonical;
use crate::arrays::{BoolArray, PrimitiveArray};
use crate::pipeline::bits::{BitVector, BitView, BitViewMut};
use crate::pipeline::operators::Operator;
use crate::pipeline::query::QueryPlan;
use crate::pipeline::types::Element;
use crate::pipeline::vec::Vector;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext, N, N_WORDS};
use crate::validity::Validity;

/// Export canonical data from a pipeline kernel with the given mask.
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

/// Export canonical data from an operator expression with a starting offset and mask.
pub fn export_canonical_pipeline_expr_offset(
    dtype: &DType,
    offset: usize,
    len: usize,
    expression: &dyn Operator,
    mask: &Mask,
) -> VortexResult<Canonical> {
    let plan = QueryPlan::new(expression)?;
    let mut pipeline = plan.executable_plan()?;
    pipeline.seek(offset)?;
    export_canonical_pipeline(dtype, len, &mut pipeline, mask)
}

/// Export canonical data from an operator expression with the given mask.
pub fn export_canonical_pipeline_expr(
    dtype: &DType,
    len: usize,
    expression: &dyn Operator,
    mask: &Mask,
) -> VortexResult<Canonical> {
    let plan = QueryPlan::new(expression)?;
    let mut pipeline = plan.executable_plan()?;
    export_canonical_pipeline(dtype, len, &mut pipeline, mask)
}

fn export_primitive_nonnull<T: Element + NativePType>(
    len: usize,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mut remaining = len;
    while remaining >= N {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, BitView::all_true(), &mut elements_view)?;
        remaining -= N;
    }

    if remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let mask = BitVector::true_until(remaining);
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask.as_view(), &mut elements_view)?;
    }

    unsafe { elements.set_len(len) };

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
    let capacity = mask.true_count().next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mask_buffer = mask.to_boolean_buffer();
    let mut mask_iter = mask_buffer.bit_chunks().iter_padded();

    let mut mask = [0usize; N_WORDS];
    let mut mask_view = BitViewMut::new(&mut mask);

    let mut offset = 0;
    let mut remaining = len;
    while remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[offset..][..N], None);

        mask_view.clear();
        mask_view.fill_with_words(&mut mask_iter);

        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask_view.as_view(), &mut elements_view)?;
        offset += mask_view.true_count();

        remaining = remaining.saturating_sub(N);
    }

    unsafe { elements.set_len(offset) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

pub fn export_bool_nonnull_masked_coll(
    mask: &Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<BoolArray> {
    let len = mask.len();
    let true_count = mask.true_count();

    let mut elements_buffer = Vector::new::<bool>();
    let mut elements_buffer_mut = elements_buffer.as_view_mut();

    let mask_buffer = mask.to_boolean_buffer();
    let mut mask_iter = mask_buffer.bit_chunks().iter_padded();

    let mut mask = [0usize; N_WORDS];
    let mut mask_view = BitViewMut::new(&mut mask);

    // Fast path: collect all bools first, then use collect_bool for optimal packing
    let mut all_bools: Vec<bool> = Vec::with_capacity(true_count);
    let mut remaining = len;

    while remaining > 0 {
        mask_view.clear();
        mask_view.fill_with_words(&mut mask_iter);

        // Handle partial iteration on the last chunk
        let current_len = remaining.min(N);
        if current_len < N {
            mask_view.intersect_prefix(current_len);
        }

        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask_view.as_view(), &mut elements_buffer_mut)?;

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

        remaining = remaining.saturating_sub(N);
    }

    // Use the new optimized collect_bool_chunked method
    let values = BooleanBuffer::collect_bool(all_bools.len(), |idx| unsafe {
        *all_bools.get_unchecked(idx)
    });

    Ok(BoolArray::new(values, Validity::NonNullable))
}

pub fn export_bool_nonnull_masked(
    mask: &Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<BoolArray> {
    let len = mask.len();
    let true_count = mask.true_count();

    let mut elements_buffer = Vector::new::<bool>();
    let mut elements_buffer_mut = elements_buffer.as_view_mut();

    let mask_buffer = mask.to_boolean_buffer();
    let mut mask_iter = mask_buffer.bit_chunks().iter_padded();

    let mut mask = [0usize; N_WORDS];
    let mut mask_view = BitViewMut::new(&mut mask);

    // Fast path: collect all bools first, then use collect_bool for optimal packing
    // let mut all_bools: Vec<bool> = Vec::with_capacity(true_count);
    let mut remaining = len;

    let mut all_bools_buf = MutableBuffer::with_capacity(len.div_ceil(64) * 8);
    let mut value_count = 0;

    while remaining > 0 {
        mask_view.clear();
        mask_view.fill_with_words(&mut mask_iter);

        // Handle partial iteration on the last chunk
        let current_len = remaining.min(N);
        if current_len < N {
            mask_view.intersect_prefix(current_len);
        }

        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask_view.as_view(), &mut elements_buffer_mut)?;

        // Collect bools efficiently with unsafe for better performance
        let bool_slice = elements_buffer_mut.as_slice::<bool>();
        value_count += mask_view.true_count();

        collect_bool(bool_slice, &mut all_bools_buf);

        // Unsafe version to avoid bounds checking in hot path
        // let old_len = all_bools.len();
        // unsafe {
        //     all_bools.set_len(old_len + count);
        //     std::ptr::copy_nonoverlapping(
        //         bool_slice.as_ptr(),
        //         all_bools.as_mut_ptr().add(old_len),
        //         count,
        //     );
        // }

        remaining = remaining.saturating_sub(N);
    }

    // Use the new optimized collect_bool_chunked method
    // let values = collect_bool_chunked(&all_bools);
    let values = BooleanBuffer::new(all_bools_buf.into(), 0, value_count);

    Ok(BoolArray::new(values, Validity::NonNullable))
}

/// Collect boolean values into a BooleanBuffer using 1024-element chunks
fn collect_bool_chunked(all_bools: &[bool]) -> BooleanBuffer {
    const CHUNK_SIZE: usize = 1024;
    let len = all_bools.len();

    // Pre-allocate a single shared buffer for all chunks
    let byte_capacity = len.div_ceil(8);
    let mut buffer = MutableBuffer::with_capacity(byte_capacity);
    // ::with_capacity(byte_capacity);

    // Process full chunks of 1024 elements
    let full_chunks = len / CHUNK_SIZE;
    let mut offset = 0;

    collect_bool(all_bools, &mut buffer);

    // for _ in 0..full_chunks {
    //     let chunk = &all_bools[offset..offset + CHUNK_SIZE];
    //     collect_bool_1k::<CHUNK_SIZE>(chunk, &mut buffer);
    //     offset += CHUNK_SIZE;
    // }
    //
    // Process the remaining elements (less than 1024)
    // let remaining = len - offset;
    // if remaining > 0 {
    //     let chunk = &all_bools[offset..offset + remaining];
    //     collect_bool(chunk, &mut buffer);
    // }

    // Create BooleanBuffer from the packed bytes
    BooleanBuffer::new(buffer.into(), 0, len)
}

/// Collect boolean values into a shared buffer using the Arrow buffer approach
fn collect_bool_1k<const CHUNK: usize>(all_bools: &[bool], buffer: &mut MutableBuffer) {
    assert_eq!(CHUNK % 64, 0);
    // Process in chunks of 64 bits (8 bytes) at a time for optimal performance
    let chunks = CHUNK / 64;
    let remainder = CHUNK % 64;

    for chunk in 0..chunks {
        let mut packed: u64 = 0;
        let base_idx = chunk * 64;

        // Pack 64 bools into a single u64
        for bit_idx in 0..64 {
            let i = base_idx + bit_idx;
            unsafe {
                // Use unsafe indexing for performance
                packed |= (*all_bools.get_unchecked(i) as u64) << bit_idx;
            }
        }

        // Push the packed u64 as 8 bytes (little-endian)
        unsafe { buffer.push_unchecked(packed) }
    }
}

/// Collect boolean values into a shared buffer using the Arrow buffer approach
fn collect_bool(all_bools: &[bool], buffer: &mut MutableBuffer) {
    let len = all_bools.len();

    // Process in chunks of 64 bits (8 bytes) at a time for optimal performance
    let chunks = len / 64;
    let remainder = len % 64;

    for chunk in 0..chunks {
        let mut packed: u64 = 0;
        let base_idx = chunk * 64;

        // Pack 64 bools into a single u64
        for bit_idx in 0..64 {
            let i = base_idx + bit_idx;
            unsafe {
                // Use unsafe indexing for performance
                packed |= (*all_bools.get_unchecked(i) as u64) << bit_idx;
            }
        }

        // Push the packed u64 as 8 bytes (little-endian)
        unsafe { buffer.push_unchecked(packed) }
    }

    // Handle remaining bits
    if remainder != 0 {
        let mut packed: u64 = 0;
        let base_idx = chunks * 64;

        for bit_idx in 0..remainder {
            let i = base_idx + bit_idx;
            unsafe {
                packed |= (*all_bools.get_unchecked(i) as u64) << bit_idx;
            }
        }

        // For partial bytes, we need to be careful about how many bytes to write
        let bytes_needed = remainder.div_ceil(8);
        let packed_bytes = packed.to_le_bytes();
        unsafe { buffer.push_unchecked(packed) }
    }
}
