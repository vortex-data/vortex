// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encode pass leaf kernels: per-row byte writers for each canonical variant, plus the
//! variable-length block body encoder.

use vortex_array::arrays::decimal::cast_decimal_values;

use super::*;

pub(super) fn encode_null(
    arr: &NullArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
) {
    let sentinel = fixed_null_sentinel(field);
    for i in 0..arr.len() {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = sentinel;
        col_offset[i] += 1;
    }
}

pub(super) fn encode_bool(
    arr: &BoolArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let bits = arr.clone().into_bit_buffer();
    let non_null = FIXED_NON_NULL_SENTINEL;
    let xor = if field.descending { 0xFF } else { 0x00 };
    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for i in 0..bits.len() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                out[pos] = non_null;
                // false=0x01, true=0x02 so false < true; XOR for descending
                let raw = u8::from(bits.value(i)) + 1;
                out[pos + 1] = raw ^ xor;
                col_offset[i] += BOOL_ENCODED_SIZE;
            }
        }
        ValidityKind::Mask(mask) => {
            let null = fixed_null_sentinel(field);
            for i in 0..bits.len() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                if mask.value(i) {
                    out[pos] = non_null;
                    let raw = u8::from(bits.value(i)) + 1;
                    out[pos + 1] = raw ^ xor;
                } else {
                    out[pos] = null;
                    out[pos + 1] = 0;
                }
                col_offset[i] += BOOL_ENCODED_SIZE;
            }
        }
    }
    Ok(())
}

pub(super) fn encode_primitive(
    arr: &PrimitiveArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_native_ptype!(arr.ptype(), |T| {
        encode_primitive_typed::<T>(arr, field, row_offsets, col_offset, out, ctx)?;
    });
    Ok(())
}

fn encode_primitive_typed<T: NativePType + RowEncode>(
    arr: &PrimitiveArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let slice: &[T] = arr.as_slice();
    let non_null = FIXED_NON_NULL_SENTINEL;
    let value_bytes = size_of::<T>();
    let stride = encoded_size_for_fixed(byte_width_u32(value_bytes));
    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for (i, &v) in slice.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                out[pos] = non_null;
                v.encode_to(&mut out[pos + 1..pos + 1 + value_bytes], field.descending);
                col_offset[i] += stride;
            }
        }
        ValidityKind::Mask(mask) => {
            let null = fixed_null_sentinel(field);
            for (i, &v) in slice.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                if mask.value(i) {
                    out[pos] = non_null;
                    v.encode_to(&mut out[pos + 1..pos + 1 + value_bytes], field.descending);
                } else {
                    out[pos] = null;
                    // Zero-fill the value bytes.
                    for b in &mut out[pos + 1..pos + 1 + value_bytes] {
                        *b = 0;
                    }
                }
                col_offset[i] += stride;
            }
        }
    }
    Ok(())
}

/// Narrow a decimal array whose physical `values_type` is wider than its precision-minimal
/// type down to that minimal type, returning `None` when it already uses the minimal width.
///
/// Row-encoded widths are a pure function of the logical dtype: [`row_width_for_dtype`] sizes a
/// decimal column from [`DecimalType::smallest_decimal_value_type`] (the smallest physical type
/// that can hold the declared precision), independent of how the producer happened to store the
/// values. A `DecimalArray` may legally carry a wider `values_type` than its precision requires,
/// so without this normalization the encode pass would write more bytes than the size pass
/// reserved. The narrowing is always lossless because a decimal's precision bounds the magnitude
/// of every valid *non-null* value, so the precision-minimal type can represent it. Null slots
/// are unconstrained and may hold values that do not fit; [`cast_decimal_values`] narrows them
/// to zero instead of casting (the encoder zero-fills null bodies anyway).
fn narrow_decimal_to_smallest(
    arr: &DecimalArray,
    mask: &vortex_mask::Mask,
) -> VortexResult<Option<DecimalArray>> {
    let target = DecimalType::smallest_decimal_value_type(&arr.decimal_dtype());
    if arr.values_type() == target {
        return Ok(None);
    }
    cast_decimal_values(arr, target, mask).map(Some)
}

pub(super) fn encode_decimal(
    arr: &DecimalArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // Normalize to the precision-minimal physical type so the bytes we write match the width the
    // size pass reserved (see `narrow_decimal_to_smallest`).
    let mask = arr.as_ref().validity()?.execute_mask(arr.len(), ctx)?;
    let narrowed = narrow_decimal_to_smallest(arr, &mask)?;
    let arr = narrowed.as_ref().unwrap_or(arr);
    match arr.values_type() {
        DecimalType::I8 => {
            encode_decimal_typed::<i8>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I16 => {
            encode_decimal_typed::<i16>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I32 => {
            encode_decimal_typed::<i32>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I64 => {
            encode_decimal_typed::<i64>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I128 => {
            encode_decimal_typed::<i128>(arr, &mask, field, row_offsets, col_offset, out)
        }
        DecimalType::I256 => {
            vortex_bail!("row encoding for Decimal256 is not yet implemented")
        }
    }
    Ok(())
}

fn encode_decimal_typed<T>(
    arr: &DecimalArray,
    mask: &vortex_mask::Mask,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
) where
    T: NativeDecimalType + RowEncode,
{
    let non_null = FIXED_NON_NULL_SENTINEL;
    let null = fixed_null_sentinel(field);
    let value_bytes = size_of::<T>();
    let total = encoded_size_for_fixed(byte_width_u32(value_bytes));
    let slice = arr.buffer::<T>();
    for i in 0..slice.len() {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        if mask.value(i) {
            out[pos] = non_null;
            slice[i].encode_to(&mut out[pos + 1..pos + 1 + value_bytes], field.descending);
        } else {
            out[pos] = null;
            for b in &mut out[pos + 1..pos + 1 + value_bytes] {
                *b = 0;
            }
        }
        col_offset[i] += total;
    }
}

pub(super) fn encode_varbinview(
    arr: &VarBinViewArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let null_byte = varlen_null_sentinel(field);
    let empty_byte = varlen_empty_sentinel(field);
    let non_empty_byte = varlen_non_empty_sentinel(field);
    let descending = field.descending;

    let views = arr.views();
    // Cache the data-buffer slices once. Inlined views (len <= 12) carry their bytes inline,
    // so they never touch `buffers`; referenced views index into the pre-validated buffer at
    // `offset..offset + len`. Walking views directly avoids the per-row bounds and branch work
    // of `with_iterator`.
    let buffers: smallvec::SmallVec<[&[u8]; 4]> = (0..arr.data_buffers().len())
        .map(|i| arr.buffer(i).as_slice())
        .collect();

    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for (i, view) in views.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                let len = view.len() as usize;
                if len == 0 {
                    out[pos] = empty_byte;
                    col_offset[i] += VARLEN_EMPTY_SIZE;
                    continue;
                }
                let bytes: &[u8] = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let r = view.as_view();
                    let off = r.offset as usize;
                    &buffers[r.buffer_index as usize][off..off + len]
                };
                out[pos] = non_empty_byte;
                let written = encode_non_empty_varlen_body(bytes, &mut out[pos + 1..], descending)?;
                col_offset[i] += 1 + written;
            }
        }
        ValidityKind::Mask(mask) => {
            for (i, view) in views.iter().enumerate() {
                let pos = (row_offsets[i] + col_offset[i]) as usize;
                if !mask.value(i) {
                    out[pos] = null_byte;
                    col_offset[i] += VARLEN_NULL_SIZE;
                    continue;
                }
                let len = view.len() as usize;
                if len == 0 {
                    out[pos] = empty_byte;
                    col_offset[i] += VARLEN_EMPTY_SIZE;
                    continue;
                }
                let bytes: &[u8] = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let r = view.as_view();
                    let off = r.offset as usize;
                    &buffers[r.buffer_index as usize][off..off + len]
                };
                out[pos] = non_empty_byte;
                let written = encode_non_empty_varlen_body(bytes, &mut out[pos + 1..], descending)?;
                col_offset[i] += 1 + written;
            }
        }
    }
    Ok(())
}

pub(super) fn encode_struct(
    arr: &StructArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    let non_null = FIXED_NON_NULL_SENTINEL;
    let null = fixed_null_sentinel(field);

    // Write the outer sentinel for each row.
    for i in 0..n {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = if mask.value(i) { non_null } else { null };
        col_offset[i] += 1;
    }

    // Encode each child. For non-null parent rows the child contributes its actual encoding;
    // for null parent rows the child contributes its canonical null encoding so that two null
    // parent rows produce byte-equal output regardless of underlying child values.
    for child in arr.iter_unmasked_fields() {
        match row_width_for_dtype(child.dtype())? {
            RowWidth::Fixed(w) => {
                let canonical = child.clone().execute::<Canonical>(ctx)?;
                field_encode(&canonical, field, row_offsets, col_offset, out, ctx)?;
                // Replace null parent rows with the canonical null encoding (the same as a
                // child-level null: null sentinel followed by zero-padded value bytes).
                let null_byte = child_canonical_null_byte(child.dtype(), field);
                for i in 0..n {
                    if !mask.value(i) {
                        let end = (row_offsets[i] + col_offset[i]) as usize;
                        let start = end - w as usize;
                        out[start] = null_byte;
                        for b in &mut out[start + 1..end] {
                            *b = 0;
                        }
                    }
                }
            }
            RowWidth::Variable => {
                encode_variable_child(child, field, &mask, row_offsets, col_offset, out, ctx)?;
            }
        }
    }

    Ok(())
}

pub(super) fn encode_fsl(
    arr: &FixedSizeListArray,
    field: RowSortFieldOptions,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let nrows = arr.len();
    // `list_size` is natively a `u32`; keep both forms (see `add_size_fsl`).
    let list_size_u32 = arr.list_size();
    let list_size = list_size_u32 as usize;
    let mask = arr.as_ref().validity()?.execute_mask(nrows, ctx)?;
    let non_null = FIXED_NON_NULL_SENTINEL;
    let null = fixed_null_sentinel(field);
    let elem_dtype = arr.elements().dtype().clone();

    // Outer sentinel.
    for i in 0..nrows {
        let pos = (row_offsets[i] + col_offset[i]) as usize;
        out[pos] = if mask.value(i) { non_null } else { null };
        col_offset[i] += 1;
    }

    match row_width_for_dtype(&elem_dtype)? {
        RowWidth::Fixed(w) => {
            // Fixed-width elements: encode the elements array directly (its length is
            // nrows * list_size) using a derived (offsets, cursors) pair. Then overwrite
            // the body of null parent rows with the canonical null encoding per element.
            let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
            debug_assert_eq!(elements.len(), nrows * list_size);
            let row_body_bytes = w
                .checked_mul(list_size_u32)
                .ok_or_else(|| vortex_err!("FSL body width overflow"))?;
            let mut elem_offsets = vec![0u32; nrows * list_size];
            for i in 0..nrows {
                let base = row_offsets[i] + col_offset[i];
                for j in 0u32..list_size_u32 {
                    elem_offsets[i * list_size + j as usize] = base + j * w;
                }
            }
            let mut elem_cursors = vec![0u32; nrows * list_size];
            field_encode(&elements, field, &elem_offsets, &mut elem_cursors, out, ctx)?;
            for i in 0..nrows {
                col_offset[i] = col_offset[i]
                    .checked_add(row_body_bytes)
                    .ok_or_else(|| vortex_err!("FSL row body overflow"))?;
            }
            // Canonical null body for null parent rows: one null encoding per element.
            let null_byte = child_canonical_null_byte(&elem_dtype, field);
            let elem_width = w as usize;
            for i in 0..nrows {
                if !mask.value(i) {
                    let end = (row_offsets[i] + col_offset[i]) as usize;
                    let start = end - row_body_bytes as usize;
                    let mut pos = start;
                    for _ in 0..list_size {
                        out[pos] = null_byte;
                        for b in &mut out[pos + 1..pos + elem_width] {
                            *b = 0;
                        }
                        pos += elem_width;
                    }
                }
            }
        }
        RowWidth::Variable => {
            // Variable-width elements: for null parent rows the canonical body is exactly
            // `list_size` null sentinel bytes (one per element). For non-null parent rows,
            // encode each element via a scratch buffer and copy into out.
            let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
            debug_assert_eq!(elements.len(), nrows * list_size);
            let mut elem_sizes = vec![0u32; nrows * list_size];
            field_size(&elements, field, &mut elem_sizes, ctx)?;
            let total: u64 = elem_sizes.iter().map(|&s| u64::from(s)).sum();
            let total_usize =
                usize::try_from(total).vortex_expect("FSL scratch buffer size fits usize");
            let mut scratch = vec![0u8; total_usize];
            let mut scratch_offsets = Vec::with_capacity(nrows * list_size);
            let mut acc: u32 = 0;
            for &s in &elem_sizes {
                scratch_offsets.push(acc);
                acc = acc
                    .checked_add(s)
                    .ok_or_else(|| vortex_err!("FSL scratch offset overflow"))?;
            }
            let mut scratch_cursors = vec![0u32; nrows * list_size];
            field_encode(
                &elements,
                field,
                &scratch_offsets,
                &mut scratch_cursors,
                &mut scratch,
                ctx,
            )?;
            let null_byte = child_canonical_null_byte(&elem_dtype, field);
            for i in 0..nrows {
                let dst = (row_offsets[i] + col_offset[i]) as usize;
                if mask.value(i) {
                    let mut body_bytes: u32 = 0;
                    for j in 0..list_size {
                        let k = i * list_size + j;
                        let src = scratch_offsets[k] as usize;
                        let sz = elem_sizes[k] as usize;
                        out[dst + body_bytes as usize..dst + body_bytes as usize + sz]
                            .copy_from_slice(&scratch[src..src + sz]);
                        body_bytes = body_bytes
                            .checked_add(elem_sizes[k])
                            .ok_or_else(|| vortex_err!("FSL body bytes overflow"))?;
                    }
                    col_offset[i] = col_offset[i]
                        .checked_add(body_bytes)
                        .ok_or_else(|| vortex_err!("FSL row offset overflow"))?;
                } else {
                    for offset in 0..list_size {
                        out[dst + offset] = null_byte;
                    }
                    col_offset[i] = col_offset[i]
                        .checked_add(list_size_u32)
                        .ok_or_else(|| vortex_err!("FSL row offset overflow"))?;
                }
            }
        }
    }

    Ok(())
}

/// Encode one variable-width child of a struct: for non-null parent rows, copy the child's
/// natural encoding from a scratch buffer; for null parent rows, write a single
/// `child_canonical_null_byte`.
fn encode_variable_child(
    child: &vortex_array::ArrayRef,
    field: RowSortFieldOptions,
    parent_mask: &vortex_mask::Mask,
    row_offsets: &[u32],
    col_offset: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = child.len();
    let canonical = child.clone().execute::<Canonical>(ctx)?;

    // Size and encode the child into a sequential scratch buffer.
    let mut child_sizes = vec![0u32; n];
    field_size(&canonical, field, &mut child_sizes, ctx)?;
    let total: u64 = child_sizes.iter().map(|&s| u64::from(s)).sum();
    let total_usize = usize::try_from(total).vortex_expect("child scratch buffer size fits usize");
    let mut scratch = vec![0u8; total_usize];
    let mut scratch_offsets = Vec::with_capacity(n);
    let mut acc: u32 = 0;
    for &s in &child_sizes {
        scratch_offsets.push(acc);
        acc = acc
            .checked_add(s)
            .ok_or_else(|| vortex_err!("child scratch offset overflow"))?;
    }
    let mut scratch_cursors = vec![0u32; n];
    field_encode(
        &canonical,
        field,
        &scratch_offsets,
        &mut scratch_cursors,
        &mut scratch,
        ctx,
    )?;

    let null_byte = child_canonical_null_byte(child.dtype(), field);
    for i in 0..n {
        let dst = (row_offsets[i] + col_offset[i]) as usize;
        if parent_mask.value(i) {
            let src = scratch_offsets[i] as usize;
            let sz = child_sizes[i] as usize;
            out[dst..dst + sz].copy_from_slice(&scratch[src..src + sz]);
            col_offset[i] = col_offset[i]
                .checked_add(child_sizes[i])
                .ok_or_else(|| vortex_err!("col_offset overflow"))?;
        } else {
            out[dst] = null_byte;
            col_offset[i] = col_offset[i]
                .checked_add(1)
                .ok_or_else(|| vortex_err!("col_offset overflow"))?;
        }
    }
    Ok(())
}

/// Arithmetic-write primitive encoder: writes each row's `sentinel + value` slot at a
/// constant within-row offset, iterating the output in `row_stride`-sized chunks so the
/// compiler can drop the per-row offset/cursor indirection.
pub(super) fn encode_primitive_arith(
    arr: &PrimitiveArray,
    field: RowSortFieldOptions,
    col_prefix: u32,
    row_stride: u32,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_native_ptype!(arr.ptype(), |T| {
        encode_primitive_arith_typed::<T>(arr, field, col_prefix, row_stride, out, ctx)?;
    });
    Ok(())
}

fn encode_primitive_arith_typed<T: NativePType + RowEncode>(
    arr: &PrimitiveArray,
    field: RowSortFieldOptions,
    col_prefix: u32,
    row_stride: u32,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let slice: &[T] = arr.as_slice();
    let non_null = FIXED_NON_NULL_SENTINEL;
    let value_bytes = size_of::<T>();
    let slot_size = 1 + value_bytes;
    let stride = row_stride as usize;
    let prefix = col_prefix as usize;
    let descending = field.descending;

    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            // Hot path: each row's slot is a fixed window inside its `stride`-sized chunk,
            // so the inner write vectorizes the same way as `arrow-row`'s not-null path.
            for (chunk, &v) in out.chunks_exact_mut(stride).zip(slice.iter()) {
                let slot = &mut chunk[prefix..prefix + slot_size];
                slot[0] = non_null;
                v.encode_to(&mut slot[1..], descending);
            }
        }
        ValidityKind::Mask(mask) => {
            let null = fixed_null_sentinel(field);
            for (i, (chunk, &v)) in out.chunks_exact_mut(stride).zip(slice.iter()).enumerate() {
                let slot = &mut chunk[prefix..prefix + slot_size];
                if mask.value(i) {
                    slot[0] = non_null;
                    v.encode_to(&mut slot[1..], descending);
                } else {
                    slot[0] = null;
                    for b in &mut slot[1..] {
                        *b = 0;
                    }
                }
            }
        }
    }
    Ok(())
}
