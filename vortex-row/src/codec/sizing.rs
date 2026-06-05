// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Size pass leaf kernels: per-row byte-size accumulation for each canonical variant.
//!
//! Every accumulator returns [`VortexResult`] and uses checked arithmetic, so an input whose
//! per-row encoding would exceed `u32::MAX` bytes surfaces a [`VortexError`](vortex_error::VortexError)
//! instead of overflowing or panicking.

use super::*;

pub(super) fn add_size_const(sizes: &mut [u32], add: u32) -> VortexResult<()> {
    for s in sizes.iter_mut() {
        *s = s
            .checked_add(add)
            .ok_or_else(|| vortex_err!("per-row size overflow"))?;
    }
    Ok(())
}

pub(super) fn add_size_null(arr: &NullArray, sizes: &mut [u32]) -> VortexResult<()> {
    debug_assert_eq!(arr.len(), sizes.len());
    // Just a sentinel byte per row.
    add_size_const(sizes, 1)
}

pub(super) fn add_size_primitive(arr: &PrimitiveArray, sizes: &mut [u32]) -> VortexResult<()> {
    let width = byte_width_u32(arr.ptype().byte_width());
    add_size_const(sizes, encoded_size_for_fixed(width))
}

pub(super) fn add_size_decimal(arr: &DecimalArray, sizes: &mut [u32]) -> VortexResult<()> {
    // Size from the precision-minimal type, not the physical `values_type`, so the size pass
    // agrees with `row_width_for_dtype` (and the encode pass) regardless of how the producer
    // stored the values. See `narrow_decimal_to_smallest`.
    let vt = DecimalType::smallest_decimal_value_type(&arr.decimal_dtype());
    let width = byte_width_u32(vt.byte_width());
    add_size_const(sizes, encoded_size_for_fixed(width))
}

pub(super) fn add_size_varbinview(
    arr: &VarBinViewArray,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let views = arr.views();
    match resolve_validity(arr.as_ref().validity()?, arr.len(), ctx)? {
        ValidityKind::AllValid => {
            for (i, view) in views.iter().enumerate() {
                let contribution = if view.is_empty() {
                    VARLEN_EMPTY_SIZE
                } else {
                    encoded_size_for_non_empty_varlen(view.len() as usize)?
                };
                sizes[i] = sizes[i]
                    .checked_add(contribution)
                    .ok_or_else(|| vortex_err!("per-row size overflow"))?;
            }
        }
        ValidityKind::Mask(mask) => {
            for (i, view) in views.iter().enumerate() {
                let contribution = if !mask.value(i) {
                    VARLEN_NULL_SIZE
                } else if view.is_empty() {
                    VARLEN_EMPTY_SIZE
                } else {
                    encoded_size_for_non_empty_varlen(view.len() as usize)?
                };
                sizes[i] = sizes[i]
                    .checked_add(contribution)
                    .ok_or_else(|| vortex_err!("per-row size overflow"))?;
            }
        }
    }
    Ok(())
}

pub(super) fn add_size_struct(
    arr: &StructArray,
    field: RowSortFieldOptions,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    // Outer sentinel: 1 byte per row.
    add_size_const(sizes, 1)?;
    // Each child contributes its per-row size when the parent is non-null, and a canonical
    // null contribution when the parent is null. For fixed-width children both are equal,
    // so we can simply add the fixed width to every row. For variable-width children the
    // null contribution collapses to 1 byte, ensuring null parent rows have a constant body.
    for child in arr.iter_unmasked_fields() {
        match row_width_for_dtype(child.dtype())? {
            RowWidth::Fixed(w) => add_size_const(sizes, w)?,
            RowWidth::Variable => {
                let canonical = child.clone().execute::<Canonical>(ctx)?;
                let mut child_sizes = vec![0u32; n];
                field_size(&canonical, field, &mut child_sizes, ctx)?;
                for i in 0..n {
                    let contribution = if mask.value(i) { child_sizes[i] } else { 1u32 };
                    sizes[i] = sizes[i]
                        .checked_add(contribution)
                        .ok_or_else(|| vortex_err!("per-row size overflow"))?;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn add_size_fsl(
    arr: &FixedSizeListArray,
    field: RowSortFieldOptions,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n = arr.len();
    debug_assert_eq!(n, sizes.len());
    // `list_size` is natively a `u32`; keep both forms so element indexing stays `usize` while
    // width arithmetic avoids a fallible `usize -> u32` conversion.
    let list_size_u32 = arr.list_size();
    let list_size = list_size_u32 as usize;
    let mask = arr.as_ref().validity()?.execute_mask(n, ctx)?;
    let elem_dtype = arr.elements().dtype();
    // Outer sentinel: 1 byte per row.
    add_size_const(sizes, 1)?;
    match row_width_for_dtype(elem_dtype)? {
        RowWidth::Fixed(w) => {
            // Each row has `list_size` fixed-width elements regardless of null parent mask.
            let body = w
                .checked_mul(list_size_u32)
                .ok_or_else(|| vortex_err!("FSL body width overflow"))?;
            add_size_const(sizes, body)?;
        }
        RowWidth::Variable => {
            let elements = arr.elements().clone().execute::<Canonical>(ctx)?;
            debug_assert_eq!(elements.len(), n * list_size);
            let mut elem_sizes = vec![0u32; n * list_size];
            field_size(&elements, field, &mut elem_sizes, ctx)?;
            for i in 0..n {
                let body: u32 = if mask.value(i) {
                    let base = i * list_size;
                    let mut sum: u32 = 0;
                    for j in 0..list_size {
                        sum = sum
                            .checked_add(elem_sizes[base + j])
                            .ok_or_else(|| vortex_err!("FSL row body overflow"))?;
                    }
                    sum
                } else {
                    // Canonical null body for FSL with variable element: one null sentinel
                    // per element. (Each element contributes `child_null_width = 1`.)
                    list_size_u32
                };
                sizes[i] = sizes[i]
                    .checked_add(body)
                    .ok_or_else(|| vortex_err!("FSL per-row size overflow"))?;
            }
        }
    }
    Ok(())
}
