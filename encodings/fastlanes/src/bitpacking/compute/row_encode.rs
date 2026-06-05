// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernel for `BitPackedArray`.
//!
//! Walks the bit-packed storage in 1024-element chunks, unpacks each chunk into a
//! stack-local buffer, and writes the row-encoded bytes in one pass. Avoids
//! materializing a canonical `PrimitiveArray` first.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "row encoding indexes into u32-sized buffers and bit-packed widths are small"
)]
#![allow(
    unused_imports,
    reason = "Item is consumed by the #[gat(Item)] macro expansion"
)]

use lending_iterator::gat;
#[allow(unused_imports)]
use lending_iterator::prelude::Item;
#[gat(Item)]
use lending_iterator::prelude::LendingIterator;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_row::RowEncodeRegistration;
use vortex_row::RowSortField;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::row_encode_common::PrimRowEncode;
use crate::row_encode_common::encode_primitive_chunk;
use crate::row_encode_common::encode_primitive_chunk_arith;
use crate::row_encode_common::encoded_size_for_ptype;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// Per-row size contribution for a `BitPacked` column.
fn bitpacked_size_contribution(
    column: &ArrayRef,
    _field: RowSortField,
    sizes: &mut [u32],
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<BitPacked>() else {
        return Ok(None);
    };
    let add = encoded_size_for_ptype(view.dtype().as_ptype());
    for s in sizes.iter_mut().take(view.as_ref().len()) {
        *s += add;
    }
    Ok(Some(()))
}

fn supported_ptype(ptype: PType) -> bool {
    matches!(
        ptype,
        PType::I8
            | PType::I16
            | PType::I32
            | PType::I64
            | PType::U8
            | PType::U16
            | PType::U32
            | PType::U64
    )
}

/// Materialize the null mask and patch slices once, outside the hot loop.
#[allow(clippy::type_complexity)]
fn bitpacked_prep(
    view: vortex_array::ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(
    Option<vortex_mask::Mask>,
    Option<(PrimitiveArray, PrimitiveArray, usize)>,
)> {
    // The explicit Ext method returns a `Validity` (the inherent `validity()` on `ArrayView`
    // returns `VortexResult<Validity>`).
    let validity = BitPackedArrayExt::validity(&view);
    let mask = match &validity {
        Validity::NonNullable | Validity::AllValid => None,
        _ => Some(validity.execute_mask(view.as_ref().len(), ctx)?),
    };
    let patch_pairs = if let Some(p) = view.patches() {
        let indices = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = p.values().clone().execute::<PrimitiveArray>(ctx)?;
        Some((indices, values, p.offset()))
    } else {
        None
    };
    Ok((mask, patch_pairs))
}

/// Walk the bit-packed storage in 1024-element chunks (initial sliced chunk, full middle
/// chunks, trailing sliced chunk), unpacking each chunk into a stack buffer, applying any
/// patches, and handing the chunk plus its starting logical row to `write`.
fn walk_bitpacked<T, F>(
    arr_view: vortex_array::ArrayView<'_, BitPacked>,
    patch_pairs: Option<&(PrimitiveArray, PrimitiveArray, usize)>,
    out: &mut [u8],
    mut write: F,
) -> VortexResult<()>
where
    T: BitPackedUnpack + NativePType,
    F: FnMut(&[T], usize, &mut [u8]),
{
    let total_len = arr_view.as_ref().len();
    let mut local_idx: usize = 0;
    let mut unpacked = arr_view.unpacked_chunks::<T>()?;

    if let Some(initial) = unpacked.initial() {
        let len_chunk = initial.len();
        apply_patches_in_range::<T>(initial, patch_pairs, local_idx, local_idx + len_chunk);
        write(initial, local_idx, out);
        local_idx += len_chunk;
    }

    let mut chunks_iter = unpacked.full_chunks();
    while let Some(chunk) = chunks_iter.next() {
        let len_chunk = 1024.min(total_len - local_idx);
        apply_patches_in_range::<T>(
            &mut chunk[..len_chunk],
            patch_pairs,
            local_idx,
            local_idx + len_chunk,
        );
        write(&chunk[..len_chunk], local_idx, out);
        local_idx += len_chunk;
    }

    if let Some(trailer) = unpacked.trailer() {
        let len_chunk = trailer.len();
        apply_patches_in_range::<T>(trailer, patch_pairs, local_idx, local_idx + len_chunk);
        write(trailer, local_idx, out);
        local_idx += len_chunk;
    }

    debug_assert_eq!(local_idx, total_len);
    Ok(())
}

/// Per-row byte encoding for a `BitPacked` column (cursor path).
fn bitpacked_encode_into(
    column: &ArrayRef,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<BitPacked>() else {
        return Ok(None);
    };
    let ptype = view.dtype().as_ptype();
    if !supported_ptype(ptype) {
        return Ok(None);
    }
    let (mask, patch_pairs) = bitpacked_prep(view, ctx)?;
    let descending = field.descending;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();

    match_each_integer_ptype!(ptype, |T| {
        let value_bytes = size_of::<T>();
        let stride = (1 + value_bytes) as u32;
        walk_bitpacked::<T, _>(view, patch_pairs.as_ref(), out, |chunk, row_start, out| {
            encode_primitive_chunk::<T>(
                chunk,
                row_start,
                offsets,
                cursors,
                out,
                mask.as_ref(),
                non_null,
                null,
                descending,
                value_bytes,
                stride,
            );
        })?;
    });
    Ok(Some(()))
}

/// Fixed-width arithmetic encoding for a `BitPacked` column: fuse decompression with the row
/// write at `i * row_stride + col_prefix (+ var_prefix[i])`, skipping the canonical array and
/// the per-row cursor entirely.
#[allow(clippy::too_many_arguments)]
fn bitpacked_encode_fixed_arith(
    column: &ArrayRef,
    field: RowSortField,
    col_prefix: u32,
    row_stride: u32,
    var_prefix: Option<&[u32]>,
    _nrows: usize,
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<BitPacked>() else {
        return Ok(None);
    };
    let ptype = view.dtype().as_ptype();
    if !supported_ptype(ptype) {
        return Ok(None);
    }
    let (mask, patch_pairs) = bitpacked_prep(view, ctx)?;
    let descending = field.descending;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();

    match_each_integer_ptype!(ptype, |T| {
        let value_bytes = size_of::<T>();
        walk_bitpacked::<T, _>(view, patch_pairs.as_ref(), out, |chunk, row_start, out| {
            encode_primitive_chunk_arith::<T>(
                chunk,
                row_start,
                col_prefix,
                row_stride,
                var_prefix,
                out,
                mask.as_ref(),
                non_null,
                null,
                descending,
                value_bytes,
            );
        })?;
    });
    Ok(Some(()))
}

/// Overwrite values in `chunk` (which covers logical rows `[chunk_start, chunk_end)`) with
/// any patch values that fall in that range.
fn apply_patches_in_range<T: NativePType>(
    chunk: &mut [T],
    patch_pairs: Option<&(PrimitiveArray, PrimitiveArray, usize)>,
    chunk_start: usize,
    chunk_end: usize,
) {
    let Some((indices_p, values_p, patch_offset)) = patch_pairs else {
        return;
    };
    let values: &[T] = values_p.as_slice();
    // Indices may be u32 or u64. We search for the first index >= chunk_start + patch_offset.
    // For simplicity, scan linearly per chunk; patches are rare.
    let logical_start = chunk_start + *patch_offset;
    let logical_end = chunk_end + *patch_offset;
    let indices_ptype = indices_p.ptype();
    match indices_ptype {
        PType::U32 => {
            let idx: &[u32] = indices_p.as_slice();
            for (i, &raw_idx) in idx.iter().enumerate() {
                let raw_idx = raw_idx as usize;
                if raw_idx < logical_start {
                    continue;
                }
                if raw_idx >= logical_end {
                    break;
                }
                let local = raw_idx - logical_start;
                chunk[local] = values[i];
            }
        }
        PType::U64 => {
            let idx: &[u64] = indices_p.as_slice();
            for (i, &raw_idx) in idx.iter().enumerate() {
                let raw_idx = raw_idx as usize;
                if raw_idx < logical_start {
                    continue;
                }
                if raw_idx >= logical_end {
                    break;
                }
                let local = raw_idx - logical_start;
                chunk[local] = values[i];
            }
        }
        PType::U16 => {
            let idx: &[u16] = indices_p.as_slice();
            for (i, &raw_idx) in idx.iter().enumerate() {
                let raw_idx = raw_idx as usize;
                if raw_idx < logical_start {
                    continue;
                }
                if raw_idx >= logical_end {
                    break;
                }
                let local = raw_idx - logical_start;
                chunk[local] = values[i];
            }
        }
        PType::U8 => {
            let idx: &[u8] = indices_p.as_slice();
            for (i, &raw_idx) in idx.iter().enumerate() {
                let raw_idx = raw_idx as usize;
                if raw_idx < logical_start {
                    continue;
                }
                if raw_idx >= logical_end {
                    break;
                }
                let local = raw_idx - logical_start;
                chunk[local] = values[i];
            }
        }
        _ => {}
    }
}

fn bitpacked_array_id() -> ArrayId {
    use vortex_session::registry::CachedId;
    static ID: CachedId = CachedId::new("fastlanes.bitpacked");
    *ID
}

inventory::submit! {
    RowEncodeRegistration {
        id: bitpacked_array_id,
        size: bitpacked_size_contribution,
        encode: bitpacked_encode_into,
        encode_fixed_arith: Some(bitpacked_encode_fixed_arith),
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::listview::ListViewArrayExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_row::RowSortField;
    use vortex_row::convert_columns;

    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

    fn collect_rows(arr: &ListViewArray) -> Vec<Vec<u8>> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let n = arr.len();
        (0..n)
            .map(|i| {
                let slice = arr.list_elements_at(i).unwrap();
                let p = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
                p.as_slice::<u8>().to_vec()
            })
            .collect()
    }

    #[test]
    fn bitpacked_row_encode_matches_canonical() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let raw = buffer![1u32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
        let bp = BitPackedData::encode(&raw, 4, &mut ctx)?.into_array();

        let by_canonical = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_bp = convert_columns(&[bp], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_canonical), collect_rows(&by_bp));
        Ok(())
    }

    #[test]
    fn bitpacked_row_encode_with_patches() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<u32> = (0..200)
            .map(|i| if i % 30 == 0 { 5000 + i } else { i % 16 })
            .collect();
        let raw = PrimitiveArray::from_iter(values).into_array();
        let bp = BitPackedData::encode(&raw, 4, &mut ctx)?.into_array();
        assert!(bp.as_opt::<crate::BitPacked>().unwrap().patches().is_some());
        let by_canonical = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_bp = convert_columns(&[bp], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_canonical), collect_rows(&by_bp));
        Ok(())
    }

    #[test]
    fn bitpacked_row_encode_multi_chunk() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<u32> = (0..3000).map(|i| i % 64).collect();
        let raw = PrimitiveArray::from_iter(values).into_array();
        let bp = BitPackedData::encode(&raw, 6, &mut ctx)?.into_array();
        let by_canonical = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_bp = convert_columns(&[bp], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_canonical), collect_rows(&by_bp));
        Ok(())
    }

    /// A fixed-width BitPacked column placed *before* a variable-length column exercises the
    /// arithmetic path's `var_prefix` branch (irregularly-spaced row positions).
    #[test]
    fn bitpacked_row_encode_before_varlen_uses_var_prefix() -> VortexResult<()> {
        use vortex_array::arrays::VarBinViewArray;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let nums: Vec<u32> = (0..64).map(|i| i % 50).collect();
        let raw = PrimitiveArray::from_iter(nums).into_array();
        let bp = BitPackedData::encode(&raw, 6, &mut ctx)?.into_array();
        let words = VarBinViewArray::from_iter_str((0..64).map(|i| "x".repeat((i % 7) + 1)));
        let words = words.into_array();
        let fields = [RowSortField::default(), RowSortField::default()];

        let by_canonical = convert_columns(&[raw, words.clone()], &fields, &mut ctx)?;
        let by_bp = convert_columns(&[bp, words], &fields, &mut ctx)?;
        assert_eq!(collect_rows(&by_canonical), collect_rows(&by_bp));
        Ok(())
    }
}
