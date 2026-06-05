// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernel for `FoRArray`.
//!
//! For the common fused path (`encoded` is a `BitPacked` with unsigned encoded values), this
//! walks the bit-packed storage in 1024-element chunks, applies the FoR base inline via a
//! custom `UnpackStrategy`, and writes the row-encoded bytes in one pass. Other shapes fall
//! through to the canonicalize path.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "row encoding indexes into u32-sized buffers"
)]
use std::mem::MaybeUninit;

use fastlanes::FoR as FoRTrait;
use num_traits::WrappingAdd;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_row::RowEncodeRegistration;
use vortex_row::RowSortField;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FoR;
use crate::r#for::array::FoRArrayExt;
use crate::row_encode_common::PrimRowEncode;
use crate::row_encode_common::encode_primitive_chunk;
use crate::row_encode_common::encoded_size_for_ptype;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// Per-row size contribution for a `FoR` column.
fn for_size_contribution(
    column: &ArrayRef,
    _field: RowSortField,
    sizes: &mut [u32],
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<FoR>() else {
        return Ok(None);
    };
    let add = encoded_size_for_ptype(view.as_ref().dtype().as_ptype());
    for s in sizes.iter_mut().take(view.as_ref().len()) {
        *s += add;
    }
    Ok(Some(()))
}

/// Per-row byte encoding for a `FoR` column.
fn for_encode_into(
    column: &ArrayRef,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<FoR>() else {
        return Ok(None);
    };
    let ptype = view.as_ref().dtype().as_ptype();

    // Fast path: encoded is a BitPacked with unsigned encoded values (the case used by the
    // standard compressor). We do fused unpack + base-add + row-write in one pass.
    if view.reference_scalar().dtype().is_unsigned_int()
        && let Some(bp) = view.encoded().as_opt::<BitPacked>()
    {
        match_each_unsigned_integer_ptype!(ptype, |T| {
            encode_for_bitpacked::<T>(view, bp, field, offsets, cursors, out, ctx)?;
        });
        return Ok(Some(()));
    }

    // Slower path: encoded is already a primitive array (or a non-BitPacked encoded). Walk
    // the canonical primitive buffer directly and add the base.
    if view.encoded().as_opt::<Primitive>().is_some() {
        match_each_integer_ptype!(ptype, |T| {
            encode_for_primitive::<T>(view, field, offsets, cursors, out, ctx)?;
        });
        return Ok(Some(()));
    }

    // Decline; the default canonicalization path will handle it.
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn encode_for_bitpacked<T>(
    for_view: vortex_array::ArrayView<'_, FoR>,
    bp_view: vortex_array::ArrayView<'_, BitPacked>,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    T: BitPackedUnpack
        + PhysicalPType<Physical = T>
        + UnsignedPType
        + FoRTrait
        + WrappingAdd
        + NativePType
        + PrimRowEncode,
{
    let ref_value: T = for_view
        .reference_scalar()
        .as_primitive()
        .as_::<T>()
        .vortex_expect("FoR reference cannot be null");

    let total_len = bp_view.as_ref().len();
    let descending = field.descending;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let value_bytes = size_of::<T>();
    let stride = (1 + value_bytes) as u32;

    // Materialize validity once.
    let validity = BitPackedArrayExt::validity(&bp_view);
    let mask = match &validity {
        Validity::NonNullable | Validity::AllValid => None,
        _ => Some(validity.execute_mask(total_len, ctx)?),
    };

    // Pre-canonicalize patches: rare. They have already been wrapping-added with the base by
    // the FoR fused path; here we mirror that contract by adding the base after looking up
    // the patch value.
    let patches = bp_view.patches();
    let patch_pairs = if let Some(p) = patches {
        let indices = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = p.values().clone().execute::<PrimitiveArray>(ctx)?;
        Some((indices, values, p.offset()))
    } else {
        None
    };

    // Walk chunks directly: 1024 elements per chunk, FoR base added inline.
    let bit_width = bp_view.bit_width() as usize;
    let offset = bp_view.offset() as usize;
    let packed_bytes = bp_view.packed().as_host();
    // SAFETY: packed bytes are aligned as `T` per FastLanes layout invariants.
    let packed_slice: &[T] = unsafe {
        std::slice::from_raw_parts(
            packed_bytes.as_ptr().cast::<T>(),
            packed_bytes.len() / size_of::<T>(),
        )
    };
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    let num_chunks = (offset + total_len).div_ceil(1024);

    let mut buf: [MaybeUninit<T>; 1024] = [const { MaybeUninit::<T>::uninit() }; 1024];
    let mut local_idx: usize = 0;
    for chunk_idx in 0..num_chunks {
        // SAFETY: `chunk` covers `elems_per_chunk` packed elements; `buf` is exactly 1024 entries.
        unsafe {
            let chunk = &packed_slice[chunk_idx * elems_per_chunk..][..elems_per_chunk];
            FoRTrait::unchecked_unfor_pack(
                bit_width,
                chunk,
                ref_value,
                std::mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(&mut buf),
            );
        }
        // SAFETY: just initialized 1024 elements.
        let unpacked: &mut [T; 1024] =
            unsafe { std::mem::transmute::<&mut [MaybeUninit<T>; 1024], &mut [T; 1024]>(&mut buf) };

        // Determine the logical range within this chunk.
        let chunk_offset = if chunk_idx == 0 { offset } else { 0 };
        let chunk_logical_start = chunk_idx * 1024;
        let chunk_logical_end = ((chunk_idx + 1) * 1024).min(offset + total_len);
        let usable = &mut unpacked[chunk_offset..(chunk_logical_end - chunk_idx * 1024)];

        // Apply patches that fall in this chunk.
        apply_patches_in_range_for::<T>(
            usable,
            patch_pairs.as_ref(),
            local_idx,
            local_idx + usable.len(),
            ref_value,
        );

        encode_primitive_chunk::<T>(
            usable,
            local_idx,
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
        local_idx += usable.len();
        let _ = chunk_logical_start;
    }

    debug_assert_eq!(local_idx, total_len);
    Ok(())
}

fn apply_patches_in_range_for<T>(
    chunk: &mut [T],
    patch_pairs: Option<&(PrimitiveArray, PrimitiveArray, usize)>,
    chunk_start: usize,
    chunk_end: usize,
    ref_value: T,
) where
    T: NativePType + WrappingAdd,
{
    let Some((indices_p, values_p, patch_offset)) = patch_pairs else {
        return;
    };
    let values: &[T] = values_p.as_slice();
    let logical_start = chunk_start + *patch_offset;
    let logical_end = chunk_end + *patch_offset;
    macro_rules! walk {
        ($idx_ty:ty) => {{
            let idx: &[$idx_ty] = indices_p.as_slice();
            for (i, &raw_idx) in idx.iter().enumerate() {
                let raw_idx = raw_idx as usize;
                if raw_idx < logical_start {
                    continue;
                }
                if raw_idx >= logical_end {
                    break;
                }
                let local = raw_idx - logical_start;
                chunk[local] = values[i].wrapping_add(&ref_value);
            }
        }};
    }
    match indices_p.ptype() {
        PType::U64 => walk!(u64),
        PType::U32 => walk!(u32),
        PType::U16 => walk!(u16),
        PType::U8 => walk!(u8),
        _ => {}
    }
}

fn encode_for_primitive<T>(
    for_view: vortex_array::ArrayView<'_, FoR>,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    T: NativePType + PrimRowEncode + WrappingAdd,
{
    let encoded = for_view.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let total_len = encoded.len();
    let descending = field.descending;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let value_bytes = size_of::<T>();
    let stride = (1 + value_bytes) as u32;
    let ref_value: T = for_view
        .reference_scalar()
        .as_primitive()
        .as_::<T>()
        .vortex_expect("FoR reference cannot be null");

    let validity = encoded.validity()?;
    let mask = match &validity {
        Validity::NonNullable | Validity::AllValid => None,
        _ => Some(validity.execute_mask(total_len, ctx)?),
    };

    let slice: &[T] = encoded.as_slice();
    match mask {
        None => {
            for (i, &v) in slice.iter().enumerate() {
                let val = v.wrapping_add(&ref_value);
                let pos = (offsets[i] + cursors[i]) as usize;
                out[pos] = non_null;
                val.row_encode_to(&mut out[pos + 1..pos + 1 + value_bytes], descending);
                cursors[i] += stride;
            }
        }
        Some(m) => {
            for (i, &v) in slice.iter().enumerate() {
                let pos = (offsets[i] + cursors[i]) as usize;
                if m.value(i) {
                    let val = v.wrapping_add(&ref_value);
                    out[pos] = non_null;
                    val.row_encode_to(&mut out[pos + 1..pos + 1 + value_bytes], descending);
                } else {
                    out[pos] = null;
                    for b in &mut out[pos + 1..pos + 1 + value_bytes] {
                        *b = 0;
                    }
                }
                cursors[i] += stride;
            }
        }
    }
    Ok(())
}

fn for_array_id() -> ArrayId {
    use vortex_session::registry::CachedId;
    static ID: CachedId = CachedId::new("fastlanes.for");
    *ID
}

inventory::submit! {
    RowEncodeRegistration {
        id: for_array_id,
        size: for_size_contribution,
        encode: for_encode_into,
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
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_row::RowSortField;
    use vortex_row::convert_columns;

    use crate::BitPackedData;
    use crate::FoR;

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
    fn for_row_encode_matches_canonical_primitive_encoded() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // FoR with a plain primitive `encoded` array (no BitPacked underneath).
        let encoded = buffer![5i64, 6, 7, 8, 9].into_array();
        let arr = FoR::try_new(encoded, Scalar::from(100i64))?.into_array();
        let raw = buffer![105i64, 106, 107, 108, 109].into_array();

        let by_raw = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_for = convert_columns(&[arr], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_raw), collect_rows(&by_for));
        Ok(())
    }

    #[test]
    fn for_row_encode_matches_canonical_bitpacked_encoded() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let n: usize = 1500;
        let raw_values: Vec<u32> = (0..n as u32).map(|i| 10_000 + i).collect();
        let raw_arr = PrimitiveArray::from_iter(raw_values).into_array();
        let unsigned_encoded =
            PrimitiveArray::from_iter((0..n as u32).collect::<Vec<u32>>()).into_array();
        let bp = BitPackedData::encode(&unsigned_encoded, 11, &mut ctx)?.into_array();
        let arr = FoR::try_new(bp, Scalar::from(10_000u32))?.into_array();

        let by_raw = convert_columns(&[raw_arr], &[RowSortField::default()], &mut ctx)?;
        let by_for = convert_columns(&[arr], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_raw), collect_rows(&by_for));
        Ok(())
    }
}
