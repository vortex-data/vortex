// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernel for `DeltaArray`.
//!
//! Decompresses the deltas directly into a temporary primitive buffer (avoiding the
//! `PrimitiveArray` wrapper allocation and validity attachment) and then walks the buffer
//! once to produce row-encoded bytes.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "row encoding indexes into u32-sized buffers"
)]

use fastlanes::Delta as DeltaTrait;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_row::RowEncodeRegistration;
use vortex_row::RowSortField;

use crate::Delta;
use crate::bit_transpose::untranspose_validity;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::delta_decompress::decompress_primitive;
use crate::row_encode_common::PrimRowEncode;
use crate::row_encode_common::encode_primitive_chunk;
use crate::row_encode_common::encoded_size_for_ptype;

/// Per-row size contribution for a `Delta` column.
fn delta_size_contribution(
    column: &ArrayRef,
    _field: RowSortField,
    sizes: &mut [u32],
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<Delta>() else {
        return Ok(None);
    };
    let add = encoded_size_for_ptype(view.as_ref().dtype().as_ptype());
    for s in sizes.iter_mut().take(view.as_ref().len()) {
        *s += add;
    }
    Ok(Some(()))
}

/// Per-row byte encoding for a `Delta` column.
fn delta_encode_into(
    column: &ArrayRef,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<Delta>() else {
        return Ok(None);
    };

    // Materialize bases and deltas (these are already primitive arrays).
    let bases = view.bases().clone().execute::<PrimitiveArray>(ctx)?;
    let deltas = view.deltas().clone().execute::<PrimitiveArray>(ctx)?;
    let start = view.offset();
    let total_len = view.as_ref().len();
    let end = start + total_len;

    // Following delta_decompress: validity is transposed on the deltas, untranspose it.
    let validity = untranspose_validity(&deltas.validity()?, ctx)?;
    let validity = validity.slice(start..end)?;

    let descending = field.descending;
    let non_null = field.non_null_sentinel();
    let null = field.null_sentinel();
    let original_ptype = deltas.ptype();
    let value_bytes = original_ptype.byte_width();
    let stride = (1 + value_bytes) as u32;

    let mask = match &validity {
        Validity::NonNullable | Validity::AllValid => None,
        _ => Some(validity.execute_mask(total_len, ctx)?),
    };

    // Operate on the unsigned reinterpretation (matches `delta_decompress`).
    let bases_u = bases.reinterpret_cast(original_ptype.to_unsigned());
    let deltas_u = deltas.reinterpret_cast(original_ptype.to_unsigned());
    let is_signed = original_ptype.is_signed_int();

    match_each_unsigned_integer_ptype!(deltas_u.ptype(), |T| {
        const LANES: usize = T::LANES;
        let buffer = decompress_primitive::<T, LANES>(bases_u.as_slice(), deltas_u.as_slice());
        let slice = &buffer.as_slice()[start..end];
        if is_signed {
            // Reinterpret each unsigned element as its signed counterpart for encoding.
            // SAFETY: `T` and its signed counterpart have the same size and alignment.
            let signed: &[<T as ToSigned>::Signed] = unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr().cast::<<T as ToSigned>::Signed>(),
                    slice.len(),
                )
            };
            encode_primitive_chunk::<<T as ToSigned>::Signed>(
                signed,
                0,
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
        } else {
            encode_primitive_chunk::<T>(
                slice,
                0,
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
        }
    });

    Ok(Some(()))
}

/// Helper trait mapping unsigned types to their signed counterparts so we can encode signed
/// values without losing the sign-bit-flip semantics of `PrimRowEncode`.
trait ToSigned: Copy {
    type Signed: Copy + NativePType + PrimRowEncode;
}
impl ToSigned for u8 {
    type Signed = i8;
}
impl ToSigned for u16 {
    type Signed = i16;
}
impl ToSigned for u32 {
    type Signed = i32;
}
impl ToSigned for u64 {
    type Signed = i64;
}

fn delta_array_id() -> ArrayId {
    use vortex_session::registry::CachedId;
    static ID: CachedId = CachedId::new("fastlanes.delta");
    *ID
}

inventory::submit! {
    RowEncodeRegistration {
        id: delta_array_id,
        size: delta_size_contribution,
        encode: delta_encode_into,
    }
}

// Silence the warning about `Transpose` / `FastLanes` being unused: they are referenced via
// the trait bound chain on `decompress_primitive::<T, LANES>`.
#[allow(dead_code)]
const fn _trait_dep<T: DeltaTrait + Transpose + FastLanes>() {}

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

    use crate::Delta;

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
    fn delta_row_encode_matches_canonical_u64() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let raw = buffer![1u64, 2, 3, 5, 10, 11, 20].into_array();
        let p = PrimitiveArray::from_iter([1u64, 2, 3, 5, 10, 11, 20]);
        let delta = Delta::try_from_primitive_array(&p, &mut ctx)?.into_array();

        let by_raw = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_delta = convert_columns(&[delta], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_raw), collect_rows(&by_delta));
        Ok(())
    }

    #[test]
    fn delta_row_encode_matches_canonical_i64() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let raw = buffer![-3i64, -2, -1, 0, 1, 2].into_array();
        let p = PrimitiveArray::from_iter([-3i64, -2, -1, 0, 1, 2]);
        let delta = Delta::try_from_primitive_array(&p, &mut ctx)?.into_array();

        let by_raw = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_delta = convert_columns(&[delta], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_raw), collect_rows(&by_delta));
        Ok(())
    }

    #[test]
    fn delta_row_encode_multi_chunk_i64() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<i64> = (0..3000).map(|i| 1000 + i as i64 * 3).collect();
        let raw = PrimitiveArray::from_iter(values.clone()).into_array();
        let p = PrimitiveArray::from_iter(values);
        let delta = Delta::try_from_primitive_array(&p, &mut ctx)?.into_array();

        let by_raw = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_delta = convert_columns(&[delta], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_raw), collect_rows(&by_delta));
        Ok(())
    }
}
