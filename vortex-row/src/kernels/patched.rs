// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernels for `Patched`.
//!
//! Row size is identical to the underlying `inner` array (patches don't change dtype). For
//! row encoding, we first delegate to the inner array's row-encode path, then overlay each
//! patched row's value directly into the output, overwriting the few bytes that the inner
//! encoder wrote at that row's slot.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "row encoding indexes into u32-sized buffers; lengths are validated to fit in u32"
)]

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::Patched;
use vortex_array::arrays::patched::PatchedArrayExt;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::codec::RowEncode;
use crate::encode::RowEncodeKernel;
use crate::encode::dispatch_encode;
use crate::options::SortField;
use crate::size::RowSizeKernel;
use crate::size::dispatch_size;

impl RowSizeKernel for Patched {
    fn row_size_contribution(
        column: ArrayView<'_, Self>,
        field: SortField,
        sizes: &mut [u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        // Per-row size matches the inner array; patches share its dtype.
        dispatch_size(column.inner(), field, sizes, ctx)?;
        Ok(Some(()))
    }
}

impl RowEncodeKernel for Patched {
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: SortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        let DType::Primitive(ptype, _) = *column.as_ref().dtype() else {
            return Ok(None);
        };
        let value_bytes = ptype.byte_width();

        // Snapshot per-row write start positions before the inner encoder advances cursors.
        let pre_cursors: Vec<u32> = cursors.to_vec();
        dispatch_encode(column.inner(), field, offsets, cursors, out, ctx)?;

        overlay_patches(
            column,
            ptype,
            value_bytes,
            field,
            offsets,
            &pre_cursors,
            out,
            ctx,
        )?;
        Ok(Some(()))
    }
}

/// Overlay patch values onto rows whose inner-encoded bytes need to be replaced.
#[allow(clippy::too_many_arguments)]
fn overlay_patches(
    column: ArrayView<'_, Patched>,
    ptype: vortex_array::dtype::PType,
    value_bytes: usize,
    field: SortField,
    offsets: &[u32],
    pre_cursors: &[u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let patch_indices: PrimitiveArray = column
        .patch_indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    if patch_indices.is_empty() {
        return Ok(());
    }
    let patch_values: PrimitiveArray = column
        .patch_values()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let lane_offsets: PrimitiveArray = column
        .lane_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let patch_indices_slice: &[u16] = patch_indices.as_slice();
    let lane_offsets_slice: &[u32] = lane_offsets.as_slice();
    let n_lanes = column.n_lanes();
    let patched_offset = column.offset();
    let array_len = column.as_ref().len();
    let n_chunks = (array_len + patched_offset).div_ceil(1024);
    let non_null = field.non_null_sentinel();
    let descending = field.descending;

    match_each_native_ptype!(ptype, |T| {
        let values_slice: &[T] = patch_values.as_slice();
        overlay_chunks::<T>(
            values_slice,
            patch_indices_slice,
            lane_offsets_slice,
            n_lanes,
            patched_offset,
            array_len,
            n_chunks,
            offsets,
            pre_cursors,
            out,
            value_bytes,
            non_null,
            descending,
        );
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn overlay_chunks<T: Copy + RowEncode>(
    values_slice: &[T],
    patch_indices_slice: &[u16],
    lane_offsets_slice: &[u32],
    n_lanes: usize,
    patched_offset: usize,
    array_len: usize,
    n_chunks: usize,
    offsets: &[u32],
    pre_cursors: &[u32],
    out: &mut [u8],
    value_bytes: usize,
    non_null: u8,
    descending: bool,
) {
    for chunk in 0..n_chunks {
        for lane in 0..n_lanes {
            let slot = chunk * n_lanes + lane;
            if slot + 1 >= lane_offsets_slice.len() {
                break;
            }
            let start = lane_offsets_slice[slot] as usize;
            let stop = lane_offsets_slice[slot + 1] as usize;
            for k in start..stop {
                let chunk_local = patch_indices_slice[k] as usize;
                let logical_idx = chunk * 1024 + chunk_local;
                if logical_idx < patched_offset {
                    continue;
                }
                let row = logical_idx - patched_offset;
                if row >= array_len {
                    continue;
                }
                let slot_start = (offsets[row] + pre_cursors[row]) as usize;
                out[slot_start] = non_null;
                values_slice[k].encode_to(
                    &mut out[slot_start + 1..slot_start + 1 + value_bytes],
                    descending,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::Patched;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::listview::ListViewArrayExt;
    use vortex_array::patches::Patches;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::SortField;
    use crate::convert_columns;

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
    fn patched_row_encode_matches_canonical() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let inner = buffer![0u32; 32].into_array();
        let patches = Patches::new(
            32,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![100u32, 200, 300].into_array(),
            None,
        )?;
        let patched = Patched::from_array_and_patches(inner, &patches, &mut ctx)?.into_array();

        let mut canonical_vals = vec![0u32; 32];
        canonical_vals[1] = 100;
        canonical_vals[2] = 200;
        canonical_vals[3] = 300;
        let canonical = PrimitiveArray::from_iter(canonical_vals).into_array();

        let by_canonical = convert_columns(&[canonical], &[SortField::default()], &mut ctx)?;
        let by_patched = convert_columns(&[patched], &[SortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_canonical), collect_rows(&by_patched));
        Ok(())
    }

    #[test]
    fn patched_row_encode_multi_chunk() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let n: usize = 4096;
        let inner = PrimitiveArray::from_iter(vec![0u32; n]).into_array();
        let indices: Vec<u32> = (0..n as u32).step_by(503).collect();
        let values: Vec<u32> = indices.iter().map(|i| i + 1000).collect();
        let patches = Patches::new(
            n,
            0,
            PrimitiveArray::from_iter(indices.clone()).into_array(),
            PrimitiveArray::from_iter(values.clone()).into_array(),
            None,
        )?;
        let patched = Patched::from_array_and_patches(inner, &patches, &mut ctx)?.into_array();

        let mut canonical_vals = vec![0u32; n];
        for (idx, &i) in indices.iter().enumerate() {
            canonical_vals[i as usize] = values[idx];
        }
        let canonical = PrimitiveArray::from_iter(canonical_vals).into_array();

        let by_canonical = convert_columns(&[canonical], &[SortField::default()], &mut ctx)?;
        let by_patched = convert_columns(&[patched], &[SortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_canonical), collect_rows(&by_patched));
        Ok(())
    }
}
