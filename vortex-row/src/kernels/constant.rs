// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernels for `ConstantArray`.
//!
//! A constant column holds a single scalar repeated for every row, so the row-encoded bytes
//! are identical for all rows. We encode that scalar once (by canonicalizing a one-element
//! array and reusing the shared codec, so the bytes are byte-identical to the canonical path)
//! and then broadcast it, paying the encode cost once rather than once per row.

#![allow(
    clippy::cast_possible_truncation,
    reason = "row encoding indexes into u32-sized buffers; lengths are validated to fit in u32"
)]

use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_error::VortexResult;

use crate::codec;
use crate::encode::RowEncodeKernel;
use crate::options::RowSortField;
use crate::size::RowSizeKernel;

/// Canonicalize the constant's single scalar into a one-element array and run the shared codec
/// to obtain the per-row encoded length and, when `encode` is true, the encoded bytes.
fn encode_scalar_once(
    column: &ArrayView<'_, Constant>,
    field: RowSortField,
    encode: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(u32, Vec<u8>)> {
    let one = ConstantArray::new(column.scalar().clone(), 1).into_array();
    let canonical = one.execute::<Canonical>(ctx)?;
    let mut size = [0u32; 1];
    codec::field_size(&canonical, field, &mut size, ctx)?;
    let len = size[0];
    let mut buf = Vec::new();
    if encode && len > 0 {
        buf = vec![0u8; len as usize];
        let offsets = [0u32; 1];
        let mut cursors = [0u32; 1];
        codec::field_encode(&canonical, field, &offsets, &mut cursors, &mut buf, ctx)?;
    }
    Ok((len, buf))
}

impl RowSizeKernel for Constant {
    fn row_size_contribution(
        column: ArrayView<'_, Self>,
        field: RowSortField,
        sizes: &mut [u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        let (add, _) = encode_scalar_once(&column, field, false, ctx)?;
        for s in sizes.iter_mut().take(column.len()) {
            *s += add;
        }
        Ok(Some(()))
    }
}

impl RowEncodeKernel for Constant {
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: RowSortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        let (len, bytes) = encode_scalar_once(&column, field, true, ctx)?;
        if len == 0 {
            return Ok(Some(()));
        }
        let len_usize = len as usize;
        for i in 0..column.len() {
            let pos = (offsets[i] + cursors[i]) as usize;
            out[pos..pos + len_usize].copy_from_slice(&bytes);
            cursors[i] += len;
        }
        Ok(Some(()))
    }

    fn row_encode_fixed_arith(
        column: ArrayView<'_, Self>,
        field: RowSortField,
        col_prefix: u32,
        row_stride: u32,
        var_prefix: Option<&[u32]>,
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        let (len, bytes) = encode_scalar_once(&column, field, true, ctx)?;
        let len_usize = len as usize;
        if len == 0 {
            return Ok(Some(()));
        }
        let n = column.len();
        match var_prefix {
            None => {
                for i in 0..n {
                    let pos = (i as u32 * row_stride + col_prefix) as usize;
                    out[pos..pos + len_usize].copy_from_slice(&bytes);
                }
            }
            Some(vp) => {
                for i in 0..n {
                    let pos = (i as u32 * row_stride + col_prefix + vp[i]) as usize;
                    out[pos..pos + len_usize].copy_from_slice(&bytes);
                }
            }
        }
        Ok(Some(()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::listview::ListViewArrayExt;
    use vortex_error::VortexResult;

    use crate::RowSortField;
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
    fn constant_utf8_row_encode_matches_canonical() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // A varlen (Utf8) constant exercises the Constant kernel via the variable-width
        // dispatch path; the equivalent VarBinView falls through to the canonical codec.
        let canonical = VarBinViewArray::from_iter_str(["hello"; 7]).into_array();
        let scalar = canonical.execute_scalar(0, &mut ctx)?;
        let constant = ConstantArray::new(scalar, 7).into_array();

        let by_const = convert_columns(&[constant], &[RowSortField::default()], &mut ctx)?;
        let by_canon = convert_columns(&[canonical], &[RowSortField::default()], &mut ctx)?;
        assert_eq!(collect_rows(&by_const), collect_rows(&by_canon));
        Ok(())
    }
}
