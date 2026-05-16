// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernels for `DictArray`.
//!
//! These kernels skip canonicalization by encoding each *unique value* once into a small
//! per-value buffer keyed by code, then materializing the per-row contribution via the codes
//! array. The per-unique-value cost is amortized over the dictionary cardinality rather than
//! the row count.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "row encoding indexes into u32-sized buffers; codes are non-negative indices into the values array"
)]

use vortex_error::VortexResult;

use super::Dict;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::row::encode::RowEncodeKernel;
use crate::row::encode::dispatch_encode;
use crate::row::options::SortField;
use crate::row::size::RowSizeKernel;
use crate::row::size::dispatch_size;

impl RowSizeKernel for Dict {
    fn row_size_contribution(
        column: ArrayView<'_, Self>,
        field: SortField,
        sizes: &mut [u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        if column.values().len() > column.codes().len() {
            return Ok(None);
        }
        let n_values = column.values().len();
        let mut value_sizes = vec![0u32; n_values];
        dispatch_size(column.values(), field, &mut value_sizes, ctx)?;

        let codes_prim = column.codes().clone().execute::<PrimitiveArray>(ctx)?;
        let ptype = codes_prim.ptype();
        match_each_integer_ptype!(ptype, |T| {
            add_codes_sizes::<T>(&codes_prim, &value_sizes, sizes);
        });
        Ok(Some(()))
    }
}

impl RowEncodeKernel for Dict {
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: SortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        if column.values().len() > column.codes().len() {
            return Ok(None);
        }

        let n_values = column.values().len();
        let mut value_sizes = vec![0u32; n_values];
        dispatch_size(column.values(), field, &mut value_sizes, ctx)?;

        // Build per-value offsets and a small contiguous per-value encoded buffer.
        let mut value_offsets = vec![0u32; n_values + 1];
        let mut total: u64 = 0;
        for i in 0..n_values {
            value_offsets[i] = total as u32;
            total += u64::from(value_sizes[i]);
        }
        value_offsets[n_values] = total as u32;

        let mut value_buf = vec![0u8; total as usize];
        // Inner dispatch uses zero base offsets (small buffer) with per-value start cursors.
        let zero_offsets = vec![0u32; n_values];
        let mut inner_cursors = value_offsets[..n_values].to_vec();
        dispatch_encode(
            column.values(),
            field,
            &zero_offsets,
            &mut inner_cursors,
            &mut value_buf,
            ctx,
        )?;

        let codes_prim = column.codes().clone().execute::<PrimitiveArray>(ctx)?;
        let ptype = codes_prim.ptype();
        match_each_integer_ptype!(ptype, |T| {
            copy_codes::<T>(
                &codes_prim,
                &value_buf,
                &value_offsets,
                &value_sizes,
                offsets,
                cursors,
                out,
            );
        });
        Ok(Some(()))
    }
}

#[inline]
fn add_codes_sizes<T>(codes: &PrimitiveArray, value_sizes: &[u32], sizes: &mut [u32])
where
    T: NativePType + Copy + TryInto<usize>,
{
    let slice: &[T] = codes.as_slice();
    debug_assert_eq!(slice.len(), sizes.len());
    if T::PTYPE == PType::U8 {
        // SAFETY: T == u8
        let raw = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len()) };
        for (i, &c) in raw.iter().enumerate() {
            sizes[i] += value_sizes[c as usize];
        }
        return;
    }
    for (i, &c) in slice.iter().enumerate() {
        let idx: usize = c
            .try_into()
            .unwrap_or_else(|_| vortex_error::vortex_panic!("dict code does not fit in usize"));
        sizes[i] += value_sizes[idx];
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn copy_codes<T>(
    codes: &PrimitiveArray,
    value_buf: &[u8],
    value_offsets: &[u32],
    value_sizes: &[u32],
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
) where
    T: NativePType + Copy + TryInto<usize>,
{
    let slice: &[T] = codes.as_slice();
    debug_assert_eq!(slice.len(), cursors.len());
    for (i, &c) in slice.iter().enumerate() {
        let idx: usize = c
            .try_into()
            .unwrap_or_else(|_| vortex_error::vortex_panic!("dict code does not fit in usize"));
        let v_start = value_offsets[idx] as usize;
        let v_size = value_sizes[idx] as usize;
        let dst = (offsets[i] + cursors[i]) as usize;
        out[dst..dst + v_size].copy_from_slice(&value_buf[v_start..v_start + v_size]);
        cursors[i] += v_size as u32;
    }
}
