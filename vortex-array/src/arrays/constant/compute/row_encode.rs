// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernels for `ConstantArray`.

#![allow(
    clippy::cast_possible_truncation,
    reason = "row encoding indexes into u32-sized buffers; lengths are validated to fit in u32"
)]

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::row::codec;
use crate::row::encode::RowEncodeKernel;
use crate::row::options::SortField;
use crate::row::size::RowSizeKernel;

impl RowSizeKernel for Constant {
    fn row_size_contribution(
        column: ArrayView<'_, Self>,
        field: SortField,
        sizes: &mut [u32],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        let add = codec::encoded_size_for_scalar(column.scalar(), field)?;
        for s in sizes.iter_mut().take(column.len()) {
            *s += add;
        }
        Ok(Some(()))
    }
}

impl RowEncodeKernel for Constant {
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: SortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        let bytes = codec::encode_scalar(column.scalar(), field)?;
        let len = bytes.len();
        let len_u32 = len as u32;
        let n = column.len();
        if len == 0 {
            return Ok(Some(()));
        }
        // SAFETY: bytes is len bytes; offsets[i] + cursors[i] + len <= out.len() by
        // construction of the buffer (the size pass already accounted for this column's
        // contribution). copy_nonoverlapping elides the bounds check + slice creation
        // that copy_from_slice would do per row.
        unsafe {
            let src = bytes.as_ptr();
            let out_ptr = out.as_mut_ptr();
            for i in 0..n {
                let pos = (offsets[i] + cursors[i]) as usize;
                std::ptr::copy_nonoverlapping(src, out_ptr.add(pos), len);
                cursors[i] += len_u32;
            }
        }
        Ok(Some(()))
    }
}
