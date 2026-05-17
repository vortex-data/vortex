// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernels for `Patched`.
//!
//! Stubs in this commit return `Ok(None)` so the dispatch loop falls back to
//! canonicalization. The real impls land in a follow-up commit.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::patched::Patched;
use vortex_error::VortexResult;

use crate::encode::RowEncodeKernel;
use crate::options::SortField;
use crate::size::RowSizeKernel;

impl RowSizeKernel for Patched {
    fn row_size_contribution(
        _column: ArrayView<'_, Self>,
        _field: SortField,
        _sizes: &mut [u32],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        Ok(None)
    }
}

impl RowEncodeKernel for Patched {
    fn row_encode_into(
        _column: ArrayView<'_, Self>,
        _field: SortField,
        _offsets: &[u32],
        _cursors: &mut [u32],
        _out: &mut [u8],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>> {
        Ok(None)
    }
}
