// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::compress::DEFAULT_DICT12_CONFIG;
use crate::compress::onpair_compress_array;

impl SliceReduce for OnPair {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // OnPair columns are not slice-cheap: the packed token stream is keyed
        // by per-row offsets stored inside the C++ object. We canonicalise the
        // requested range to a VarBinView and re-compress with the same config.
        //
        // For workloads with frequent sub-range scans this round-trip should be
        // replaced by a native `OnPairColumnView::slice` API exposed through
        // the shim; this is tracked as future work.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        slice_with_ctx(array, range, &mut ctx).map(Some)
    }
}

fn slice_with_ctx(
    array: ArrayView<'_, OnPair>,
    range: Range<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let canonical = array
        .array()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_array();
    let sliced = canonical.slice(range)?;
    Ok(onpair_compress_array(&sliced, DEFAULT_DICT12_CONFIG, ctx)?.into_array())
}
