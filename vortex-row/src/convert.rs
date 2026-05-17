// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! User-facing entry point: turn N columnar arrays into one row-encoded `ListView<u8>`.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListViewArray;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::encode::RowEncode;
use crate::options::RowEncodeOptions;
use crate::options::SortField;
use crate::size::RowSize;

/// Convert N columnar arrays into a single row-oriented [`ListViewArray`] of `u8` whose
/// bytes are lexicographically comparable in the same order as a tuple comparison of the
/// input values according to `fields`.
pub fn convert_columns(
    cols: &[ArrayRef],
    fields: &[SortField],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    if cols.len() != fields.len() {
        vortex_bail!(
            "convert_columns: cols.len() ({}) does not match fields.len() ({})",
            cols.len(),
            fields.len()
        );
    }
    if cols.is_empty() {
        vortex_bail!("convert_columns: at least one column is required");
    }
    let nrows = cols[0].len();
    for (i, col) in cols.iter().enumerate() {
        if col.len() != nrows {
            vortex_bail!(
                "convert_columns: column {} has length {} but expected {}",
                i,
                col.len(),
                nrows
            );
        }
    }

    let options = RowEncodeOptions::new(fields.to_vec());
    let args = VecExecutionArgs::new(cols.to_vec(), nrows);
    let result = RowEncode.execute(&options, &args, ctx)?;
    result.execute::<ListViewArray>(ctx)
}

/// Compute only the per-row sizes (in bytes) of the row-encoded form for N columns.
pub fn compute_row_sizes(
    cols: &[ArrayRef],
    fields: &[SortField],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if cols.len() != fields.len() {
        vortex_bail!(
            "compute_row_sizes: cols.len() ({}) does not match fields.len() ({})",
            cols.len(),
            fields.len()
        );
    }
    if cols.is_empty() {
        vortex_bail!("compute_row_sizes: at least one column is required");
    }
    let nrows = cols[0].len();
    let options = RowEncodeOptions::new(fields.to_vec());
    let args = VecExecutionArgs::new(cols.to_vec(), nrows);
    RowSize.execute(&options, &args, ctx)
}
