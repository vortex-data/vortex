// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! User-facing entry point: turn N columnar arrays into one row-encoded `ListView<u8>`.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListViewArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::encode::RowEncode;
use crate::options::RowEncodingOptions;
use crate::options::RowSortField;
use crate::size::RowSize;

/// Encodes N columnar arrays into a single row-oriented [`ListViewArray`] of `u8` whose row
/// byte slices compare lexicographically in the same order as a tuple comparison of the input
/// values under the configured [`RowSortField`]s.
///
/// Construct with [`RowEncoder::new`] or [`RowEncoder::with_options`] to pin the per-column
/// sort options, or use [`RowEncoder::default`] to apply ascending, nulls-first ordering to
/// every column. The same encoder can be reused across calls.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RowEncoder {
    options: Option<RowEncodingOptions>,
}

impl RowEncoder {
    /// Construct a `RowEncoder` from one [`RowSortField`] per input column.
    pub fn new(fields: impl IntoIterator<Item = RowSortField>) -> Self {
        Self {
            options: Some(RowEncodingOptions::new(fields)),
        }
    }

    /// Construct a `RowEncoder` from an explicit [`RowEncodingOptions`].
    pub fn with_options(options: RowEncodingOptions) -> Self {
        Self {
            options: Some(options),
        }
    }

    /// Borrow the configured options, or `None` when the encoder applies default
    /// (ascending, nulls-first) ordering inferred from the column count at encode time.
    pub fn options(&self) -> Option<&RowEncodingOptions> {
        self.options.as_ref()
    }

    /// Encode `cols` into a single row-oriented [`ListViewArray`] of `u8`.
    pub fn encode(&self, cols: &[ArrayRef], ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
        let (options, args) = self.prepare(cols)?;
        RowEncode
            .execute(&options, &args, ctx)?
            .execute::<ListViewArray>(ctx)
    }

    /// Compute only the per-row sizes (the `Struct { fixed: u32, var: u32 }` produced by
    /// [`RowSize`]) without materializing the encoded rows.
    pub fn row_sizes(&self, cols: &[ArrayRef], ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let (options, args) = self.prepare(cols)?;
        RowSize.execute(&options, &args, ctx)
    }

    /// Validate the input columns and resolve the options + execution args shared by
    /// [`encode`](Self::encode) and [`row_sizes`](Self::row_sizes).
    fn prepare(&self, cols: &[ArrayRef]) -> VortexResult<(RowEncodingOptions, VecExecutionArgs)> {
        if cols.is_empty() {
            vortex_bail!("RowEncoder: at least one column is required");
        }
        let options = match &self.options {
            Some(options) => {
                if options.len() != cols.len() {
                    vortex_bail!(
                        "RowEncoder: options describe {} columns but {} were provided",
                        options.len(),
                        cols.len()
                    );
                }
                options.clone()
            }
            None => RowEncodingOptions::default_for_columns(cols.len()),
        };
        let nrows = cols[0].len();
        for (i, col) in cols.iter().enumerate() {
            reject_extension_dtype(col.dtype())?;
            if col.len() != nrows {
                vortex_bail!(
                    "RowEncoder: column {} has length {} but expected {}",
                    i,
                    col.len(),
                    nrows
                );
            }
        }
        Ok((options, VecExecutionArgs::new(cols.to_vec(), nrows)))
    }
}

fn reject_extension_dtype(dtype: &DType) -> VortexResult<()> {
    match dtype {
        DType::Extension(ext_dtype) => {
            vortex_bail!(
                "row encoding does not support Extension arrays yet: {}",
                ext_dtype.id()
            )
        }
        DType::Struct(fields, _) => {
            for field_dtype in fields.fields() {
                reject_extension_dtype(&field_dtype)?;
            }
        }
        DType::FixedSizeList(elem, ..) | DType::List(elem, _) => {
            reject_extension_dtype(elem)?;
        }
        _ => {}
    }
    Ok(())
}

/// Convert N columnar arrays into a single row-oriented [`ListViewArray`] of `u8` whose bytes
/// are lexicographically comparable in the same order as a tuple comparison of the input
/// values according to `fields`. Convenience wrapper over [`RowEncoder::encode`].
pub fn convert_columns(
    cols: &[ArrayRef],
    fields: &[RowSortField],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    RowEncoder::new(fields.iter().copied()).encode(cols, ctx)
}

/// Like [`convert_columns`] but takes a prebuilt [`RowEncodingOptions`].
pub fn convert_columns_with_options(
    cols: &[ArrayRef],
    options: &RowEncodingOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    RowEncoder::with_options(options.clone()).encode(cols, ctx)
}

/// Compute only the per-row sizes (in bytes) of the row-encoded form for N columns.
/// Convenience wrapper over [`RowEncoder::row_sizes`].
pub fn compute_row_sizes(
    cols: &[ArrayRef],
    fields: &[RowSortField],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    RowEncoder::new(fields.iter().copied()).row_sizes(cols, ctx)
}

/// Like [`compute_row_sizes`] but takes a prebuilt [`RowEncodingOptions`].
pub fn compute_row_sizes_with_options(
    cols: &[ArrayRef],
    options: &RowEncodingOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    RowEncoder::with_options(options.clone()).row_sizes(cols, ctx)
}
