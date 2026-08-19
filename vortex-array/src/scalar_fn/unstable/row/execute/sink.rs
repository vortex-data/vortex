// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executes row kernels that write through an [`OutputSink`].
//!
//! Dense execution visits every row. Skip-invalid execution initializes skipped output rows and
//! visits only rows that are valid in every input. Skip-invalid execution declines when either the
//! input representation or sink cannot support that path.

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::SinkResult;
use crate::scalar_fn::unstable::row::ViewLen;

/// Decode inputs once, then write one sink row for each input row.
///
/// The executor owns the sink and passes each output row to `apply`. This keeps `apply` as [`Fn`].
/// Capturing the sink would require [`FnMut`] and put its buffer metadata behind loop-carried
/// mutable closure state, which can prevent LLVM from treating that metadata as loop-invariant.
pub(crate) fn execute_sink<Args, Prepared, Sink, ApplyResult, Options>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>, <Sink as OutputSink<Options>>::Row<'_>) -> ApplyResult,
) -> VortexResult<ArrayRef>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
    ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>,
{
    let columns = Args::decode(args, ctx)?;

    let row_count = args.row_count();
    let const_values = Args::const_values(&columns);
    let prepared = prepare(const_values);

    let mut sink = <Sink as OutputSink<Options>>::with_capacity(row_count)?;

    // Keep `rows` scoped so its borrow ends before `finish`, which consumes the sink.
    {
        let mut rows = <Sink as OutputSink<Options>>::rows(&mut sink);

        // This equality proves to LLVM that `0..row_count` is in bounds for `rows`.
        let sink_row_count = rows.len();
        vortex_ensure_eq!(
            sink_row_count,
            row_count,
            "the output sink must address exactly {row_count} rows, got {sink_row_count}",
        );

        let views = Args::views_if_no_consts(&columns);
        if let Some(views) = views {
            if !Args::view_lens_match(&views, row_count) {
                decoded_length_error(row_count)?;
            }

            for index in 0..row_count {
                // SAFETY: `view_lens_match` checked that these exact retained views address
                // `row_count` rows before the loop.
                let elements = unsafe { Args::get_from_views_unchecked(&views, index) };
                // SAFETY: the sink row-count check above proved every loop index is in bounds.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };

                apply(&prepared, elements, output).into_result()?;
            }
        } else {
            if !Args::decoded_lens_match(&columns, row_count) {
                decoded_length_error(row_count)?;
            }

            for index in 0..row_count {
                // SAFETY: the sink row-count check above proved every loop index is in bounds.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };

                // LLVM unswitches the batch-constant checks in `Args::get` before vectorizing the
                // loop.
                apply(&prepared, Args::get(&columns, index), output).into_result()?;
            }
        }
    }

    // SAFETY: every row callback completed successfully, so each returned the required write token.
    unsafe { <Sink as OutputSink<Options>>::finish(sink) }
}

/// Write only the rows set in `valid`, or decline when the inputs or sink cannot support
/// skip-invalid execution.
///
/// `Ok(None)` signals that direct skip-invalid execution is unavailable. Batch execution decides
/// how to handle the decline.
pub(crate) fn execute_sink_valid_rows<Args, Prepared, Sink, ApplyResult, Options>(
    args: &dyn ExecutionArgs,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>, <Sink as OutputSink<Options>>::Row<'_>) -> ApplyResult,
) -> VortexResult<Option<ArrayRef>>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
    ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>,
{
    let Some(ValidRowsSetup {
        initialize_skipped_rows,
        columns,
        valid_rows,
        row_count,
        mut sink,
    }) = setup_sink_valid_rows::<Args, Sink, Options>(args, valid, ctx)?
    else {
        return Ok(None);
    };

    let views = Args::views_if_no_consts(&columns);
    let const_values = Args::const_values(&columns);
    let prepared = prepare(const_values);

    // Keep `rows` scoped so its borrow ends before `finish`. With multiple CGUs and no LTO, using
    // `drop(rows)` duplicates `Args::get` in every sparse callback.
    {
        // Initialize every slot before visiting only valid rows.
        let mut rows = <Sink as OutputSink<Options>>::rows(&mut sink);
        initialize_skipped_rows(&mut rows);

        // The initializer can change addressability. Recheck it so LLVM can prove every mask
        // index is in bounds.
        let initialized_row_count = rows.len();
        vortex_ensure_eq!(
            initialized_row_count,
            row_count,
            "the initialized output sink must address exactly {row_count} rows, got {initialized_row_count}",
        );

        if let Some(views) = views {
            if !Args::view_lens_match(&views, row_count) {
                decoded_length_error(row_count)?;
            }

            valid_rows.try_for_each_set_index(|index| {
                // SAFETY: the post-initialization row-count check proved that the sink addresses
                // every mask index, which is below the mask's validated `row_count`.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };

                // SAFETY: `view_lens_match` checked that these exact retained views address
                // `row_count` rows, and mask indices are below `row_count`.
                let elements = unsafe { Args::get_from_views_unchecked(&views, index) };

                apply(&prepared, elements, output).into_result()
            })?;
        } else {
            if !Args::decoded_lens_match(&columns, row_count) {
                decoded_length_error(row_count)?;
            }

            valid_rows.try_for_each_set_index(|index| {
                // SAFETY: the post-initialization row-count check proved that the sink addresses
                // every mask index, which is below the mask's validated `row_count`.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };

                apply(&prepared, Args::get(&columns, index), output).into_result()
            })?;
        }
    }

    // SAFETY: the initializer completed before traversal, and every visited callback completed
    // successfully and returned the required write token.
    unsafe { <Sink as OutputSink<Options>>::finish(sink) }.map(Some)
}

/// Construct a decoded-length error outside the traversal branches.
///
/// Owned execution (`owned.rs`) derives its index from an output-slice iterator. Sink execution
/// only has indexed row access, so `row_count` remains the loop bound. Formatting the error inside
/// either branch takes the address of that bound and prevents LLVM from vectorizing some sink
/// loops.
#[cold]
#[inline(never)]
fn decoded_length_error(row_count: usize) -> VortexResult<()> {
    vortex_bail!("a decoded row input does not address exactly {row_count} rows")
}

/// State resolved before preparing the skip-invalid row loop.
struct ValidRowsSetup<'valid, Args, Sink, Options>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
{
    initialize_skipped_rows: for<'rows> fn(&mut <Sink as OutputSink<Options>>::Rows<'rows>),
    columns: Args::Columns,
    valid_rows: &'valid BitBuffer,
    row_count: usize,
    sink: Sink,
}

/// Resolve the capabilities, inputs, sink, and validity mask for skip-invalid execution.
fn setup_sink_valid_rows<'valid, Args, Sink, Options>(
    args: &dyn ExecutionArgs,
    valid: &'valid Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ValidRowsSetup<'valid, Args, Sink, Options>>>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
{
    // The initializer both declares support for skipping rows and initializes those rows.
    let Some(initialize_skipped_rows) = <Sink as OutputSink<Options>>::skipped_rows_initializer()
    else {
        return Ok(None);
    };

    // Null-tolerant decoding exposes values behind nulls without filtering. Decline when any input
    // cannot provide those values safely.
    let Some(columns) = Args::decode_null_tolerant(args, ctx)? else {
        return Ok(None);
    };

    let row_count = args.row_count();

    // Keep allocation before the validity and length checks. With multiple CGUs and no LTO,
    // moving it later inlines `Args::get` into every sparse callback, duplicating its bounds
    // checks.
    let sink = <Sink as OutputSink<Options>>::with_capacity(row_count)?;

    // Batch execution resolves all-valid and all-null inputs before selecting this path.
    let AllOr::Some(valid_rows) = valid.bit_buffer() else {
        vortex_bail!(
            "execute_sink_valid_rows requires valid and invalid rows, got an all-valid or all-invalid mask"
        );
    };
    vortex_ensure_eq!(
        valid_rows.len(),
        row_count,
        "the validity mask must address exactly {row_count} rows, got {}",
        valid_rows.len(),
    );

    Ok(Some(ValidRowsSetup {
        initialize_skipped_rows,
        columns,
        valid_rows,
        row_count,
        sink,
    }))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::execute_sink_valid_rows;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::NativePType;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::VecExecutionArgs;
    use crate::scalar_fn::unstable::row::OutputSink;
    use crate::validity::Validity;

    struct NonSkippingSink;

    struct ShrinkingSink(Vec<i64>);

    // SAFETY: `with_capacity` always returns an error, so no sink value can reach `rows`, `row`, or
    // `finish` through the executor. The row-initialization requirements are therefore vacuous.
    unsafe impl<Options> OutputSink<Options> for NonSkippingSink {
        type Rows<'a> = ();
        type Row<'a> = ();
        type WriteToken = ();

        fn return_dtype(_options: &Options) -> VortexResult<DType> {
            Ok(DType::from(i64::PTYPE))
        }

        fn with_capacity(_rows: usize) -> VortexResult<Self> {
            Err(vortex_err!(
                "a non-skipping sink must decline before allocation"
            ))
        }

        fn rows(&mut self) -> Self::Rows<'_> {}

        unsafe fn row_unchecked<'a>(_rows: &'a mut Self::Rows<'_>, _index: usize) -> Self::Row<'a> {
        }

        unsafe fn finish(self) -> VortexResult<ArrayRef> {
            Err(vortex_err!("a non-skipping sink must not finish"))
        }
    }

    // SAFETY: the initializer deliberately shrinks the row collection to exercise the executor's
    // post-initialization length check. If execution incorrectly continues, safe indexing in
    // `row_unchecked` panics instead of accessing invalid memory.
    unsafe impl<Options> OutputSink<Options> for ShrinkingSink {
        type Rows<'a> = &'a mut Vec<i64>;
        type Row<'a> = &'a mut i64;
        type WriteToken = ();

        fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
            Some(|rows| {
                rows.pop();
            })
        }

        fn return_dtype(_options: &Options) -> VortexResult<DType> {
            Ok(DType::from(i64::PTYPE))
        }

        fn with_capacity(rows: usize) -> VortexResult<Self> {
            Ok(Self(vec![0; rows]))
        }

        fn rows(&mut self) -> Self::Rows<'_> {
            &mut self.0
        }

        unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
            &mut rows[index]
        }

        unsafe fn finish(self) -> VortexResult<ArrayRef> {
            Ok(PrimitiveArray::from_iter(self.0).into_array())
        }
    }

    #[test]
    fn test_non_skipping_sink_declines_before_allocation() -> VortexResult<()> {
        let input = PrimitiveArray::new(vec![1_i64, 2], Validity::NonNullable).into_array();
        let args = VecExecutionArgs::new(vec![input], 2);
        let valid = Mask::from_iter([true, false]);
        let mut ctx = array_session().create_execution_ctx();

        let execution = execute_sink_valid_rows::<(i64,), (), NonSkippingSink, (), EmptyOptions>(
            &args,
            &valid,
            &mut ctx,
            |_| (),
            |_, _, _| (),
        )?;

        assert!(execution.is_none());

        Ok(())
    }

    #[test]
    fn test_skip_invalid_sink_rechecks_rows_after_initialization() -> VortexResult<()> {
        let input = PrimitiveArray::from_iter([10_i64, 20]).into_array();
        let args = VecExecutionArgs::new(vec![input], 2);
        let valid = Mask::from_iter([false, true]);
        let mut ctx = array_session().create_execution_ctx();

        let result = execute_sink_valid_rows::<(i64,), (), ShrinkingSink, (), EmptyOptions>(
            &args,
            &valid,
            &mut ctx,
            |_| (),
            |_, (value,), output| {
                *output = value;
            },
        );

        let error = match result {
            Err(error) => error,
            Ok(_) => vortex_bail!("the sink must reject rows changed by its initializer"),
        };
        assert!(
            error
                .to_string()
                .contains("initialized output sink must address exactly 2 rows, got 1"),
            "unexpected error: {error}",
        );

        Ok(())
    }
}
