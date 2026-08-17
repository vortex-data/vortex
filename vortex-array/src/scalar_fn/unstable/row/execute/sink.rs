// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executes row kernels that write through an [`OutputSink`].
//!
//! Dense execution visits every row. Skip-invalid execution can instead initialize omitted output
//! positions and visit only rows that are valid in every input, falling back when either the input
//! representation or sink lacks that capability.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::RowExecution;
use crate::ExecutionCtx;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::SinkResult;

fn ensure_decoded_lengths<Args: ElementTuple>(
    columns: &Args::Columns,
    views: Option<&Args::Views<'_>>,
    row_count: usize,
) -> VortexResult<()> {
    let lengths_match = match views {
        Some(views) => Args::view_lens_match(views, row_count),
        None => Args::decoded_lens_match(columns, row_count),
    };
    vortex_ensure!(
        lengths_match,
        "a decoded row input does not address exactly {row_count} rows",
    );

    Ok(())
}

/// Decode every input column and allocate one sink for one kernel invocation.
///
/// The sink lives here rather than in the closure, so `apply` stays [`Fn`] and mutable output state
/// does not need to be captured by the closure.
pub(crate) fn execute_sink<Args, Prepared, Sink, ApplyResult, Options>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>, <Sink as OutputSink<Options>>::Row<'_>) -> ApplyResult,
) -> VortexResult<RowExecution>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
    ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>,
{
    let row_count = args.row_count();
    let mut sink = <Sink as OutputSink<Options>>::with_capacity(row_count)?;
    let columns = Args::decode(args, ctx)?;
    let constants = Args::constants(&columns);
    let views = Args::per_row_views(&columns);
    ensure_decoded_lengths::<Args>(&columns, views.as_ref(), row_count)?;
    let prepared = prepare(constants);

    {
        // Borrow the sink once so its shape and buffer descriptor remain loop invariants. This
        // scope releases the borrow before `finish_sink` consumes the sink.
        let mut rows = <Sink as OutputSink<Options>>::rows(&mut sink);
        let sink_row_count = <Sink as OutputSink<Options>>::row_count(&rows);
        vortex_ensure_eq!(
            sink_row_count,
            row_count,
            "the output sink must address exactly {row_count} rows, got {sink_row_count}",
        );

        // The all-per-row representation removes argument-shape dispatch from the hot loop. The
        // constant-and-per-row path instead reads collapsed batch constants at row zero.
        if let Some(views) = views {
            for index in 0..row_count {
                // SAFETY: `ensure_decoded_lengths` proved every view has `row_count` rows before
                // the loop.
                let elements = unsafe { Args::get_from_views_unchecked(&views, index) };
                // SAFETY: the sink row-count check above proved every loop index is in bounds.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };
                apply(&prepared, elements, output).into_result()?;
            }
        } else {
            for index in 0..row_count {
                // SAFETY: the sink row-count check above proved every loop index is in bounds.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };
                apply(&prepared, Args::get(&columns, index), output).into_result()?;
            }
        }
    }

    finish_sink::<Sink, Options>(sink)
}

/// Run a prepared sink over only the rows set in `valid`, or decline when the sink cannot skip.
pub(crate) fn execute_sink_valid_rows<Args, Prepared, Sink, ApplyResult, Options>(
    args: &dyn ExecutionArgs,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>, <Sink as OutputSink<Options>>::Row<'_>) -> ApplyResult,
) -> VortexResult<Option<RowExecution>>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
    ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>,
{
    // Decline before input decoding or sink allocation when this sink cannot initialize rows that
    // the mask skips. The capability and the operation are the same function pointer.
    let Some(initialize_skipped_rows) = <Sink as OutputSink<Options>>::skipped_rows_initializer()
    else {
        return Ok(None);
    };

    // Null-tolerant decoding exposes values behind nulls without filtering the inputs first. An
    // element representation may decline when it cannot provide those values safely.
    let Some(columns) = Args::decode_null_tolerant(args, ctx)? else {
        return Ok(None);
    };
    let constants = Args::constants(&columns);
    let row_count = args.row_count();
    let mut sink = <Sink as OutputSink<Options>>::with_capacity(row_count)?;

    // Batch execution resolves all-valid and all-null inputs before selecting this path.
    let AllOr::Some(valid) = valid.bit_buffer() else {
        vortex_bail!(
            "execute_sink_valid_rows requires valid and invalid rows, got an all-valid or all-invalid mask"
        );
    };
    vortex_ensure_eq!(
        valid.len(),
        row_count,
        "the validity mask must address exactly {row_count} rows, got {}",
        valid.len(),
    );

    let views = Args::per_row_views(&columns);
    ensure_decoded_lengths::<Args>(&columns, views.as_ref(), row_count)?;
    let prepared = prepare(constants);

    {
        let mut rows = <Sink as OutputSink<Options>>::rows(&mut sink);

        // Initialize every slot before skipping rows. Recheck addressability afterward because the
        // initializer mutably borrows the row representation.
        initialize_skipped_rows(&mut rows);
        let initialized_row_count = <Sink as OutputSink<Options>>::row_count(&rows);
        vortex_ensure_eq!(
            initialized_row_count,
            row_count,
            "the initialized output sink must address exactly {row_count} rows, got {initialized_row_count}",
        );

        // Mask traversal is callback-based and cannot return a `VortexResult`. Record the first
        // immediate error, turn later callbacks into no-ops, and return before finishing the sink.
        let mut error = None;
        valid.for_each_set_index(|index| {
            if error.is_some() {
                return;
            }

            // SAFETY: the post-initialization row-count check proved that the sink addresses every
            // mask index, which is below the mask's validated `row_count`.
            let output = unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };
            let result = match &views {
                Some(views) => {
                    // SAFETY: `ensure_decoded_lengths` proved every view has `row_count` rows, and
                    // mask indices are below `row_count`.
                    let elements = unsafe { Args::get_from_views_unchecked(views, index) };
                    apply(&prepared, elements, output)
                }
                None => apply(&prepared, Args::get(&columns, index), output),
            };
            if let Err(row_error) = result.into_result() {
                error = Some(row_error);
            }
        });

        if let Some(error) = error {
            return Err(error);
        }
    }

    finish_sink::<Sink, Options>(sink).map(Some)
}

fn finish_sink<Sink, Options>(sink: Sink) -> VortexResult<RowExecution>
where
    Sink: OutputSink<Options>,
{
    // SAFETY: callers reach this helper only after every completed callback returned the sink's
    // write token. Skipped-row traversal also ran the sink's initializer before visiting its mask.
    // The sink contract defines how that evidence establishes initialization of its row storage.
    unsafe { <Sink as OutputSink<Options>>::finish(sink) }.map(RowExecution::Output)
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::RowExecution;
    use super::execute_sink_valid_rows;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::NativePType;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::VecExecutionArgs;
    use crate::scalar_fn::unstable::row::InitializedElement;
    use crate::scalar_fn::unstable::row::OutputSink;
    use crate::scalar_fn::unstable::row::UninitElementSink;
    use crate::validity::Validity;

    struct NonSkippingSink;

    struct ShrinkingSink(Vec<i64>);

    // SAFETY: `with_capacity` always returns an error, so no sink value can reach `rows`, `row`, or
    // `finish` through the executor. The row-initialization requirements are therefore vacuous.
    unsafe impl<Options> OutputSink<Options> for NonSkippingSink {
        type Rows<'a> = ();
        type Row<'a> = ();
        type WriteToken = ();

        fn output_dtype(_options: &Options, _args: &[DType]) -> VortexResult<DType> {
            Ok(DType::from(i64::PTYPE))
        }

        fn with_capacity(_rows: usize) -> VortexResult<Self> {
            Err(vortex_err!(
                "a non-skipping sink must decline before allocation"
            ))
        }

        fn rows(&mut self) -> Self::Rows<'_> {}

        fn row_count(_rows: &Self::Rows<'_>) -> usize {
            0
        }

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

        fn output_dtype(_options: &Options, _args: &[DType]) -> VortexResult<DType> {
            Ok(DType::from(i64::PTYPE))
        }

        fn with_capacity(rows: usize) -> VortexResult<Self> {
            Ok(Self(vec![0; rows]))
        }

        fn rows(&mut self) -> Self::Rows<'_> {
            &mut self.0
        }

        fn row_count(rows: &Self::Rows<'_>) -> usize {
            rows.len()
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
    fn test_skip_invalid_sink_initializes_and_writes_addressed_rows() -> VortexResult<()> {
        let input = PrimitiveArray::from_iter([10_i64, 20, 30]).into_array();
        let args = VecExecutionArgs::new(vec![input], 3);
        let valid = Mask::from_iter([true, false, true]);
        let mut ctx = array_session().create_execution_ctx();

        let execution = execute_sink_valid_rows::<
            (i64,),
            (),
            UninitElementSink<i64>,
            InitializedElement,
            EmptyOptions,
        >(
            &args,
            &valid,
            &mut ctx,
            |_| (),
            |_, (value,), output| {
                // SAFETY: `output` is the row supplied to this callback.
                unsafe { InitializedElement::write(output, value * 2) }
            },
        )?;
        let Some(RowExecution::Output(actual)) = execution else {
            vortex_bail!("the skip-invalid sink must produce an output");
        };
        let expected = PrimitiveArray::from_iter([20_i64, 0, 60]);

        assert_arrays_eq!(&actual, expected.as_ref(), &mut ctx);
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
