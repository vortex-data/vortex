// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Executes row kernels that return one independent owned value per row.
//!
//! [`execute_owned`] decodes inputs once, prepares constant state, writes into spare vector
//! capacity, and reduces compact failure evidence without putting error construction in the hot
//! loop. [`execute_owned_infallible`] removes that failure path for infallible kernels.

use std::ops::BitOrAssign;

use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::RowExecution;
use crate::ExecutionCtx;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::ViewLen;
use crate::scalar_fn::unstable::row::visitor::assert_owned_output_needs_no_drop;

/// Zero-sized failure accumulator for infallible owned visits.
#[derive(Clone, Copy, Default)]
struct NoFailure;

impl BitOrAssign for NoFailure {
    fn bitor_assign(&mut self, _rhs: Self) {}
}

/// Decode every input column, then store one output per row from an infallible kernel.
pub(crate) fn execute_owned_infallible<Args, Out, Prepared>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> Out,
) -> VortexResult<RowExecution>
where
    Args: IndexedElementTuple,
    Out: OutputElement,
{
    execute_owned::<Args, Out, Prepared, NoFailure>(
        args,
        ctx,
        prepare,
        move |prepared, args| (apply(prepared, args), NoFailure),
        |_| Ok(()),
    )
}

/// Decode every input column, then store outputs and combine per-row failure evidence.
pub(crate) fn execute_owned<Args, Out, Prepared, Fail>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
    finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
) -> VortexResult<RowExecution>
where
    Args: IndexedElementTuple,
    Out: OutputElement,
    Fail: FailureEvidence,
{
    // The output vector stays at length zero until every slot is initialized so that an unwind
    // abandons partially initialized spare capacity. This no-drop assertion proves that no
    // initialized value requires a destructor to run.
    const { assert_owned_output_needs_no_drop::<Out>() };

    let columns = Args::decode(args, ctx)?;
    let prepared = prepare(Args::const_values(&columns));

    let row_count = args.row_count();
    let mut values = Vec::<Out>::with_capacity(row_count);
    let output = &mut values.spare_capacity_mut()[..row_count];

    let failure = if let Some(views) = Args::views_if_no_consts(&columns) {
        // Keep this validation beside the views so LLVM sees their common length here.
        vortex_ensure!(
            Args::ARITY == 0 || views.len() == row_count,
            "a decoded row input does not address exactly {row_count} rows",
        );

        // SAFETY: the tuple length check proved every non-nullary view addresses exactly
        // `row_count` rows immediately above. Nullary tuples do not access an input view.
        let source = unsafe { Args::indexed_source(views, row_count) };

        source.map_checked_into(output, |elements| apply(&prepared, elements))
    } else {
        // Keep this proof branch-local. Shared validation prevents LLVM from specializing this
        // loop for each batch-constant arrangement, leaving it scalar under multiple CGUs without
        // LTO.
        vortex_ensure!(
            Args::decoded_lens_match(&columns, row_count),
            "a decoded row input does not address exactly {row_count} rows",
        );

        let mut accumulated = Fail::default();

        // Iterate over `output` directly. A `0..row_count` range reuses the address-taken value
        // from the validation error formatter and retains an output bounds check.
        for (index, slot) in output.iter_mut().enumerate() {
            // LLVM unswitches the batch-constant checks in `Args::get` before vectorizing the loop.
            let (value, row_failure) = apply(&prepared, Args::get(&columns, index));

            slot.write(value);
            accumulated |= row_failure;
        }

        accumulated
    };

    // SAFETY: normal completion of either execution path initializes `0..row_count` exactly
    // once, and `values` was allocated with at least `row_count` capacity.
    unsafe { values.set_len(row_count) };

    // Defer failures so batch execution can retry with only valid rows.
    match finish_failure(failure) {
        Ok(()) => Ok(RowExecution::Output(Out::build(values))),
        Err(error) => Ok(RowExecution::DeferredError(error)),
    }
}
