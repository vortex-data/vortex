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
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::visitor::assert_owned_output_needs_no_drop;

/// Zero-sized evidence used to erase failure reduction from infallible owned visits.
#[derive(Clone, Copy, Default)]
struct NoFailure;

impl BitOrAssign for NoFailure {
    fn bitor_assign(&mut self, _rhs: Self) {}
}

/// Decode every input column for one kernel invocation, then store one infallible output per row.
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

/// Decode every input column for one kernel invocation, then store outputs and reduce failures.
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
    Fail: Copy + Default + BitOrAssign,
{
    const { assert_owned_output_needs_no_drop::<Out>() };

    // Keep the vector length at zero until every row succeeds. An unwind then abandons partially
    // initialized spare capacity without treating it as initialized output. The no-drop assertion
    // above proves that no initialized value requires its destructor to run.
    let row_count = args.row_count();
    let mut values = Vec::<Out>::with_capacity(row_count);
    let columns = Args::decode(args, ctx)?;
    let prepared = prepare(Args::constants(&columns));
    let failure;

    {
        let output = &mut values.spare_capacity_mut()[..row_count];

        // When every input stores one value per row, the indexed source removes argument-shape
        // dispatch from the hot loop and lets the lane kernel optimize the traversal as one
        // operation. Keep view construction and its length proof in this branch. Hoisting them
        // through the shared validation helper changed add, subtract, and multiply with
        // batch-constant and per-row arguments from 9.219, 9.229, and 18.94 us to 30.46, 31.11,
        // and 37.73 us on a Ryzen 9 7950X with rustc 1.91.0 and LLVM 21.1.2.
        // Restoring this placement recovered the fast code under the 16-CGU, no-LTO bench profile.
        if let Some(views) = Args::per_row_views(&columns) {
            vortex_ensure!(
                Args::view_lens_match(&views, row_count),
                "a decoded row input does not address exactly {row_count} rows",
            );

            // SAFETY: `view_lens_match` proved every view addresses exactly `row_count` rows
            // immediately above.
            failure = unsafe { Args::indexed_source(views, row_count) }
                .map_checked_into(output, |elements| apply(&prepared, elements));
        } else {
            // A batch-constant input was collapsed to one row during decoding. This path reads that
            // row repeatedly while indexing only the per-row inputs.
            vortex_ensure!(
                Args::decoded_lens_match(&columns, row_count),
                "a decoded row input does not address exactly {row_count} rows",
            );

            // Keep the output-slot iterator as the loop bound. `row_count` is address-taken by the
            // validation error formatting above. With rustc 1.97.1 and LLVM 22.1.6 under 16 CGUs
            // without LTO, indexing `output` by a `0..row_count` range retains an early-exit bounds
            // check and prevents vectorization with batch-constant and per-row arguments. Recheck
            // the optimized IR and those benchmarks before restoring that range loop.
            let mut accumulated = Fail::default();
            for (index, slot) in output.iter_mut().enumerate() {
                let (value, row_failure) = apply(&prepared, Args::get(&columns, index));
                slot.write(value);
                accumulated |= row_failure;
            }
            failure = accumulated;
        }
    }

    // SAFETY: normal completion of either loop initializes every slot in `0..row_count` exactly
    // once, and `values` was allocated with at least `row_count` capacity.
    unsafe { values.set_len(row_count) };

    // Failure evidence is reduced inside the loop so its richer error construction stays cold.
    // Preserve that provenance so batch execution may retry over only valid rows.
    match finish_failure(failure) {
        Ok(()) => Ok(RowExecution::Output(Out::build(values))),
        Err(error) => Ok(RowExecution::DeferredError(error)),
    }
}
