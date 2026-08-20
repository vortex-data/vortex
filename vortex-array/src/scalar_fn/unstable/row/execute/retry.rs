// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dense owned execution that can request a valid-row retry.
//!
//! [`execute_owned_with_retry`] decodes and evaluates every row while reducing compact failure
//! evidence. Accepted evidence returns the dense output. Rejected evidence discards that output
//! and tells batch execution to retry only the valid rows. Decode, validation, and allocation
//! errors remain terminal.

use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::visitor::assert_owned_output_needs_no_drop;

/// Decode every input column and report rejected failure evidence to the batch executor.
///
/// Returns `None` only when `finish_failure` rejects the reduced evidence. Decode, validation, and
/// allocation errors remain ordinary [`VortexResult`] errors.
pub(crate) fn execute_owned_with_retry<Args, Out, Prepared, Fail>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
    finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
) -> VortexResult<Option<ArrayRef>>
where
    Args: IndexedElementTuple,
    Out: OutputElement,
    Fail: FailureEvidence,
{
    // Keep this loop separate from `execute_owned`. Returning its state through a shared helper
    // changes the optimized dense kernel even when the helper is inlined.
    const { assert_owned_output_needs_no_drop::<Out>() };

    let columns = Args::decode(args, ctx)?;
    let prepared = prepare(Args::const_values(&columns));

    let row_count = args.row_count();
    let mut values = Vec::<Out>::with_capacity(row_count);
    let output = &mut values.spare_capacity_mut()[..row_count];

    let failure_evidence = if let Some(views) = Args::views_if_no_consts(&columns) {
        // Keep this validation beside the views so LLVM sees their common length here.
        vortex_ensure!(
            Args::view_lens_match(&views, row_count),
            "a decoded row input does not address exactly {row_count} rows",
        );

        // SAFETY: `view_lens_match` checked that these exact retained views address `row_count`
        // rows.
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

        let mut accumulated_failure = Fail::default();

        // Iterate over `output` directly. A `0..row_count` range reuses the address-taken value
        // from the validation error formatter and retains an output bounds check.
        for (index, slot) in output.iter_mut().enumerate() {
            // LLVM unswitches the batch-constant checks in `Args::get` before vectorizing the loop.
            let (value, row_failure) = apply(&prepared, Args::get(&columns, index));

            slot.write(value);
            accumulated_failure |= row_failure;
        }

        accumulated_failure
    };

    // SAFETY: normal completion of either execution path initializes `0..row_count` exactly
    // once, and `values` was allocated with at least `row_count` capacity.
    unsafe { values.set_len(row_count) };

    match finish_failure(failure_evidence) {
        Ok(()) => Ok(Some(Out::build(values))),
        Err(_) => Ok(None),
    }
}
