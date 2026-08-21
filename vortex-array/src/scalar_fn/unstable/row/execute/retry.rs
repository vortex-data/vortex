// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dense owned execution that can request a valid-row retry.
//!
//! [`execute_owned_dense_attempt`] decodes and evaluates every row while reducing compact failure
//! evidence. Accepted evidence returns dense values. Rejected evidence discards those values and
//! preserves the error while batch execution resolves input validity. Decode, validation, and
//! allocation errors remain terminal.

use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::visitor::assert_owned_output_needs_no_drop;

/// The result of attempting dense execution for a deferred row kernel.
pub(in crate::scalar_fn::unstable::row) enum DenseAttempt {
    /// Dense values whose reduced failure evidence was accepted.
    ///
    /// Batch execution must still validate these values and attach input validity.
    Values(ArrayRef),

    /// An error produced from the reduced failure evidence.
    ///
    /// Batch execution must resolve input validity before deciding whether this error is
    /// observable.
    DeferredError(VortexError),
}

/// Decode every input column and report rejected failure evidence to the batch executor.
///
/// Returns [`DenseAttempt::DeferredError`] only when `finish_failure` rejects the reduced
/// evidence. Decode, validation, and allocation errors remain ordinary [`VortexResult`] errors.
pub(in crate::scalar_fn::unstable::row) fn execute_owned_dense_attempt<Args, Out, Prepared, Fail>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
    finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
) -> VortexResult<DenseAttempt>
where
    Args: IndexedElementTuple,
    Out: OutputElement,
    Fail: FailureEvidence,
{
    // The output vector stays at length zero until every slot is initialized so that an unwind
    // abandons partially initialized spare capacity. This no-drop assertion proves that no
    // initialized value requires a destructor to run.
    const { assert_owned_output_needs_no_drop::<Out>() };

    // Keep this dense row loop separate from `execute_owned`. Factoring their shared state into a
    // helper changes LLVM's optimized dense kernel even when the helper is inlined.
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
        Ok(()) => Ok(DenseAttempt::Values(Out::build(values))),
        Err(error) => Ok(DenseAttempt::DeferredError(error)),
    }
}
