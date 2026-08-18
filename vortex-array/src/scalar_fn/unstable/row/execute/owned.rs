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
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
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
) -> VortexResult<ArrayRef>
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

/// Decode nullable inputs, then store one output for each valid row from an infallible kernel.
pub(crate) fn execute_owned_infallible_valid_rows<Args, Out, Prepared>(
    args: &dyn ExecutionArgs,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> Out,
) -> VortexResult<Option<ArrayRef>>
where
    Args: IndexedElementTuple,
    Out: OutputElement,
{
    execute_owned_valid_rows::<Args, Out, Prepared, NoFailure>(
        args,
        valid,
        ctx,
        prepare,
        move |prepared, args| (apply(prepared, args), NoFailure),
        |_| Ok(()),
    )
}

/// Decode nullable inputs, then store outputs and combine failure evidence for valid rows.
pub(crate) fn execute_owned_valid_rows<Args, Out, Prepared, Fail>(
    args: &dyn ExecutionArgs,
    valid: &Mask,
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
    const { assert_owned_output_needs_no_drop::<Out>() };

    let Some(columns) = Args::decode_null_tolerant(args, ctx)? else {
        return Ok(None);
    };

    let row_count = args.row_count();
    let AllOr::Some(valid_rows) = valid.bit_buffer() else {
        vortex_bail!(
            "execute_owned_valid_rows requires valid and invalid rows, got an all-valid or all-invalid mask"
        );
    };
    vortex_ensure_eq!(
        valid_rows.len(),
        row_count,
        "the validity mask must address exactly {row_count} rows, got {}",
        valid_rows.len(),
    );

    let prepared = prepare(Args::const_values(&columns));
    let mut values: Vec<Out> = std::iter::repeat_with(Out::default)
        .take(row_count)
        .collect();
    let mut failure = Fail::default();

    if let Some(views) = Args::views_if_no_consts(&columns) {
        vortex_ensure!(
            Args::view_lens_match(&views, row_count),
            "a decoded row input does not address exactly {row_count} rows",
        );

        valid_rows.for_each_set_index(|index| {
            // SAFETY: the tuple-wide length check proved every view has `row_count` rows, and mask
            // indices are below `row_count`. Nullary tuples do not access an input view.
            let elements = unsafe { Args::get_from_views_unchecked(&views, index) };
            let (value, row_failure) = apply(&prepared, elements);

            // SAFETY: the mask length check proved that every set index is below `row_count`.
            unsafe { *values.get_unchecked_mut(index) = value };
            failure |= row_failure;
        });
    } else {
        vortex_ensure!(
            Args::decoded_lens_match(&columns, row_count),
            "a decoded row input does not address exactly {row_count} rows",
        );

        valid_rows.for_each_set_index(|index| {
            let (value, row_failure) = apply(&prepared, Args::get(&columns, index));

            // SAFETY: the mask length check proved that every set index is below `row_count`.
            unsafe { *values.get_unchecked_mut(index) = value };
            failure |= row_failure;
        });
    }

    finish_failure(failure)?;

    Ok(Some(Out::build(values)))
}

/// Decode every input column, then store outputs and combine per-row failure evidence.
pub(crate) fn execute_owned<Args, Out, Prepared, Fail>(
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
    finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
) -> VortexResult<ArrayRef>
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

    // Defer rich error construction until after the row loop.
    finish_failure(failure)?;

    Ok(Out::build(values))
}
