// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMax;
use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::ExtensionAggregate;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;

/// Accumulate the extrema of an extension batch through the session's
/// `ExtensionAggregate<MinMax>` for its type.
///
/// Extension types without one registered have no extrema: their storage order need not be their
/// own.
pub(super) fn accumulate_extension(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    ext_dtype: &ExtDTypeRef,
    batch: &Columnar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let ext_min_max = ExtensionAggregate::for_ext_id(MinMax, ext_dtype.id());
    if ctx
        .session()
        .aggregate_fns()
        .find_plugin(&ext_min_max.id())
        .is_none()
    {
        return Ok(());
    }

    let mut acc = Accumulator::try_new(ext_min_max, *args.options, batch.dtype().clone())?;
    let array = match batch {
        Columnar::Canonical(canonical) => canonical.clone().into_array(),
        Columnar::Constant(constant) => constant.clone().into_array(),
    };
    acc.accumulate(&array, ctx)?;
    partial.merge(args, MinMaxResult::from_scalar(&acc.finish()?)?);
    Ok(())
}

/// Accumulate groups of extension elements through the session's `ExtensionAggregate<MinMax>` for
/// their type, so grouped kernels for the storage encoding fire.
///
/// Returns `None` for other elements, and for extension types without one registered, which have
/// no extrema.
pub(super) fn accumulate_grouped_extension(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    groups: &GroupedArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let DType::Extension(ext_dtype) = groups.elements().dtype() else {
        return Ok(None);
    };
    let ext_min_max = ExtensionAggregate::for_ext_id(MinMax, ext_dtype.id());
    if ctx
        .session()
        .aggregate_fns()
        .find_plugin(&ext_min_max.id())
        .is_none()
    {
        return Ok(None);
    }

    let mut acc = GroupedAccumulator::try_new(ext_min_max, *args.options, args.dtype.clone())?;
    acc.accumulate_grouped_array(groups.clone(), ctx)?;
    acc.flush().map(Some)
}
