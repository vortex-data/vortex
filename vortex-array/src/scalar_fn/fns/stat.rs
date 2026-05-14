// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function implementation for aggregate-backed stat expressions.

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::stats::legacy::legacy_stat_for_aggregate;

/// Options for the `stat` scalar function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StatOptions {
    aggregate_fn: AggregateFnRef,
}

impl StatOptions {
    /// Creates options for the provided aggregate statistic.
    pub fn new(aggregate_fn: AggregateFnRef) -> Self {
        Self { aggregate_fn }
    }

    /// Returns the aggregate function backing this statistic lookup.
    pub fn aggregate_fn(&self) -> &AggregateFnRef {
        &self.aggregate_fn
    }
}

impl Display for StatOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.aggregate_fn, f)
    }
}

/// Scalar function that broadcasts a stored aggregate statistic over the input rows.
///
/// The only current consumer is **row-wise pruning**: substituting `stat(col, agg)` into a
/// predicate produces a cheap, row-aligned approximation whose constant runs let downstream
/// filters drop entire stretches at once. For example, `value < 10` is prunable as
/// `stat(value, max) < 10` (rows where the bound is false are guaranteed false) or
/// `stat(value, min) >= 10` (rows where it is true are guaranteed true) — the zone-map /
/// min-max-index pattern, expressed as an ordinary expression so the existing scalar
/// machinery can rewrite, fold, and execute it.
///
/// The result is row-aligned with the input, at whatever granularity the input carries the
/// stat at: e.g. a flat array yields a single broadcast `ConstantArray`; a chunked array
/// yields a constant per chunk; a zone-mapped array would yield a run-end-encoded array,
/// one run per zone. If the requested stat is not available, the result is a null constant.
///
/// Pruning only makes sense for aggregates that bound individual rows — `min`, `max`,
/// `has_nulls`, bloom filters, etc. Non-idempotent aggregates like `sum`, `count`, `mean`,
/// `null_count`, and `nan_count` still produce a meaningful per-chunk value but do **not**
/// bound any single row.
#[derive(Clone)]
pub struct StatFn;

impl ScalarFnVTable for StatFn {
    type Options = StatOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.stat")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Stat expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "stat(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", {})", options.aggregate_fn())
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        stat_dtype(options.aggregate_fn(), &arg_dtypes[0])
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let dtype = stat_dtype(options.aggregate_fn(), input.dtype())?;
        stat_array(&input, options.aggregate_fn(), dtype, args.row_count())
    }
}

fn stat_dtype(aggregate_fn: &AggregateFnRef, input_dtype: &DType) -> VortexResult<DType> {
    let Some(dtype) = aggregate_fn.return_dtype(input_dtype) else {
        vortex_bail!(
            "Aggregate function {} does not support input dtype {}",
            aggregate_fn,
            input_dtype
        );
    };
    Ok(dtype.as_nullable())
}

fn stat_array(
    array: &ArrayRef,
    aggregate_fn: &AggregateFnRef,
    dtype: DType,
    len: usize,
) -> VortexResult<ArrayRef> {
    let value = legacy_stat_for_aggregate(aggregate_fn)
        .and_then(|stat| {
            array
                .statistics()
                .with_typed_stats_set(|stats| stats.get(stat))
        })
        // We don't mind whether the stat is approxed or not, since these are row-wise bounds
        .map(|stat| stat.into_inner())
        .and_then(Scalar::into_value);

    let scalar = Scalar::try_new(dtype, value)?;
    Ok(ConstantArray::new(scalar, len).into_array())
}
