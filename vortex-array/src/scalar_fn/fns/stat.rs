// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function implementation for aggregate-backed stat expressions.

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::display::ExprDisplay;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

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

/// A reference to stored aggregate metadata, broadcast over the input rows.
///
/// File and zone binders replace these expressions with available summaries. An unresolved
/// expression evaluates to a typed null without scanning array values. Its type is the aggregate's
/// nullable state type. Binders may use a finalized result only when its semantics match that state.
#[derive(Clone)]
pub struct StatFn;

impl ScalarFnVTable for StatFn {
    type Options = StatOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.stat");
        *ID
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
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "stat(")?;
        Display::fmt(expr.display_child(0), f)?;
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
        Ok(ConstantArray::new(Scalar::null(dtype), args.row_count()).into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        false
    }
}

fn stat_dtype(aggregate_fn: &AggregateFnRef, input_dtype: &DType) -> VortexResult<DType> {
    let Some(dtype) = aggregate_fn.state_dtype(input_dtype) else {
        vortex_bail!(
            "Aggregate function {} does not support input dtype {}",
            aggregate_fn,
            input_dtype
        );
    };
    Ok(dtype.as_nullable())
}
