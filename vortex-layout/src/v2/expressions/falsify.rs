// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;

/// An expression that evaluates to true when the predicate is provably false, without evaluating
/// it.
///
/// Falsify typically reduces to operations over statistics expressions. For example,
/// the expression `falsify(col > 5)` may reduce to `col.max() <= 5`.
///
/// If a falsify expression cannot be reduced, it evaluates to `false` for all inputs.
#[derive(Clone)]
pub struct Falsify;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FalsifyOptions {
    predicate: Expression,
}

impl Display for FalsifyOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "predicate={}", self.predicate)
    }
}

impl ScalarFnVTable for Falsify {
    // FIXME(ngates): should the predicate be a child expression, or live like this in the options.
    //  It's a bit weird? Maybe it makes implementing the optimizer rules a little more fiddly?
    //  But it's weird to have a child expression that we know is never executed.
    type Options = FalsifyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("falsify")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("predicate")
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "falsify(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Unless falsify has been reduced by another expression, we cannot prove the predicate
        // is false. Therefore, we return a constant false array.
        Ok(ConstantArray::new(false, args.row_count()).into_array())
    }
}
