// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::expr::Arity;
use vortex_array::expr::ChildName;
use vortex_array::expr::ExecutionArgs;
use vortex_array::expr::ExprId;
use vortex_array::expr::Expression;
use vortex_array::expr::VTable;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Datum;
use vortex_vector::Scalar;
use vortex_vector::bool::BoolScalar;

/// An expression that evaluates to true when the predicate is provably false, without evaluating
/// it.
///
/// Falsify typically reduces to operations over statistics expressions. For example,
/// the expression `falsify(col > 5)` may reduce to `col.max() <= 5`.
///
/// If a falsify expression cannot be reduced, it evaluates to `false` for all inputs.
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

impl VTable for Falsify {
    // FIXME(ngates): should the predicate be a child expression, or live like this in the options.
    //  It's a bit weird? Maybe it makes implementing the optimizer rules a little more fiddly?
    //  But it's weird to have a child expression that we know is never executed.
    type Options = FalsifyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("falsify")
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

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        if !arg_dtypes[0].is_boolean() {
            vortex_bail!("falsify() requires a boolean argument");
        }
        Ok(DType::Bool(Nullability::NonNullable))
    }

    // NOTE(ngates): do we prefer evaluate or execute semantics???
    fn evaluate(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        // Unless falsify has been reduced by another expression, we cannot prove the predicate
        // is false. Therefore, we return a constant false array.
        Ok(ConstantArray::new(false, scope.len()).into_array())
    }

    fn execute(&self, _options: &Self::Options, _args: ExecutionArgs) -> VortexResult<Datum> {
        Ok(Datum::Scalar(Scalar::Bool(BoolScalar::new(Some(false)))))
    }
}
