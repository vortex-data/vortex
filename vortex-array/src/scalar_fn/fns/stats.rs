// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::SimplifyCtx;

/// Creates a new expression that returns a statistic of its input.
pub fn statistic(stat: Stat, child: Expression) -> Expression {
    Statistic.new_expr(stat, vec![child])
}

/// A scalar function vtable for statistics expressions.
#[derive(Clone)]
pub struct Statistic;

impl ScalarFnVTable for Statistic {
    type Options = Stat;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("statistic")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        _expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "statistic({options})")
    }

    fn return_dtype(&self, stat: &Stat, arg_dtypes: &[DType]) -> VortexResult<DType> {
        stat.dtype(&arg_dtypes[0])
            .ok_or_else(|| {
                vortex_err!(
                    "statistic {:?} not supported for dtype {:?}",
                    stat,
                    arg_dtypes[0]
                )
            })
            // We make all statistics types nullable in case there is no reduction rule to handle
            // the statistic expression.
            .map(|dt| dt.as_nullable())
    }

    fn execute(
        &self,
        _stat: &Stat,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let return_dtype = self.return_dtype(_stat, &[])?;
        Ok(ConstantArray::new(Scalar::null(return_dtype), args.row_count()).into_array())
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        _ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        // FIXME(ngates): we really want to implement a reduction rule for all arrays? But it's an array.
        //  And it's a reduction rule. How do we do this without reduce_parent on everything..?
        Ok(None)
    }
}
