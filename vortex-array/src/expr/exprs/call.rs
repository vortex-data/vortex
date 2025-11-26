// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::{
    ChildName, ExecutionArgs, ExprId, Expression, ExpressionView, StatsCatalog, VTable,
};
use crate::functions::{ScalarFunction, ScalarFunctionVTable};
use crate::stats::Stat;
use crate::ArrayRef;
use itertools::Itertools;
use std::fmt::Formatter;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Vector;

/// An expression representing a call to a scalar function.
pub struct Call {
    vtable: ScalarFunctionVTable,
}

impl VTable for Call {
    type Instance = ScalarFunction;

    fn id(&self) -> ExprId {
        self.vtable.id()
    }

    fn serialize(&self, func: &ScalarFunction) -> VortexResult<Option<Vec<u8>>> {
        func.serialize_options()
    }

    fn deserialize(&self, bytes: &[u8]) -> VortexResult<Self::Instance> {
        self.vtable.deserialize(bytes)
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        // TODO(ngates): check against the function signature.
        Ok(())
    }

    fn child_name(&self, _func: &ScalarFunction, _child_idx: usize) -> ChildName {
        // TODO(ngates): fetch from the function signature.
        ChildName::from("unknown")
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.id())?;
        for (i, child) in expr.children().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            child.fmt_sql(f)?;
        }

        write!(f, " options: ")?;
        expr.data().fmt_options(f)?;

        write!(f, ")")
    }

    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let arg_dtypes: Vec<_> = expr
            .children()
            .iter()
            .map(|c| c.return_dtype(scope))
            .try_collect()?;
        expr.data().return_dtype(&arg_dtypes)
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        // TODO(ngates): we evaluate a function by wrapping it in a CallArray.
        todo!()
    }

    fn execute(&self, _data: &Self::Instance, _args: ExecutionArgs) -> VortexResult<Vector> {
        // In theory, expressions shouldn't have any execute function at all.
        todo!()
    }

    fn stat_falsification(
        &self,
        _expr: &ExpressionView<Self>,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        todo!()
    }

    fn stat_expression(
        &self,
        _expr: &ExpressionView<Self>,
        _stat: Stat,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        todo!()
    }
}
