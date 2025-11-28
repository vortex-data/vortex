// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::SessionVar;
use vortex_vector::Vector;

use crate::ArrayRef;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::ExpressionView;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::functions;
use crate::expr::functions::ScalarFnVTable;
use crate::expr::functions::scalar::ScalarFn;
use crate::expr::stats::Stat;
use crate::expr::transform::rules::Matcher;

/// An expression that wraps arbitrary scalar functions.
///
/// Note that for backwards-compatibility, the `id` of this expression is the same as the
/// `id` of the underlying scalar function vtable, rather than being something constant like
/// `vortex.scalar_fn`.
pub struct ScalarFnExpr {
    /// The vtable of the particular scalar function represented by this expression.
    vtable: ScalarFnVTable,
}

impl VTable for ScalarFnExpr {
    type Instance = ScalarFn;

    fn id(&self) -> ExprId {
        self.vtable.id()
    }

    fn serialize(&self, func: &ScalarFn) -> VortexResult<Option<Vec<u8>>> {
        func.options().serialize()
    }

    fn deserialize(&self, bytes: &[u8]) -> VortexResult<Option<Self::Instance>> {
        self.vtable.deserialize(bytes).map(Some)
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        vortex_ensure!(
            expr.data()
                .signature()
                .arity()
                .matches(expr.children().len()),
            "invalid number of arguments for scalar function"
        );
        Ok(())
    }

    fn child_name(&self, _func: &ScalarFn, _child_idx: usize) -> ChildName {
        "unknown".into()
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", expr.data())?;
        for (i, child) in expr.children().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            child.fmt_sql(f)?;
        }
        write!(f, ")")
    }

    fn fmt_data(&self, func: &ScalarFn, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", func)
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let arg_dtypes: Vec<_> = expr
            .children()
            .iter()
            .map(|e| e.return_dtype(scope))
            .try_collect()?;
        expr.data().return_dtype(&arg_dtypes)
    }

    fn evaluate(&self, _expr: &ExpressionView<Self>, _scope: &ArrayRef) -> VortexResult<ArrayRef> {
        vortex_bail!("Scalar function evaluation not yet implemented")
    }

    fn execute(&self, _data: &Self::Instance, _args: ExecutionArgs) -> VortexResult<Vector> {
        vortex_bail!("Scalar function execution not yet implemented")
    }

    fn stat_falsification(
        &self,
        _expr: &ExpressionView<Self>,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // TODO(ngates): ideally this is implemented as optimizer rules over a `falsify` and
        //  `verify` expressions.
        todo!()
    }

    fn stat_expression(
        &self,
        _expr: &ExpressionView<Self>,
        _stat: Stat,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // TODO(ngates): ideally this is implemented specifically for the Zoned layout, no one
        //  else needs to know what a specific stat over a column resolves to.
        todo!()
    }

    fn is_null_sensitive(&self, _func: &ScalarFn) -> bool {
        todo!()
    }
}

/// A matcher that matches any scalar function expression.
#[derive(Debug)]
pub struct AnyScalarFn;
impl Matcher for AnyScalarFn {
    type View<'a> = &'a ScalarFn;

    fn try_match(parent: &Expression) -> Option<Self::View<'_>> {
        Some(parent.as_opt::<ScalarFnExpr>()?.data())
    }
}

/// A matcher that matches a specific scalar function expression.
#[derive(Debug)]
pub struct ExactScalarFn<F: functions::VTable>(PhantomData<F>);
impl<F: functions::VTable> Matcher for ExactScalarFn<F> {
    type View<'a> = &'a F::Options;

    fn try_match(parent: &Expression) -> Option<Self::View<'_>> {
        let expr_view = parent.as_opt::<ScalarFnExpr>()?;
        expr_view.data().as_any().downcast_ref::<F::Options>()
    }
}
