// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_dtype::DType;
use vortex_dtype::FieldPath;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::expr::ExprId;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;
use crate::expr::stats::Stat;

/// An expression that returns the full scope of the expression evaluation.
// TODO(ngates): rename to "Scope"
pub struct Root;

impl VTable for Root {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.root")
    }

    fn serialize(&self, _instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        unreachable!(
            "Root expression does not have children, got index {}",
            child_idx
        )
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "$")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_bail!("Root expression does not support return_dtype")
    }

    fn execute(
        &self,
        _data: &Self::Options,
        _args: ExecutionArgs,
    ) -> VortexResult<ExecutionResult> {
        vortex_bail!("Root expression is not executable")
    }

    fn stat_expression(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        catalog.stats_ref(&FieldPath::root(), stat)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Creates an expression that references the root scope.
///
/// Returns the entire input array as passed to the expression evaluator.
/// This is commonly used as the starting point for field access and other operations.
pub fn root() -> Expression {
    Root.try_new_expr(EmptyOptions, vec![])
        .vortex_expect("Failed to create Root expression")
}

/// Return whether the expression is a root expression.
pub fn is_root(expr: &Expression) -> bool {
    expr.is::<Root>()
}

#[cfg(feature = "arbitrary")]
mod arb {
    use arbitrary::Result as AResult;
    use arbitrary::Unstructured;
    use vortex_dtype::DType;

    use super::Root;
    use super::root;
    use crate::expr::Expression;
    use crate::expr::arbitrary::ArbExpr;
    use crate::expr::arbitrary::ArbExprCtx;

    impl ArbExpr for Root {
        fn arb_wrap(
            &self,
            _u: &mut Unstructured,
            _scope: &DType,
            _child: Expression,
            _child_type: &DType,
            _ctx: &dyn ArbExprCtx,
        ) -> AResult<Option<(Expression, DType)>> {
            // root() cannot wrap anything - it's a leaf
            Ok(None)
        }

        fn arb_gen(
            &self,
            _u: &mut Unstructured,
            scope: &DType,
            target: &DType,
            _depth: u8,
            _ctx: &dyn ArbExprCtx,
        ) -> AResult<Option<Expression>> {
            // Can only produce scope type
            if scope.eq_ignore_nullability(target) {
                Ok(Some(root()))
            } else {
                Ok(None)
            }
        }
    }
}
