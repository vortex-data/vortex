// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_dtype::DType;
use vortex_dtype::FieldPath;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Vector;

use crate::ArrayRef;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::ExpressionView;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;
use crate::expr::stats::Stat;

/// An expression that returns the full scope of the expression evaluation.
// TODO(ngates): rename to "Scope"
pub struct Root;

impl VTable for Root {
    type Instance = ();

    fn id(&self) -> ExprId {
        ExprId::from("vortex.root")
    }

    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        Ok(Some(()))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if !expr.children().is_empty() {
            vortex_bail!(
                "Root expression does not have children, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        unreachable!(
            "Root expression does not have children, got index {}",
            child_idx
        )
    }

    fn fmt_sql(&self, _expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "$")
    }

    fn return_dtype(&self, _expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        Ok(scope.clone())
    }

    fn evaluate(&self, _expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        Ok(scope.clone())
    }

    fn execute(&self, _data: &Self::Instance, _args: ExecutionArgs) -> VortexResult<Vector> {
        vortex_bail!("Root expression is not executable")
    }

    fn stat_expression(
        &self,
        _expr: &ExpressionView<Self>,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        catalog.stats_ref(&FieldPath::root(), stat)
    }

    fn is_null_sensitive(&self, _instance: &Self::Instance) -> bool {
        false
    }

    fn is_fallible(&self, _instance: &Self::Instance) -> bool {
        false
    }
}

/// Creates an expression that references the root scope.
///
/// Returns the entire input array as passed to the expression evaluator.
/// This is commonly used as the starting point for field access and other operations.
pub fn root() -> Expression {
    Root.try_new_expr((), vec![])
        .vortex_expect("Failed to create Root expression")
}

/// Return whether the expression is a root expression.
pub fn is_root(expr: &Expression) -> bool {
    expr.is::<Root>()
}
