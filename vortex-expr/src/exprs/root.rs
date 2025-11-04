// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use vortex_array::stats::Stat;
use vortex_array::ArrayRef;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::v2::Expression;
use crate::{
    AnalysisVTable, ChildName, ExprId, ExprInstance, NotSupported, StatsCatalog, VTable, VTableExt,
};

/// An expression that returns the full scope of the expression evaluation.
// TODO(ngates): rename to "Scope"
pub struct Root;

impl VTable for Root {
    type Instance = ();
    type AnalysisVTable = NotSupported;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.root")
    }

    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        Ok(Some(()))
    }

    fn validate(&self, expr: &ExprInstance<Self>) -> VortexResult<()> {
        if !expr.children().is_empty() {
            vortex_bail!(
                "Root expression does not have children, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        unreachable!(
            "Root expression does not have children, got index {}",
            child_idx
        )
    }

    fn fmt_compact(&self, _expr: &ExprInstance<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "$")
    }

    fn return_dtype(&self, _expr: &ExprInstance<Self>, scope: &DType) -> VortexResult<DType> {
        Ok(scope.clone())
    }

    fn evaluate(&self, _expr: &ExprInstance<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        Ok(scope.clone())
    }
}

impl AnalysisVTable<Root> for Root {
    fn max(&self, expr: &ExprInstance<Root>, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        catalog.stats_ref(&self.field_path(expr)?, Stat::Max)
    }

    fn min(&self, expr: &ExprInstance<Root>, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        catalog.stats_ref(&self.field_path(expr)?, Stat::Min)
    }

    fn nan_count(
        &self,
        expr: &ExprInstance<Root>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        catalog.stats_ref(&self.field_path(expr)?, Stat::NaNCount)
    }

    fn field_path(&self, _expr: &ExprInstance<Root>) -> Option<FieldPath> {
        Some(FieldPath::root())
    }
}

/// Creates an expression that references the root scope.
///
/// Returns the entire input array as passed to the expression evaluator.
/// This is commonly used as the starting point for field access and other operations.
pub fn root() -> Expression {
    Root.try_new((), vec![])
        .vortex_expect("Failed to create Root expression")
}

/// Return whether the expression is a root expression.
pub fn is_root(expr: &Expression) -> bool {
    expr.is::<Root>()
}
