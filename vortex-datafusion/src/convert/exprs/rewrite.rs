// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DataFusion-level rewrites applied to filters before they are converted for Vortex.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TreeNode;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;
use datafusion_physical_plan::expressions as df_expr;

/// How [`substitute_scalar_subqueries`] treats a subquery whose value is not known yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PendingSubquery {
    /// Leave the subquery in place, so DataFusion evaluates it.
    Keep,
    /// Replace it with a non-null placeholder literal of its type.
    ///
    /// Pushdown runs while planning, before any subquery has executed. The placeholder lets the
    /// converter decide whether the predicate's shape is supported. The scan replaces the
    /// subquery with its real value when it opens a file, and evaluates the predicate in
    /// DataFusion if that substituted predicate does not convert.
    Placeholder,
}

/// Replaces every scalar subquery in `expr` with a literal of its value.
///
/// DataFusion's `ScalarSubqueryExec` runs all subqueries before it executes its input, so the
/// values are known by the time the scan opens a file.
pub(crate) fn substitute_scalar_subqueries(
    expr: Arc<dyn PhysicalExpr>,
    pending: PendingSubquery,
) -> DFResult<Arc<dyn PhysicalExpr>> {
    if !contains_scalar_subquery(&expr)? {
        return Ok(expr);
    }
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    expr.transform_up(|expr| {
        if !expr.is::<ScalarSubqueryExpr>() {
            return Ok(Transformed::no(expr));
        }
        // A subquery evaluates to its value once it is resolved and fails until then.
        let value = match expr.evaluate(&empty) {
            Ok(ColumnarValue::Scalar(value)) => value,
            Ok(ColumnarValue::Array(_)) | Err(_) => match pending {
                PendingSubquery::Keep => return Ok(Transformed::no(expr)),
                PendingSubquery::Placeholder => {
                    let data_type = expr.data_type(empty.schema_ref())?;
                    ScalarValue::new_default(&data_type)
                        .or_else(|_| ScalarValue::try_from(&data_type))?
                }
            },
        };
        Ok(Transformed::yes(
            Arc::new(df_expr::Literal::new(value)) as Arc<dyn PhysicalExpr>
        ))
    })
    .map(|transformed| transformed.data)
}

/// Whether `expr` contains a scalar subquery.
pub(crate) fn contains_scalar_subquery(expr: &Arc<dyn PhysicalExpr>) -> DFResult<bool> {
    expr.exists(|expr| Ok(expr.is::<ScalarSubqueryExpr>()))
}
