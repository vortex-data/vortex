// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_err;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;

/// Returns the non-null `UInt64`, file-relative physical row index in a Vortex scan.
///
/// Use this expression in runtime-created scan projections. It must be evaluated by
/// Vortex, not against an Arrow batch, and does not support plan serialization.
pub fn row_index() -> Arc<dyn PhysicalExpr> {
    Arc::new(RowIndex)
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct RowIndex;

impl fmt::Display for RowIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("row_index()")
    }
}

impl PhysicalExpr for RowIndex {
    fn data_type(&self, _input_schema: &Schema) -> DFResult<DataType> {
        Ok(DataType::UInt64)
    }

    fn nullable(&self, _input_schema: &Schema) -> DFResult<bool> {
        Ok(false)
    }

    fn evaluate(&self, _batch: &RecordBatch) -> DFResult<ColumnarValue> {
        exec_err!("row_index() must be evaluated by a Vortex scan")
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        if !children.is_empty() {
            return exec_err!("row_index() has no children");
        }
        Ok(self)
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }

    fn is_volatile_node(&self) -> bool {
        // This leaf depends on the scan, even though it has no column children.
        true
    }
}
