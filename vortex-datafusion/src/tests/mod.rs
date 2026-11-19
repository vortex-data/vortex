// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use parking_lot::Mutex;
use vortex::expr::Expression;
use vortex::session::VortexSession;

use crate::convert::DefaultExpressionConverter;
use crate::convert::ExpressionConverter;
use crate::convert::ProcessedProjection;

mod nested_projection;
mod schema_evolution;

/// Converts expressions with the default converter and records each scan projection.
struct RecordingConverter {
    inner: DefaultExpressionConverter,
    scan_projections: Mutex<Vec<Expression>>,
}

impl RecordingConverter {
    /// Converts with a default converter that resolves Arrow types through `session`.
    fn new(session: VortexSession) -> Self {
        Self {
            inner: DefaultExpressionConverter::new(session),
            scan_projections: Mutex::default(),
        }
    }
}

impl ExpressionConverter for RecordingConverter {
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        self.inner.try_convert(expr, schema)
    }

    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        let processed =
            self.inner
                .split_projection(source_projection, input_schema, output_schema)?;
        self.scan_projections
            .lock()
            .push(processed.scan_projection.clone());
        Ok(processed)
    }
}
