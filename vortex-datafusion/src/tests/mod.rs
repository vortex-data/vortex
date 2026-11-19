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
pub(crate) struct RecordingConverter {
    inner: DefaultExpressionConverter,
    /// Whether the scan evaluates the projection expressions it can, or reads raw columns.
    projection_pushdown: bool,
    pub(crate) scan_projections: Mutex<Vec<Expression>>,
}

impl RecordingConverter {
    /// Converts with a default converter that resolves Arrow types through `session`.
    pub(crate) fn new(session: VortexSession) -> Self {
        Self {
            inner: DefaultExpressionConverter::new(session),
            projection_pushdown: true,
            scan_projections: Mutex::default(),
        }
    }

    /// Converts filters like [`Self::new`], but keeps every projection expression in DataFusion,
    /// as a converter does that applies DataFusion's own field access semantics to projections.
    pub(crate) fn without_projection_pushdown(session: VortexSession) -> Self {
        Self {
            projection_pushdown: false,
            ..Self::new(session)
        }
    }

    fn record(&self, processed: &ProcessedProjection) {
        self.scan_projections
            .lock()
            .push(processed.scan_projection.clone());
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
        let processed = if self.projection_pushdown {
            self.inner
                .split_projection(source_projection, input_schema, output_schema)?
        } else {
            self.no_pushdown_projection(source_projection, input_schema)?
        };
        self.record(&processed);
        Ok(processed)
    }

    fn split_filtered_projection(
        &self,
        projection: ProjectionExprs,
        residual_filters: Vec<Arc<dyn PhysicalExpr>>,
        input_schema: &Schema,
    ) -> DFResult<(ProcessedProjection, Vec<Arc<dyn PhysicalExpr>>)> {
        let (processed, residual_filters) = if self.projection_pushdown {
            self.inner
                .split_filtered_projection(projection, residual_filters, input_schema)?
        } else {
            self.no_pushdown_filtered_projection(projection, residual_filters, input_schema)?
        };
        self.record(&processed);
        Ok((processed, residual_filters))
    }
}
