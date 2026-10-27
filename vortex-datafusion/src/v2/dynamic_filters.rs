// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::root;
use vortex::expr::transform::replace;
use vortex::scan::DynamicFilter;

use crate::convert::ExpressionConverter;

#[derive(Clone)]
pub(super) struct DynamicFilterExpr {
    pub expression: Arc<dyn PhysicalExpr>,
    pub schema: SchemaRef,
    pub projection: Expression,
    pub leftover: Option<ProjectionExprs>,
    pub converter: Arc<dyn ExpressionConverter>,
}

impl DynamicFilterExpr {
    fn current(&self) -> Option<Expression> {
        let snapshot = snapshot_physical_expr(Arc::clone(&self.expression)).ok()?;
        let snapshot = match &self.leftover {
            Some(projection) => projection.unproject_expr(&snapshot).ok()?,
            None => snapshot,
        };
        let snapshot = reassign_expr_columns(snapshot, &self.schema).ok()?;
        let converted = self.converter.try_convert(&snapshot, &self.schema).ok()??;
        Some(replace(converted, &root(), self.projection.clone()))
    }
}

#[derive(Clone, Default)]
pub(super) struct DynamicFilters(pub Vec<DynamicFilterExpr>);

impl fmt::Debug for DynamicFilters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DynamicFilters")
            .field(&self.0.len())
            .finish()
    }
}

impl DynamicFilter for DynamicFilters {
    fn current(&self) -> Option<Expression> {
        and_collect(self.0.iter().filter_map(DynamicFilterExpr::current))
    }
}
