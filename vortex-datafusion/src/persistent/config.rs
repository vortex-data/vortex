use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion_common::{Constraints, Statistics};
use datafusion_physical_expr::LexOrdering;
use vortex_dtype::FieldName;
use vortex_expr::{VortexExpr, ident, select};

/// Vortex specific methods for [`FileScanConfig`]
pub trait FileScanConfigExt {
    fn project_for_vortex(&self) -> ConfigProjection;
}

impl FileScanConfigExt for FileScanConfig {
    /// Apply the projection to the original schema and statistics, and create a [`VortexExpr`] to represent it.
    fn project_for_vortex(&self) -> ConfigProjection {
        let (arrow_schema, constraints, statistics, orderings) = self.project();
        let projection_expr = match self.projection {
            None => ident(),
            Some(_) => projection_expr(&arrow_schema),
        };

        ConfigProjection {
            arrow_schema,
            constraints,
            statistics,
            orderings,
            projection_expr,
        }
    }
}

pub struct ConfigProjection {
    pub arrow_schema: SchemaRef,
    pub constraints: Constraints,
    pub statistics: Statistics,
    pub orderings: Vec<LexOrdering>,
    pub projection_expr: Arc<dyn VortexExpr>,
}

fn projection_expr(projected_arrow_schema: &SchemaRef) -> Arc<dyn VortexExpr> {
    let fields = projected_arrow_schema
        .fields()
        .iter()
        .map(|field| FieldName::from(field.name().clone()))
        .collect::<Vec<_>>();

    select(fields, ident())
}
