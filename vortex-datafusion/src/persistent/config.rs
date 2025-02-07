use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion_common::Statistics;
use datafusion_physical_expr::LexOrdering;
use vortex_array::arrow::FromArrowType as _;
use vortex_dtype::{DType, FieldName};
use vortex_expr::{ident, select, VortexExpr};

/// Vortex specific methods for [`FileScanConfig`]
pub trait FileScanConfigExt {
    fn project_for_vortex(&self) -> ConfigProjection;
}

impl FileScanConfigExt for FileScanConfig {
    /// Apply the projection to the [`DType`] in addition to the original schema and statistics, and create a [`VortexExpr`] to represent it.
    fn project_for_vortex(&self) -> ConfigProjection {
        let (arrow_schema, statistics, orderings) = self.project();
        let dtype = DType::from_arrow(arrow_schema.as_ref());
        let projection_expr = match self.projection {
            None => ident(),
            Some(_) => projection_expr(&arrow_schema),
        };

        ConfigProjection {
            arrow_schema,
            statistics,
            orderings,
            projection_expr,
            dtype,
        }
    }
}

pub struct ConfigProjection {
    pub arrow_schema: SchemaRef,
    pub statistics: Statistics,
    pub orderings: Vec<LexOrdering>,
    pub projection_expr: Arc<dyn VortexExpr>,
    pub dtype: DType,
}

fn projection_expr(projected_arrow_schema: &SchemaRef) -> Arc<dyn VortexExpr> {
    let fields = projected_arrow_schema
        .fields()
        .iter()
        .map(|field| FieldName::from(field.name().clone()))
        .collect::<Vec<_>>();

    select(fields, ident())
}
