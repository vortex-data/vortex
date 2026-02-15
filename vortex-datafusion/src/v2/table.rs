// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VortexTable`] implements DataFusion's [`TableProvider`] trait, providing a direct
//! integration between a Vortex [`DataSource`] and DataFusion's query engine.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_catalog::TableProvider;
use datafusion_common::ColumnStatistics;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Expr;
use datafusion_expr::TableType;
use datafusion_physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::scan::api::DataSourceRef;
use vortex::scan::api::ScanRequest;
use vortex::session::VortexSession;

use crate::v2::source::VortexScanSource;

fn vx_err(e: vortex::error::VortexError) -> datafusion_common::DataFusionError {
    datafusion_common::DataFusionError::External(Box::new(e))
}

/// A DataFusion [`TableProvider`] backed by a Vortex [`DataSourceRef`].
///
/// Maps each Vortex scan split to one DataFusion partition, letting DataFusion's scheduler
/// control concurrency.
pub struct VortexTable {
    data_source: DataSourceRef,
    session: VortexSession,
    arrow_schema: SchemaRef,
}

impl fmt::Debug for VortexTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexTable")
            .field("schema", &self.arrow_schema)
            .finish()
    }
}

impl VortexTable {
    /// Creates a new [`VortexTable`] from a Vortex data source and session.
    ///
    /// The Arrow schema will be used to emit the correct column names and types to DataFusion.
    /// The Vortex DType of the data source should be compatible with this Arrow schema.
    pub fn new(
        data_source: DataSourceRef,
        session: VortexSession,
        arrow_schema: SchemaRef,
    ) -> Self {
        Self {
            data_source,
            session,
            arrow_schema,
        }
    }
}

#[async_trait]
impl TableProvider for VortexTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.arrow_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Build the projection expression and projected arrow schema.
        let (vx_projection, projected_schema) = if let Some(indices) = projection {
            let projected_fields: Vec<_> = indices
                .iter()
                .map(|&i| self.arrow_schema.field(i).clone())
                .collect();
            let projected_schema = Arc::new(arrow_schema::Schema::new(projected_fields));

            let elements: Vec<(String, Expression)> = indices
                .iter()
                .map(|&i| {
                    let name = self.arrow_schema.field(i).name().clone();
                    (name.clone(), get_item(name, root()))
                })
                .collect();
            let expr = pack(elements, Nullability::NonNullable);
            (Some(expr), projected_schema)
        } else {
            (None, self.arrow_schema.clone())
        };

        // Build the scan request.
        let scan_request = ScanRequest {
            projection: vx_projection,
            limit: limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
            ..Default::default()
        };

        // Create the scan and collect splits.
        let scan = self.data_source.scan(scan_request).map_err(vx_err)?;
        let splits: Vec<_> = scan.splits().try_collect().await.map_err(vx_err)?;

        Ok(DataSourceExec::from_data_source(VortexScanSource::new(
            splits,
            projected_schema,
            self.session.clone(),
        )))
    }

    fn statistics(&self) -> Option<Statistics> {
        let num_rows = match self.data_source.row_count_estimate() {
            Some(vortex::expr::stats::Precision::Exact(v)) => {
                usize::try_from(v).map(Precision::Exact).unwrap_or_default()
            }
            _ => Precision::Absent,
        };

        let total_byte_size = match self.data_source.byte_size_estimate() {
            Some(vortex::expr::stats::Precision::Exact(v)) => {
                usize::try_from(v).map(Precision::Exact).unwrap_or_default()
            }
            _ => Precision::Absent,
        };

        let column_statistics =
            vec![ColumnStatistics::new_unknown(); self.arrow_schema.fields.len()];

        Some(Statistics {
            num_rows,
            total_byte_size,
            column_statistics,
        })
    }
}
