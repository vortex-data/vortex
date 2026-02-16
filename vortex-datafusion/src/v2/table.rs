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
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Expr;
use datafusion_expr::TableType;
use datafusion_physical_plan::ExecutionPlan;
use vortex::scan::api::DataSourceRef;
use vortex::session::VortexSession;

use crate::v2::source::VortexDataSource;

/// A DataFusion [`TableProvider`] backed by a Vortex [`DataSourceRef`].
///
/// Passes the [`DataSourceRef`] to [`VortexDataSource`], which defers scan construction to
/// [`open`](datafusion_datasource::source::DataSource::open) so that pushed-down filters and
/// limits are included in the scan request.
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
        // Unlike filters and limit, we _do_ apply the projection at this stage since DataFusion's
        // physical projection expression push-down is still in its early stages. In theory, we
        // could also wait to apply the projection until we can push down over the physical plan.
        projection: Option<&Vec<usize>>,
        // We ignore push-down of logical filters since Vortex requires a physical
        //  expression (i.e. we require that coercion semantics have already been performed by the
        //  engine). Instead, DataFusion will push down filters through the physical plan via
        //  the VortexScanSource DataSource.
        _filters: &[Expr],
        // Similarly for limit, we wait until we can push down over the physical plan.
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let data_source = VortexDataSource::builder(self.data_source.clone(), self.session.clone())
            .with_arrow_schema(self.arrow_schema.clone())
            .with_some_projection(projection.cloned())
            .build()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(DataSourceExec::from_data_source(data_source))
    }

    /// Returns statistics for the full table, before any projection.
    /// To keep this reasonably cheap, we just return cardinality and byte size estimates.
    /// We provide full statistics from the physical plan.
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
