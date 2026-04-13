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
use vortex::scan::DataSourceRef;
use vortex::session::VortexSession;

use crate::v2::source::VortexDataSource;

/// A DataFusion [`TableProvider`] backed by a Vortex [`DataSourceRef`].
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
        Arc::clone(&self.arrow_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Construct the physical node representing this table.
        let data_source =
            VortexDataSource::builder(Arc::clone(&self.data_source), self.session.clone())
                .with_arrow_schema(Arc::clone(&self.arrow_schema))
                // We push down the projection now since it can make building the physical plan a lot
                // cheaper, e.g. by only computing stats for the projected columns.
                .with_some_projection(projection.cloned())
                // We don't push down filters for two reasons:
                //  1. Vortex requires a physical expression, not logical. DataFusion will try to push
                //     the physical filters later.
                //  2. There's nothing useful we can do with filters now to reduce the amount of work
                //     we have to do.
                //
                // We also don't push down the limit for the same reason, there's nothing useful we
                // can do with it.
                .build()
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(DataSourceExec::from_data_source(data_source))
    }

    /// Returns statistics for the full table, prior to any projection.
    ///
    /// We should not (and actually, cannot) perform I/O here, so the best we can do is return
    /// cardinality and byte size estimates.
    ///
    // NOTE(ngates): it's not obvious these are actually used? I think DataFusion does join
    //  planning over stats from the physical plan?
    fn statistics(&self) -> Option<Statistics> {
        let num_rows = match self.data_source.row_count() {
            Some(vortex::expr::stats::Precision::Exact(v)) => {
                usize::try_from(v).map(Precision::Exact).unwrap_or_default()
            }
            _ => Precision::Absent,
        };

        let total_byte_size = match self.data_source.byte_size() {
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
