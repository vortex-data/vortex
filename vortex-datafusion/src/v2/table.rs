// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VortexTable`] adapts a Vortex [`DataSourceRef`] into a DataFusion
//! [`TableProvider`].
//!
//! [`DataSourceRef`]: vortex::scan::DataSourceRef
//! [`TableProvider`]: datafusion_catalog::TableProvider

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
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::scan::DataSourceRef;
use vortex::session::VortexSession;

use crate::v2::source::VortexDataSource;

/// DataFusion [`TableProvider`] backed by a Vortex
/// [`DataSourceRef`].
///
/// `VortexTable` is the usual entry point into [`crate::v2`] when you want to
/// register an existing Vortex source with DataFusion.
///
/// Use it when another part of the system has already built a Vortex source and
/// you want to expose that source through a
/// [`SessionContext`].
///
/// `VortexTable` handles the `TableProvider` side of the integration:
///
/// - it exposes the table schema and coarse statistics to DataFusion,
/// - it seeds the initial top-level projection during `scan`,
/// - it hands execution off to [`VortexDataSource`] for later pushdown and
///   execution.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
///
/// use arrow_schema::Schema;
/// use datafusion::prelude::SessionContext;
/// use vortex::VortexSessionDefault;
/// use vortex::scan::DataSourceRef;
/// use vortex::session::VortexSession;
/// use vortex_datafusion::v2::VortexTable;
///
/// # let data_source: DataSourceRef = todo!();
/// let table = Arc::new(VortexTable::new(
///     data_source,
///     VortexSession::default(),
///     Arc::new(Schema::empty()),
/// ));
///
/// let ctx = SessionContext::new();
/// ctx.register_table("vortex_data", table)?;
/// # Ok::<(), datafusion_common::DataFusionError>(())
/// ```
///
/// [`DataSourceRef`]: vortex::scan::DataSourceRef
/// [`SessionContext`]: https://docs.rs/datafusion/latest/datafusion/prelude/struct.SessionContext.html
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
    /// The Arrow schema is the schema DataFusion will observe for this table.
    /// It should be compatible with the Vortex dtype exposed by `data_source`.
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
    // NOTE(ngates): it's not obvious these are actually used? I think DataFusion does join
    //  planning over stats from the physical plan?
    fn statistics(&self) -> Option<Statistics> {
        let num_rows = match self.data_source.row_count() {
            VortexPrecision::Exact(v) => {
                usize::try_from(v).map(Precision::Exact).unwrap_or_default()
            }
            _ => Precision::Absent,
        };

        let total_byte_size = match self.data_source.byte_size() {
            VortexPrecision::Exact(v) => {
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
