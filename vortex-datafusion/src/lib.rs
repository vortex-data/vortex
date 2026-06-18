// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integrations between [`Vortex`] and [DataFusion].
//!
//! The crate exposes two main entry points:
//!
//! - [`VortexFormatFactory`] for the file-based integration used by SQL,
//!   `CREATE EXTERNAL TABLE`, and
//!   [`ListingTable`].
//! - [`v2`] for direct integration from an existing Vortex
//!   [`DataSourceRef`].
//!
//! # Registering The File Format
//!
//! Most applications register [`VortexFormatFactory`] with a DataFusion
//! [`SessionContext`] and then let DataFusion create [`VortexFormat`] and
//! [`VortexSource`] instances as queries are planned:
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use datafusion::datasource::provider::DefaultTableFactory;
//! use datafusion::execution::SessionStateBuilder;
//! use datafusion::prelude::SessionContext;
//! use datafusion_common::GetExt;
//! use vortex_datafusion::VortexFormatFactory;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let factory = Arc::new(VortexFormatFactory::new());
//! let mut state_builder = SessionStateBuilder::new()
//!     .with_default_features()
//!     .with_table_factory(
//!         factory.get_ext().to_uppercase(),
//!         Arc::new(DefaultTableFactory::new()),
//!     );
//!
//! if let Some(file_formats) = state_builder.file_formats() {
//!     file_formats.push(factory.clone() as _);
//! }
//!
//! let ctx = SessionContext::new_with_state(state_builder.build()).enable_url_table();
//! ctx.sql(
//!     "CREATE EXTERNAL TABLE metrics (service VARCHAR, value BIGINT) \
//!      STORED AS vortex LOCATION 'file:///tmp/metrics/'",
//! )
//! .await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Registering An Existing Vortex Data Source
//!
//! If your application already has a Vortex [`DataSourceRef`], use
//! [`v2::VortexTable`] to register it directly with DataFusion:
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use arrow_schema::Schema;
//! use datafusion::prelude::SessionContext;
//! use vortex::VortexSessionDefault;
//! use vortex::scan::DataSourceRef;
//! use vortex::session::VortexSession;
//! use vortex_datafusion::v2::VortexTable;
//!
//! # let data_source: DataSourceRef = todo!();
//! let table = Arc::new(VortexTable::new(
//!     data_source,
//!     VortexSession::default(),
//!     Arc::new(Schema::empty()),
//! ));
//!
//! let ctx = SessionContext::new();
//! ctx.register_table("vortex_data", table)?;
//! # Ok::<(), datafusion_common::DataFusionError>(())
//! ```
//!
//! [`Vortex`]: https://docs.rs/crate/vortex/latest
//! [DataFusion]: https://docs.rs/datafusion/latest/datafusion/
//! [`ListingTable`]: https://docs.rs/datafusion/latest/datafusion/datasource/listing/struct.ListingTable.html
//! [`DataSourceRef`]: vortex::scan::DataSourceRef
//! [`SessionContext`]: https://docs.rs/datafusion/latest/datafusion/prelude/struct.SessionContext.html
#![deny(missing_docs)]
use std::fmt::Debug;

use datafusion_common::stats::Precision as DFPrecision;
use vortex::expr::stats::Precision;

pub mod convert;
mod persistent;
pub mod v2;

#[cfg(test)]
mod tests;

pub use persistent::*;

/// Extension trait to convert our [`Precision`] to DataFusion's
/// [`DataFusionPrecision`].
///
/// [`Precision`]: vortex::expr::stats::Precision
/// [`DataFusionPrecision`]: datafusion_common::stats::Precision
trait PrecisionExt<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    /// Convert `Precision` to the datafusion equivalent.
    fn to_df(self) -> DFPrecision<T>;
}

impl<T> PrecisionExt<T> for Precision<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Precision::Exact(v) => DFPrecision::Exact(v),
            Precision::Inexact(v) => DFPrecision::Inexact(v),
            Precision::Absent => DFPrecision::Absent,
        }
    }
}

#[cfg(test)]
mod common_tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use datafusion::arrow::array::RecordBatch;
    use datafusion::datasource::provider::DefaultTableFactory;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_catalog::TableProvider;
    use datafusion_common::DFSchema;
    use datafusion_common::GetExt;
    use datafusion_expr::CreateExternalTable;
    use object_store::ObjectStore;
    use object_store::memory::InMemory;
    use url::Url;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::arrow::FromArrowArray;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexWrite;
    use vortex::io::object_store::ObjectStoreWrite;
    use vortex::session::VortexSession;

    use crate::VortexFormatFactory;
    use crate::VortexTableOptions;

    static VX_SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

    pub struct TestSessionContext {
        pub store: Arc<dyn ObjectStore>,
        pub session: SessionContext,
    }

    impl Default for TestSessionContext {
        fn default() -> Self {
            Self::new(false)
        }
    }

    impl TestSessionContext {
        /// Create a new test session context with the given projection pushdown setting.
        pub fn new(projection_pushdown: bool) -> Self {
            let store = Arc::new(InMemory::new());
            let opts = VortexTableOptions {
                projection_pushdown,
                ..Default::default()
            };
            let factory = Arc::new(VortexFormatFactory::new().with_options(opts));
            let mut session_state_builder = SessionStateBuilder::new()
                .with_default_features()
                .with_table_factory(
                    factory.get_ext().to_uppercase(),
                    Arc::new(DefaultTableFactory::new()),
                )
                .with_object_store(
                    &Url::try_from("file://").unwrap(),
                    Arc::<InMemory>::clone(&store),
                );

            if let Some(file_formats) = session_state_builder.file_formats() {
                file_formats.push(factory as _);
            }

            let session: SessionContext =
                SessionContext::new_with_state(session_state_builder.build()).enable_url_table();

            Self { store, session }
        }

        // Write arrow data into a vortex file.
        pub async fn write_arrow_batch<P>(&self, path: P, batch: &RecordBatch) -> anyhow::Result<()>
        where
            P: Into<object_store::path::Path>,
        {
            let array = ArrayRef::from_arrow(batch, false)?;
            let mut write = ObjectStoreWrite::new(Arc::clone(&self.store), &path.into()).await?;
            VX_SESSION
                .write_options()
                .write(&mut write, array.to_array_stream())
                .await?;
            write.shutdown().await?;

            Ok(())
        }

        /// Creates a ListingTable provider targeted at the provided path
        pub async fn table_provider<S>(
            &self,
            name: &str,
            location: impl Into<String>,
            schema: S,
        ) -> anyhow::Result<Arc<dyn TableProvider>>
        where
            DFSchema: TryFrom<S>,
            anyhow::Error: From<<S as TryInto<DFSchema>>::Error>,
        {
            let factory = self.session.table_factory("VORTEX").unwrap();

            let cmd = CreateExternalTable::builder(
                name,
                location.into(),
                "vortex",
                DFSchema::try_from(schema)?.into(),
            )
            .build();

            let table = factory.create(&self.session.state(), &cmd).await?;

            Ok(table)
        }
    }
}
