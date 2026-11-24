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
//! # Schemas and extension types
//!
//! A table's _logical schema_ describes the fields a query expects. Each file's _physical schema_
//! describes the fields its reader produces. Both use Arrow's [`Schema`]. DataFusion's
//! [`PhysicalExprAdapter`] reconciles differences with casts and nulls for missing nullable columns.
//! [`DFSchema`] adds logical planning information, such as table qualifiers.
//!
//! Vortex's [`DType`] omits some Arrow layout choices, so the reader also needs the logical schema.
//! For example, [`DType::Utf8`] can export to multiple Arrow string layouts.
//!
//! An [Arrow extension type] adds identity to its storage field through metadata. JSON, for example,
//! uses string storage with extension name `arrow.json`. [`ArrowSession`] defines how Vortex maps
//! these fields to dtypes. The reader checks extension identity before copying table metadata, so
//! plain strings cannot be mislabeled as JSON or JSON read as plain strings. This preserves type
//! identity during scans, but does not define valid operations on extension values.
//!
//! [`Schema`]: arrow_schema::Schema
//! [`DFSchema`]: datafusion_common::DFSchema
//! [`PhysicalExprAdapter`]: datafusion_physical_expr_adapter::PhysicalExprAdapter
//! [`DType`]: vortex::dtype::DType
//! [`DType::Utf8`]: vortex::dtype::DType::Utf8
//! [Arrow extension type]: https://arrow.apache.org/docs/format/Columnar.html#extension-types
//! [`ArrowSession`]: vortex_arrow::ArrowSession
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
mod row_index;
pub mod v2;

#[cfg(test)]
mod tests;

pub use persistent::*;
pub use row_index::row_index;

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
    use vortex::dtype::DType;
    use vortex::dtype::extension::ExtDType;
    use vortex::dtype::extension::ExtDTypeRef;
    use vortex::dtype::extension::ExtId;
    use vortex::dtype::extension::ExtVTable;
    use vortex::error::VortexResult;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexWrite;
    use vortex::io::object_store::ObjectStoreWrite;
    use vortex::scalar::ScalarValue;
    use vortex::session::VortexSession;
    use vortex::session::registry::CachedId;
    use vortex_arrow::ArrowSessionExt;

    use crate::VortexFormatFactory;
    use crate::VortexTableOptions;

    static VX_SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

    /// The Vortex ID of [`AnyStorage`], which is also its Arrow extension name.
    pub static ANY_STORAGE_ID: CachedId = CachedId::new("test.any_storage");

    /// An extension type over any storage dtype, without Arrow plugins.
    ///
    /// Its metadata is its storage dtype, which `metadata_from_storage` reads back from the
    /// storage, so the serialized metadata is empty.
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    pub struct AnyStorage;

    impl AnyStorage {
        /// An [`AnyStorage`] dtype over `storage`.
        pub fn dtype(storage: DType) -> VortexResult<ExtDTypeRef> {
            Ok(ExtDType::try_from_storage_with_vtable(Self, (), storage)?.erased())
        }
    }

    impl ExtVTable for AnyStorage {
        type Metadata = DType;
        type StoredMetadata = ();
        type NativeValue<'a> = &'a ScalarValue;

        fn id(&self) -> ExtId {
            *ANY_STORAGE_ID
        }

        fn serialize_metadata(&self, _metadata: &DType) -> VortexResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<()> {
            Ok(())
        }

        fn metadata_from_storage(
            &self,
            _stored_metadata: (),
            storage_dtype: &DType,
        ) -> VortexResult<DType> {
            Ok(storage_dtype.clone())
        }

        fn storage_dtype(metadata: &DType) -> DType {
            metadata.clone()
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<&'a ScalarValue> {
            Ok(storage_value)
        }
    }

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
            let opts = VortexTableOptions {
                projection_pushdown,
                ..Default::default()
            };
            let factory = Arc::new(VortexFormatFactory::new().with_options(opts));

            Self::new_with_factory(factory)
        }

        /// Create a new test session context with the given Vortex format factory.
        pub fn new_with_factory(factory: Arc<VortexFormatFactory>) -> Self {
            let store = Arc::new(InMemory::new());
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
            let schema = batch.schema();
            let array = VX_SESSION
                .arrow()
                .from_arrow_record_batch(batch.clone(), &schema)?;
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
