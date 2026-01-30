// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Connectors to enable [DataFusion](https://docs.rs/datafusion/latest/datafusion/) to read [`Vortex`](https://docs.rs/crate/vortex/latest) data.
#![deny(missing_docs)]
use std::fmt::Debug;

use datafusion_common::stats::Precision as DFPrecision;
use vortex::expr::stats::Precision;

mod convert;
mod persistent;

#[cfg(test)]
mod tests;

pub use convert::exprs::ExpressionConvertor;
pub use persistent::*;

/// Extension trait to convert our [`Precision`](vortex::stats::Precision) to Datafusion's [`Precision`](datafusion_common::stats::Precision)
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
        }
    }
}

impl<T> PrecisionExt<T> for Option<Precision<T>>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Some(v) => v.to_df(),
            None => DFPrecision::Absent,
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
    use vortex::io::ObjectStoreWriter;
    use vortex::io::VortexWrite;
    use vortex::session::VortexSession;

    use crate::VortexFormatFactory;

    static VX_SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

    pub struct TestSessionContext {
        pub store: Arc<dyn ObjectStore>,
        pub session: SessionContext,
    }

    impl Default for TestSessionContext {
        fn default() -> Self {
            let store = Arc::new(InMemory::new());
            let factory = Arc::new(VortexFormatFactory::new());
            let session_state_builder = SessionStateBuilder::new()
                .with_default_features()
                .with_table_factory(
                    factory.get_ext().to_uppercase(),
                    Arc::new(DefaultTableFactory::new()),
                )
                .with_file_formats(vec![factory])
                .with_object_store(&Url::try_from("file://").unwrap(), store.clone());

            let session: SessionContext =
                SessionContext::new_with_state(session_state_builder.build()).enable_url_table();

            Self { store, session }
        }
    }

    impl TestSessionContext {
        // Write arrow data into a vortex file.
        pub async fn write_arrow_batch<P>(&self, path: P, batch: &RecordBatch) -> anyhow::Result<()>
        where
            P: Into<object_store::path::Path>,
        {
            let array = ArrayRef::from_arrow(batch, false)?;
            let mut write = ObjectStoreWriter::new(self.store.clone(), &path.into()).await?;
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
