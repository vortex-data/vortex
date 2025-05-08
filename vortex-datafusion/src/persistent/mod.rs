//! Persistent implementation of a Vortex table provider.
mod cache;
mod config;
mod format;
pub mod metrics;
mod opener;
mod sink;
mod source;

pub use format::{VortexFormat, VortexFormatFactory, VortexFormatOptions};

#[cfg(test)]
/// Utility function to register Vortex with a [`SessionStateBuilder`]
fn register_vortex_format_factory(
    factory: VortexFormatFactory,
    session_state_builder: &mut datafusion::execution::SessionStateBuilder,
) {
    if let Some(table_factories) = session_state_builder.table_factories() {
        table_factories.insert(
            datafusion_common::GetExt::get_ext(&factory).to_uppercase(), // Has to be uppercase
            std::sync::Arc::new(datafusion::datasource::provider::DefaultTableFactory::new()),
        );
    }

    if let Some(file_formats) = session_state_builder.file_formats() {
        file_formats.push(std::sync::Arc::new(factory));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::datasource::listing::{
        ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
    };
    use datafusion::prelude::SessionContext;
    use tempfile::tempdir;
    use tokio::fs::OpenOptions;
    use vortex_array::arrays::{ChunkedArray, StructArray, VarBinArray};
    use vortex_array::stream::ArrayStreamArrayExt;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray};
    use vortex_buffer::buffer;
    use vortex_error::vortex_err;
    use vortex_file::VortexWriteOptions;

    use crate::persistent::VortexFormat;

    #[tokio::test]
    async fn query_file() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let strings = ChunkedArray::from_iter([
            VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
            VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        ])
        .into_array();

        let numbers = ChunkedArray::from_iter([
            buffer![1u32, 2, 3, 4].into_array(),
            buffer![5u32, 6, 7, 8].into_array(),
        ])
        .into_array();

        let st = StructArray::try_new(
            ["strings".into(), "numbers".into()].into(),
            vec![strings, numbers],
            8,
            Validity::NonNullable,
        )?;

        let filepath = temp_dir.path().join("data.vortex");

        let f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&filepath)
            .await?;

        VortexWriteOptions::default()
            .write(f, st.to_array_stream())
            .await?;

        let ctx = SessionContext::default();
        let format = Arc::new(VortexFormat::default());
        let table_url = ListingTableUrl::parse(
            temp_dir
                .path()
                .to_str()
                .ok_or_else(|| vortex_err!("Path is not valid UTF-8"))?,
        )?;
        assert!(table_url.is_collection());

        let config = ListingTableConfig::new(table_url)
            .with_listing_options(ListingOptions::new(format))
            .infer_schema(&ctx.state())
            .await?;

        let listing_table = Arc::new(ListingTable::try_new(config)?);

        ctx.register_table("vortex_tbl", listing_table as _)?;
        let row_count = ctx.table("vortex_tbl").await?.count().await?;
        assert_eq!(row_count, 8);

        Ok(())
    }
}
